"""Local sequence classification, independent pairs in length-sorted microbatches."""

from dataclasses import dataclass
import math
from typing import Protocol

from .config import Settings
from .errors import InputTooLong


@dataclass(frozen=True)
class NLIProbabilities:
    contradiction: float
    entailment: float
    neutral: float

    def __post_init__(self):
        values = (self.contradiction, self.entailment, self.neutral)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid NLI probability")
        if not math.isclose(sum(values), 1, abs_tol=1e-4):
            raise ValueError("NLI probabilities must sum to one")


class NLIBackend(Protocol):
    ready: bool

    def load(self) -> None: ...

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIProbabilities]: ...


class TransformersNLI:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.ready = False

    def load(self) -> None:
        if self.ready:
            return
        import torch
        import transformers
        from packaging.version import Version
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

        if Version(transformers.__version__) < Version("5.17.0"):
            raise RuntimeError("Qwen3.5 NLI requires transformers>=5.17.0; install requirements-model.txt")
        settings = self.settings
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but no CUDA device is available")
        if settings.quantization != "none" and device != "cuda":
            raise RuntimeError("Quantization requires CUDA in this service")
        dtype = settings.dtype
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else (
                "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            )
        if device == "cpu" and dtype != "float32":
            raise RuntimeError("Use DTYPE=float32 or auto for the supported CPU path")
        if device == "cuda" and dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16; use DTYPE=auto")

        source = settings.model_path or settings.model_id
        kwargs = {"local_files_only": settings.local_files_only, "trust_remote_code": False}
        if not settings.model_path:
            kwargs.update(subfolder=settings.model_subfolder, revision=settings.model_revision)
        config = AutoConfig.from_pretrained(source, **kwargs)
        labels = {str(label).lower(): int(index) for index, label in config.id2label.items()}
        if set(labels) != {"contradiction", "entailment", "neutral"} or set(labels.values()) != {0, 1, 2}:
            raise RuntimeError("Checkpoint must have the three named NLI labels, not a generative LM head")
        tokenizer = AutoTokenizer.from_pretrained(source, **kwargs)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        # The tokenizer and checkpoint may declare different PAD ids. Keep both
        # top-level and text configs aligned for last-non-padding-token pooling.
        config.pad_token_id = tokenizer.pad_token_id
        config.get_text_config().pad_token_id = tokenizer.pad_token_id
        config.use_cache = False
        config.get_text_config().use_cache = False
        model_kwargs = {"config": config, "dtype": getattr(torch, dtype), "attn_implementation": "sdpa"}
        if settings.quantization != "none":
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=settings.quantization == "4bit",
                load_in_8bit=settings.quantization == "8bit",
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=getattr(torch, dtype),
                llm_int8_skip_modules=["score"],
            )
            model_kwargs["device_map"] = {"": 0}
        model, loading_info = AutoModelForSequenceClassification.from_pretrained(
            source, **kwargs, **model_kwargs, output_loading_info=True,
        )
        if loading_info.get("missing_keys") or loading_info.get("mismatched_keys"):
            raise RuntimeError("Checkpoint has missing/mismatched weights; refusing an untrained classification head")
        if settings.quantization == "none":
            model.to(device)
        model.eval()
        template = getattr(config, "nli_template", "Premise: {premise}\nHypothesis: {hypothesis}")
        if "{premise}" not in template or "{hypothesis}" not in template:
            raise RuntimeError("Invalid NLI template in checkpoint")
        self.torch, self.device = torch, device
        self.model, self.tokenizer = model, tokenizer
        self.template, self.labels = template, labels
        self.ready = True

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIProbabilities]:
        if not pairs:
            return []
        self.load()
        texts = [self.template.format(premise=p, hypothesis=h) for p, h in pairs]
        encoded = self.tokenizer(texts, padding=False, truncation=False, return_attention_mask=True)
        lengths = [len(ids) for ids in encoded["input_ids"]]
        if max(lengths) > self.settings.max_length:
            raise InputTooLong(self.settings.max_length)
        order = sorted(range(len(texts)), key=lengths.__getitem__)
        results = [None] * len(texts)
        with self.torch.inference_mode():
            for start in range(0, len(order), self.settings.batch_size):
                indices = order[start:start + self.settings.batch_size]
                batch = self.tokenizer.pad(
                    [{"input_ids": encoded["input_ids"][i], "attention_mask": encoded["attention_mask"][i]}
                     for i in indices],
                    padding=True, return_tensors="pt",
                )
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch, use_cache=False).logits
                if tuple(logits.shape) != (len(indices), 3):
                    raise RuntimeError("Invalid classification output shape")
                probabilities = self.torch.softmax(logits.float(), dim=-1).cpu().tolist()
                for index, row in zip(indices, probabilities, strict=True):
                    results[index] = NLIProbabilities(**{label: row[i] for label, i in self.labels.items()})
        return results
