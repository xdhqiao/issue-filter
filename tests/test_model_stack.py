"""No downloads: exercise actual Qwen3.5 classes with tiny random local weights."""

from dataclasses import replace
import importlib.util
import os

import pytest

from issue_filter.config import Settings
from issue_filter.errors import InputTooLong
from issue_filter.nli import TransformersNLI


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers", minversion="5.17.0")
    from tokenizers import Tokenizer, models, pre_tokenizers

    path = tmp_path_factory.mktemp("tiny-qwen-nli")
    tokenizer = Tokenizer(models.WordLevel({
        "[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "Premise": 3, "Hypothesis": 4,
        ":": 5, "code": 6, "fails": 7, "build": 8, "works": 9,
    }, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tokenizer, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]",
    )
    fast.save_pretrained(path)
    config = transformers.Qwen3_5Config(
        text_config={
            "vocab_size": 64, "hidden_size": 64, "intermediate_size": 128,
            "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 16, "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 2, "linear_num_value_heads": 4,
            "linear_key_head_dim": 16, "linear_value_head_dim": 16,
            "max_position_embeddings": 128, "pad_token_id": 63,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000,
                                "partial_rotary_factor": .5, "mrope_section": [2, 1, 1]},
        },
        vision_config={
            "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
            "out_hidden_size": 64, "num_position_embeddings": 16, "patch_size": 2,
            "spatial_merge_size": 2, "deepstack_visual_indexes": [],
        },
        # Deliberately shuffle label order to catch hard-coded class indices.
        id2label={0: "neutral", 1: "contradiction", 2: "entailment"},
        label2id={"neutral": 0, "contradiction": 1, "entailment": 2},
        pad_token_id=63, image_token_id=60, video_token_id=61,
        vision_start_token_id=62, vision_end_token_id=59,
        nli_template="Premise: {premise}\nHypothesis: {hypothesis}",
    )
    transformers.Qwen3_5ForSequenceClassification(config).save_pretrained(path)
    return path


@pytest.mark.model_stack
def test_real_transformers_loading_padding_batching_and_label_mapping(tiny_checkpoint):
    backend = TransformersNLI(Settings(model_path=str(tiny_checkpoint), local_files_only=True,
                                      device="cpu", batch_size=2, max_length=128))
    pairs = [("code fails build", "build fails"), ("code", "code"), ("code works", "build fails")]
    together = backend.predict(pairs)
    alone = [backend.predict([pair])[0] for pair in pairs]
    assert backend.ready
    assert backend.model.config.pad_token_id == backend.tokenizer.pad_token_id
    assert backend.model.config.get_text_config().pad_token_id == backend.tokenizer.pad_token_id
    assert backend.labels == {"neutral": 0, "contradiction": 1, "entailment": 2}
    for batch_result, single_result in zip(together, alone, strict=True):
        assert batch_result.entailment == pytest.approx(single_result.entailment, abs=1e-5)
        assert batch_result.contradiction == pytest.approx(single_result.contradiction, abs=1e-5)
    backend.settings = replace(backend.settings, max_length=4)
    with pytest.raises(InputTooLong):
        backend.predict(pairs)
    assert backend.predict([]) == []


@pytest.mark.real_model
@pytest.mark.skipif(not os.getenv("RUN_REAL_MODEL_TESTS"), reason="Set RUN_REAL_MODEL_TESTS=1 and MODEL_PATH explicitly")
def test_downloaded_checkpoint_smoke():
    if not os.getenv("MODEL_PATH"):
        pytest.fail("Set MODEL_PATH to an existing local checkpoint; this test must not download weights")
    assert importlib.util.find_spec("torch")
    settings = replace(Settings.from_env(), local_files_only=True)
    backend = TransformersNLI(settings)
    scores = backend.predict([("编译不通过", "编译不通过"), ("编译可以通过", "编译不通过")])
    assert len(scores) == 2
    assert scores[0].entailment > scores[0].contradiction
    assert scores[1].contradiction > scores[1].entailment
