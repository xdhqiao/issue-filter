"""Validated configuration; importing the API does not import torch."""

from dataclasses import dataclass
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name, str(default)).lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError(f"{name} must be true/false or 1/0")
    return value in {"true", "1"}


@dataclass(frozen=True)
class Settings:
    history_path: Path = ROOT / "data/historical_issues.json"
    model_id: str = "AlexWortega/openjev"
    model_subfolder: str = "qwen3.5-4b-nli-v5"
    model_revision: str = "5df9876e03397de71504a8f156509e6529c263d8"
    model_path: str = ""
    local_files_only: bool = False
    device: str = "auto"
    dtype: str = "auto"
    quantization: str = "none"
    batch_size: int = 4
    max_length: int = 2048
    max_description_chars: int = 8000
    max_candidates: int = 1000
    max_content_length: int = 4 * 1024 * 1024
    threshold: float = 0.85
    max_contradiction: float = 0.10
    match_mode: str = "directional"
    keyword_mode: str = "any"
    keyword_case_sensitive: bool = True
    inference_lock_timeout: float = 1.0
    preload_model: bool = False
    api_key: str = ""

    def __post_init__(self):
        for name, choices in {
            "device": {"auto", "cpu", "cuda"},
            "dtype": {"auto", "float32", "float16", "bfloat16"},
            "quantization": {"none", "4bit", "8bit"},
            "match_mode": {"directional", "bidirectional"},
            "keyword_mode": {"any", "all"},
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")
        for name in ("batch_size", "max_length", "max_description_chars", "max_candidates", "max_content_length"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("threshold", "max_contradiction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if not math.isfinite(self.inference_lock_timeout) or self.inference_lock_timeout < 0:
            raise ValueError("inference_lock_timeout must be finite and nonnegative")
        if not self.model_path and (not self.model_id or not self.model_revision):
            raise ValueError("model_id and model_revision are required without model_path")
        if self.device == "cpu" and self.quantization != "none":
            raise ValueError("This service supports bitsandbytes quantization only on CUDA")

    @classmethod
    def from_env(cls):
        defaults = cls()
        values = {}
        for name, field in cls.__dataclass_fields__.items():
            key = name.upper()
            default = getattr(defaults, name)
            if isinstance(default, bool):
                values[name] = env_bool(key, default)
            elif key in os.environ:
                values[name] = type(default)(os.environ[key])
        return cls(**values)
