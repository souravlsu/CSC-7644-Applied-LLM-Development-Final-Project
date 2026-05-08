"""Central configuration resolved from environment variables with sensible defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from dotenv import load_dotenv

load_dotenv()
# Use values from `.env` even when the shell / Windows User env defines the same keys
# (e.g. stray CHUNK_* from an old session). Explicit CLI overrides still apply in scripts.
load_dotenv(override=True)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
INDEX_DIR = PROJECT_ROOT / "index"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
EXAMPLES_DIR = DATA_DIR / "examples"

MANUAL_PDF = PROJECT_ROOT / "Ansys_Fluent_UDF_Manual.pdf"


def _env(key: str, default: str) -> str:
    value = os.getenv(key)
    return cast(str, value) if value not in (None, "") else default


def _env_int(key: str, default: int) -> int:
    try:
        return int(_env(key, str(default)))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).strip().lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Config:
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    llm_temperature: float
    llm_max_tokens: int

    embed_model: str
    embed_device: str
    embed_batch_size: int

    top_k: int
    use_hybrid: bool
    chunk_tokens: int
    chunk_overlap: int


def load_config() -> Config:
    return Config(
        llm_base_url=_env("LLM_BASE_URL", "http://localhost:11434/v1"),
        llm_api_key=_env("LLM_API_KEY", "ollama"),
        llm_model=_env("LLM_MODEL", "llama3.2:3b"),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.2),
        llm_max_tokens=_env_int("LLM_MAX_TOKENS", 900),
        embed_model=_env("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        embed_device=_env("EMBED_DEVICE", "auto"),
        embed_batch_size=_env_int("EMBED_BATCH_SIZE", 64),
        top_k=_env_int("TOP_K", 5),
        use_hybrid=_env_bool("USE_HYBRID", True),
        chunk_tokens=_env_int("CHUNK_TOKENS", 200),
        chunk_overlap=_env_int("CHUNK_OVERLAP", 80),
    )


CONFIG = load_config()
