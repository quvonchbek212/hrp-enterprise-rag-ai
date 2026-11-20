"""Configuration helpers for the HR chatbot module."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class HRChatbotConfig:
    """Runtime settings for document indexing and LLM/Ollama requests."""

    manuals_dir: Path
    vector_store_path: Path
    # Fallback Ollama host (used for embeddings when embedding_base_url is empty).
    ollama_host: str
    # OpenAI-compatible (vLLM / Ollama / OpenAI) server: response generation and embeddings.
    chat_base_url: str
    chat_api_key: str
    embedding_base_url: str
    chat_model: str
    embedding_model: str
    top_k: int
    min_relevance_score: float
    # Manuals are in Uzbek: Russian or English questions naturally have lower cosine scores,
    # hence a lower threshold is used for cross-lingual queries.
    cross_lingual_min_relevance_score: float
    chunk_size: int
    chunk_overlap: int
    timeout_seconds: float
    num_ctx: int
    num_predict: int
    temperature: float


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _derive_ollama_host() -> str:
    explicit = os.getenv("HR_CHATBOT_OLLAMA_HOST") or os.getenv("OLLAMA_HOST")
    if explicit:
        return explicit.rstrip("/")

    qwen_endpoint = os.getenv("QWEN_ENDPOINT", "")
    if qwen_endpoint:
        parsed = urlparse(qwen_endpoint)
        if parsed.scheme and parsed.netloc:
            return urlunparse(parsed._replace(path="", params="", query="", fragment="")).rstrip("/")
    return "http://127.0.0.1:11434"


def get_hr_chatbot_config() -> HRChatbotConfig:
    """Build module config from environment variables with safe defaults."""

    manuals_dir = Path(os.getenv("HR_CHATBOT_MANUALS_DIR", str(PROJECT_ROOT / "manuals"))).expanduser()
    vector_store_path = Path(
        os.getenv("HR_CHATBOT_VECTOR_STORE_PATH", str(PROJECT_ROOT / "storage" / "vector_store.json"))
    ).expanduser()

    chat_base = (
        os.getenv("HR_CHATBOT_CHAT_BASE_URL")
        or os.getenv("CHAT_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or ""
    ).rstrip("/")
    chat_key = (
        os.getenv("HR_CHATBOT_CHAT_API_KEY")
        or os.getenv("CHAT_API_KEY")
        or os.getenv("VLLM_API_KEY")
        or ""
    )
    embed_base = (
        os.getenv("HR_CHATBOT_EMBEDDING_BASE_URL")
        or os.getenv("EMBEDDING_BASE_URL")
        or os.getenv("VLLM_BASE_URL")
        or ""
    ).rstrip("/")
    chat_mod = (
        os.getenv("HR_CHATBOT_CHAT_MODEL")
        or os.getenv("CHAT_MODEL")
        or os.getenv("VLLM_MODEL")
        or "qwen2.5:14b-instruct"
    )
    embed_mod = os.getenv("HR_CHATBOT_EMBEDDING_MODEL") or os.getenv("EMBEDDING_MODEL") or "bge-m3"

    return HRChatbotConfig(
        manuals_dir=manuals_dir,
        vector_store_path=vector_store_path,
        ollama_host=_derive_ollama_host(),
        chat_base_url=chat_base,
        chat_api_key=chat_key,
        embedding_base_url=embed_base,
        chat_model=chat_mod,
        embedding_model=embed_mod,
        top_k=max(1, _int_env("HR_CHATBOT_TOP_K", 5)),
        min_relevance_score=min(1.0, max(0.0, _float_env("HR_CHATBOT_MIN_RELEVANCE_SCORE", 0.58))),
        cross_lingual_min_relevance_score=min(
            1.0,
            max(0.0, _float_env("HR_CHATBOT_CROSS_LINGUAL_MIN_RELEVANCE_SCORE", 0.46)),
        ),
        chunk_size=max(600, _int_env("HR_CHATBOT_CHUNK_SIZE", 1200)),
        chunk_overlap=max(0, _int_env("HR_CHATBOT_CHUNK_OVERLAP", 180)),
        timeout_seconds=max(10.0, _float_env("HR_CHATBOT_TIMEOUT_SECONDS", 120.0)),
        num_ctx=max(2048, _int_env("HR_CHATBOT_NUM_CTX", 8192)),
        num_predict=max(256, _int_env("HR_CHATBOT_NUM_PREDICT", 1024)),
        temperature=max(0.0, _float_env("HR_CHATBOT_TEMPERATURE", 0.0)),
    )
