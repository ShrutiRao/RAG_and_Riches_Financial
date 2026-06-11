from __future__ import annotations

import os
from dataclasses import dataclass, field

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    def load_dotenv(*_args, **_kwargs):
        return False


def load_environment() -> bool:
    return load_dotenv()


load_environment()


@dataclass(frozen=True)
class AppConfig:
    pinecone_api_key: str | None = field(default_factory=lambda: os.getenv("PINECONE_API_KEY"))
    pinecone_index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "rag-and-riches-financial"))
    pinecone_index_host: str | None = field(default_factory=lambda: os.getenv("PINECONE_INDEX_HOST"))
    nebius_api_key: str | None = field(default_factory=lambda: os.getenv("NEBIUS_API_KEY") or os.getenv("NEBIUS_TOKEN_FACTORY_API_KEY"))
    nebius_base_url: str | None = field(default_factory=lambda: os.getenv("NEBIUS_BASE_URL"))
    nebius_model: str = field(default_factory=lambda: os.getenv("NEBIUS_MODEL", "Qwen/Qwen3-0.6B"))
    llamaparse_api_key: str | None = field(default_factory=lambda: os.getenv("LLAMA_CLOUD_API_KEY") or os.getenv("LLAMAPARSE_API_KEY"))
