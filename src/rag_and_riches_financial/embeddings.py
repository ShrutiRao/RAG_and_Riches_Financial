from __future__ import annotations

import hashlib
from dataclasses import dataclass

def _zeros(dimension: int) -> list[float]:
    return [0.0] * dimension


def _normalize_vector(vector: list[float]) -> list[float]:
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0:
        return vector
    return [value / norm for value in vector]


@dataclass
class BaseEmbedder:
    provider_name: str
    dimension: int = 128

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]


class OpenAIEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 128):
        super().__init__(provider_name="openai", dimension=dimension)

    def embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vector = _zeros(self.dimension)
        for i, token in enumerate(tokens[: self.dimension]):
            vector[i] = len(token)
        return vector


class BGEEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 128):
        super().__init__(provider_name="bge", dimension=dimension)

    def _token_index_and_weight(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % self.dimension
        weight = 1.0 + (digest[4] / 255.0)
        if digest[5] % 2:
            weight *= -1.0
        return index, weight

    def embed(self, text: str) -> list[float]:
        tokens = text.lower().split()
        vector = _zeros(self.dimension)
        for token in tokens:
            index, weight = self._token_index_and_weight(token)
            vector[index] += weight
        return _normalize_vector(vector)


class FinancialEmbedder(OpenAIEmbedder):
    """Backwards-compatible deterministic embedder used by existing tests."""


def get_embedder(provider_name: str = "openai", dimension: int = 128) -> BaseEmbedder:
    provider = provider_name.lower().strip()
    if provider == "bge":
        return BGEEmbedder(dimension=dimension)
    return OpenAIEmbedder(dimension=dimension)
