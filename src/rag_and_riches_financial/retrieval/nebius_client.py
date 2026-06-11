from __future__ import annotations

import json
import os
from functools import lru_cache
from dataclasses import dataclass
from typing import Any


def _local_rewrite(query: str) -> str:
    return f"financially focused: {query}"


def _local_rerank(query: str, candidates: list[str]) -> list[str]:
    if not candidates:
        return []
    query_terms = set(query.lower().split())
    return sorted(
        candidates,
        key=lambda text: (len(query_terms.intersection(set(text.lower().split()))), -len(text)),
        reverse=True,
    )


@dataclass
class NebiusChatClient:
    client: Any | None = None
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None

    @classmethod
    def from_env(cls) -> "NebiusChatClient":
        return cls(
            api_key=os.getenv("NEBIUS_API_KEY") or os.getenv("NEBIUS_TOKEN_FACTORY_API_KEY"),
            base_url=os.getenv("NEBIUS_BASE_URL"),
            model=os.getenv("NEBIUS_MODEL", "Qwen/Qwen3-0.6B"),
        )

    def _resolve_client(self) -> Any | None:
        if self.client is not None:
            return self.client
        if not self.api_key or not self.base_url:
            return None
        try:
            from openai import OpenAI
        except ImportError:
            return None
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def rewrite_query(self, query: str) -> str:
        client = self._resolve_client()
        if client is None:
            return _local_rewrite(query)

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": "Rewrite the user query for financial document retrieval. Return only the rewritten query.",
                    },
                    {
                        "role": "user",
                        "content": query,
                    },
                ],
            )
        except Exception:
            return _local_rewrite(query)
        return response.choices[0].message.content.strip()

    def rerank_candidates(self, query: str, candidates: list[str]) -> list[str]:
        client = self._resolve_client()
        if client is None or not candidates:
            return _local_rerank(query, candidates)

        numbered_candidates = "\n".join(f"{i}. {text}" for i, text in enumerate(candidates, start=1))
        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Rank the candidate passages for answering the financial question. "
                            "Return only a JSON array containing the candidate texts in best-to-worst order."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Question:\n{query}\n\nCandidates:\n{numbered_candidates}",
                    },
                ],
            )
        except Exception:
            return _local_rerank(query, candidates)
        content = response.choices[0].message.content.strip()
        try:
            ranked_texts = json.loads(content)
        except json.JSONDecodeError:
            return _local_rerank(query, candidates)

        if not isinstance(ranked_texts, list):
            return _local_rerank(query, candidates)

        ordered: list[str] = []
        remaining = list(candidates)
        for candidate in ranked_texts:
            if candidate in remaining:
                ordered.append(candidate)
                remaining.remove(candidate)
        return ordered + remaining

    def probe_connection(self) -> bool:
        client = self._resolve_client()
        if client is None:
            return False

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=1,
                messages=[
                    {
                        "role": "system",
                        "content": "Reply with a single word confirming connectivity.",
                    },
                    {
                        "role": "user",
                        "content": "ping",
                    },
                ],
            )
        except Exception:
            return False

        choices = getattr(response, "choices", [])
        return bool(choices and getattr(choices[0], "message", None))


@lru_cache(maxsize=32)
def probe_nebius_connection(api_key: str | None, base_url: str | None, model: str | None) -> bool:
    return NebiusChatClient(api_key=api_key, base_url=base_url, model=model).probe_connection()


def rewrite_query(query: str, client: NebiusChatClient | None = None) -> str:
    return (client or NebiusChatClient.from_env()).rewrite_query(query)


def rerank_candidates(query: str, candidates: list[str], client: NebiusChatClient | None = None) -> list[str]:
    return (client or NebiusChatClient.from_env()).rerank_candidates(query, candidates)
