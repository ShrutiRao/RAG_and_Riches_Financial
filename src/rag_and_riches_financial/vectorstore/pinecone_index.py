from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from rag_and_riches_financial.embeddings import get_embedder
from rag_and_riches_financial.models.chunks import ChunkRecord


def _local_score(query: str, text: str, embedder: Any) -> float:
    query_vector = embedder.embed_query(query)
    text_vector = embedder.embed_query(text)
    if not any(query_vector) or not any(text_vector):
        return 0.0
    return float(sum(q * t for q, t in zip(query_vector, text_vector)))


@dataclass
class PineconeIndex:
    index: Any | None = None
    index_name: str = "rag-and-riches-financial"
    index_host: str | None = None
    api_key: str | None = None
    embedding_provider: str = "openai"
    embedder: Any | None = None
    fixed_namespace: str = "fixed"
    semantic_namespace: str = "semantic"
    _store: dict[str, list[dict[str, Any]]] = field(default_factory=lambda: {"fixed": [], "semantic": []})

    def _resolve_index(self) -> Any | None:
        if self.index is not None:
            return self.index
        if not self.api_key:
            return None
        try:
            from pinecone import Pinecone
        except ImportError:
            return None
        pc = Pinecone(api_key=self.api_key)
        if self.index_host:
            return pc.Index(host=self.index_host)
        return pc.Index(name=self.index_name)

    def namespace_for(self, chunk: ChunkRecord) -> str:
        return self.fixed_namespace if chunk.chunking_strategy == "fixed" else self.semantic_namespace

    def upsert_chunk(self, chunk: ChunkRecord) -> None:
        namespace = self.namespace_for(chunk)
        record = {
            "_id": chunk.chunk_id,
            "text": chunk.text,
            "chunk_text": chunk.text,
            "doc_id": chunk.doc_id,
            "chunk_index": chunk.chunk_index,
            "chunking_strategy": chunk.chunking_strategy,
            "section": chunk.section,
            **chunk.metadata,
        }
        index = self._resolve_index()
        if index is None:
            self._store.setdefault(namespace, []).append(record)
            return

        self._upsert_records(index, namespace, [record])

    def _upsert_records(self, index: Any, namespace: str, records: list[dict[str, Any]]) -> Any:
        try:
            return index.upsert_records(namespace=namespace, records=records)
        except TypeError:
            # Backward-compatible fallback for older fakes/tests.
            return index.upsert_records(namespace, records)

    def search(self, query: str, namespace: str, top_k: int = 3) -> list[ChunkRecord]:
        index = self._resolve_index()
        if index is None:
            candidates = self._store.get(namespace, [])
            embedder = self.embedder or get_embedder(self.embedding_provider)
            scored = sorted(
                candidates,
                key=lambda item: (_local_score(query, item["chunk_text"], embedder), -len(item["chunk_text"])),
                reverse=True,
            )
            return [self._record_to_chunk(item) for item in scored[:top_k]]

        response = self._search_records(index, namespace, query, top_k)
        return self._extract_chunks(response)

    def _search_records(self, index: Any, namespace: str, query: str, top_k: int) -> Any:
        fields = [
            "text",
            "chunk_text",
            "doc_id",
            "chunk_index",
            "chunking_strategy",
            "section",
            "doc_type",
            "source_name",
            "date",
            "title",
            "tags",
        ]
        try:
            return index.search(namespace=namespace, top_k=top_k, inputs={"text": query}, fields=fields)
        except TypeError:
            # Backward-compatible fallback for older fakes/tests.
            return index.search(
                namespace,
                {"inputs": {"text": query}, "top_k": top_k},
                fields=fields,
            )

    def _record_to_chunk(self, record: dict[str, Any]) -> ChunkRecord:
        metadata = {
            "doc_type": record.get("doc_type"),
            "source_name": record.get("source_name"),
            "date": record.get("date"),
            "title": record.get("title"),
            "tags": record.get("tags", []),
            "chunking_strategy": record.get("chunking_strategy", "fixed"),
        }
        return ChunkRecord(
            chunk_id=record["_id"],
            doc_id=record.get("doc_id", ""),
            chunk_index=record.get("chunk_index", 0),
            chunking_strategy=record.get("chunking_strategy", "fixed"),
            section=record.get("section", ""),
            text=record.get("chunk_text") or record.get("text", ""),
            metadata=metadata,
        )

    def _extract_chunks(self, response: Any) -> list[ChunkRecord]:
        if isinstance(response, dict):
            hits = response.get("result", {}).get("hits", [])
        else:
            hits = getattr(getattr(response, "result", None), "hits", [])

        chunks: list[ChunkRecord] = []
        for hit in hits:
            fields = hit.get("fields", {}) if isinstance(hit, dict) else getattr(hit, "fields", {})
            chunks.append(
                ChunkRecord(
                    chunk_id=hit.get("_id") if isinstance(hit, dict) else getattr(hit, "_id", ""),
                    doc_id=fields.get("doc_id", ""),
                    chunk_index=fields.get("chunk_index", 0),
                    chunking_strategy=fields.get("chunking_strategy", "fixed"),
                    section=fields.get("section", ""),
                    text=fields.get("chunk_text") or fields.get("text", ""),
                    metadata={
                        "doc_type": fields.get("doc_type"),
                        "source_name": fields.get("source_name"),
                        "date": fields.get("date"),
                        "title": fields.get("title"),
                        "tags": fields.get("tags", []),
                        "chunking_strategy": fields.get("chunking_strategy", "fixed"),
                        "_score": hit.get("_score") if isinstance(hit, dict) else getattr(hit, "_score", None),
                    },
                )
            )
        return chunks

    def probe_connection(self) -> bool:
        index = self._resolve_index()
        if index is None:
            return False

        try:
            self._search_records(index, namespace=self.fixed_namespace, query="connectivity probe", top_k=1)
        except Exception:
            return False
        return True


@lru_cache(maxsize=32)
def probe_pinecone_connection(api_key: str | None, index_name: str, index_host: str | None) -> bool:
    return PineconeIndex(
        api_key=api_key,
        index_name=index_name,
        index_host=index_host,
    ).probe_connection()
