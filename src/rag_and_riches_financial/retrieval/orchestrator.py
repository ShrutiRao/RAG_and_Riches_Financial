from __future__ import annotations

from hashlib import sha256
import re
from typing import Any

from rag_and_riches_financial.chunking.fixed_size import chunk_fixed_size
from rag_and_riches_financial.chunking.semantic import chunk_semantic
from rag_and_riches_financial.config import AppConfig
from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.embeddings import get_embedder
from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.retrieval.nebius_client import NebiusChatClient, rerank_candidates, rewrite_query
from rag_and_riches_financial.vectorstore.pinecone_index import PineconeIndex


_DEFAULT_INDEX_CACHE: dict[tuple[str | None, str, str | None, str], PineconeIndex] = {}
_INGESTED_CORPUS_KEYS: set[tuple[tuple[Any, ...], str]] = set()


def _chunk_document(document, chunking_strategy: str) -> list[ChunkRecord]:
    cleaned_text = document.text
    if chunking_strategy == "hybrid":
        return [
            *chunk_fixed_size(document, cleaned_text, chunk_size=40, overlap=10),
            *chunk_semantic(document, cleaned_text),
        ]
    if chunking_strategy == "semantic":
        return chunk_semantic(document, cleaned_text)
    return chunk_fixed_size(document, cleaned_text, chunk_size=40, overlap=10)


def _search_chunks(index: PineconeIndex, query: str, namespace: str, top_k: int) -> list[ChunkRecord]:
    return index.search(query, namespace=namespace, top_k=top_k)


def _extract_query_identifiers(query: str) -> list[str]:
    return [match.upper() for match in re.findall(r"\b[A-Z]{2,}-\d{4}-\d{4}\b", query.upper())]


def _chunk_search_blob(chunk: ChunkRecord) -> str:
    metadata_parts = []
    for value in chunk.metadata.values():
        if isinstance(value, str):
            metadata_parts.append(value)
        elif isinstance(value, list):
            metadata_parts.extend(str(item) for item in value)
    return " ".join([chunk.chunk_id, chunk.doc_id, chunk.section, chunk.text, *metadata_parts]).lower()


def _filter_chunks_by_identifiers(chunks: list[ChunkRecord], identifiers: list[str]) -> list[ChunkRecord]:
    if not identifiers:
        return chunks

    filtered = []
    for chunk in chunks:
        blob = _chunk_search_blob(chunk)
        if any(identifier.lower() in blob for identifier in identifiers):
            filtered.append(chunk)
    return filtered


def _collect_exact_identifier_chunks(corpus, chunking_strategy: str, identifiers: list[str]) -> list[ChunkRecord]:
    if not identifiers:
        return []

    strategies = ("fixed", "semantic") if chunking_strategy == "hybrid" else (chunking_strategy,)

    matched: list[ChunkRecord] = []
    for document in corpus:
        for strategy in strategies:
            for chunk in _chunk_document(document, strategy):
                if any(identifier.lower() in _chunk_search_blob(chunk) for identifier in identifiers):
                    matched.append(chunk)
    return matched


def _default_index_cache_key(config: AppConfig, embedding_provider: str) -> tuple[str | None, str, str | None, str]:
    return (
        config.pinecone_api_key,
        config.pinecone_index_name,
        config.pinecone_index_host,
        embedding_provider,
    )


def _get_default_index(config: AppConfig, embedding_provider: str) -> tuple[PineconeIndex, tuple[str | None, str, str | None, str]]:
    cache_key = _default_index_cache_key(config, embedding_provider)
    index = _DEFAULT_INDEX_CACHE.get(cache_key)
    if index is None:
        index = PineconeIndex(
            api_key=config.pinecone_api_key,
            index_name=config.pinecone_index_name,
            index_host=config.pinecone_index_host,
            embedding_provider=embedding_provider,
            embedder=get_embedder(embedding_provider),
        )
        _DEFAULT_INDEX_CACHE[cache_key] = index
    return index, cache_key


def _corpus_signature(corpus, chunking_strategy: str) -> str:
    digest = sha256()
    digest.update(chunking_strategy.encode("utf-8"))
    for document in corpus:
        digest.update(document.doc_id.encode("utf-8"))
        digest.update(document.doc_type.encode("utf-8"))
        digest.update(document.source_name.encode("utf-8"))
        digest.update(document.company.encode("utf-8"))
        digest.update(document.date.encode("utf-8"))
        digest.update(document.section.encode("utf-8"))
        digest.update(document.title.encode("utf-8"))
        digest.update(document.text.encode("utf-8"))
        for tag in document.tags:
            digest.update(tag.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _ingest_corpus(index: PineconeIndex, corpus, chunking_strategy: str, cache_key: tuple[Any, ...]) -> None:
    ingestion_key = (cache_key, _corpus_signature(corpus, chunking_strategy))
    if ingestion_key in _INGESTED_CORPUS_KEYS:
        return

    for document in corpus:
        for chunk in _chunk_document(document, chunking_strategy):
            index.upsert_chunk(chunk)

    _INGESTED_CORPUS_KEYS.add(ingestion_key)


def _auto_ingest_local_corpus(index: PineconeIndex, corpus, chunking_strategy: str, cache_key: tuple[Any, ...]) -> None:
    resolve_index = getattr(index, "_resolve_index", None)
    if not callable(resolve_index):
        return

    if resolve_index() is not None:
        return

    store = getattr(index, "_store", None)
    if not isinstance(store, dict):
        return

    strategies_to_ingest: list[str] = []
    if chunking_strategy == "hybrid":
        if not store.get("fixed"):
            strategies_to_ingest.append("fixed")
        if not store.get("semantic"):
            strategies_to_ingest.append("semantic")
    else:
        namespace = "fixed" if chunking_strategy == "fixed" else "semantic"
        if not store.get(namespace):
            strategies_to_ingest.append(chunking_strategy)

    for strategy in strategies_to_ingest:
        _ingest_corpus(index, corpus, strategy, cache_key)


def _combine_unique_chunks(*candidate_lists: list[ChunkRecord]) -> list[ChunkRecord]:
    combined: list[ChunkRecord] = []
    seen_ids: set[str] = set()
    for candidates in candidate_lists:
        for chunk in candidates:
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            combined.append(chunk)
    return combined


def _normalize_rrf_text(text: str) -> str:
    return " ".join(text.lower().split())


def _rrf_candidate_key(chunk: ChunkRecord) -> tuple[str, str, str, str]:
    return (
        chunk.doc_id,
        chunk.section,
        _normalize_rrf_text(chunk.text),
        chunk.metadata.get("source_name") or "",
    )


def _reciprocal_rank_fusion(*candidate_lists: list[ChunkRecord], rank_constant: int = 60) -> list[ChunkRecord]:
    fused_scores: dict[tuple[str, str, str, str], float] = {}
    first_seen: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    representatives: dict[tuple[str, str, str, str], ChunkRecord] = {}

    for list_index, candidates in enumerate(candidate_lists):
        for rank_index, chunk in enumerate(candidates, start=1):
            key = _rrf_candidate_key(chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (rank_constant + rank_index))
            first_seen.setdefault(key, (list_index, rank_index))
            representatives.setdefault(key, chunk)

    ordered_keys = sorted(
        representatives,
        key=lambda key: (-fused_scores[key], first_seen[key][0], first_seen[key][1], representatives[key].chunk_id),
    )
    return [representatives[key] for key in ordered_keys]


def retrieve_context(
    query: str,
    corpus=None,
    chunking_strategy: str = "fixed",
    retrieval_mode: str = "rewrite",
    allow_rerank: bool = True,
    embedding_provider: str = "openai",
    top_k: int | None = None,
    index: PineconeIndex | None = None,
    llm_client: NebiusChatClient | None = None,
) -> list[ChunkRecord]:
    corpus = corpus or build_sample_corpus()
    config = AppConfig()
    if index is None:
        index, cache_key = _get_default_index(config, embedding_provider)
    else:
        cache_key = ("provided", id(index), embedding_provider)
    llm_client = llm_client or NebiusChatClient.from_env()
    query_identifiers = _extract_query_identifiers(query)

    _auto_ingest_local_corpus(index, corpus, chunking_strategy, cache_key)

    exact_matches = _collect_exact_identifier_chunks(corpus, chunking_strategy, query_identifiers)
    if exact_matches:
        if retrieval_mode == "rerank" and allow_rerank:
            reranked_texts = rerank_candidates(query, [chunk.text for chunk in exact_matches], client=llm_client)
            reranked: list[ChunkRecord] = []
            for text in reranked_texts:
                reranked.extend([chunk for chunk in exact_matches if chunk.text == text])
            return reranked
        return exact_matches

    search_query = query if query_identifiers else (rewrite_query(query, client=llm_client) if retrieval_mode in {"rewrite", "rerank"} else query)
    effective_top_k = top_k if top_k is not None else (5 if retrieval_mode == "rerank" and allow_rerank else 3)
    if chunking_strategy == "hybrid":
        fixed_candidates = _search_chunks(index, search_query, namespace="fixed", top_k=effective_top_k)
        semantic_candidates = _search_chunks(index, search_query, namespace="semantic", top_k=effective_top_k)
        candidates = _reciprocal_rank_fusion(fixed_candidates, semantic_candidates)
    else:
        namespace = "fixed" if chunking_strategy == "fixed" else "semantic"
        candidates = _search_chunks(index, search_query, namespace=namespace, top_k=effective_top_k)

    candidates = _filter_chunks_by_identifiers(candidates, query_identifiers)

    if retrieval_mode == "rerank" and allow_rerank:
        reranked_texts = rerank_candidates(query, [chunk.text for chunk in candidates], client=llm_client)
        reranked: list[ChunkRecord] = []
        for text in reranked_texts:
            reranked.extend([chunk for chunk in candidates if chunk.text == text])
        return reranked
    return candidates
