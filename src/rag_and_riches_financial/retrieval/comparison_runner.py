from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.generation.generator import generate_answer
from rag_and_riches_financial.retrieval.orchestrator import retrieve_context


@dataclass(frozen=True)
class ComparisonRow:
    embedding_provider: str
    top_k: int
    retrieved_count: int
    source_summary: str
    answer_summary: str
    label: str
    chunks: list


@dataclass(frozen=True)
class ComparisonResult:
    query: str
    rows: list[ComparisonRow]


@dataclass(frozen=True)
class RetrievalBenchmarkQuery:
    label: str
    query: str
    expected_doc_types: tuple[str, ...] = ()
    expected_source_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalBenchmarkRow:
    query_label: str
    query: str
    chunking_strategy: str
    reranked: bool
    top_k: int
    retrieved_count: int
    matched_count: int
    first_match_rank: int | None
    reciprocal_rank: float
    matched_sources: str
    label: str
    chunks: list


@dataclass(frozen=True)
class RetrievalBenchmarkSummaryRow:
    label: str
    chunking_strategy: str
    reranked: bool
    average_reciprocal_rank: float
    hit_rate: float
    improvement: float


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    queries: list[RetrievalBenchmarkQuery]
    rows: list[RetrievalBenchmarkRow]
    summary_rows: list[RetrievalBenchmarkSummaryRow]


DEFAULT_RETRIEVAL_BENCHMARK_QUERIES: tuple[RetrievalBenchmarkQuery, ...] = (
    RetrievalBenchmarkQuery(
        label="Liquidity",
        query="What are the liquidity risks?",
        expected_doc_types=("sec_filing", "loan_document"),
        expected_source_names=("10-K", "Credit Agreement 001"),
    ),
    RetrievalBenchmarkQuery(
        label="Earnings",
        query="What do the earnings calls say about margins and guidance?",
        expected_doc_types=("earnings_transcript", "sec_filing"),
        expected_source_names=("Q1 Earnings Call", "Q2 Earnings Call", "Q3 Earnings Call", "Q4 Earnings Call"),
    ),
    RetrievalBenchmarkQuery(
        label="Claims",
        query="How are reserve movements and settlement timing changing in claims?",
        expected_doc_types=("insurance_claim",),
        expected_source_names=("Claim File 001", "Claim File 005", "Claim File 010"),
    ),
    RetrievalBenchmarkQuery(
        label="Loans",
        query="What covenant obligations should we watch in the loan documents?",
        expected_doc_types=("loan_document",),
        expected_source_names=("Credit Agreement 001", "Credit Agreement 005", "Credit Agreement 008"),
    ),
)


def _top_sources_summary(chunks) -> str:
    if not chunks:
        return "none"
    pairs = []
    for chunk in chunks[:3]:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        source_name = chunk.metadata.get("source_name", "unknown source")
        pairs.append(f"{doc_type} | {source_name}")
    return ", ".join(pairs)


def _answer_summary(answer: str) -> str:
    summary = answer.splitlines()[0].strip() if answer else ""
    return summary[:160]


def _provider_label(provider_name: str) -> str:
    provider = provider_name.lower().strip()
    if provider == "openai":
        return "OpenAI"
    return provider_name.upper()


def _source_label(chunk) -> str:
    doc_type = chunk.metadata.get("doc_type", "unknown")
    source_name = chunk.metadata.get("source_name", "unknown source")
    return f"{doc_type} | {source_name}"


def _matches_benchmark_expectation(chunk, query: RetrievalBenchmarkQuery) -> bool:
    doc_type = chunk.metadata.get("doc_type", "")
    source_name = chunk.metadata.get("source_name", "")
    if query.expected_doc_types and doc_type in query.expected_doc_types:
        return True
    if query.expected_source_names and source_name in query.expected_source_names:
        return True
    return False


def _score_benchmark_chunks(chunks, query: RetrievalBenchmarkQuery) -> tuple[int | None, float, int, str]:
    matched_sources: list[str] = []
    first_match_rank: int | None = None
    matched_count = 0

    for rank, chunk in enumerate(chunks, start=1):
        if not _matches_benchmark_expectation(chunk, query):
            continue
        matched_count += 1
        matched_sources.append(_source_label(chunk))
        if first_match_rank is None:
            first_match_rank = rank

    reciprocal_rank = 1.0 / first_match_rank if first_match_rank else 0.0
    return first_match_rank, reciprocal_rank, matched_count, ", ".join(matched_sources) if matched_sources else "none"


def _coerce_benchmark_query(query) -> RetrievalBenchmarkQuery:
    if isinstance(query, RetrievalBenchmarkQuery):
        return query
    return RetrievalBenchmarkQuery(
        label=query["label"],
        query=query["query"],
        expected_doc_types=tuple(query.get("expected_doc_types", ())),
        expected_source_names=tuple(query.get("expected_source_names", ())),
    )


def compare_chunking_and_rerank(
    queries: list[RetrievalBenchmarkQuery | dict] | None = None,
    corpus=None,
    top_k: int = 5,
) -> RetrievalBenchmarkResult:
    corpus = corpus or build_sample_corpus(include_pdf_docs=False)
    queries = [ _coerce_benchmark_query(query) for query in (queries or DEFAULT_RETRIEVAL_BENCHMARK_QUERIES) ]
    strategies = ("fixed", "semantic")

    rows: list[RetrievalBenchmarkRow] = []
    for query_item in queries:
        for chunking_strategy in strategies:
            for reranked in (False, True):
                retrieval_mode = "rerank" if reranked else "rewrite"
                chunks = retrieve_context(
                    query_item.query,
                    corpus=corpus,
                    chunking_strategy=chunking_strategy,
                    retrieval_mode=retrieval_mode,
                    allow_rerank=reranked,
                    top_k=top_k,
                )
                first_match_rank, reciprocal_rank, matched_count, matched_sources = _score_benchmark_chunks(chunks, query_item)
                rows.append(
                    RetrievalBenchmarkRow(
                        query_label=query_item.label,
                        query=query_item.query,
                        chunking_strategy=chunking_strategy,
                        reranked=reranked,
                        top_k=top_k,
                        retrieved_count=len(chunks),
                        matched_count=matched_count,
                        first_match_rank=first_match_rank,
                        reciprocal_rank=reciprocal_rank,
                        matched_sources=matched_sources,
                        label=f"{query_item.label} | {chunking_strategy.title()}{' + rerank' if reranked else ''}",
                        chunks=chunks,
                    )
                )

    summary_rows: list[RetrievalBenchmarkSummaryRow] = []
    for chunking_strategy in strategies:
        raw_rows = [row for row in rows if row.chunking_strategy == chunking_strategy and not row.reranked]
        reranked_rows = [row for row in rows if row.chunking_strategy == chunking_strategy and row.reranked]
        raw_rr = mean([row.reciprocal_rank for row in raw_rows]) if raw_rows else 0.0
        reranked_rr = mean([row.reciprocal_rank for row in reranked_rows]) if reranked_rows else 0.0
        raw_hit_rate = mean([1.0 if row.reciprocal_rank > 0 else 0.0 for row in raw_rows]) if raw_rows else 0.0
        reranked_hit_rate = mean([1.0 if row.reciprocal_rank > 0 else 0.0 for row in reranked_rows]) if reranked_rows else 0.0
        summary_rows.extend(
            [
                RetrievalBenchmarkSummaryRow(
                    label=f"{chunking_strategy} raw",
                    chunking_strategy=chunking_strategy,
                    reranked=False,
                    average_reciprocal_rank=raw_rr,
                    hit_rate=raw_hit_rate,
                    improvement=0.0,
                ),
                RetrievalBenchmarkSummaryRow(
                    label=f"{chunking_strategy} reranked",
                    chunking_strategy=chunking_strategy,
                    reranked=True,
                    average_reciprocal_rank=reranked_rr,
                    hit_rate=reranked_hit_rate,
                    improvement=reranked_rr - raw_rr,
                ),
            ]
        )

    return RetrievalBenchmarkResult(queries=queries, rows=rows, summary_rows=summary_rows)


def compare_embedding_and_topk(
    query: str,
    corpus=None,
    strategies: list[tuple[str, int]] | None = None,
    chunking_strategy: str = "fixed",
) -> ComparisonResult:
    corpus = corpus or build_sample_corpus()
    strategies = strategies or [("openai", 3), ("openai", 5), ("bge", 3), ("bge", 5)]

    rows: list[ComparisonRow] = []
    for embedding_provider, top_k in strategies:
        chunks = retrieve_context(
            query,
            corpus=corpus,
            chunking_strategy=chunking_strategy,
            retrieval_mode="direct",
            allow_rerank=False,
            embedding_provider=embedding_provider,
            top_k=top_k,
        )
        answer = generate_answer(query, chunks, retrieval_mode="direct", chunking_strategy=chunking_strategy)
        rows.append(
            ComparisonRow(
                embedding_provider=embedding_provider,
                top_k=top_k,
                retrieved_count=len(chunks),
                source_summary=_top_sources_summary(chunks),
                answer_summary=_answer_summary(answer),
                label=f"{_provider_label(embedding_provider)} | Top-K={top_k}",
                chunks=chunks,
            )
        )

    return ComparisonResult(query=query, rows=rows)
