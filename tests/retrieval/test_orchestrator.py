from rag_and_riches_financial.retrieval import orchestrator
from rag_and_riches_financial.retrieval.orchestrator import retrieve_context
from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.models.documents import FinancialDocument


def test_retrieve_context_returns_strategy_tagged_candidates():
    corpus = build_sample_corpus()
    candidates = retrieve_context("What are the liquidity risks?", corpus, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=False)

    assert candidates
    assert all("chunking_strategy" in item.metadata for item in candidates)


def test_retrieve_context_can_skip_rerank_even_when_rerank_mode_is_selected():
    corpus = build_sample_corpus()
    candidates = retrieve_context("What are the liquidity risks?", corpus, chunking_strategy="fixed", retrieval_mode="rerank", allow_rerank=False)

    assert candidates
    assert all(chunk.chunking_strategy == "fixed" for chunk in candidates)


def test_retrieve_context_hybrid_combines_fixed_and_semantic_candidates(monkeypatch):
    corpus = build_sample_corpus()

    class FakeIndex:
        def __init__(self, *args, **kwargs):
            self.upsert_calls = 0

        def upsert_chunk(self, chunk):
            self.upsert_calls += 1

        def search(self, query, namespace, top_k):
            if namespace == "fixed":
                return [
                    orchestrator.ChunkRecord(
                        chunk_id="sec-001-fixed-0",
                        doc_id="sec-001",
                        chunk_index=0,
                        chunking_strategy="fixed",
                        section="Risk Factors",
                        text="Fixed liquidity evidence.",
                        metadata={"doc_type": "sec_filing", "source_name": "10-K", "date": "2025-01-01"},
                    )
                ]
            return [
                orchestrator.ChunkRecord(
                    chunk_id="sec-001-semantic-0",
                    doc_id="sec-001",
                    chunk_index=0,
                    chunking_strategy="semantic",
                    section="Risk Factors",
                    text="Semantic liquidity evidence.",
                    metadata={"doc_type": "sec_filing", "source_name": "10-K", "date": "2025-01-01"},
                )
            ]

    monkeypatch.setattr(orchestrator, "PineconeIndex", FakeIndex)
    monkeypatch.setattr(orchestrator, "_DEFAULT_INDEX_CACHE", {})
    monkeypatch.setattr(orchestrator, "_INGESTED_CORPUS_KEYS", set())

    candidates = retrieve_context("What are the liquidity risks?", corpus, chunking_strategy="hybrid", retrieval_mode="direct", allow_rerank=False)

    assert candidates
    assert any(chunk.chunking_strategy == "fixed" for chunk in candidates)
    assert any(chunk.chunking_strategy == "semantic" for chunk in candidates)
    assert any(chunk.text == "Fixed liquidity evidence." for chunk in candidates)
    assert any(chunk.text == "Semantic liquidity evidence." for chunk in candidates)


def test_retrieve_context_hybrid_uses_rrf_to_promote_shared_evidence(monkeypatch):
    corpus = build_sample_corpus()

    class FakeIndex:
        def __init__(self, *args, **kwargs):
            self.upsert_calls = 0

        def upsert_chunk(self, chunk):
            self.upsert_calls += 1

        def search(self, query, namespace, top_k):
            if namespace == "fixed":
                return [
                    orchestrator.ChunkRecord(
                        chunk_id="fixed-1",
                        doc_id="sec-001",
                        chunk_index=0,
                        chunking_strategy="fixed",
                        section="Risk Factors",
                        text="Shared liquidity evidence.",
                        metadata={"doc_type": "sec_filing", "source_name": "10-K", "date": "2025-01-01"},
                    ),
                    orchestrator.ChunkRecord(
                        chunk_id="fixed-2",
                        doc_id="loan-001",
                        chunk_index=0,
                        chunking_strategy="fixed",
                        section="Covenants",
                        text="Fixed-only covenant evidence.",
                        metadata={"doc_type": "loan_document", "source_name": "Credit Agreement 001", "date": "2025-01-01"},
                    ),
                ]
            return [
                orchestrator.ChunkRecord(
                    chunk_id="semantic-1",
                    doc_id="sec-001",
                    chunk_index=0,
                    chunking_strategy="semantic",
                    section="Risk Factors",
                    text="Shared liquidity evidence.",
                    metadata={"doc_type": "sec_filing", "source_name": "10-K", "date": "2025-01-01"},
                ),
                orchestrator.ChunkRecord(
                    chunk_id="semantic-2",
                    doc_id="earn-001",
                    chunk_index=0,
                    chunking_strategy="semantic",
                    section="Prepared Remarks",
                    text="Semantic-only earnings evidence.",
                    metadata={"doc_type": "earnings_transcript", "source_name": "Q1 Earnings Call", "date": "2025-01-01"},
                ),
            ]

    monkeypatch.setattr(orchestrator, "PineconeIndex", FakeIndex)
    monkeypatch.setattr(orchestrator, "_DEFAULT_INDEX_CACHE", {})
    monkeypatch.setattr(orchestrator, "_INGESTED_CORPUS_KEYS", set())

    candidates = retrieve_context("What are the liquidity risks?", corpus, chunking_strategy="hybrid", retrieval_mode="direct", allow_rerank=False)

    assert candidates
    assert candidates[0].text == "Shared liquidity evidence."
    assert len(candidates) == 3
    assert any(chunk.text == "Fixed-only covenant evidence." for chunk in candidates)
    assert any(chunk.text == "Semantic-only earnings evidence." for chunk in candidates)


def test_retrieve_context_does_not_ingest_on_every_query(monkeypatch):
    corpus = [
        FinancialDocument(
            doc_id="sec-001",
            doc_type="sec_filing",
            source_name="10-K",
            company="RAG & Riches Financial",
            date="2025-01-01",
            section="Risk Factors",
            title="Liquidity and Credit Risk",
            text="Liquidity risk increased as loan delinquencies rose.",
            tags=["sec", "risk"],
        )
    ]

    class FakeIndex:
        instances = 0

        def __init__(self, *args, **kwargs):
            FakeIndex.instances += 1
            self.upsert_calls = 0
            self.search_calls = 0

        def upsert_chunk(self, chunk):
            self.upsert_calls += 1

        def search(self, query, namespace, top_k):
            self.search_calls += 1
            return []

    monkeypatch.setattr(orchestrator, "PineconeIndex", FakeIndex)
    monkeypatch.setattr(orchestrator, "build_sample_corpus", lambda include_pdf_docs=None: corpus)
    monkeypatch.setattr(orchestrator, "_DEFAULT_INDEX_CACHE", {})
    monkeypatch.setattr(orchestrator, "_INGESTED_CORPUS_KEYS", set())

    retrieve_context("What are the liquidity risks?", chunking_strategy="fixed", retrieval_mode="direct", allow_rerank=False)
    retrieve_context("What are the liquidity risks?", chunking_strategy="fixed", retrieval_mode="direct", allow_rerank=False)

    assert FakeIndex.instances == 1
    cached_index = next(iter(orchestrator._DEFAULT_INDEX_CACHE.values()))
    assert cached_index.upsert_calls == 0
    assert cached_index.search_calls == 2


def test_retrieve_context_exact_claim_identifier_skips_rewrite_and_filters_to_matching_chunks(monkeypatch):
    corpus = [
        FinancialDocument(
            doc_id="claim-1048",
            doc_type="insurance_claim",
            source_name="Claim File 1048",
            company="RAG & Riches Financial",
            date="2026-03-01",
            section="Adjuster Notes",
            title="Claim CLM-2026-1048",
            text="Claim CLM-2026-1048 discussed reserve movement, settlement timing, and mitigation steps.",
            tags=["claim", "reserve"],
        ),
        FinancialDocument(
            doc_id="claim-2104",
            doc_type="insurance_claim",
            source_name="Claim File 2104",
            company="RAG & Riches Financial",
            date="2026-03-02",
            section="Adjuster Notes",
            title="Claim CLM-2026-2104",
            text="Claim CLM-2026-2104 discussed reserve movement, settlement timing, and mitigation steps.",
            tags=["claim", "reserve"],
        ),
    ]

    class FakeIndex:
        def __init__(self, *args, **kwargs):
            self.upsert_calls = 0
            self.search_calls = []

        def upsert_chunk(self, chunk):
            self.upsert_calls += 1

        def search(self, query, namespace, top_k):
            self.search_calls.append((query, namespace, top_k))
            return [
                orchestrator.ChunkRecord(
                    chunk_id=f"{namespace}-1048",
                    doc_id="claim-1048",
                    chunk_index=0,
                    chunking_strategy="hybrid" if namespace == "semantic" else "fixed",
                    section="Adjuster Notes",
                    text="Claim CLM-2026-1048 discussed reserve movement, settlement timing, and mitigation steps.",
                    metadata={
                        "doc_type": "insurance_claim",
                        "source_name": "Claim File 1048",
                        "date": "2026-03-01",
                        "title": "Claim CLM-2026-1048",
                    },
                ),
                orchestrator.ChunkRecord(
                    chunk_id=f"{namespace}-2104",
                    doc_id="claim-2104",
                    chunk_index=0,
                    chunking_strategy="hybrid" if namespace == "semantic" else "fixed",
                    section="Adjuster Notes",
                    text="Claim CLM-2026-2104 discussed reserve movement, settlement timing, and mitigation steps.",
                    metadata={
                        "doc_type": "insurance_claim",
                        "source_name": "Claim File 2104",
                        "date": "2026-03-02",
                        "title": "Claim CLM-2026-2104",
                    },
                ),
            ]

    monkeypatch.setattr(orchestrator, "PineconeIndex", FakeIndex)
    monkeypatch.setattr(orchestrator, "build_sample_corpus", lambda include_pdf_docs=None: corpus)
    monkeypatch.setattr(orchestrator, "_DEFAULT_INDEX_CACHE", {})
    monkeypatch.setattr(orchestrator, "_INGESTED_CORPUS_KEYS", set())
    rewrite_calls = []
    monkeypatch.setattr(orchestrator, "rewrite_query", lambda query, client=None: rewrite_calls.append(query) or f"rewritten: {query}")

    candidates = retrieve_context(
        "What happened with claim CLM-2026-1048?",
        chunking_strategy="hybrid",
        retrieval_mode="rewrite",
        allow_rerank=False,
    )

    assert rewrite_calls == []
    assert candidates
    assert all("CLM-2026-1048" in chunk.text or "CLM-2026-1048" in chunk.metadata.get("title", "") for chunk in candidates)
    assert all("CLM-2026-2104" not in chunk.text for chunk in candidates)


def test_retrieve_context_exact_claim_identifier_returns_matching_corpus_chunks_without_vector_search(monkeypatch):
    corpus = [
        FinancialDocument(
            doc_id="claim-1048",
            doc_type="insurance_claim",
            source_name="Claim File 1048",
            company="RAG & Riches Financial",
            date="2026-03-01",
            section="Adjuster Notes",
            title="Claim CLM-2026-1048",
            text="Claim CLM-2026-1048 discussed reserve movement, settlement timing, and mitigation steps.",
            tags=["claim", "reserve"],
        )
    ]

    class FakeIndex:
        def __init__(self, *args, **kwargs):
            self.upsert_calls = 0
            self.search_calls = 0

        def upsert_chunk(self, chunk):
            self.upsert_calls += 1

        def search(self, query, namespace, top_k):
            self.search_calls += 1
            return []

    monkeypatch.setattr(orchestrator, "PineconeIndex", FakeIndex)
    monkeypatch.setattr(orchestrator, "build_sample_corpus", lambda include_pdf_docs=None: corpus)
    monkeypatch.setattr(orchestrator, "_DEFAULT_INDEX_CACHE", {})
    monkeypatch.setattr(orchestrator, "_INGESTED_CORPUS_KEYS", set())

    candidates = retrieve_context(
        "What happened with claim CLM-2026-1048?",
        chunking_strategy="fixed",
        retrieval_mode="rewrite",
        allow_rerank=False,
    )

    assert candidates
    assert any("CLM-2026-1048" in chunk.text for chunk in candidates)
    assert orchestrator._DEFAULT_INDEX_CACHE
    cached_index = next(iter(orchestrator._DEFAULT_INDEX_CACHE.values()))
    assert cached_index.search_calls == 0
