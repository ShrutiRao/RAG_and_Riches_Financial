from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.retrieval.comparison_runner import compare_embedding_and_topk, compare_chunking_and_rerank


def test_compare_embeddings_and_top_k_returns_four_rows():
    result = compare_embedding_and_topk(
        query="What liquidity risks are disclosed?",
        corpus=build_sample_corpus(),
        strategies=[("openai", 3), ("openai", 5), ("bge", 3), ("bge", 5)],
    )

    assert len(result.rows) == 4
    assert {(row.embedding_provider, row.top_k) for row in result.rows} == {
        ("openai", 3),
        ("openai", 5),
        ("bge", 3),
        ("bge", 5),
    }
    assert all(row.retrieved_count >= 0 for row in result.rows)
    assert all(row.answer_summary for row in result.rows)


def test_compare_embeddings_and_top_k_uses_distinct_labels():
    result = compare_embedding_and_topk(
        query="What liquidity risks are disclosed?",
        corpus=build_sample_corpus(),
    )

    labels = [row.label for row in result.rows]

    assert "OpenAI | Top-K=3" in labels
    assert "OpenAI | Top-K=5" in labels
    assert "BGE | Top-K=3" in labels
    assert "BGE | Top-K=5" in labels


def test_compare_chunking_and_rerank_reports_improvement(monkeypatch):
    def fake_retrieve_context(
        query,
        corpus=None,
        chunking_strategy="fixed",
        retrieval_mode="rewrite",
        allow_rerank=True,
        embedding_provider="openai",
        top_k=None,
        index=None,
        llm_client=None,
    ):
        if "liquidity" in query.lower():
            if chunking_strategy == "fixed" and retrieval_mode == "rewrite":
                return [
                    ChunkRecord("fixed-1", "sec-001", 0, "fixed", "Risk Factors", "Irrelevant fixed chunk.", {"doc_type": "loan_document", "source_name": "Credit Agreement 001"}),
                    ChunkRecord("fixed-2", "sec-001", 1, "fixed", "Risk Factors", "Liquidity risk increased as loan delinquencies rose.", {"doc_type": "sec_filing", "source_name": "10-K"}),
                ]
            if chunking_strategy == "fixed" and retrieval_mode == "rerank":
                return [
                    ChunkRecord("fixed-2", "sec-001", 1, "fixed", "Risk Factors", "Liquidity risk increased as loan delinquencies rose.", {"doc_type": "sec_filing", "source_name": "10-K"}),
                    ChunkRecord("fixed-1", "sec-001", 0, "fixed", "Risk Factors", "Irrelevant fixed chunk.", {"doc_type": "loan_document", "source_name": "Credit Agreement 001"}),
                ]
            if chunking_strategy == "semantic" and retrieval_mode == "rewrite":
                return [
                    ChunkRecord("semantic-1", "sec-001", 0, "semantic", "Risk Factors", "Irrelevant semantic chunk.", {"doc_type": "loan_document", "source_name": "Credit Agreement 001"}),
                    ChunkRecord("semantic-2", "sec-001", 1, "semantic", "Risk Factors", "Liquidity risk increased as loan delinquencies rose.", {"doc_type": "sec_filing", "source_name": "10-K"}),
                ]
            if chunking_strategy == "semantic" and retrieval_mode == "rerank":
                return [
                    ChunkRecord("semantic-2", "sec-001", 1, "semantic", "Risk Factors", "Liquidity risk increased as loan delinquencies rose.", {"doc_type": "sec_filing", "source_name": "10-K"}),
                    ChunkRecord("semantic-1", "sec-001", 0, "semantic", "Risk Factors", "Irrelevant semantic chunk.", {"doc_type": "loan_document", "source_name": "Credit Agreement 001"}),
                ]
        return []

    monkeypatch.setattr("rag_and_riches_financial.retrieval.comparison_runner.retrieve_context", fake_retrieve_context)
    result = compare_chunking_and_rerank(
        queries=[
            {
                "label": "Liquidity",
                "query": "What are the liquidity risks?",
                "expected_doc_types": ("sec_filing",),
                "expected_source_names": ("10-K",),
            }
        ],
        corpus=build_sample_corpus(),
        top_k=2,
    )

    summary_map = {row.label: row for row in result.summary_rows}

    assert len(result.summary_rows) == 4
    assert "fixed raw" in summary_map
    assert "fixed reranked" in summary_map
    assert summary_map["fixed reranked"].average_reciprocal_rank > summary_map["fixed raw"].average_reciprocal_rank
    assert summary_map["semantic reranked"].average_reciprocal_rank > summary_map["semantic raw"].average_reciprocal_rank
    assert set(summary_map) == {"fixed raw", "fixed reranked", "semantic raw", "semantic reranked"}
