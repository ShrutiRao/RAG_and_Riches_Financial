from rag_and_riches_financial.evaluation.benchmarks import build_benchmark_questions, build_benchmark_question_groups
from rag_and_riches_financial.evaluation.metrics import compute_faithfulness
from rag_and_riches_financial.evaluation.run_evaluation import run_benchmark
from rag_and_riches_financial.models.chunks import ChunkRecord


def test_benchmark_returns_scores_for_each_strategy():
    results = run_benchmark()

    assert "fixed" in results
    assert "semantic" in results
    assert "faithfulness" in results["fixed"]


def test_benchmark_question_pack_covers_all_task_types():
    questions = build_benchmark_questions()
    groups = build_benchmark_question_groups()

    assert len(questions) >= 11
    assert {"retrieval", "hybrid", "rerank", "exact_id"} <= set(groups)
    assert any("CLM-2026-1048" in question for question in questions)
    assert any("covenant" in question.lower() for question in questions)


def test_faithfulness_scores_supported_answers_higher_than_unsupported_answers():
    chunks = [
        ChunkRecord(
            chunk_id="claim-1048-fixed-0",
            doc_id="claim-1048",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Adjuster Notes",
            text="Claim CLM-2026-1048 discussed reserve movement and settlement timing.",
            metadata={"doc_type": "insurance_claim", "source_name": "Claim File 1048", "title": "Claim CLM-2026-1048"},
        )
    ]
    supported = "Bottom line: The retrieved passages point to CLM-2026-1048.\nEvidence:\n- Claim File 1048 | Adjuster Notes: Claim CLM-2026-1048 discussed reserve movement and settlement timing."
    unsupported = "Bottom line: The retrieved passages point to CLM-2026-9999.\nEvidence:\n- Claim File 1048 | Adjuster Notes: Claim CLM-2026-1048 discussed reserve movement and settlement timing."

    assert compute_faithfulness(supported, chunks) > compute_faithfulness(unsupported, chunks)
