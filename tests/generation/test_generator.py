from rag_and_riches_financial.generation.generator import generate_answer
from rag_and_riches_financial.generation.prompt import build_prompt
from rag_and_riches_financial.models.chunks import ChunkRecord


def test_generator_returns_a_concise_summary():
    chunks = [
        ChunkRecord(
            chunk_id="sec-001-fixed-0",
            doc_id="sec-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Risk Factors",
            text="Liquidity risk increased.",
            metadata={"doc_type": "sec_filing", "source_name": "10-K"},
        )
    ]

    answer = generate_answer("What is the liquidity risk?", chunks, retrieval_mode="rewrite", chunking_strategy="fixed")

    assert "Bottom line:" in answer
    assert "Evidence:" in answer
    assert "Liquidity risk increased." in answer
    assert "10-K" in answer
    assert "Why it matters:" not in answer
    assert "Recommendation:" not in answer


def test_generator_summarizes_retrieved_chunks_concisely():
    chunks = [
        ChunkRecord(
            chunk_id="sec-001-fixed-0",
            doc_id="sec-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Risk Factors",
            text="Liquidity risk increased as loan delinquencies rose and capital ratios narrowed.",
            metadata={"doc_type": "sec_filing", "source_name": "10-K"},
        ),
        ChunkRecord(
            chunk_id="loan-001-fixed-0",
            doc_id="loan-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Covenants",
            text="The borrower must maintain leverage and interest coverage covenants and report any covenant breach within five business days.",
            metadata={"doc_type": "loan_document", "source_name": "Credit Agreement"},
        ),
    ]

    answer = generate_answer("What are the key risks?", chunks, retrieval_mode="rewrite", chunking_strategy="fixed")

    assert "Bottom line:" in answer
    assert "Evidence:" in answer
    assert "Liquidity risk increased" in answer
    assert "Credit Agreement" in answer
    assert "reduce financial flexibility" not in answer.lower()
    assert "loss outlook" not in answer.lower()
    assert len(answer.split()) < 140
    assert "Why it matters:" not in answer
    assert "Recommendation:" not in answer


def test_prompt_contains_refusal_rule_and_phone_like_tone():
    prompt = build_prompt("What is the weather today?", [], retrieval_mode="rewrite", chunking_strategy="fixed")

    assert "Final reminder of Refusal Rule" in prompt
    assert "Talk like you're speaking on the phone" in prompt
    assert "Use short sentences" in prompt
    assert "include only the most relevant details" in prompt


def test_generator_refuses_irrelevant_questions():
    chunks = [
        ChunkRecord(
            chunk_id="sec-001-fixed-0",
            doc_id="sec-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Risk Factors",
            text="Liquidity risk increased.",
            metadata={"doc_type": "sec_filing", "source_name": "10-K"},
        )
    ]

    answer = generate_answer("What is the weather today?", chunks, retrieval_mode="rewrite", chunking_strategy="fixed")

    assert "I can only answer questions grounded in the provided financial documents" in answer
    assert "weather" not in answer.lower()
    assert "Final reminder of Refusal Rule" not in answer
    assert "You're a helpful financial analyst chatbot" not in answer


def test_generator_refuses_non_financial_question_with_common_words():
    chunks = [
        ChunkRecord(
            chunk_id="sec-001-fixed-0",
            doc_id="sec-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Risk Factors",
            text="Liquidity risk increased.",
            metadata={"doc_type": "sec_filing", "source_name": "10-K"},
        )
    ]

    answer = generate_answer("Hiw is my dress", chunks, retrieval_mode="rewrite", chunking_strategy="fixed")

    assert "I can only answer questions grounded in the provided financial documents" in answer
    assert "dress" not in answer.lower()


def test_generator_refuses_when_retrieved_chunks_do_not_support_a_financial_question():
    chunks = [
        ChunkRecord(
            chunk_id="loan-001-fixed-0",
            doc_id="loan-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Operations",
            text="The board approved a new dividend policy and share repurchase program.",
            metadata={"doc_type": "loan_document", "source_name": "Credit Agreement"},
        ),
        ChunkRecord(
            chunk_id="claim-001-semantic-0",
            doc_id="claim-001",
            chunk_index=0,
            chunking_strategy="semantic",
            section="Overview",
            text="The claim file tracked routine premium adjustments and administrative updates.",
            metadata={"doc_type": "insurance_claim", "source_name": "Claim Summary"},
        ),
    ]

    answer = generate_answer("What are the liquidity risks?", chunks, retrieval_mode="rewrite", chunking_strategy="hybrid")

    assert "I can only answer questions grounded in the provided financial documents" in answer
    assert "dividend" not in answer.lower()
    assert "premium" not in answer.lower()


def test_generator_prefers_the_exact_claim_identifier_when_present():
    chunks = [
        ChunkRecord(
            chunk_id="claim-1048-fixed-0",
            doc_id="claim-1048",
            chunk_index=0,
            chunking_strategy="hybrid",
            section="Adjuster Notes",
            text="Claim CLM-2026-1048 noted reserve movement and settlement timing.",
            metadata={"doc_type": "insurance_claim", "source_name": "Claim File 1048", "title": "Claim CLM-2026-1048"},
        ),
        ChunkRecord(
            chunk_id="claim-2104-fixed-0",
            doc_id="claim-2104",
            chunk_index=0,
            chunking_strategy="hybrid",
            section="Adjuster Notes",
            text="Claim CLM-2026-2104 noted reserve movement and settlement timing.",
            metadata={"doc_type": "insurance_claim", "source_name": "Claim File 2104", "title": "Claim CLM-2026-2104"},
        ),
    ]

    answer = generate_answer("What happened with claim CLM-2026-1048?", chunks, retrieval_mode="rewrite", chunking_strategy="hybrid")

    assert "CLM-2026-1048" in answer
    assert "CLM-2026-2104" not in answer
