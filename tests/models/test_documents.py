from rag_and_riches_financial.models.documents import FinancialDocument
from rag_and_riches_financial.models.chunks import ChunkRecord


def test_document_and_chunk_models_store_metadata():
    doc = FinancialDocument(
        doc_id="sec-001",
        doc_type="sec_filing",
        source_name="10-K",
        company="RAG & Riches Financial",
        date="2025-12-31",
        section="Risk Factors",
        title="Risk Factors",
        text="Liquidity risk and credit risk are material.",
        tags=["risk", "liquidity"],
    )
    chunk = ChunkRecord(
        chunk_id="chunk-001",
        doc_id="sec-001",
        chunk_index=0,
        chunking_strategy="fixed",
        section="Risk Factors",
        text="Liquidity risk and credit risk are material.",
        metadata={"source_name": "10-K"},
    )

    assert doc.company == "RAG & Riches Financial"
    assert chunk.chunking_strategy == "fixed"
