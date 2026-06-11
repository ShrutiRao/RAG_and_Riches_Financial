from rag_and_riches_financial.ingestion.pdf_loader import load_pdf_documents


def test_pdf_manifest_loader_returns_named_documents():
    docs = load_pdf_documents()

    assert len(docs) >= 6
    assert all(doc.doc_id for doc in docs)
    assert any(doc.doc_type == "sec_filing" for doc in docs)
    assert any(doc.doc_type == "earnings_transcript" for doc in docs)
    assert any(doc.doc_type == "insurance_claim" for doc in docs)
    assert any(doc.doc_type == "loan_document" for doc in docs)
    assert all(doc.text.strip() for doc in docs)


def test_pdf_loader_uses_manifest_paths():
    docs = load_pdf_documents()

    assert any(doc.source_name.endswith(".pdf") for doc in docs)
