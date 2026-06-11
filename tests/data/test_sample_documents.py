from rag_and_riches_financial.data import sample_documents
from rag_and_riches_financial.data.sample_documents import build_sample_corpus


def test_sample_corpus_includes_all_document_types():
    corpus = build_sample_corpus()
    doc_types = {doc.doc_type for doc in corpus}

    assert {"sec_filing", "earnings_transcript", "insurance_claim", "loan_document"} <= doc_types
    assert len([doc for doc in corpus if doc.doc_type == "sec_filing"]) >= 10
    assert len([doc for doc in corpus if doc.doc_type == "earnings_transcript"]) >= 10
    assert len([doc for doc in corpus if doc.doc_type == "insurance_claim"]) >= 10
    assert len([doc for doc in corpus if doc.doc_type == "loan_document"]) >= 10
    assert any("liquidity" in doc.text.lower() for doc in corpus)
    assert any("covenant" in doc.text.lower() for doc in corpus)


def test_sample_corpus_can_include_pdf_documents():
    corpus = build_sample_corpus(include_pdf_docs=True)

    assert len(corpus) > 40
    assert any(doc.source_name.endswith(".pdf") for doc in corpus)


def test_sample_corpus_includes_larger_synthetic_benchmark_set():
    corpus = build_sample_corpus(include_pdf_docs=False)

    assert len(corpus) >= 80
    assert any("CLM-2026-1004" in doc.text for doc in corpus)
    assert any("Liquidity Stress Test" in doc.title for doc in corpus)


def test_sample_corpus_loads_pdfs_only_once(monkeypatch):
    calls = []

    def fake_load_pdf_documents():
        calls.append(True)
        return []

    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "test-key")
    monkeypatch.setattr(sample_documents, "load_pdf_documents", fake_load_pdf_documents)
    sample_documents._build_sample_corpus_cached.cache_clear()

    build_sample_corpus(include_pdf_docs=True)
    build_sample_corpus(include_pdf_docs=True)

    assert calls == [True]
