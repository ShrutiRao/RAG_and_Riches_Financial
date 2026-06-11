# LlamaParse PDF Ingestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ingest additional PDF documents from `src/rag_and_riches_financial/data/` using LlamaParse when available, with a local PDF fallback, and normalize them into the existing `FinancialDocument` pipeline.

**Architecture:** We will keep the synthetic sample corpus separate from real PDF ingestion. A small manifest will map each PDF to metadata, while a new ingestion module will parse PDF text with LlamaParse first and fall back to local extraction if the API is unavailable. The parsed content will be normalized into `FinancialDocument` objects so chunking, retrieval, and the UI can reuse the same downstream code without special cases.

**Tech Stack:** Python, LlamaParse / LlamaIndex, PyPDF or pdfplumber fallback, dataclasses, pytest, existing financial document models.

---

### Task 1: Add a PDF manifest and parsed-document loader

**Files:**
- Create: `src/rag_and_riches_financial/data/pdf_manifest.json`
- Create: `src/rag_and_riches_financial/ingestion/pdf_loader.py`
- Create: `tests/ingestion/test_pdf_loader.py`
- Modify: `src/rag_and_riches_financial/data/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pdf_manifest_loader_returns_named_documents():
    docs = load_pdf_documents()

    assert docs
    assert all(doc.doc_id for doc in docs)
    assert any(doc.doc_type == "sec_filing" for doc in docs)
    assert any(doc.doc_type == "earnings_transcript" for doc in docs)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ingestion.test_pdf_loader as t; t.test_pdf_manifest_loader_returns_named_documents(); print('ok')"`
Expected: FAIL because the loader and manifest do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import asdict
import json
from pathlib import Path


def load_pdf_documents() -> list[FinancialDocument]:
    manifest = json.loads(Path(__file__).with_name("pdf_manifest.json").read_text(encoding="utf-8"))
    documents = []
    for entry in manifest["documents"]:
        documents.append(
            FinancialDocument(
                doc_id=entry["doc_id"],
                doc_type=entry["doc_type"],
                source_name=entry["source_name"],
                company=entry["company"],
                date=entry["date"],
                section=entry["section"],
                title=entry["title"],
                text=entry["text"],
                tags=entry.get("tags", []),
            )
        )
    return documents
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ingestion.test_pdf_loader as t; t.test_pdf_manifest_loader_returns_named_documents(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/ingestion/pdf_loader.py src/rag_and_riches_financial/data/pdf_manifest.json src/rag_and_riches_financial/data/__init__.py tests/ingestion/test_pdf_loader.py
git commit -m "feat: add pdf manifest loader"
```

### Task 2: Add LlamaParse with local fallback parsing

**Files:**
- Create: `src/rag_and_riches_financial/ingestion/pdf_parser.py`
- Modify: `src/rag_and_riches_financial/config.py`
- Modify: `src/rag_and_riches_financial/ingestion/pdf_loader.py`
- Test: `tests/ingestion/test_pdf_parser.py`

- [ ] **Step 1: Write the failing test**

```python
def test_pdf_parser_falls_back_when_llamaparse_is_unavailable():
    text = parse_pdf(Path("src/rag_and_riches_financial/data/SEC Filing 10-K Excerpt.pdf"))

    assert text
    assert isinstance(text, str)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ingestion.test_pdf_parser as t; t.test_pdf_parser_falls_back_when_llamaparse_is_unavailable(); print('ok')"`
Expected: FAIL because the parser module does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def parse_pdf(path: Path, use_llamaparse: bool = True) -> str:
    if use_llamaparse and config.llamaparse_api_key:
        # call LlamaParse here
        ...
    # fallback to local text extraction
    return extract_text_locally(path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ingestion.test_pdf_parser as t; t.test_pdf_parser_falls_back_when_llamaparse_is_unavailable(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/ingestion/pdf_parser.py src/rag_and_riches_financial/config.py tests/ingestion/test_pdf_parser.py
git commit -m "feat: add llamaparse pdf parsing fallback"
```

### Task 3: Normalize parsed PDFs into the existing corpus flow

**Files:**
- Modify: `src/rag_and_riches_financial/data/sample_documents.py`
- Modify: `src/rag_and_riches_financial/data/__init__.py`
- Modify: `src/rag_and_riches_financial/retrieval/orchestrator.py`
- Test: `tests/data/test_sample_documents.py`

- [ ] **Step 1: Write the failing test**

```python
def test_combined_corpus_includes_real_pdfs():
    corpus = build_sample_corpus(include_pdf_docs=True)

    assert any(doc.source_name.endswith(".pdf") for doc in corpus)
    assert any(doc.doc_type == "sec_filing" for doc in corpus)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.data.test_sample_documents as t; t.test_combined_corpus_includes_real_pdfs(); print('ok')"`
Expected: FAIL because the corpus builder does not yet include parsed PDFs.

- [ ] **Step 3: Write minimal implementation**

```python
def build_sample_corpus(include_pdf_docs: bool = False) -> list[FinancialDocument]:
    docs = _sec_filings() + _earnings_calls() + _insurance_claims() + _loan_documents()
    if include_pdf_docs:
        docs.extend(load_pdf_documents())
    return docs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.data.test_sample_documents as t; t.test_combined_corpus_includes_real_pdfs(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/data/sample_documents.py src/rag_and_riches_financial/data/__init__.py src/rag_and_riches_financial/retrieval/orchestrator.py tests/data/test_sample_documents.py
git commit -m "feat: include parsed pdf documents in corpus"
```

### Task 4: Update the UI and README for PDF-backed documents

**Files:**
- Modify: `src/rag_and_riches_financial/ui/chat_app.py`
- Modify: `README.md`
- Test: `tests/ui/test_chat_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_ui_mentions_pdf_docs_in_source_help():
    fake_streamlit = FakeStreamlit()
    render_chat_app(fake_streamlit, retrieve_context_fn=fake_retrieve_context, generate_answer_fn=fake_generate_answer)

    assert any("PDF" in info or "documents" in info for info in fake_streamlit.sidebar.infos)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ui.test_chat_app as t; t.test_ui_mentions_pdf_docs_in_source_help(); print('ok')"`
Expected: FAIL until the UI/help text is updated.

- [ ] **Step 3: Write minimal implementation**

```markdown
Update the README ingestion section to say:
- PDFs in `src/rag_and_riches_financial/data/` can be parsed with LlamaParse
- local fallback extraction is available
- parsed documents flow into the same `FinancialDocument` pipeline
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ui.test_chat_app as t; t.test_ui_mentions_pdf_docs_in_source_help(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/ui/chat_app.py README.md tests/ui/test_chat_app.py
git commit -m "docs: describe pdf-backed ingestion flow"
```

## Self-Review

- Spec coverage: the plan covers manifest loading, LlamaParse parsing, fallback extraction, corpus normalization, and UI/docs updates.
- Placeholder scan: there are no TBD/TODO markers in the plan.
- Type consistency: `FinancialDocument` remains the single normalized output type for parsed PDFs.
- Scope check: the plan is focused and can be implemented incrementally.

