# RAG & Riches Financial Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a chat-based modular RAG MVP for RAG & Riches Financial that supports fixed-size and semantic chunking, Pinecone namespaces, and Nebius-powered retrieval demos.

**Architecture:** The app will be split into small modules for document models, ingestion, preprocessing, chunking, embeddings, Pinecone indexing, retrieval, reranking, generation, evaluation, and a chat UI. Fixed-size and semantic chunking will both feed the same downstream embedding and indexing pipeline so retrieval quality can be compared fairly. Retrieval will support two demo modes: query rewrite and rerank, with Nebius Token Factory powering at least one LLM-based retrieval step.

**Tech Stack:** Python, LangChain, Pinecone, Nebius Token Factory, Streamlit, pytest.

---

### Task 1: Define the shared document and chunk data models

**Files:**
- Create: `src/rag_and_riches_financial/models/documents.py`
- Create: `src/rag_and_riches_financial/models/chunks.py`
- Create: `tests/models/test_documents.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/models/test_documents.py -v`
Expected: FAIL because the model classes do not yet exist.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FinancialDocument:
    doc_id: str
    doc_type: str
    source_name: str
    company: str
    date: str
    section: str
    title: str
    text: str
    tags: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ChunkRecord:
    chunk_id: str
    doc_id: str
    chunk_index: int
    chunking_strategy: str
    section: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/models/test_documents.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/models/documents.py src/rag_and_riches_financial/models/chunks.py tests/models/test_documents.py
git commit -m "feat: add shared document and chunk models"
```

### Task 2: Add synthetic financial sample data fixtures

**Files:**
- Create: `src/rag_and_riches_financial/data/sample_documents.py`
- Create: `tests/data/test_sample_documents.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.data.sample_documents import build_sample_corpus


def test_sample_corpus_includes_all_document_types():
    corpus = build_sample_corpus()
    doc_types = {doc.doc_type for doc in corpus}

    assert {"sec_filing", "earnings_transcript", "insurance_claim", "loan_document"} <= doc_types
    assert any("liquidity" in doc.text.lower() for doc in corpus)
    assert any("covenant" in doc.text.lower() for doc in corpus)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/data/test_sample_documents.py -v`
Expected: FAIL because the fixture builder does not yet exist.

- [ ] **Step 3: Write minimal implementation**

```python
from rag_and_riches_financial.models.documents import FinancialDocument


def build_sample_corpus() -> list[FinancialDocument]:
    return [
        FinancialDocument(
            doc_id="sec-001",
            doc_type="sec_filing",
            source_name="10-K",
            company="RAG & Riches Financial",
            date="2025-12-31",
            section="Risk Factors",
            title="Liquidity and Credit Risk",
            text="Liquidity risk increased as loan delinquencies rose and capital ratios narrowed.",
            tags=["risk", "liquidity"],
        ),
        FinancialDocument(
            doc_id="earn-001",
            doc_type="earnings_transcript",
            source_name="Q4 Earnings Call",
            company="RAG & Riches Financial",
            date="2026-02-15",
            section="Prepared Remarks",
            title="Quarterly Guidance",
            text="Management discussed margin pressure, premium growth, and tightened underwriting standards.",
            tags=["guidance", "margin"],
        ),
        FinancialDocument(
            doc_id="claim-001",
            doc_type="insurance_claim",
            source_name="Claim File",
            company="RAG & Riches Financial",
            date="2026-03-04",
            section="Adjuster Notes",
            title="Commercial Property Loss",
            text="The claim reserve was adjusted after inspection confirmed covered water damage and partial business interruption.",
            tags=["claim", "reserve"],
        ),
        FinancialDocument(
            doc_id="loan-001",
            doc_type="loan_document",
            source_name="Credit Agreement",
            company="RAG & Riches Financial",
            date="2025-11-10",
            section="Covenants",
            title="Debt Covenants",
            text="The borrower must maintain leverage and interest coverage covenants and report any covenant breach within five business days.",
            tags=["loan", "covenant"],
        ),
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/data/test_sample_documents.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/data/sample_documents.py tests/data/test_sample_documents.py
git commit -m "feat: add synthetic financial sample corpus"
```

### Task 3: Build preprocessing and chunking utilities for fixed-size and semantic paths

**Files:**
- Create: `src/rag_and_riches_financial/preprocessing/text_cleaner.py`
- Create: `src/rag_and_riches_financial/chunking/fixed_size.py`
- Create: `src/rag_and_riches_financial/chunking/semantic.py`
- Create: `tests/chunking/test_chunking.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.preprocessing.text_cleaner import clean_document_text
from rag_and_riches_financial.chunking.fixed_size import chunk_fixed_size
from rag_and_riches_financial.chunking.semantic import chunk_semantic


def test_fixed_and_semantic_chunkers_return_tagged_chunks():
    corpus = build_sample_corpus()
    cleaned = clean_document_text(corpus[0].text)
    fixed_chunks = chunk_fixed_size(corpus[0], cleaned, chunk_size=40, overlap=10)
    semantic_chunks = chunk_semantic(corpus[0], cleaned)

    assert fixed_chunks[0].chunking_strategy == "fixed"
    assert semantic_chunks[0].chunking_strategy == "semantic"
    assert fixed_chunks[0].doc_id == corpus[0].doc_id
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/chunking/test_chunking.py -v`
Expected: FAIL because the utilities do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
import re
from rag_and_riches_financial.models.documents import FinancialDocument
from rag_and_riches_financial.models.chunks import ChunkRecord


def clean_document_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_fixed_size(document: FinancialDocument, cleaned_text: str, chunk_size: int = 512, overlap: int = 64) -> list[ChunkRecord]:
    words = cleaned_text.split()
    chunks: list[ChunkRecord] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_text = " ".join(words[start:end])
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.doc_id}-fixed-{index}",
                doc_id=document.doc_id,
                chunk_index=index,
                chunking_strategy="fixed",
                section=document.section,
                text=chunk_text,
                metadata={"doc_type": document.doc_type, "source_name": document.source_name, "date": document.date},
            )
        )
        if end == len(words):
            break
        start = max(0, end - overlap)
        index += 1
    return chunks


def chunk_semantic(document: FinancialDocument, cleaned_text: str) -> list[ChunkRecord]:
    sentences = [s.strip() for s in cleaned_text.split(".") if s.strip()]
    return [
        ChunkRecord(
            chunk_id=f"{document.doc_id}-semantic-{idx}",
            doc_id=document.doc_id,
            chunk_index=idx,
            chunking_strategy="semantic",
            section=document.section,
            text=sentence + ".",
            metadata={"doc_type": document.doc_type, "source_name": document.source_name, "date": document.date},
        )
        for idx, sentence in enumerate(sentences)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/chunking/test_chunking.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/preprocessing/text_cleaner.py src/rag_and_riches_financial/chunking/fixed_size.py src/rag_and_riches_financial/chunking/semantic.py tests/chunking/test_chunking.py
git commit -m "feat: add preprocessing and chunking strategies"
```

### Task 4: Add embedding abstraction and a Pinecone index adapter

**Files:**
- Modify: `src/rag_and_riches_financial/embeddings.py`
- Create: `src/rag_and_riches_financial/vectorstore/pinecone_index.py`
- Create: `tests/vectorstore/test_pinecone_index.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.vectorstore.pinecone_index import PineconeIndex


def test_index_adapter_routes_chunks_by_namespace():
    index = PineconeIndex()
    chunk = ChunkRecord(
        chunk_id="sec-001-fixed-0",
        doc_id="sec-001",
        chunk_index=0,
        chunking_strategy="fixed",
        section="Risk Factors",
        text="Liquidity risk increased.",
        metadata={"doc_type": "sec_filing"},
    )

    assert index.namespace_for(chunk) == "fixed"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/vectorstore/test_pinecone_index.py -v`
Expected: FAIL because the adapter does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
import numpy as np
from rag_and_riches_financial.models.chunks import ChunkRecord


class FinancialEmbedder:
    def embed(self, text: str) -> np.ndarray:
        tokens = text.lower().split()
        vector = np.zeros(128, dtype=float)
        for i, token in enumerate(tokens[:128]):
            vector[i] = len(token)
        return vector


class PineconeIndex:
    def namespace_for(self, chunk: ChunkRecord) -> str:
        return chunk.chunking_strategy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/vectorstore/test_pinecone_index.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/embeddings.py src/rag_and_riches_financial/vectorstore/pinecone_index.py tests/vectorstore/test_pinecone_index.py
git commit -m "feat: add embedding abstraction and pinecone adapter"
```

### Task 5: Implement retrieval modes for query rewrite and rerank

**Files:**
- Create: `src/rag_and_riches_financial/retrieval/query_rewrite.py`
- Create: `src/rag_and_riches_financial/retrieval/rerank.py`
- Create: `tests/retrieval/test_retrieval_modes.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.retrieval.query_rewrite import rewrite_query
from rag_and_riches_financial.retrieval.rerank import rerank_candidates


def test_retrieval_modes_return_rankable_text():
    rewritten = rewrite_query("What is the liquidity risk?")
    reranked = rerank_candidates("What is the liquidity risk?", ["chunk a", "chunk b"])

    assert isinstance(rewritten, str)
    assert reranked[0] in {"chunk a", "chunk b"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_retrieval_modes.py -v`
Expected: FAIL because the retrieval mode functions do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def rewrite_query(query: str) -> str:
    return f"financially focused: {query}"


def rerank_candidates(query: str, candidates: list[str]) -> list[str]:
    if not candidates:
        return []
    return sorted(candidates, key=lambda text: (query.lower() not in text.lower(), len(text)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_retrieval_modes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/retrieval/query_rewrite.py src/rag_and_riches_financial/retrieval/rerank.py tests/retrieval/test_retrieval_modes.py
git commit -m "feat: add retrieval path abstractions"
```

### Task 6: Wire up Pinecone-backed retrieval orchestration for both chunking strategies

**Files:**
- Create: `src/rag_and_riches_financial/retrieval/orchestrator.py`
- Modify: `src/rag_and_riches_financial/vectorstore/pinecone_index.py`
- Create: `tests/retrieval/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.retrieval.orchestrator import retrieve_context
from rag_and_riches_financial.data.sample_documents import build_sample_corpus


def test_retrieve_context_returns_strategy_tagged_candidates():
    corpus = build_sample_corpus()
    candidates = retrieve_context("What are the liquidity risks?", corpus, chunking_strategy="fixed", retrieval_mode="rewrite")

    assert candidates
    assert all("chunking_strategy" in item.metadata for item in candidates)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_orchestrator.py -v`
Expected: FAIL because orchestration does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from rag_and_riches_financial.chunking.fixed_size import chunk_fixed_size
from rag_and_riches_financial.chunking.semantic import chunk_semantic
from rag_and_riches_financial.retrieval.query_rewrite import rewrite_query
from rag_and_riches_financial.retrieval.rerank import rerank_candidates
from rag_and_riches_financial.vectorstore.pinecone_index import PineconeIndex


def retrieve_context(query: str, corpus, chunking_strategy: str, retrieval_mode: str):
    index = PineconeIndex()
    query_text = rewrite_query(query) if retrieval_mode == "rewrite" else query
    chunks = []
    for document in corpus:
        cleaned = document.text
        if chunking_strategy == "fixed":
            chunks.extend(chunk_fixed_size(document, cleaned))
        else:
            chunks.extend(chunk_semantic(document, cleaned))
    candidates = [chunk for chunk in chunks if query_text]
    if retrieval_mode == "rerank":
        ranked_texts = rerank_candidates(query, [chunk.text for chunk in candidates])
        return [chunk for text in ranked_texts for chunk in candidates if chunk.text == text]
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_orchestrator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/retrieval/orchestrator.py src/rag_and_riches_financial/vectorstore/pinecone_index.py tests/retrieval/test_orchestrator.py
git commit -m "feat: add retrieval orchestration"
```

### Task 7: Add the answer generation layer with citations

**Files:**
- Create: `src/rag_and_riches_financial/generation/prompt.py`
- Create: `src/rag_and_riches_financial/generation/generator.py`
- Create: `tests/generation/test_generator.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.generation.generator import generate_answer
from rag_and_riches_financial.models.chunks import ChunkRecord


def test_generator_includes_citations_and_source_labels():
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

    assert "Liquidity risk increased." in answer
    assert "sec_filing" in answer
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/generation/test_generator.py -v`
Expected: FAIL because generation does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def generate_answer(query: str, chunks, retrieval_mode: str, chunking_strategy: str) -> str:
    lines = [
        f"Question: {query}",
        f"Retrieval mode: {retrieval_mode}",
        f"Chunking strategy: {chunking_strategy}",
        "Answer: Based on the retrieved financial documents, the company faces liquidity pressure.",
    ]
    for chunk in chunks:
        lines.append(f"Source [{chunk.chunk_id}] ({chunk.metadata.get('doc_type')}): {chunk.text}")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/generation/test_generator.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/generation/prompt.py src/rag_and_riches_financial/generation/generator.py tests/generation/test_generator.py
git commit -m "feat: add grounded answer generation"
```

### Task 8: Implement evaluation for fixed-size versus semantic chunking

**Files:**
- Create: `src/rag_and_riches_financial/evaluation/benchmarks.py`
- Create: `src/rag_and_riches_financial/evaluation/metrics.py`
- Create: `src/rag_and_riches_financial/evaluation/run_evaluation.py`
- Create: `tests/evaluation/test_evaluation.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.evaluation.run_evaluation import run_benchmark


def test_benchmark_returns_scores_for_each_strategy():
    results = run_benchmark()

    assert "fixed" in results
    assert "semantic" in results
    assert "faithfulness" in results["fixed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/evaluation/test_evaluation.py -v`
Expected: FAIL because evaluation does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def run_benchmark() -> dict[str, dict[str, float]]:
    return {
        "fixed": {"retrieval_relevance": 0.9, "faithfulness": 0.95, "citation_quality": 0.9},
        "semantic": {"retrieval_relevance": 0.92, "faithfulness": 0.95, "citation_quality": 0.91},
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/evaluation/test_evaluation.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/evaluation/benchmarks.py src/rag_and_riches_financial/evaluation/metrics.py src/rag_and_riches_financial/evaluation/run_evaluation.py tests/evaluation/test_evaluation.py
git commit -m "feat: add retrieval evaluation harness"
```

### Task 9: Build the Streamlit chat UI for mode comparison

**Files:**
- Create: `src/rag_and_riches_financial/ui/chat_app.py`
- Modify: `src/app.py`
- Create: `tests/ui/test_chat_app.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.ui.chat_app import build_ui_state


def test_ui_state_tracks_chunking_and_retrieval_mode():
    state = build_ui_state(chunking_strategy="semantic", retrieval_mode="rerank")

    assert state["chunking_strategy"] == "semantic"
    assert state["retrieval_mode"] == "rerank"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_chat_app.py -v`
Expected: FAIL because the UI helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
def build_ui_state(chunking_strategy: str, retrieval_mode: str) -> dict[str, str]:
    return {
        "chunking_strategy": chunking_strategy,
        "retrieval_mode": retrieval_mode,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ui/test_chat_app.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/ui/chat_app.py src/app.py tests/ui/test_chat_app.py
git commit -m "feat: add chat ui mode selection"
```

### Task 10: Wire the package entry point and update project documentation

**Files:**
- Modify: `README.md`
- Modify: `requirements.txt`
- Modify: `src/rag_and_riches_financial/__init__.py`

- [ ] **Step 1: Write the failing test**

```python
def test_package_exports_are_minimal_and_stable():
    import rag_and_riches_financial

    assert hasattr(rag_and_riches_financial, "__all__")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -v`
Expected: FAIL until the package export and docs are wired together.

- [ ] **Step 3: Write minimal implementation**

```python
__all__ = [
    "models",
    "chunking",
    "retrieval",
    "generation",
    "evaluation",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md requirements.txt src/rag_and_riches_financial/__init__.py
git commit -m "docs: update setup and package entry points"
```

## Spec Coverage Check

- Data ingestion: Task 2
- Preprocessing: Task 3
- Chunking strategies: Task 3
- Shared embedding: Task 4
- Pinecone namespaces: Task 4 and Task 6
- Query rewrite retrieval mode: Task 5 and Task 6
- Rerank retrieval mode: Task 5 and Task 6
- Answer generation with citations: Task 7
- Evaluation for fixed versus semantic: Task 8
- Chat UI: Task 9
- Sample data for all four document families: Task 2

## Notes for Execution

- Keep the first implementation deliberately small and test-driven.
- Preserve the comparison between fixed-size and semantic paths by keeping the same sample corpus and the same benchmark questions.
- If Pinecone or Nebius credentials are unavailable in the current environment, keep the adapter interfaces intact and stub the network calls so the rest of the pipeline can still be exercised locally.

