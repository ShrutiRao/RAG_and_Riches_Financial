# Embeddings and Top-K Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a four-way benchmark view that compares OpenAI vs BGE embeddings and Top-K=3 vs Top-K=5 in a Streamlit table.

**Architecture:** We will keep the current RAG pipeline modular by introducing an embedding provider interface that can switch between OpenAI and BGE without changing retrieval callers. Retrieval comparison will run the same query across four combinations and return a normalized results matrix that the UI can render as a table with expandable detail rows. Local fallback implementations will remain available so the demo works without live API keys.

**Tech Stack:** Python, LangChain, Streamlit, Pinecone, OpenAI SDK, Nebius Token Factory, Hugging Face sentence-transformers or BGE-style embeddings, pytest.

---

### Task 1: Add embedding provider adapters and a selection layer

**Files:**
- Create: `src/rag_and_riches_financial/embeddings/providers.py`
- Create: `src/rag_and_riches_financial/embeddings/openai_embedder.py`
- Create: `src/rag_and_riches_financial/embeddings/bge_embedder.py`
- Modify: `src/rag_and_riches_financial/embeddings.py`
- Modify: `src/rag_and_riches_financial/config.py`
- Test: `tests/embeddings/test_providers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_embedding_provider_factory_returns_openai_and_bge():
    openai_embedder = get_embedder("openai")
    bge_embedder = get_embedder("bge")

    assert openai_embedder.provider_name == "openai"
    assert bge_embedder.provider_name == "bge"
    assert openai_embedder.embed_query("risk disclosure")
    assert bge_embedder.embed_query("risk disclosure")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.embeddings.test_providers as t; t.test_embedding_provider_factory_returns_openai_and_bge(); print('ok')"`
Expected: FAIL because the factory and provider modules do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass


@dataclass
class BaseEmbedder:
    provider_name: str

    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]
```

```python
def get_embedder(provider_name: str):
    if provider_name == "bge":
        return BGEEmbedder()
    return OpenAIEmbedder()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.embeddings.test_providers as t; t.test_embedding_provider_factory_returns_openai_and_bge(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/embeddings.py src/rag_and_riches_financial/embeddings/ tests/embeddings/test_providers.py src/rag_and_riches_financial/config.py
git commit -m "feat: add embedding provider adapters"
```

### Task 2: Add four-way comparison retrieval for embeddings and top-k

**Files:**
- Create: `src/rag_and_riches_financial/retrieval/comparison_runner.py`
- Modify: `src/rag_and_riches_financial/retrieval/orchestrator.py`
- Modify: `src/rag_and_riches_financial/vectorstore/pinecone_index.py`
- Test: `tests/retrieval/test_comparison_runner.py`
- Test: `tests/retrieval/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.retrieval.test_comparison_runner as t; t.test_compare_embeddings_and_top_k_returns_four_rows(); print('ok')"`
Expected: FAIL because the comparison runner does not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
@dataclass
class ComparisonRow:
    embedding_provider: str
    top_k: int
    retrieved_count: int
    source_summary: str
    answer_summary: str
    chunks: list[ChunkRecord]
```

```python
def compare_embedding_and_topk(query, corpus, strategies):
    rows = []
    for embedding_provider, top_k in strategies:
        chunks = retrieve_context(
            query,
            corpus=corpus,
            embedding_provider=embedding_provider,
            top_k=top_k,
        )
        rows.append(...)
    return ComparisonResult(rows=rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.retrieval.test_comparison_runner as t; t.test_compare_embeddings_and_top_k_returns_four_rows(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/retrieval/comparison_runner.py src/rag_and_riches_financial/retrieval/orchestrator.py src/rag_and_riches_financial/vectorstore/pinecone_index.py tests/retrieval/test_comparison_runner.py tests/retrieval/test_orchestrator.py
git commit -m "feat: add embedding and top-k comparison retrieval"
```

### Task 3: Render the four combinations as a Streamlit comparison table

**Files:**
- Modify: `src/rag_and_riches_financial/ui/chat_app.py`
- Test: `tests/ui/test_chat_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_compare_table_renders_four_rows():
    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_compare_runner,
        generate_answer_fn=fake_generate_answer,
        compare_mode=True,
    )

    assert result["comparison_matrix"]["rows"]
    assert len(result["comparison_matrix"]["rows"]) == 4
    assert any("OpenAI" in row["label"] for row in result["comparison_matrix"]["rows"])
    assert any("BGE" in row["label"] for row in result["comparison_matrix"]["rows"])
    assert any("Top-K=3" in row["label"] for row in result["comparison_matrix"]["rows"])
    assert any("Top-K=5" in row["label"] for row in result["comparison_matrix"]["rows"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ui.test_chat_app as t; t.test_compare_table_renders_four_rows(); print('ok')"`
Expected: FAIL because the UI still renders the existing compare layout.

- [ ] **Step 3: Write minimal implementation**

```python
def _render_comparison_table(target, comparison_result):
    rows = []
    for row in comparison_result.rows:
        rows.append(
            {
                "Embedding": row.embedding_provider,
                "Top-K": row.top_k,
                "Chunks": row.retrieved_count,
                "Sources": row.source_summary,
                "Answer": row.answer_summary,
            }
        )
    target.table(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "import sys; sys.path.insert(0,'src'); sys.path.insert(0,'.'); import tests.ui.test_chat_app as t; t.test_compare_table_renders_four_rows(); print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/ui/chat_app.py tests/ui/test_chat_app.py
git commit -m "feat: render embedding and top-k comparison table"
```

### Task 4: Update docs and demo guidance

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-06-07-rag-and-riches-financial-design.md`

- [ ] **Step 1: Write the failing test**

```python
def test_readme_mentions_openai_bge_and_top_k_comparison():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "OpenAI" in text
    assert "BGE" in text
    assert "Top-K=3" in text
    assert "Top-K=5" in text
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -B -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8'); assert 'OpenAI' in text and 'BGE' in text and 'Top-K=3' in text and 'Top-K=5' in text; print('ok')"`
Expected: FAIL until the README is updated.

- [ ] **Step 3: Write minimal implementation**

```markdown
Update the README comparison section to describe:
- OpenAI embeddings vs BGE embeddings
- Top-K=3 vs Top-K=5
- the four-row table in the UI
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -B -c "from pathlib import Path; text = Path('README.md').read_text(encoding='utf-8'); assert 'OpenAI' in text and 'BGE' in text and 'Top-K=3' in text and 'Top-K=5' in text; print('ok')"`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add README.md docs/superpowers/specs/2026-06-07-rag-and-riches-financial-design.md
git commit -m "docs: describe embedding and top-k comparison demo"
```

## Self-Review

- Spec coverage: the plan covers provider adapters, comparison retrieval, UI rendering, and documentation.
- Placeholder scan: there are no TBD/TODO markers or vague task descriptions.
- Type consistency: the plan uses `embedding_provider`, `top_k`, and `ComparisonResult` consistently across tasks.
- Scope check: this is one coherent feature set for a single implementation cycle.

