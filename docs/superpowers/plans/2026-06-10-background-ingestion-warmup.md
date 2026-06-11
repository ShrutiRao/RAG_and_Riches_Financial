# Background Ingestion Warmup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move corpus ingestion off the user query path by starting it automatically in the background at app startup, and expose a clear ingestion status badge in the hero header.

**Architecture:** The app will launch a one-time, in-process warmup job when the UI starts. That warmup job will build the sample corpus, ingest it into the active Pinecone-backed index, and publish a small status object that the UI can render as `warming up`, `done`, or `failed`. The query path will become search-only so user questions no longer wait on ingestion work; the hero status strip will show both service connectivity and ingestion progress.

**Tech Stack:** Python 3.14, Streamlit, threading, Pinecone, pytest.

---

### Task 1: Add a warmup manager that owns background ingestion and status

**Files:**
- Create: `src/rag_and_riches_financial/retrieval/ingestion_warmup.py`
- Modify: `src/rag_and_riches_financial/retrieval/orchestrator.py`
- Test: `tests/retrieval/test_ingestion_warmup.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.retrieval import ingestion_warmup


def test_background_ingestion_starts_once_and_tracks_status(monkeypatch):
    events = []

    class FakeFuture:
        def __init__(self):
            self.done_value = False

        def done(self):
            return self.done_value

    class FakeExecutor:
        def __init__(self, max_workers):
            events.append(("executor", max_workers))
            self.submissions = 0

        def submit(self, fn, *args, **kwargs):
            self.submissions += 1
            events.append(("submit", self.submissions))
            fn(*args, **kwargs)
            future = FakeFuture()
            future.done_value = True
            return future

    monkeypatch.setattr(ingestion_warmup, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(ingestion_warmup, "_warmup_sample_corpus_index", lambda *args, **kwargs: events.append("warmup"))

    status_1 = ingestion_warmup.start_background_ingestion()
    status_2 = ingestion_warmup.start_background_ingestion()

    assert status_1.state in {"warming", "done"}
    assert status_2.state == status_1.state
    assert events.count("warmup") == 1
    assert events.count(("submit", 1)) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_ingestion_warmup.py::test_background_ingestion_starts_once_and_tracks_status -v`
Expected: FAIL because `ingestion_warmup.py` and `start_background_ingestion()` do not exist yet.

- [ ] **Step 3: Write minimal implementation**

```python
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from threading import Lock

from rag_and_riches_financial.config import AppConfig
from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.retrieval.orchestrator import _get_default_index, _ingest_corpus


class IngestionPhase(str, Enum):
    IDLE = "idle"
    WARMING = "warming"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class IngestionStatus:
    state: str
    detail: str = ""


_status = IngestionStatus(state=IngestionPhase.IDLE.value, detail="not started")
_status_lock = Lock()
_warmup_future = None
_warmup_executor = ThreadPoolExecutor(max_workers=1)


def get_background_ingestion_status() -> IngestionStatus:
    return _status


def _set_status(state: str, detail: str = "") -> None:
    global _status
    with _status_lock:
        _status = IngestionStatus(state=state, detail=detail)


def _warmup_sample_corpus_index() -> None:
    try:
        config = AppConfig()
        corpus = build_sample_corpus()
        index, cache_key = _get_default_index(config, embedding_provider="openai")
        _ingest_corpus(index, corpus, "fixed", cache_key)
        _ingest_corpus(index, corpus, "semantic", cache_key)
    except Exception as exc:
        _set_status(IngestionPhase.FAILED.value, str(exc))
        return

    _set_status(IngestionPhase.DONE.value, "sample corpus ingested")


def start_background_ingestion() -> IngestionStatus:
    global _warmup_future
    with _status_lock:
        if _status.state in {IngestionPhase.WARMING.value, IngestionPhase.DONE.value}:
            return _status
        _set_status(IngestionPhase.WARMING.value, "warming sample corpus")
        if _warmup_future is None:
            _warmup_future = _warmup_executor.submit(_warmup_sample_corpus_index)
    return _status
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_ingestion_warmup.py::test_background_ingestion_starts_once_and_tracks_status -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/retrieval/ingestion_warmup.py tests/retrieval/test_ingestion_warmup.py src/rag_and_riches_financial/retrieval/orchestrator.py
git commit -m "feat: add background ingestion warmup manager"
```

### Task 2: Make query retrieval search-only so it no longer waits on ingestion

**Files:**
- Modify: `src/rag_and_riches_financial/retrieval/orchestrator.py`
- Modify: `src/rag_and_riches_financial/vectorstore/pinecone_index.py` only if a small helper is needed for warmup reuse
- Test: `tests/retrieval/test_orchestrator.py`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.retrieval.orchestrator import retrieve_context


def test_retrieve_context_does_not_ingest_on_every_query(monkeypatch):
    class FakeIndex:
        def __init__(self, *args, **kwargs):
            self.upsert_calls = 0
            self.search_calls = 0

        def upsert_chunk(self, chunk):
            self.upsert_calls += 1

        def search(self, query, namespace, top_k):
            self.search_calls += 1
            return []

    fake_index = FakeIndex()
    monkeypatch.setattr("rag_and_riches_financial.retrieval.orchestrator._get_default_index", lambda config, embedding_provider: (fake_index, ("cache",)))
    monkeypatch.setattr("rag_and_riches_financial.retrieval.orchestrator.build_sample_corpus", lambda include_pdf_docs=None: [])
    monkeypatch.setattr("rag_and_riches_financial.retrieval.orchestrator.rewrite_query", lambda query, client=None: query)
    monkeypatch.setattr("rag_and_riches_financial.retrieval.orchestrator.rerank_candidates", lambda query, candidates, client=None: candidates)

    retrieve_context("What are the liquidity risks?", chunking_strategy="fixed", retrieval_mode="direct", allow_rerank=False, index=fake_index)

    assert fake_index.upsert_calls == 0
    assert fake_index.search_calls == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/retrieval/test_orchestrator.py::test_retrieve_context_does_not_ingest_on_every_query -v`
Expected: FAIL because `retrieve_context()` still ingests before it searches.

- [ ] **Step 3: Write minimal implementation**

```python
def retrieve_context(
    query: str,
    corpus=None,
    chunking_strategy: str = "fixed",
    retrieval_mode: str = "rewrite",
    allow_rerank: bool = True,
    embedding_provider: str = "openai",
    top_k: int | None = None,
    index: PineconeIndex | None = None,
    llm_client: NebiusChatClient | None = None,
) -> list[ChunkRecord]:
    corpus = corpus or build_sample_corpus()
    config = AppConfig()
    if index is None:
        index, cache_key = _get_default_index(config, embedding_provider)
    else:
        cache_key = ("provided", id(index), embedding_provider)
    llm_client = llm_client or NebiusChatClient.from_env()

    search_query = rewrite_query(query, client=llm_client) if retrieval_mode in {"rewrite", "rerank"} else query
    effective_top_k = top_k if top_k is not None else (5 if retrieval_mode == "rerank" and allow_rerank else 3)
    if chunking_strategy == "hybrid":
        fixed_candidates = _search_chunks(index, search_query, namespace="fixed", top_k=effective_top_k)
        semantic_candidates = _search_chunks(index, search_query, namespace="semantic", top_k=effective_top_k)
        candidates = _combine_unique_chunks(fixed_candidates, semantic_candidates)
    else:
        namespace = "fixed" if chunking_strategy == "fixed" else "semantic"
        candidates = _search_chunks(index, search_query, namespace=namespace, top_k=effective_top_k)

    if retrieval_mode == "rerank" and allow_rerank:
        reranked_texts = rerank_candidates(query, [chunk.text for chunk in candidates], client=llm_client)
        reranked: list[ChunkRecord] = []
        for text in reranked_texts:
            reranked.extend([chunk for chunk in candidates if chunk.text == text])
        return reranked
    return candidates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/retrieval/test_orchestrator.py::test_retrieve_context_does_not_ingest_on_every_query -v`
Expected: PASS, and the existing retrieval tests should still pass.

- [ ] **Step 5: Commit**

```bash
git add src/rag_and_riches_financial/retrieval/orchestrator.py tests/retrieval/test_orchestrator.py
git commit -m "refactor: remove ingestion from query hot path"
```

### Task 3: Start warmup automatically in the UI and render an ingestion badge in the hero

**Files:**
- Modify: `src/rag_and_riches_financial/ui/chat_app.py`
- Modify: `tests/ui/test_chat_app.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing test**

```python
from rag_and_riches_financial.ui.chat_app import render_chat_app


def test_render_chat_app_shows_ingestion_status_badge(monkeypatch):
    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.start_background_ingestion", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "rag_and_riches_financial.ui.chat_app.get_background_ingestion_status",
        lambda: type("Status", (), {"state": "done", "detail": "sample corpus ingested"})(),
    )

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    main_markdown = "\n".join(fake_streamlit.markdowns)
    assert "Ingestion" in main_markdown
    assert "done" in main_markdown
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/ui/test_chat_app.py::test_render_chat_app_shows_ingestion_status_badge -v`
Expected: FAIL because the hero does not yet render an ingestion badge and startup does not kick off warmup.

- [ ] **Step 3: Write minimal implementation**

```python
from rag_and_riches_financial.retrieval.ingestion_warmup import (
    get_background_ingestion_status,
    start_background_ingestion,
)


def _render_service_status_hero(config: AppConfig) -> str:
    statuses = _service_statuses(config)
    ingestion = get_background_ingestion_status()
    status_colors = {
        "connected": ("#22c55e", "#1f2937"),
        "connection failed": ("#ef4444", "#7f1d1d"),
        "not configured": ("#94a3b8", "#3f3f46"),
        "warming": ("#f59e0b", "#422006"),
        "done": ("#22c55e", "#1f2937"),
        "failed": ("#ef4444", "#7f1d1d"),
    }
    pills = []
    for item in statuses:
        dot_color, pill_bg = status_colors.get(item["status"], ("#94a3b8", "#334155"))
        pills.append(
            f"""
<span class="rrf-status-pill" style="background:{pill_bg};">
  <span class="rrf-status-pill__dot" style="background:{dot_color};"></span>
  <span>{escape(item["name"])}</span>
  <span class="rrf-status-pill__sub">{escape(item["status"])}</span>
</span>
"""
        )

    ingestion_dot, ingestion_bg = status_colors.get(ingestion.state, ("#94a3b8", "#334155"))
    pills.append(
        f"""
<span class="rrf-status-pill" style="background:{ingestion_bg};">
  <span class="rrf-status-pill__dot" style="background:{ingestion_dot};"></span>
  <span>Ingestion</span>
  <span class="rrf-status-pill__sub">{escape(ingestion.state)}</span>
</span>
"""
    )

    return (
        '<div class="rrf-status-strip">'
        '<div class="rrf-status-strip__label">Service Status</div>'
        + "".join(pills)
        + "</div>"
    )


def render_chat_app(
    st,
    *,
    corpus=None,
    retrieve_context_fn=retrieve_context,
    generate_answer_fn=generate_answer,
    compare_mode: bool | None = None,
    columns_factory=None,
):
    st.set_page_config(page_title="RAG & Riches Financial", page_icon="RF", layout="wide")
    _inject_premium_styles(st)
    config = AppConfig()
    start_background_ingestion()
    # The rest of the function keeps the existing sidebar and answer rendering flow.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/ui/test_chat_app.py::test_render_chat_app_shows_ingestion_status_badge -v`
Expected: PASS, and the existing hero/status tests should still pass.

- [ ] **Step 5: Update the README**

```markdown
The app starts a background ingestion warmup at launch so the UI stays responsive while Pinecone is populated. The hero header shows an ingestion status badge that changes from warming up to done or failed.
```

- [ ] **Step 6: Commit**

```bash
git add src/rag_and_riches_financial/ui/chat_app.py tests/ui/test_chat_app.py README.md
git commit -m "feat: warm ingestion in background and show status"
```

### Task 4: Run the targeted verification suite and fix any regressions

**Files:**
- No new files expected
- Modify any file that fails verification

- [ ] **Step 1: Run the focused tests**

Run:
```powershell
$env:PYTHONPATH='src'; & '.\.venv\Scripts\python.exe' -m pytest `
  tests/retrieval/test_ingestion_warmup.py `
  tests/retrieval/test_orchestrator.py `
  tests/ui/test_chat_app.py -k 'service_status or ingestion or not_installed or connected' -v
```
Expected: all targeted tests pass.

- [ ] **Step 2: Run a quick manual sanity check**

Run:
```powershell
$env:PYTHONPATH='src'; & '.\.venv\Scripts\python.exe' -c "from rag_and_riches_financial.ui.chat_app import render_chat_app; print('import ok')"
```
Expected: prints `import ok` with no import errors.

- [ ] **Step 3: Commit the final verification fixes**

```bash
git add .
git commit -m "test: verify background ingestion warmup flow"
```

## Self-Review

- Spec coverage: The plan covers automatic warmup startup, a separate background ingestion path, removal of ingestion from the query hot path, and a visible ingestion status badge in the hero.
- Placeholder scan: No TBD/TODO placeholders remain in the tasks or code snippets.
- Type consistency: The new status object is named consistently as `IngestionStatus`, and the UI imports `start_background_ingestion()` plus `get_background_ingestion_status()` exactly as used in the task steps.
- Scope check: This is one focused feature with a single data-flow change; it does not expand into unrelated vectorstore or model work.
