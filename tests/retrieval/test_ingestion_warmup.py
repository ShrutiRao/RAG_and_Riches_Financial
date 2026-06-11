from __future__ import annotations

from rag_and_riches_financial.models.documents import FinancialDocument
from rag_and_riches_financial.retrieval import ingestion_warmup, orchestrator


def _reset_warmup_state(monkeypatch):
    monkeypatch.setattr(
        ingestion_warmup,
        "_status",
        ingestion_warmup.IngestionStatus(state="idle", detail="not started"),
    )
    monkeypatch.setattr(ingestion_warmup, "_warmup_future", None)
    monkeypatch.setattr(ingestion_warmup, "_warmup_executor", None)


def test_background_ingestion_starts_once_and_tracks_status(monkeypatch):
    _reset_warmup_state(monkeypatch)
    events: list[str] = []

    class FakeFuture:
        def __init__(self):
            self.done_value = False

        def done(self):
            return self.done_value

        def exception(self):
            return None

    class FakeExecutor:
        def __init__(self, max_workers):
            events.append(f"executor:{max_workers}")
            self.submissions = 0

        def submit(self, fn, *args, **kwargs):
            self.submissions += 1
            events.append(f"submit:{self.submissions}")
            fn(*args, **kwargs)
            future = FakeFuture()
            future.done_value = True
            return future

    monkeypatch.setattr(ingestion_warmup, "ThreadPoolExecutor", FakeExecutor)
    monkeypatch.setattr(ingestion_warmup, "_warmup_sample_corpus_index", lambda: events.append("warmup"))

    initial_status = ingestion_warmup.get_background_ingestion_status()
    first_status = ingestion_warmup.start_background_ingestion()
    second_status = ingestion_warmup.start_background_ingestion()

    assert initial_status.state == "idle"
    assert first_status.state in {"warming", "done"}
    assert second_status.state in {first_status.state, "done"}
    assert ingestion_warmup.get_background_ingestion_status().state == "done"
    assert events.count("warmup") == 1
    assert events.count("submit:1") == 1


def test_background_warmup_ingests_fixed_and_semantic_chunks_and_marks_done(monkeypatch):
    _reset_warmup_state(monkeypatch)
    corpus = [
        FinancialDocument(
            doc_id="sample-001",
            doc_type="sec_filing",
            source_name="10-K",
            company="RAG & Riches Financial",
            date="2026-01-01",
            section="Risk Factors",
            title="Liquidity and Credit Risk",
            text="Liquidity risk increased as loan delinquencies rose. Management expects continued scrutiny.",
            tags=["sec", "risk"],
        )
    ]
    calls: list[tuple[str, tuple[FinancialDocument, ...], tuple[str, str]]] = []
    include_pdf_docs_args: list[bool] = []

    def fake_build_sample_corpus(include_pdf_docs=False):
        include_pdf_docs_args.append(include_pdf_docs)
        return corpus

    monkeypatch.setattr(ingestion_warmup, "build_sample_corpus", fake_build_sample_corpus)

    def fake_get_default_index(config, embedding_provider):
        return object(), ("cache", embedding_provider)

    def fake_ingest(index, corpus_arg, chunking_strategy, cache_key):
        calls.append((chunking_strategy, tuple(corpus_arg), cache_key))

    monkeypatch.setattr(orchestrator, "_get_default_index", fake_get_default_index)
    monkeypatch.setattr(orchestrator, "_ingest_corpus", fake_ingest)

    ingestion_warmup._warmup_sample_corpus_index()

    status = ingestion_warmup.get_background_ingestion_status()
    assert status.state == "done"
    assert status.detail == "sample corpus and PDFs ingested"
    assert include_pdf_docs_args == [True]
    assert [call[0] for call in calls] == ["fixed", "semantic"]
    assert all(call[1] == tuple(corpus) for call in calls)
    assert all(call[2] == ("cache", "openai") for call in calls)


def test_background_warmup_marks_failed_on_error(monkeypatch):
    _reset_warmup_state(monkeypatch)

    monkeypatch.setattr(ingestion_warmup, "build_sample_corpus", lambda include_pdf_docs=False: [])

    def fail_get_default_index(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(orchestrator, "_get_default_index", fail_get_default_index)

    ingestion_warmup._warmup_sample_corpus_index()

    status = ingestion_warmup.get_background_ingestion_status()
    assert status.state == "failed"
    assert "boom" in status.detail
