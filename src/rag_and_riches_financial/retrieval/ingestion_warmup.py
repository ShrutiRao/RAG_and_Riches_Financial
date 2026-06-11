from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock

from rag_and_riches_financial.config import AppConfig
from rag_and_riches_financial.data.sample_documents import build_sample_corpus


@dataclass(frozen=True)
class IngestionStatus:
    state: str
    detail: str = ""


_INGESTION_IDLE = "idle"
_INGESTION_WARMING = "warming"
_INGESTION_DONE = "done"
_INGESTION_FAILED = "failed"

_status = IngestionStatus(state=_INGESTION_IDLE, detail="not started")
_status_lock = RLock()
_warmup_executor: ThreadPoolExecutor | None = None
_warmup_future = None


def _refresh_status_from_future() -> IngestionStatus:
    global _status

    future = _warmup_future
    if future is None or not future.done() or _status.state not in {_INGESTION_WARMING, _INGESTION_IDLE}:
        return _status

    try:
        exception_fn = getattr(future, "exception", None)
        exc = exception_fn() if callable(exception_fn) else None
    except Exception as error:  # pragma: no cover - future inspection failure is defensive only
        _set_status(_INGESTION_FAILED, str(error))
        return _status

    if exc is not None:
        _set_status(_INGESTION_FAILED, str(exc))
        return _status

    if _status.state != _INGESTION_DONE:
        _set_status(_INGESTION_DONE, "sample corpus and PDFs ingested")
    return _status


def get_background_ingestion_status() -> IngestionStatus:
    return _refresh_status_from_future()


def _set_status(state: str, detail: str = "") -> None:
    global _status
    with _status_lock:
        _status = IngestionStatus(state=state, detail=detail)


def _warmup_sample_corpus_index() -> None:
    try:
        from rag_and_riches_financial.retrieval.orchestrator import _get_default_index, _ingest_corpus

        config = AppConfig()
        corpus = build_sample_corpus(include_pdf_docs=True)
        index, cache_key = _get_default_index(config, embedding_provider="openai")
        _ingest_corpus(index, corpus, "fixed", cache_key)
        _ingest_corpus(index, corpus, "semantic", cache_key)
    except Exception as exc:  # pragma: no cover - defensive path for real service failures
        _set_status(_INGESTION_FAILED, str(exc))
        return

    _set_status(_INGESTION_DONE, "sample corpus and PDFs ingested")


def start_background_ingestion() -> IngestionStatus:
    global _warmup_executor, _warmup_future

    with _status_lock:
        _refresh_status_from_future()
        if _status.state in {_INGESTION_WARMING, _INGESTION_DONE}:
            return _status

        _set_status(_INGESTION_WARMING, "warming sample corpus and PDFs")
        if _warmup_executor is None:
            _warmup_executor = ThreadPoolExecutor(max_workers=1)
        if _warmup_future is None:
            _warmup_future = _warmup_executor.submit(_warmup_sample_corpus_index)

    return _status
