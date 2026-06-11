from pathlib import Path

from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.ui.chat_app import (
    build_demo_questions,
    build_ui_state,
    get_logo_path,
    render_chat_app,
)


def test_ui_state_tracks_chunking_and_retrieval_mode():
    state = build_ui_state(chunking_strategy="semantic", retrieval_mode="rerank", allow_rerank=False)

    assert state["chunking_strategy"] == "semantic"
    assert state["retrieval_mode"] == "rerank"
    assert state["allow_rerank"] is False


def test_demo_questions_cover_all_document_families():
    questions = build_demo_questions()

    assert len(questions) >= 10
    assert any("liquidity" in question.lower() for question in questions)
    assert any("earnings" in question.lower() or "margin" in question.lower() for question in questions)
    assert any("claim" in question.lower() for question in questions)
    assert any("covenant" in question.lower() or "debt" in question.lower() for question in questions)
    assert any(("sec" in question.lower() or "filing" in question.lower()) and ("earnings" in question.lower() or "transcript" in question.lower()) for question in questions)
    assert any(
        (
            ("claim" in question.lower() or "insurance" in question.lower())
            and ("sec" in question.lower() or "filing" in question.lower() or "earnings" in question.lower())
        )
        or (
            ("loan" in question.lower() or "covenant" in question.lower())
            and ("sec" in question.lower() or "earnings" in question.lower())
        )
        for question in questions
    )


def test_demo_question_help_is_present_without_selectbox_override():
    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    assert result["demo_question"] is None
    assert len(fake_streamlit.sidebar.selectbox_calls) == 0
    assert fake_streamlit.sidebar.infos
    assert any("Demo questions" in info for info in fake_streamlit.sidebar.infos)
    assert any("Comparing fixed-size vs semantic chunking" in markdown for markdown in fake_streamlit.sidebar.markdowns)
    assert any("Financial intelligence" in markdown for markdown in fake_streamlit.sidebar.markdowns)


def test_render_chat_app_shows_service_statuses_in_hero_when_unconfigured(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_TOKEN_FACTORY_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("PINECONE_INDEX_HOST", raising=False)
    monkeypatch.delenv("LLAMA_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("LLAMAPARSE_API_KEY", raising=False)

    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    main_markdown = "\n".join(fake_streamlit.markdowns)
    sidebar_markdown = "\n".join(fake_streamlit.sidebar.markdowns)

    assert "Service Status" in main_markdown
    assert "Nebius" in main_markdown
    assert "Pinecone" in main_markdown
    assert "LlamaParse" in main_markdown
    assert "local fallback" in main_markdown or "not configured" in main_markdown
    assert "Service Status" not in sidebar_markdown


def test_render_chat_app_shows_connection_failed_when_probes_fail(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-nebius-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_HOST", "https://example.test")
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "test-llama-key")
    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.probe_nebius_connection", lambda *args, **kwargs: False)
    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.probe_pinecone_connection", lambda *args, **kwargs: False)
    monkeypatch.setattr(
        "rag_and_riches_financial.ui.chat_app.probe_llamaparse_connection_details",
        lambda *args, **kwargs: (False, "connection failed"),
    )

    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    main_markdown = "\n".join(fake_streamlit.markdowns)
    sidebar_markdown = "\n".join(fake_streamlit.sidebar.markdowns)

    assert "Service Status" in main_markdown
    assert "Nebius" in main_markdown and "connection failed" in main_markdown
    assert "Pinecone" in main_markdown and "connection failed" in main_markdown
    assert "LlamaParse" in main_markdown and "connection failed" in main_markdown
    assert "Service Status" not in sidebar_markdown


def test_render_chat_app_shows_not_installed_when_llamaparse_client_is_missing(monkeypatch):
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "test-llama-key")
    monkeypatch.setattr(
        "rag_and_riches_financial.ui.chat_app.probe_llamaparse_connection_details",
        lambda *args, **kwargs: (False, "not installed"),
    )

    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    main_markdown = "\n".join(fake_streamlit.markdowns)

    assert "LlamaParse" in main_markdown and "not installed" in main_markdown


def test_render_chat_app_shows_connected_when_probes_succeed(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-nebius-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("PINECONE_INDEX_HOST", "https://example.test")
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "test-llama-key")
    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.probe_nebius_connection", lambda *args, **kwargs: True)
    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.probe_pinecone_connection", lambda *args, **kwargs: True)
    monkeypatch.setattr(
        "rag_and_riches_financial.ui.chat_app.probe_llamaparse_connection_details",
        lambda *args, **kwargs: (True, "connected"),
    )

    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    main_markdown = "\n".join(fake_streamlit.markdowns)
    sidebar_markdown = "\n".join(fake_streamlit.sidebar.markdowns)

    assert "Service Status" in main_markdown
    assert "Nebius" in main_markdown and "connected" in main_markdown
    assert "Pinecone" in main_markdown and "connected" in main_markdown
    assert "LlamaParse" in main_markdown and "connected" in main_markdown
    assert "Service Status" not in sidebar_markdown


def test_render_chat_app_injects_dark_radio_styles():
    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    combined_markdown = "\n".join(fake_streamlit.markdowns)
    assert "[data-baseweb=\"radio\"]" in combined_markdown
    assert "input[type=\"radio\"]" in combined_markdown


def test_render_chat_app_shows_ingestion_badge_and_starts_warmup(monkeypatch):
    started = []

    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.start_background_ingestion", lambda: started.append(True))
    monkeypatch.setattr(
        "rag_and_riches_financial.ui.chat_app.get_background_ingestion_status",
        lambda: type("Status", (), {"state": "warming", "detail": "warming sample corpus and PDFs"})(),
    )

    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    main_markdown = "\n".join(fake_streamlit.markdowns)
    assert started == [True]
    assert "Ingestion" in main_markdown
    assert "warming up" in main_markdown
    assert "PDFs" in main_markdown
    assert fake_streamlit.fragment_calls


class FakeSidebar:
    def __init__(
        self,
        retrieval_mode="rewrite",
        allow_rerank=False,
        chunking_strategy="fixed",
        show_benchmark=False,
        show_retrieval_benchmark=False,
    ):
        self.retrieval_mode = retrieval_mode
        self.allow_rerank = allow_rerank
        self.chunking_strategy = chunking_strategy
        self.show_benchmark = show_benchmark
        self.show_retrieval_benchmark = show_retrieval_benchmark
        self.selectbox_calls = []
        self.radio_calls = []
        self.checkbox_calls = []
        self.markdowns = []
        self.subheaders = []
        self.infos = []

    def header(self, *_args, **_kwargs):
        return None

    def subheader(self, text):
        self.subheaders.append(text)

    def markdown(self, text, **_kwargs):
        self.markdowns.append(text)

    def info(self, text, **_kwargs):
        self.infos.append(text)

    def selectbox(self, label, options, index=0):
        self.selectbox_calls.append((label, options, index))
        if "Chunking" in label:
            return self.chunking_strategy
        return options[index]

    def radio(self, label, options, index=0):
        self.radio_calls.append((label, options, index))
        return self.retrieval_mode

    def checkbox(self, label, value=False):
        self.checkbox_calls.append((label, value))
        if "Compare" in label:
            return True
        if "retrieval benchmark" in label.lower():
            return self.show_retrieval_benchmark
        if "Show OpenAI" in label:
            return self.show_benchmark
        return self.allow_rerank


class FakeChatMessage:
    def __init__(self, role):
        self.role = role
        self.messages = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write(self, content):
        self.messages.append(content)


class FakeStreamlit:
    def __init__(self, query="What are the liquidity risks?"):
        self.sidebar = FakeSidebar()
        self.query = query
        self.page_config = None
        self.titles = []
        self.markdowns = []
        self.subheaders = []
        self.chat_messages = []
        self.texts = []
        self.answers = []
        self.captions = []
        self.tables = []
        self.dataframes = []
        self.session_state = {}
        self.fragment_calls = []
        self.text_input_value = query
        self.button_clicked = True
        self.form_called = False
        self.form_submit_clicked = True

    def set_page_config(self, **kwargs):
        self.page_config = kwargs

    def title(self, text):
        self.titles.append(text)

    def markdown(self, text, **_kwargs):
        self.markdowns.append(text)

    def subheader(self, text):
        self.subheaders.append(text)

    def chat_input(self, _prompt):
        return self.query

    def form(self, _key):
        self.form_called = True

        class Form:
            def __enter__(self_inner):
                return self

            def __exit__(self_inner, exc_type, exc, tb):
                return False

        return Form()

    def text_input(self, _label, value=""):
        return self.text_input_value if self.text_input_value is not None else value

    def button(self, _label):
        return self.button_clicked

    def form_submit_button(self, _label):
        return self.form_submit_clicked

    def columns(self, count):
        class Column:
            def __init__(self):
                self.markdowns = []
                self.subheaders = []
                self.writes = []

            def subheader(self, text):
                self.subheaders.append(text)

            def markdown(self, text, **_kwargs):
                self.markdowns.append(text)

            def write(self, text):
                self.writes.append(text)

        return [Column() for _ in range(count)]

    def chat_message(self, role):
        message = FakeChatMessage(role)
        self.chat_messages.append(message)
        return message

    def write(self, text):
        self.texts.append(text)

    def caption(self, text):
        self.captions.append(text)

    def table(self, data):
        self.tables.append(data)

    def dataframe(self, data, **kwargs):
        self.dataframes.append({"data": data, "kwargs": kwargs})

    def empty(self):
        return self

    def fragment(self, *args, **kwargs):
        self.fragment_calls.append({"args": args, "kwargs": kwargs})

        def decorator(fn):
            def wrapped(*fn_args, **fn_kwargs):
                return fn(*fn_args, **fn_kwargs)

            return wrapped

        return decorator


def test_render_chat_app_exposes_rewrite_and_rerank_controls():
    fake_streamlit = FakeStreamlit()
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rerank", allow_rerank=True, chunking_strategy="semantic")
    captured = {}

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        captured["retrieve"] = {
            "query": query,
            "chunking_strategy": chunking_strategy,
            "retrieval_mode": retrieval_mode,
            "allow_rerank": allow_rerank,
        }
        return []

    def fake_generate_answer(query, chunks, retrieval_mode, chunking_strategy):
        captured["generate"] = {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "chunking_strategy": chunking_strategy,
            "chunks": chunks,
        }
        return "generated answer"

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=fake_generate_answer,
    )

    assert result["state"]["retrieval_mode"] == "rerank"
    assert result["state"]["chunking_strategy"] == "fixed"
    assert result["state"]["allow_rerank"] is True
    assert captured["retrieve"]["retrieval_mode"] == "rerank"
    assert captured["retrieve"]["allow_rerank"] is True
    assert captured["generate"]["retrieval_mode"] == "rerank"
    assert not fake_streamlit.sidebar.selectbox_calls
    assert any("Comparing fixed-size vs semantic chunking" in markdown for markdown in fake_streamlit.sidebar.markdowns)


def test_render_chat_app_can_run_rewrite_only_mode_from_browser():
    fake_streamlit = FakeStreamlit()
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")
    captured = {}

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        captured["retrieve"] = {
            "query": query,
            "chunking_strategy": chunking_strategy,
            "retrieval_mode": retrieval_mode,
            "allow_rerank": allow_rerank,
        }
        return []

    def fake_generate_answer(query, chunks, retrieval_mode, chunking_strategy):
        captured["generate"] = {
            "query": query,
            "retrieval_mode": retrieval_mode,
            "chunking_strategy": chunking_strategy,
            "chunks": chunks,
        }
        return "generated answer"

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=fake_generate_answer,
    )

    assert result["state"]["retrieval_mode"] == "rewrite"
    assert result["state"]["allow_rerank"] is False
    assert captured["retrieve"]["retrieval_mode"] == "rewrite"
    assert captured["retrieve"]["allow_rerank"] is False


def test_render_chat_app_can_show_fixed_and_semantic_side_by_side():
    fake_streamlit = FakeStreamlit()
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")
    compare_calls = {}

    class FakeColumn:
        def __init__(self, name):
            self.name = name
            self.markdowns = []
            self.subheaders = []

        def subheader(self, text):
            self.subheaders.append(text)

        def markdown(self, text, **_kwargs):
            self.markdowns.append(text)

        def write(self, text):
            self.markdowns.append(text)

    def fake_columns(count):
        compare_calls["count"] = count
        return [FakeColumn("fixed"), FakeColumn("semantic")]

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
        compare_mode=True,
        columns_factory=fake_columns,
    )

    assert result["state"]["compare_mode"] is True
    assert compare_calls["count"] == 2
    assert "fixed" in result["chunks"]
    assert "semantic" in result["chunks"]
    assert "comparison_summary" in result
    assert "fixed" in result["comparison_summary"]
    assert "semantic" in result["comparison_summary"]
    assert result["preferred_strategy"] in {"fixed", "semantic"}
    assert "question_history" in result
    assert "best_evidence_note" in result
    assert any("Comparing fixed-size vs semantic chunking" in markdown for markdown in fake_streamlit.sidebar.markdowns)


def test_render_chat_app_tracks_recent_question_history_in_sidebar():
    fake_streamlit = FakeStreamlit(query="What liquidity risks and covenant pressures are disclosed across the SEC filings and loan documents?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    assert result["question_history"][0].startswith("What liquidity risks")
    assert fake_streamlit.sidebar.markdowns


def test_render_chat_app_processes_typed_question_from_text_box():
    fake_streamlit = FakeStreamlit(query="How do the loan covenants compare with the SEC filings?")
    fake_streamlit.text_input_value = "How do the loan covenants compare with the SEC filings?"
    fake_streamlit.form_submit_clicked = True
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")
    captured = {}

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        captured["query"] = query
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    assert captured["query"] == "How do the loan covenants compare with the SEC filings?"
    assert result["query"] == "How do the loan covenants compare with the SEC filings?"
    assert result["demo_question"] is None
    assert fake_streamlit.form_called is True


def test_render_chat_app_renders_answer_and_badges_only():
    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    class SingleModeSidebar(FakeSidebar):
        def checkbox(self, label, value=False):
            self.checkbox_calls.append((label, value))
            if "Compare" in label:
                return False
            return self.allow_rerank

    fake_streamlit.sidebar = SingleModeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    chunks = [
        ChunkRecord(
            chunk_id="sec-001-fixed-0",
            doc_id="sec-001",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Risk Factors",
            text="Liquidity risk increased as loan delinquencies rose and capital ratios narrowed.",
            metadata={"doc_type": "sec_filing", "source_name": "10-K", "date": "2025-02-15"},
        )
    ]

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return chunks

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "Liquidity risk increased.",
    )

    assert result["answer"] == "Liquidity risk increased."
    assert any("Short answer" in markdown for markdown in fake_streamlit.markdowns)
    assert any("sec_filing" in markdown for markdown in fake_streamlit.markdowns)
    assert not any("Retrieved chunks" in markdown for markdown in fake_streamlit.markdowns)
    assert not any("Sources" in markdown for markdown in fake_streamlit.markdowns)


def test_render_chat_app_shows_exact_claim_id_cue_for_claim_queries():
    fake_streamlit = FakeStreamlit(query="What is the status of CLM-2026-1048?")

    class SingleModeSidebar(FakeSidebar):
        def checkbox(self, label, value=False):
            self.checkbox_calls.append((label, value))
            if "Compare" in label:
                return False
            return self.allow_rerank

    fake_streamlit.sidebar = SingleModeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed")

    chunks = [
        ChunkRecord(
            chunk_id="claim-1048-fixed-0",
            doc_id="clm-2026-1048",
            chunk_index=0,
            chunking_strategy="fixed",
            section="Claim Summary",
            text="Claim CLM-2026-1048 was approved for reserve review after the loss estimate changed.",
            metadata={"doc_type": "insurance_claim", "source_name": "Claim Log", "date": "2026-03-14"},
        )
    ]

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        return chunks

    render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "Claim CLM-2026-1048 was approved for reserve review.",
    )

    assert any("Exact claim ID" in markdown for markdown in fake_streamlit.markdowns)
    assert any("CLM-2026-1048" in markdown for markdown in fake_streamlit.markdowns)


def test_render_chat_app_shows_embedding_benchmark_table():
    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="fixed", show_benchmark=True)

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, embedding_provider="openai", top_k=None, index=None, llm_client=None):
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    assert result["embedding_benchmark"] is not None
    assert fake_streamlit.dataframes
    assert len(fake_streamlit.dataframes[0]["data"]) == 4
    assert fake_streamlit.dataframes[0]["kwargs"]["use_container_width"] is True
    assert fake_streamlit.dataframes[0]["kwargs"]["hide_index"] is True
    assert any("OpenAI" in row["Embedding"] for row in fake_streamlit.dataframes[0]["data"])
    assert any("BGE" in row["Embedding"] for row in fake_streamlit.dataframes[0]["data"])
    assert all("Answer snippet" in row for row in fake_streamlit.dataframes[0]["data"])


def test_render_chat_app_shows_retrieval_benchmark_table(monkeypatch):
    fake_streamlit = FakeStreamlit(query="What is the liquidity risk?")
    fake_streamlit.sidebar = FakeSidebar(
        retrieval_mode="rewrite",
        allow_rerank=False,
        chunking_strategy="fixed",
        show_retrieval_benchmark=True,
    )

    fake_result = type(
        "BenchmarkResult",
        (),
        {
            "summary_rows": [
                type("SummaryRow", (), {"label": "fixed raw", "average_reciprocal_rank": 0.5, "hit_rate": 1.0, "improvement": 0.0})(),
                type("SummaryRow", (), {"label": "fixed reranked", "average_reciprocal_rank": 1.0, "hit_rate": 1.0, "improvement": 0.5})(),
            ],
            "rows": [
                type("DetailRow", (), {"query_label": "Liquidity", "chunking_strategy": "fixed", "reranked": False, "top_k": 5, "reciprocal_rank": 0.5, "first_match_rank": 2, "matched_sources": "sec_filing | 10-K"})(),
            ],
            "queries": [type("QueryRow", (), {"label": "Liquidity", "query": "What are the liquidity risks?"})()],
        },
    )()
    monkeypatch.setattr("rag_and_riches_financial.ui.chat_app.compare_chunking_and_rerank", lambda corpus=None: fake_result)

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, embedding_provider="openai", top_k=None, index=None, llm_client=None):
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
    )

    assert result["retrieval_benchmark"] is not None
    assert fake_streamlit.dataframes
    assert any(row["Strategy"] == "fixed raw" for row in fake_streamlit.dataframes[0]["data"])
    assert any(row["Strategy"] == "fixed reranked" for row in fake_streamlit.dataframes[0]["data"])


def test_render_chat_app_compare_mode_uses_fixed_and_semantic():
    fake_streamlit = FakeStreamlit(query="What are the liquidity risks?")
    fake_streamlit.sidebar = FakeSidebar(retrieval_mode="rewrite", allow_rerank=False, chunking_strategy="semantic")
    compare_calls = []

    class FakeColumn:
        def __init__(self):
            self.markdowns = []
            self.subheaders = []

        def subheader(self, text):
            self.subheaders.append(text)

        def markdown(self, text, **_kwargs):
            self.markdowns.append(text)

        def write(self, text):
            self.markdowns.append(text)

    def fake_retrieve_context(query, corpus=None, chunking_strategy="fixed", retrieval_mode="rewrite", allow_rerank=True, index=None, llm_client=None):
        compare_calls.append(chunking_strategy)
        return []

    result = render_chat_app(
        fake_streamlit,
        retrieve_context_fn=fake_retrieve_context,
        generate_answer_fn=lambda *args, **kwargs: "generated answer",
        compare_mode=True,
        columns_factory=lambda count: [FakeColumn(), FakeColumn()],
    )

    assert result["state"]["compare_mode"] is True
    assert compare_calls == ["fixed", "semantic"]
    assert set(result["chunks"].keys()) == {"fixed", "semantic"}


def test_logo_path_points_to_the_png_asset():
    logo_path = get_logo_path()

    assert logo_path.name == "Vault_Mind.png"
    assert logo_path.suffix == ".png"
    assert logo_path.exists()
