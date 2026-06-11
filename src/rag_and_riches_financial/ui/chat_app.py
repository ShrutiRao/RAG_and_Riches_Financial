from __future__ import annotations

from html import escape
import re
from pathlib import Path

from rag_and_riches_financial.config import AppConfig
from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.evaluation.benchmarks import build_benchmark_questions
from rag_and_riches_financial.ingestion.pdf_parser import probe_llamaparse_connection_details
from rag_and_riches_financial.generation.generator import generate_answer
from rag_and_riches_financial.retrieval.ingestion_warmup import get_background_ingestion_status, start_background_ingestion
from rag_and_riches_financial.retrieval.comparison_runner import compare_chunking_and_rerank, compare_embedding_and_topk
from rag_and_riches_financial.retrieval.nebius_client import probe_nebius_connection
from rag_and_riches_financial.retrieval.orchestrator import retrieve_context
from rag_and_riches_financial.vectorstore.pinecone_index import probe_pinecone_connection


def get_logo_path() -> Path:
    return Path(__file__).with_name("Vault_Mind.png")


def _inject_premium_styles(target) -> None:
    target.markdown(
        """
<style>
  .stApp {
    background:
      radial-gradient(circle at top left, rgba(96, 165, 250, 0.04), transparent 22%),
      radial-gradient(circle at top right, rgba(148, 163, 184, 0.035), transparent 18%),
      linear-gradient(180deg, #0a0f19 0%, #0e1624 48%, #111827 100%);
    color: #e5e7eb;
  }

  .stApp,
  .stApp p,
  .stApp li,
  .stApp label,
  .stApp span,
  .stApp div {
    color: #e5e7eb;
  }

  [data-testid="stHeader"] {
    background: rgba(10, 15, 25, 0.94) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  }

  [data-testid="stHeader"] [data-testid="stToolbar"] {
    background: rgba(10, 15, 25, 0.94) !important;
  }

  [data-testid="stToolbar"] {
    background: rgba(10, 15, 25, 0.94) !important;
    border-bottom: 1px solid rgba(148, 163, 184, 0.12);
  }

  [data-testid="stToolbar"] button,
  [data-testid="stHeader"] button {
    background: transparent !important;
    color: #e5e7eb !important;
    border: 0 !important;
    box-shadow: none !important;
  }

  [data-testid="stToolbar"] svg,
  [data-testid="stHeader"] svg {
    fill: #cbd5e1 !important;
    color: #cbd5e1 !important;
  }

  [data-testid="stSidebar"] {
    background:
      linear-gradient(180deg, rgba(8, 12, 20, 0.98) 0%, rgba(15, 23, 42, 0.98) 100%);
    border-right: 1px solid rgba(148, 163, 184, 0.14);
  }

  [data-testid="stSidebar"] .stMarkdown,
  [data-testid="stSidebar"] .stCaption,
  [data-testid="stSidebar"] label,
  [data-testid="stSidebar"] p {
    color: #d1d5db;
  }

  [data-testid="stSidebar"] .st-emotion-cache-16idsys,
  [data-testid="stSidebar"] .st-emotion-cache-1q1n0ol {
    color: #d1d5db;
  }

  .rrf-hero {
    margin: 0.5rem 0 1.1rem 0;
    padding: 1.25rem 1.35rem;
    border-radius: 24px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: rgba(15, 23, 42, 0.92);
    box-shadow: 0 18px 44px rgba(2, 6, 23, 0.28);
  }

  .rrf-status-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
    align-items: center;
    margin-bottom: 0.8rem;
    padding-bottom: 0.85rem;
    border-bottom: 1px solid rgba(148, 163, 184, 0.14);
  }

  .rrf-status-strip__label {
    font-size: 0.68rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 900;
    color: #94a3b8;
    margin-right: 0.15rem;
  }

  .rrf-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.36rem 0.62rem;
    border-radius: 999px;
    border: 1px solid rgba(148, 163, 184, 0.18);
    background: rgba(15, 23, 42, 0.72);
    color: #e5e7eb;
    font-size: 0.76rem;
    font-weight: 800;
    line-height: 1;
  }

  .rrf-status-pill__dot {
    width: 0.55rem;
    height: 0.55rem;
    border-radius: 999px;
    flex: 0 0 auto;
  }

  .rrf-status-pill__sub {
    color: #94a3b8;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }

  .rrf-hero__eyebrow {
    font-size: 0.72rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-weight: 800;
    color: #93c5fd;
    margin-bottom: 0.35rem;
  }

  .rrf-hero__title {
    font-size: clamp(1.7rem, 2.5vw, 2.4rem);
    line-height: 1.12;
    font-weight: 800;
    color: #f3f4f6;
    margin: 0;
  }

  .rrf-hero__subtitle {
    margin-top: 0.45rem;
    max-width: 64rem;
    color: #cbd5e1;
    font-size: 1rem;
    line-height: 1.6;
  }

  .rrf-sidebar-caption {
    margin-top: -0.35rem;
    margin-bottom: 0.75rem;
    padding: 0 0.2rem;
    color: #9ca3af;
    font-size: 0.86rem;
    font-style: italic;
    letter-spacing: 0.01em;
  }

  .rrf-section-label {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    margin: 0.25rem 0 0.4rem 0;
    padding: 0.35rem 0.7rem;
    border-radius: 999px;
    background: rgba(30, 41, 59, 0.72);
    border: 1px solid rgba(148, 163, 184, 0.16);
    color: #e5e7eb;
    font-size: 0.74rem;
    font-weight: 900;
    letter-spacing: 0.14em;
    text-transform: uppercase;
  }

  .stMain {
    background:
      linear-gradient(180deg, rgba(15, 23, 42, 0.12) 0%, rgba(15, 23, 42, 0.04) 100%);
  }

  .stMainBlockContainer,
  [data-testid="stVerticalBlock"] {
    gap: 0.7rem;
  }

  [data-testid="stForm"] {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(148, 163, 184, 0.14);
    border-radius: 20px;
    padding: 0.9rem 1rem 1rem;
    box-shadow: 0 14px 30px rgba(2, 6, 23, 0.18);
  }

  [data-testid="stForm"] label,
  [data-testid="stTextInput"] label,
  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] label,
  [data-testid="stRadio"] label,
  [data-testid="stCheckbox"] label {
    color: #e5e7eb !important;
  }

  [data-testid="stTextInput"] input,
  [data-testid="stSelectbox"] div,
  [data-testid="stRadio"] div,
  [data-testid="stCheckbox"] div {
    background: rgba(15, 23, 42, 0.78) !important;
    border-color: rgba(148, 163, 184, 0.18) !important;
  }

  [data-testid="stRadio"] [role="radiogroup"],
  [data-testid="stRadio"] [data-baseweb="radio"] {
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    margin-top: 0.25rem;
  }

  [data-testid="stRadio"] [role="radiogroup"] > label,
  [data-testid="stRadio"] [data-baseweb="radio"] > label {
    display: flex !important;
    align-items: center;
    gap: 0.65rem;
    padding: 0.58rem 0.72rem;
    border-radius: 16px;
    border: 1px solid rgba(148, 163, 184, 0.16);
    background: linear-gradient(180deg, rgba(15, 23, 42, 0.82) 0%, rgba(15, 23, 42, 0.66) 100%);
    box-shadow: 0 10px 22px rgba(2, 6, 23, 0.12);
    cursor: pointer;
  }

  [data-testid="stRadio"] [role="radiogroup"] > label:hover,
  [data-testid="stRadio"] [data-baseweb="radio"] > label:hover {
    border-color: rgba(148, 163, 184, 0.28);
    background: rgba(30, 41, 59, 0.78);
  }

  [data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked),
  [data-testid="stRadio"] [data-baseweb="radio"] > label:has(input:checked) {
    border-color: rgba(147, 197, 253, 0.42);
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.96) 0%, rgba(15, 23, 42, 0.9) 100%);
    box-shadow: 0 12px 26px rgba(2, 6, 23, 0.22);
  }

  [data-testid="stRadio"] input[type="radio"] {
    accent-color: #93c5fd;
    width: 1rem;
    height: 1rem;
    margin: 0;
  }

  [data-testid="stRadio"] [role="radiogroup"] > label > div,
  [data-testid="stRadio"] [data-baseweb="radio"] > label > div {
    color: #e5e7eb !important;
  }

  [data-testid="stRadio"] [role="radiogroup"] > label > div:last-child,
  [data-testid="stRadio"] [data-baseweb="radio"] > label > div:last-child {
    color: #cbd5e1 !important;
    font-weight: 700;
  }

  [data-testid="stButton"] button {
    background: linear-gradient(180deg, #475569 0%, #334155 100%);
    color: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.18);
  }

  [data-testid="stButton"] button:hover {
    border-color: rgba(148, 163, 184, 0.32);
    box-shadow: 0 8px 18px rgba(2, 6, 23, 0.24);
  }

  [data-testid="stAlert"] {
    background: rgba(15, 23, 42, 0.78) !important;
    color: #e5e7eb !important;
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
  }

  [data-testid="stAlert"] p,
  [data-testid="stAlert"] div,
  [data-testid="stAlert"] span {
    color: #e5e7eb !important;
  }
</style>
""",
        unsafe_allow_html=True,
    )


def build_demo_questions() -> list[str]:
    return build_benchmark_questions() + [
        "What liquidity risks and covenant pressures are disclosed across the SEC filings and loan documents?",
        "How did management describe margin pressure in the earnings call, and does it align with the risk factors in the SEC filings?",
        "What claim reserve adjustments were made in the insurance files, and do the earnings remarks suggest the same loss trend?",
        "What covenant obligations does the borrower have, and how do those terms compare with the funding risk discussed in the filings?",
        "Which compliance or internal control issues were highlighted in SEC filings, and were they echoed on the earnings call?",
        "What is the outlook for capital management, and how does it relate to the debt covenants and liquidity buffers?",
        "What did analysts ask about loss trends and underwriting discipline, and do the claims files show reserve movement?",
        "Are there any settlement, fraud screening, or litigation concerns in the claims, and what SEC disclosures mention related risk?",
        "Which document family has the strongest evidence of liquidity risk when you compare SEC filings, the earnings call, and the loan agreement?",
        "Do the loan documents, SEC filings, and earnings transcript all point to the same covenant or funding pressure?",
        "How do the claim reserve notes compare with the earnings call and SEC disclosure on loss severity?",
        "What cross-document evidence suggests the company is managing compliance, liquidity, and reserve pressure at the same time?",
    ]


def _render_demo_questions_help(sidebar) -> None:
    demo_text = "\n".join(
        [
            "Use these demo questions as inspiration:",
            "- What liquidity risks and covenant pressures are disclosed across the SEC filings and loan documents?",
            "- How did management describe margin pressure in the earnings call, and does it align with the risk factors in the SEC filings?",
            "- What claim reserve adjustments were made in the insurance files, and do the earnings remarks suggest the same loss trend?",
            "- What cross-document evidence suggests the company is managing compliance, liquidity, and reserve pressure at the same time?",
        ]
    )
    sidebar.info(f"ℹ️ **Demo questions**\n\n{demo_text}")


def _service_statuses(config: AppConfig) -> list[dict[str, str]]:
    nebius_connected = bool(
        config.nebius_api_key
        and config.nebius_base_url
        and probe_nebius_connection(config.nebius_api_key, config.nebius_base_url, config.nebius_model)
    )
    pinecone_connected = bool(
        config.pinecone_api_key
        and probe_pinecone_connection(config.pinecone_api_key, config.pinecone_index_name, config.pinecone_index_host)
    )
    llamaparse_connected, llamaparse_status = probe_llamaparse_connection_details(config.llamaparse_api_key)

    return [
        {
            "name": "Nebius",
            "status": "connected"
            if nebius_connected
            else ("connection failed" if config.nebius_api_key and config.nebius_base_url else "not configured"),
            "detail": "rewrite / rerank",
        },
        {
            "name": "Pinecone",
            "status": "connected"
            if pinecone_connected
            else ("connection failed" if config.pinecone_api_key else "not configured"),
            "detail": "vector search",
        },
        {
            "name": "LlamaParse",
            "status": "connected" if llamaparse_connected else llamaparse_status,
            "detail": "pdf ingest",
        },
    ]


def _render_service_status_hero(config: AppConfig) -> str:
    statuses = _service_statuses(config)
    ingestion = get_background_ingestion_status()
    status_colors = {
        "connected": ("#22c55e", "#1f2937"),
        "connection failed": ("#ef4444", "#7f1d1d"),
        "not configured": ("#94a3b8", "#3f3f46"),
        "idle": ("#94a3b8", "#3f3f46"),
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
    ingestion_label = "warming up" if ingestion.state == "warming" else ingestion.state
    ingestion_detail = ingestion.detail or "not started"
    pills.append(
        f"""
<span class="rrf-status-pill" style="background:{ingestion_bg};">
  <span class="rrf-status-pill__dot" style="background:{ingestion_dot};"></span>
  <span>Ingestion</span>
  <span class="rrf-status-pill__sub">{escape(ingestion_label)}: {escape(ingestion_detail)}</span>
</span>
"""
    )
    return (
        '<div class="rrf-status-strip">'
        '<div class="rrf-status-strip__label">Service Status</div>'
        + "".join(pills)
        + "</div>"
    )


def _render_hero_content(target) -> None:
    target.markdown(
        """
<div class="rrf-hero">
  <div class="rrf-hero__eyebrow">Analyst-grade RAG</div>
  <div class="rrf-hero__title">RAG & Riches Financial Chatbot</div>
  <div class="rrf-hero__subtitle">Chat with SEC filings, earnings transcripts, insurance claims, and loan documents. Compare retrieval strategies, inspect sources, and keep every answer grounded in evidence.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def build_ui_state(
    chunking_strategy: str,
    retrieval_mode: str,
    allow_rerank: bool = True,
    compare_mode: bool = False,
) -> dict[str, str | bool]:
    return {
        "chunking_strategy": chunking_strategy,
        "retrieval_mode": retrieval_mode,
        "allow_rerank": allow_rerank,
        "compare_mode": compare_mode,
    }


def _chunk_doc_types(chunks) -> str:
    doc_types = []
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        if doc_type not in doc_types:
            doc_types.append(doc_type)
    return ", ".join(doc_types) if doc_types else "none"


def _doc_type_badge_style(doc_type: str) -> str:
    palette = {
        "sec_filing": "#1e293b",
        "earnings_transcript": "#27272a",
        "insurance_claim": "#1f2937",
        "loan_document": "#312e81",
        "unknown": "#334155",
    }
    return palette.get(doc_type, palette["unknown"])


def _render_doc_type_badges(target, chunks) -> None:
    seen = []
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        if doc_type not in seen:
            seen.append(doc_type)
    if not seen:
        target.markdown(
            "<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;padding:0.25rem 0.55rem;border-radius:999px;background:#334155;color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.18);\">No source badges</span>",
            unsafe_allow_html=True,
        )
        return
    for doc_type in seen:
        count = sum(1 for chunk in chunks if chunk.metadata.get("doc_type", "unknown") == doc_type)
        color = _doc_type_badge_style(doc_type)
        target.markdown(
            f"<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;padding:0.25rem 0.55rem;border-radius:999px;background:{color};color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.16);\">{doc_type}: {count}</span>",
            unsafe_allow_html=True,
        )


def _render_badge_row(target, label: str, value: str) -> None:
    target.markdown(
        f"<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;padding:0.25rem 0.55rem;border-radius:999px;background:#334155;color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.16);\">{label}: {value}</span>",
        unsafe_allow_html=True,
    )


def _extract_exact_claim_ids(query: str) -> list[str]:
    return [match.upper() for match in re.findall(r"\b[A-Z]{2,}-\d{4}-\d{4}\b", query.upper())]


def _render_chip_row(target, items, *, palette=None, fallback_text: str = "none") -> None:
    if not items:
        target.markdown(
            f"<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;padding:0.25rem 0.55rem;border-radius:999px;background:#334155;color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.16);\">{fallback_text}</span>",
            unsafe_allow_html=True,
        )
        return

    chips = []
    for label, value in items:
        color = palette.get(label, "#dbeafe") if palette else "#dbeafe"
        chips.append(
            f"<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;padding:0.3rem 0.65rem;border-radius:999px;background:{color};color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.16);\">{escape(label)}: {escape(value)}</span>"
        )
    target.markdown("".join(chips), unsafe_allow_html=True)


def _render_answer_card(target, query: str, chunks, answer: str) -> None:
    doc_types = []
    for chunk in chunks:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        if doc_type not in doc_types:
            doc_types.append(doc_type)

    doc_type_chips = "".join(
        f"<span style=\"display:inline-block;margin:0 0.35rem 0.35rem 0;padding:0.3rem 0.65rem;border-radius:999px;background:{_doc_type_badge_style(doc_type)};color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.16);\">{escape(doc_type)}</span>"
        for doc_type in doc_types
    )
    answer_text = "<br/>".join(escape(answer).splitlines()) if answer else "No answer available."
    target.markdown(
        f"""
<div style="border:1px solid rgba(148,163,184,0.16);border-top:4px solid #64748b;border-radius:22px;padding:1.05rem 1.15rem;background:linear-gradient(180deg,rgba(15,23,42,0.94) 0%,rgba(17,24,39,0.92) 100%);box-shadow:0 16px 34px rgba(2,6,23,0.30);backdrop-filter:blur(10px);">
  <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
    <div>
      <div style="font-size:0.72rem;letter-spacing:0.18em;text-transform:uppercase;color:#93c5fd;font-weight:900;">Short answer</div>
      <div style="margin-top:0.42rem;font-size:1.03rem;line-height:1.72;color:#e5e7eb;">{answer_text}</div>
    </div>
  </div>
  <div style="margin-top:0.9rem;display:flex;flex-wrap:wrap;gap:0.35rem;">
    {doc_type_chips if doc_type_chips else '<span style="display:inline-block;padding:0.3rem 0.65rem;border-radius:999px;background:#334155;color:#e5e7eb;font-size:0.8rem;font-weight:700;border:1px solid rgba(148,163,184,0.16);">No source badges</span>'}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def _render_embedding_benchmark(target, comparison_result) -> None:
    rows = []
    for row in comparison_result.rows:
        rows.append(
            {
                "Embedding": row.label,
                "Top-K": row.top_k,
                "Retrieved": row.retrieved_count,
                "Sources": row.source_summary,
                "Answer snippet": row.answer_summary[:96] + ("..." if len(row.answer_summary) > 96 else ""),
            }
        )

    target.subheader("Embedding / Top-K Benchmark")
    if hasattr(target, "caption"):
        target.caption("This benchmark uses fixed-size chunks to isolate embedding and retrieval-depth effects.")
    if hasattr(target, "dataframe"):
        target.dataframe(rows, use_container_width=True, hide_index=True)
    elif hasattr(target, "table"):
        target.table(rows)
    else:
        for row in rows:
            target.markdown(
                f"- **{row['Embedding']}** | Top-K={row['Top-K']} | Retrieved={row['Retrieved']} | {row['Sources']} | {row['Answer snippet']}"
            )

    if hasattr(target, "expander"):
        for row in comparison_result.rows:
            with target.expander(row.label):
                _render_context_list(target, row.chunks)


def _render_chunking_benchmark(target, benchmark_result) -> None:
    summary_rows = [
        {
            "Strategy": row.label,
            "Avg Reciprocal Rank": round(row.average_reciprocal_rank, 3),
            "Hit Rate": round(row.hit_rate, 3),
            "Rerank Delta": round(row.improvement, 3),
        }
        for row in benchmark_result.summary_rows
    ]
    detail_rows = [
        {
            "Query": row.query_label,
            "Strategy": row.chunking_strategy,
            "Reranked": row.reranked,
            "Top-K": row.top_k,
            "RR": round(row.reciprocal_rank, 3),
            "First Hit": row.first_match_rank if row.first_match_rank is not None else "none",
            "Matched": row.matched_sources,
        }
        for row in benchmark_result.rows
    ]

    target.subheader("Chunking / Rerank Benchmark")
    if hasattr(target, "caption"):
        target.caption("This benchmark uses a curated query set to compare fixed-size, semantic, and hybrid (RRF) retrieval.")
    if hasattr(target, "dataframe"):
        target.dataframe(summary_rows, use_container_width=True, hide_index=True)
        target.dataframe(detail_rows, use_container_width=True, hide_index=True)
    elif hasattr(target, "table"):
        target.table(summary_rows)
        target.table(detail_rows)
    else:
        for row in summary_rows:
            target.markdown(
                f"- **{row['Strategy']}** | RR={row['Avg Reciprocal Rank']} | Hit rate={row['Hit Rate']} | Delta={row['Rerank Delta']}"
            )

    if hasattr(target, "expander"):
        with target.expander("Benchmark queries"):
            for row in benchmark_result.queries:
                target.markdown(f"- **{row.label}**: {row.query}")


def _build_comparison_summary(chunks_by_strategy) -> dict[str, dict[str, str | int]]:
    summary = {}
    for strategy, chunks in chunks_by_strategy.items():
        doc_type_counts = {}
        for chunk in chunks:
            doc_type = chunk.metadata.get("doc_type", "unknown")
            doc_type_counts[doc_type] = doc_type_counts.get(doc_type, 0) + 1
        top_doc_types = (
            ", ".join(
                f"{doc_type} x{count}"
                for doc_type, count in sorted(doc_type_counts.items(), key=lambda item: (-item[1], item[0]))[:3]
            )
            if doc_type_counts
            else "none"
        )
        summary[strategy] = {
            "chunk_count": len(chunks),
            "unique_doc_types": len(doc_type_counts),
            "top_doc_types": top_doc_types,
        }
    return summary


def _top_sources_summary(chunks) -> str:
    if not chunks:
        return "none"
    pairs = []
    for chunk in chunks[:3]:
        doc_type = chunk.metadata.get("doc_type", "unknown")
        source_name = chunk.metadata.get("source_name", "unknown source")
        pairs.append(f"{doc_type} | {source_name}")
    return ", ".join(pairs)


def _choose_preferred_strategy(comparison_summary) -> str:
    if not comparison_summary:
        return "semantic"

    def score(strategy: str) -> tuple[int, int]:
        stats = comparison_summary.get(strategy, {})
        return (stats.get("unique_doc_types", 0), stats.get("chunk_count", 0))

    strategies = list(comparison_summary.keys())
    if len(strategies) == 1:
        return strategies[0]
    return max(strategies, key=score)


def _build_best_evidence_note(comparison_summary, preferred_strategy: str) -> str:
    preferred = comparison_summary.get(preferred_strategy, {})
    top_doc_types = preferred.get("top_doc_types", "none")
    chunk_count = preferred.get("chunk_count", 0)
    return (
        f"Best evidence: {preferred_strategy} retrieved {chunk_count} chunks and surfaced the strongest mix of source types "
        f"({top_doc_types})."
    )


def _update_question_history(session_state, question: str, limit: int = 5) -> list[str]:
    history = list(session_state.get("question_history", []))
    if question:
        history = [question] + [item for item in history if item != question]
    session_state["question_history"] = history[:limit]
    return session_state["question_history"]


def _render_question_history(sidebar, question_history: list[str]) -> None:
    if not question_history:
        return
    sidebar.subheader("Recent questions")
    for item in question_history:
        sidebar.markdown(f"- {item}")


def _get_user_question(st) -> str:
    if hasattr(st, "form") and hasattr(st, "form_submit_button") and hasattr(st, "text_input"):
        with st.form("question_form"):
            question = st.text_input("Enter your question", value="")
            submitted = st.form_submit_button("Ask")
        if submitted and question.strip():
            return question.strip()
        return ""

    if hasattr(st, "text_input") and hasattr(st, "button"):
        question = st.text_input("Enter your question", value="")
        submitted = st.button("Ask")
        if submitted and question.strip():
            return question.strip()
        return ""

    question = st.chat_input("Ask about performance, risk, compliance, claims, or covenants")
    return question or ""


def _render_context_list(target, chunks):
    if not chunks:
        target.write("No chunks retrieved.")
        return

    for chunk in chunks:
        source_name = chunk.metadata.get("source_name", "unknown source")
        doc_type = chunk.metadata.get("doc_type", "unknown")
        date = chunk.metadata.get("date", "unknown date")
        target.markdown(
            f"**{chunk.chunk_id}** | `{chunk.chunking_strategy}` | `{doc_type}` | `{source_name}` | `{date}`\n\n{chunk.text}"
        )


def _render_result_summary(target, label: str, query: str, chunks, answer: str):
    target.markdown(f"### {label}")
    _render_badge_row(target, "Chunks", str(len(chunks)))
    _render_badge_row(target, "Document types", _chunk_doc_types(chunks))
    exact_claim_ids = _extract_exact_claim_ids(query)
    if exact_claim_ids:
        _render_badge_row(target, "Exact claim ID", ", ".join(exact_claim_ids))
    _render_doc_type_badges(target, chunks)
    _render_answer_card(target, query, chunks, answer)


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
    ingestion_status = get_background_ingestion_status()
    status_target = st.empty() if hasattr(st, "empty") else st

    def _render_status() -> None:
        status_target.markdown(_render_service_status_hero(config), unsafe_allow_html=True)

    if ingestion_status.state == "warming" and hasattr(st, "fragment"):
        @st.fragment(run_every="2s")
        def _refreshable_status_fragment():
            _render_status()

        _refreshable_status_fragment()
    else:
        _render_status()

    _render_hero_content(st)

    if hasattr(st, "sidebar"):
        if hasattr(st.sidebar, "image"):
            st.sidebar.image(str(get_logo_path()), width=220)
        if hasattr(st.sidebar, "markdown"):
            st.sidebar.markdown(
                "<div class=\"rrf-sidebar-caption\">Financial intelligence, grounded in evidence</div>",
                unsafe_allow_html=True,
            )
        elif hasattr(st.sidebar, "caption"):
            st.sidebar.caption("Financial intelligence, grounded in evidence")
    st.markdown("<div class=\"rrf-hero-anchor\"></div>", unsafe_allow_html=True)

    st.markdown("<div class=\"rrf-section-label\">Workspace</div>", unsafe_allow_html=True)

    st.sidebar.header("Retrieval Controls")
    retrieval_mode = st.sidebar.radio("Retrieval mode", ["rewrite", "rerank"], index=0)
    compare_mode = True
    if hasattr(st.sidebar, "caption"):
        st.sidebar.caption("Comparing fixed-size vs semantic chunking")
    elif hasattr(st.sidebar, "markdown"):
        st.sidebar.markdown("Comparing fixed-size vs semantic chunking")
    show_retrieval_benchmark = st.sidebar.checkbox("Show retrieval benchmark", value=False)
    show_embedding_benchmark = st.sidebar.checkbox("Show OpenAI vs BGE benchmark", value=False)
    allow_rerank = retrieval_mode == "rerank" and st.sidebar.checkbox("Enable rerank", value=True)
    _render_demo_questions_help(st.sidebar)
    state = build_ui_state(
        chunking_strategy="fixed",
        retrieval_mode=retrieval_mode,
        allow_rerank=allow_rerank,
        compare_mode=compare_mode,
    )

    st.subheader("Ask a question")
    query = _get_user_question(st)
    if not query:
        st.write("Choose a mode from the sidebar, then ask a question to begin.")
        return {"state": state, "query": None, "chunks": [], "answer": None}

    corpus = corpus or build_sample_corpus()
    question_history = _update_question_history(getattr(st, "session_state", {}), query)
    _render_question_history(st.sidebar, question_history)
    embedding_benchmark = None
    if state["compare_mode"]:
        columns_factory = columns_factory or st.columns
        strategies = ["fixed", "semantic"]
        columns = columns_factory(len(strategies))

        chunks_by_strategy = {}
        answers_by_strategy = {}
        for strategy in strategies:
            chunks_by_strategy[strategy] = retrieve_context_fn(
                query,
                corpus=corpus,
                chunking_strategy=strategy,
                retrieval_mode=state["retrieval_mode"],
                allow_rerank=state["allow_rerank"],
            )
            answers_by_strategy[strategy] = generate_answer_fn(
                query,
                chunks_by_strategy[strategy],
                retrieval_mode=state["retrieval_mode"],
                chunking_strategy=strategy,
            )

        comparison_summary = _build_comparison_summary(chunks_by_strategy)
        preferred_strategy = _choose_preferred_strategy(comparison_summary)
        best_evidence_note = _build_best_evidence_note(comparison_summary, preferred_strategy)

        st.markdown("### Compare Summary")
        st.markdown(best_evidence_note)
        _render_badge_row(st, "Preferred", preferred_strategy)
        for strategy in strategies:
            _render_badge_row(st, f"{strategy.title()} chunks", str(comparison_summary[strategy]["chunk_count"]))
            _render_badge_row(st, f"{strategy.title()} doc types", comparison_summary[strategy]["top_doc_types"])

        for column, strategy in zip(columns, strategies, strict=False):
            label = "Fixed-Size" if strategy == "fixed" else strategy.title()
            _render_result_summary(
                column,
                label + (" - Recommended" if preferred_strategy == strategy else ""),
                query,
                chunks_by_strategy[strategy],
                answers_by_strategy[strategy],
            )

        if show_embedding_benchmark:
            embedding_benchmark = compare_embedding_and_topk(query, corpus=corpus, chunking_strategy="fixed")
            _render_embedding_benchmark(st, embedding_benchmark)
        if show_retrieval_benchmark:
            retrieval_benchmark = compare_chunking_and_rerank(corpus=corpus)
            _render_chunking_benchmark(st, retrieval_benchmark)

        return {
            "state": state,
            "query": query,
            "chunks": chunks_by_strategy,
            "answer": answers_by_strategy,
            "demo_question": None,
            "embedding_benchmark": embedding_benchmark,
            "retrieval_benchmark": retrieval_benchmark if show_retrieval_benchmark else None,
            "comparison_summary": comparison_summary,
            "preferred_strategy": preferred_strategy,
            "best_evidence_note": best_evidence_note,
            "question_history": question_history,
        }

    chunks = retrieve_context_fn(
        query,
        corpus=corpus,
        chunking_strategy=state["chunking_strategy"],
        retrieval_mode=state["retrieval_mode"],
        allow_rerank=state["allow_rerank"],
    )
    answer = generate_answer_fn(
        query,
        chunks,
        retrieval_mode=state["retrieval_mode"],
        chunking_strategy=state["chunking_strategy"],
    )

    if show_embedding_benchmark:
        embedding_benchmark = compare_embedding_and_topk(query, corpus=corpus, chunking_strategy="fixed")
        _render_embedding_benchmark(st, embedding_benchmark)
    retrieval_benchmark = None
    if show_retrieval_benchmark:
        retrieval_benchmark = compare_chunking_and_rerank(corpus=corpus)
        _render_chunking_benchmark(st, retrieval_benchmark)

    with st.chat_message("assistant"):
        _render_result_summary(st, "Answer", query, chunks, answer)

    return {
        "state": state,
        "query": query,
        "chunks": chunks,
        "answer": answer,
        "demo_question": None,
        "embedding_benchmark": embedding_benchmark,
        "retrieval_benchmark": retrieval_benchmark,
        "best_evidence_note": None,
        "question_history": question_history,
    }
