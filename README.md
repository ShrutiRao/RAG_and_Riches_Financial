---
title: RAG & Riches Financial
emoji: 🏢
colorFrom: purple
colorTo: purple
sdk: streamlit
app_file: app.py
pinned: false
short_description: A demo of retrieval quality across a financial corpus
---

# RAG & Riches Financial

A modular Retrieval-Augmented Generation MVP for financial analysts and investors.

## Overview

This project demonstrates a chat-based financial RAG workflow across:

- SEC filings
- earnings call transcripts
- insurance claims
- loan documents

It compares fixed-size vs semantic chunking, adds an optional reranking step, and includes an embedding benchmark that compares OpenAI vs BGE with Top-K=3 vs Top-K=5.
The repo also ships with an expanded synthetic corpus and benchmark question pack so you can stress-test retrieval, reranking, hybrid RRF, and exact claim-ID lookups.

## Architecture at a glance

```mermaid
flowchart LR
    classDef user fill:#0f172a,stroke:#60a5fa,color:#e5e7eb,stroke-width:1px;
    classDef ui fill:#111827,stroke:#93c5fd,color:#e5e7eb,stroke-width:1px;
    classDef retrieval fill:#1f2937,stroke:#cbd5e1,color:#e5e7eb,stroke-width:1px;
    classDef data fill:#0b1220,stroke:#94a3b8,color:#e5e7eb,stroke-width:1px;
    classDef external fill:#1e293b,stroke:#fbbf24,color:#f8fafc,stroke-width:1px;
    classDef answer fill:#172554,stroke:#38bdf8,color:#f8fafc,stroke-width:1px;

    U[User question]:::user --> UI[Streamlit UI]:::ui
    UI --> S[Sidebar controls<br/>rewrite / rerank / compare]:::ui
    UI --> R["retrieve_context()"]:::retrieval

    subgraph Retrieval["Retrieval path"]
        direction LR
        R --> ID{Exact claim ID?}:::retrieval
        ID -- Yes --> EXACT[Return matching chunks directly]:::data
        ID -- No --> CHUNK[Chunking strategy]:::retrieval
        CHUNK --> FIXED[Fixed-size chunks]:::data
        CHUNK --> SEM[Semantic chunks]:::data
        CHUNK --> HYB[Hybrid retrieval<br/>RRF merge]:::data
        FIXED --> SEARCH[Vector search]:::data
        SEM --> SEARCH
        HYB --> SEARCH
        SEARCH --> RERANK{Rerank enabled?}:::retrieval
        RERANK -- Yes --> NB[Nebius rerank]:::external
        RERANK -- No --> CONTEXT[Retrieved chunks]:::data
        NB --> CONTEXT
        EXACT --> CONTEXT
    end

    CONTEXT --> G["generate_answer()"]:::answer
    G --> A[Grounded answer<br/>+ evidence bullets]:::answer
    A --> UI

    subgraph Sources["Document sources"]
        direction TB
        CORPUS[Sample corpus<br/>SEC filings / earnings / claims / loans]:::data
        PDF[Optional PDF ingest<br/>LlamaParse or text fallback]:::external
        VEC[Pinecone or local vector store]:::external
        CORPUS --> VEC
        PDF --> CORPUS
    end

    CORPUS -. feeds .-> FIXED
    CORPUS -. feeds .-> SEM
    CORPUS -. feeds .-> HYB

    REWRITE[Optional Nebius query rewrite]:::external
    R --> REWRITE
    REWRITE --> SEARCH
```

## Structure

- `src/rag_and_riches_financial/` - core application modules
- `tests/` - unit tests
- `docs/` - design and implementation plans
- `requirements.txt` - dependencies

## Quick start

1. Create a virtual environment:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
3. Run the sample app:
   ```powershell
   python src\app.py
   ```
4. Launch the browser UI:
   ```powershell
   streamlit run src\app.py
   ```

## Notes

The MVP is structured to plug into Pinecone for vector search and Nebius Token Factory for retrieval-time LLM calls.
It starts a background ingestion warmup at app launch so the main query path stays responsive, and the hero header shows when ingestion is warming, done, or failed.
It can also parse additional PDFs from `src/rag_and_riches_financial/data/` with LlamaParse when the `llama-parse` package is installed and a Llama Cloud API key is available.

### Optional environment variables

- `PINECONE_API_KEY`
- `PINECONE_INDEX_NAME`
- `PINECONE_INDEX_HOST`
- `NEBIUS_API_KEY` or `NEBIUS_TOKEN_FACTORY_API_KEY`
- `NEBIUS_BASE_URL`
- `NEBIUS_MODEL`
- `LLAMA_CLOUD_API_KEY` or `LLAMAPARSE_API_KEY`

You can place these values in a local `.env` file in the project root. The app loads `.env` automatically at startup.

If those are not set, the app falls back to its local in-memory demo path.
If the LlamaParse key is set, the parsed PDF docs in `src/rag_and_riches_financial/data/` are added to the corpus automatically.

### Demo modes

- `rewrite` uses Nebius to rewrite the query and skips reranking.
- `rerank` uses Nebius to rewrite the query and optionally rerank candidates.

In the Streamlit sidebar, retrieval mode is still configurable, but the app now always compares fixed-size vs semantic chunking side by side for the same question.

Compare mode shows fixed-size vs semantic chunking side by side so you can see how the two strategies differ on the same question.
The retrieval benchmark section compares fixed vs semantic retrieval on the same curated query set and shows the effect of reranking on each strategy.
You can also enable the embedding benchmark to see a four-row table for OpenAI vs BGE and Top-K=3 vs Top-K=5 on the same query.

The UI includes your `Vault_Mind.png` logo in the sidebar, scaled to fit the layout.

### Demo questions

The app includes preset demo questions in the sidebar for quick walkthroughs. Good starting points are:

- liquidity and covenant pressure
- margin pressure and revenue momentum
- claim reserve adjustments
- insurance fraud screening and settlement notes
- SEC compliance and internal controls
