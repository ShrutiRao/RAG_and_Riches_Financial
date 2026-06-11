# Comparison Report and Reranking Impact Analysis

We used LlamaIndex to build 3 parallel RAG pipelines over SEC filings, earnings call transcripts, insurance claims, and loan agreements. The only difference between the pipelines was the chunking strategy - fixed-size versus semantic chunking - allowing us to directly measure the impact of chunking on retrieval quality, answer relevance, and faithfulness.

## Executive Summary

This benchmark compares how retrieval behaves when the same financial corpus is split and indexed in different ways. The goal is to isolate the effect of chunking, then measure what reranking adds on top of that baseline.

The comparison report focuses on four questions:

1. Which chunking strategy retrieves the most relevant evidence?
2. Does reranking improve the order of the retrieved chunks?
3. Does hybrid retrieval, using Reciprocal Rank Fusion, produce stronger combined evidence than a single strategy alone?
4. Are the resulting answers more faithful to the underlying documents?

## Data and Corpus

The benchmark corpus includes four document families:

- SEC filings
- Earnings call transcripts
- Insurance claims
- Loan agreements

For benchmark stress testing, the sample corpus is expanded with synthetic variants so the app can be evaluated across more queries, more retrieval targets, and more exact-identifier lookups.

## Pipeline Design

Three retrieval paths are evaluated side by side:

### 1. Fixed-size chunking

Documents are split into consistent, overlapping word windows. This is the most mechanical strategy and often performs well for broad keyword matching, but it can cut across sentence or paragraph boundaries.

### 2. Semantic chunking

Documents are split along sentence-aware boundaries. This tends to preserve meaning better and can improve evidence quality when a query depends on a specific statement or claim.

### 3. Hybrid retrieval with RRF

Hybrid retrieval runs fixed-size and semantic retrieval in parallel and merges the ranked results using Reciprocal Rank Fusion (RRF). This helps recover evidence that one strategy may miss while still surfacing strong candidates from both sides.

## Tools Used

The benchmark and app workflow use the following tools and services:

- LlamaIndex for RAG orchestration and pipeline comparison
- Pinecone for vector search and indexed retrieval
- Nebius for query rewrite and reranking
- OpenAI embeddings for one embedding baseline
- BGE embeddings for the second embedding baseline
- LlamaParse for optional PDF ingestion when available
- Streamlit for the interactive UI and benchmark views

## Reranking Impact

Reranking is applied after the first retrieval pass. Its job is not to discover new documents from scratch, but to reorder the retrieved candidates so the most relevant passages rise to the top.

In practice, reranking can help when:

- the query is broad and returns several plausible candidates
- the first retrieval pass brings back mixed evidence quality
- the most relevant source is present, but not ranked first

Reranking usually improves the top of the list more than the total recall. That is why it is best evaluated together with retrieval quality, not in isolation.

## How to Read the Comparison Report

When reviewing the report, look for these patterns:

- Higher first-hit rank means the strategy found the right evidence earlier.
- Higher reciprocal rank means the right evidence is closer to the top.
- Better faithfulness means the answer stayed more tightly grounded in the retrieved chunks.
- If reranking improves rank but not faithfulness, the answer generator may still be too loose.
- If hybrid retrieval improves recall, it is often a good default for mixed document families.

## What the Benchmark Shows

The comparison is intended to answer a practical question: which retrieval path gives the most reliable evidence for a financial chatbot?

Typical interpretation:

- Fixed-size chunking is useful as a strong baseline.
- Semantic chunking can improve precision for sentence-level claims.
- Hybrid retrieval often gives the best coverage across heterogeneous documents.
- Reranking usually improves ordering and can help the final answer stay more focused.

## Faithfulness and Answer Quality

Faithfulness is measured by checking whether the answer is supported by the retrieved chunks.

In this app, that means:

- claim IDs in the answer should appear in the evidence
- numbers and dates should be grounded in retrieved text
- summary statements should be traceable to supporting passages

This makes the benchmark useful not only for retrieval evaluation, but also for answer quality analysis.

## Recommended Demo Questions

Use these questions to show the impact of chunking and reranking:

- What liquidity risks and funding pressure are disclosed across the SEC filings and loan documents?
- What do the earnings calls say about margin pressure, pricing, and guidance?
- How are claim reserves, settlement timing, and litigation exposure changing?
- What covenant obligations and default triggers should we watch in the loan documents?
- Which documents mention both liquidity pressure and compliance or control remediation?
- How do the claim reserve notes compare with the earnings commentary on loss trends?
- Which source best explains the company's liquidity and reserve pressure?
- What happened with claim CLM-2026-1048?
- What is the status of claim CLM-2026-1057?

## Final reminder of refusal rule

The assistant must stay grounded in the provided financial documents. If a question is not supported by the corpus, the answer should refuse briefly rather than inventing details.

