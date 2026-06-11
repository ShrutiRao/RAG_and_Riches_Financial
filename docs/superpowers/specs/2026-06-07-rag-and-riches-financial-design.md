# RAG & Riches Financial: Modular RAG MVP Design

## Overview

Build a chat-based Retrieval-Augmented Generation MVP for financial analysts and investors. The system will answer questions across four document families for the fictional company "RAG & Riches Financial":

- SEC filings
- earnings call transcripts
- insurance claims
- loan documents

The MVP will compare two chunking strategies:

- fixed-size chunking
- semantic chunking

It will also include an embedding and retrieval-depth benchmark that compares OpenAI vs BGE embeddings and Top-K=3 vs Top-K=5 in a four-row table.

It will also demo two retrieval paths in the same chat UI:

- query rewrite before retrieval
- retrieval candidate reranking after Pinecone search

Pinecone will be the vector database. Nebius Token Factory will power at least one LLM call in the retrieval flow, and the design assumes it can also power generation if desired later.

## Goals

- Provide a conversational chat UI for financial questions.
- Ingest multiple financial document types into a normalized pipeline.
- Support side-by-side comparison of fixed-size and semantic chunking.
- Support side-by-side comparison of query rewrite and rerank retrieval paths.
- Use Pinecone namespaces to isolate retrieval by chunking strategy.
- Include sample data for SEC filings, earnings transcripts, insurance claims, and loan documents.
- Measure retrieval relevance, faithfulness, and citation quality.

## Non-Goals

- Production authentication and authorization.
- Multi-tenant document isolation.
- Real company document ingestion from live upstream systems.
- Long-running orchestration, queues, or distributed workers.
- Full observability stack beyond lightweight logging and evaluation outputs.

## Reference Architecture

The architecture follows the same stage order as the reference diagram:

1. Data ingestion
2. Text preprocessing
3. Chunking strategies
4. Shared embedding
5. Vector store
6. Retrieval and reranking
7. Answer generation
8. Evaluation

The primary difference from the reference diagram is that the source layer is broadened from SEC-only data to four financial document families.

## User Experience

The chat UI will let the user:

- ask natural-language financial questions
- select document scope, such as SEC only or all documents
- select chunking strategy: fixed-size or semantic
- select retrieval mode: query rewrite or rerank
- view the answer with citations
- inspect retrieved chunks and the strategy used

The UI should make the comparison explicit so a demo audience can see how retrieval decisions affect the answer.

## Data Ingestion

The ingestion layer will normalize all source documents into a shared document model with metadata.

### Source Types

- SEC filings: 10-K, 10-Q, risk factors, MD&A, liquidity, and compliance sections
- Earnings transcripts: prepared remarks, Q&A, guidance commentary, and margin discussion
- Insurance claims: claim status, policy type, loss details, reserve notes, and settlement outcomes
- Loan documents: covenant language, repayment schedules, delinquency notes, and compliance clauses

### Normalized Fields

Each document should at minimum carry:

- `doc_id`
- `doc_type`
- `source_name`
- `company`
- `date`
- `section`
- `title`
- `text`
- `tags`

The ingestion step should preserve enough structure to cite the source and reason about document family, section, and reporting period.

## Preprocessing

Preprocessing will:

- strip boilerplate and repeated headers/footers
- normalize whitespace
- preserve meaningful section boundaries
- remove obvious document noise while keeping legal and financial phrasing intact
- attach document-level metadata for later retrieval and citation

This step should be deterministic so that the comparison between chunking strategies stays fair.

## Chunking Strategies

### Path A: Fixed-Size Chunking

- Use a recursive character splitter with overlap.
- Keep chunk sizes stable and predictable.
- This path is intended to be simple and reproducible.

### Path B: Semantic Chunking

- Split along topic boundaries and section semantics.
- Keep related disclosures, narrative explanations, and covenant clauses together.
- This path is intended to preserve meaning and reduce broken context.

### Comparison Rule

Both strategies should be evaluated against the same document corpus and the same question set.

## Embedding

The shared embedding stage will convert chunks from both strategies into dense vectors.

Requirements:

- a single embedding interface usable by both paths
- consistent vector dimensionality across both namespaces
- deterministic metadata association per chunk

Nebius Token Factory must power at least one LLM call somewhere in the retrieval pipeline. The design supports using it for query rewriting and/or reranking.

## Pinecone Vector Store

Pinecone will store the indexed chunks in two namespaces:

- `fixed`
- `semantic`

Each upserted vector should include metadata such as:

- document type
- source ID
- section
- chunk index
- chunking strategy
- date

This namespace split makes the comparison easy to demo and measure.

## Retrieval Paths

### Path 1: Query Rewrite

1. The user submits a question.
2. Nebius-based LLM rewrites or expands the query.
3. The rewritten query is sent to Pinecone.
4. Pinecone returns top-k chunks from the chosen namespace.
5. The generator produces the final answer from retrieved context.

### Path 2: Rerank

1. The user submits a question.
2. Pinecone returns a larger candidate set from the chosen namespace.
3. Nebius-based LLM reranks the candidates.
4. The top-ranked chunks are passed to generation.
5. The generator produces the final answer from the reranked context.

### Demo Requirement

The chat UI should expose both retrieval modes so users can switch between them during the demo.

## Answer Generation

The generation step will receive:

- the user question
- selected chunks
- chunk metadata
- retrieval mode
- chunking strategy

The output should include:

- a concise answer
- supporting citations
- source document type labels
- optional explanation of which retrieval path was used

The prompt should explicitly discourage unsupported claims and favor grounded answers.

## Evaluation

The evaluation layer will compare fixed-size and semantic chunking across a benchmark question set.

### Metrics

- retrieval relevance
- faithfulness
- citation quality
- answer completeness

### Target Bar

- 95% faithfulness
- 90% retrieval relevance

### Benchmark Coverage

The question set should cover:

- financial performance
- risk exposure
- compliance obligations
- liquidity and debt servicing
- insurance claim status and reserve behavior
- litigation or policy wording where relevant

## Sample Data

The repository should include synthetic sample data for all four document families. The data should be realistic enough for demo and evaluation use, but fully fictional.

### Example content patterns

- SEC filings: revenue trends, margin discussion, liquidity, risk factors, and compliance language
- Earnings transcripts: guidance changes, analyst questions, operational commentary
- Insurance claims: claim narrative, reserve adjustments, settlement notes, and fraud screening indicators
- Loan docs: covenant definitions, repayment schedules, delinquency, and covenant breach language

The sample data should be large enough to exercise both chunking strategies and produce visibly different retrieval behavior.

## File Structure

The code should be organized into small modules with clear boundaries:

- `src/rag_and_riches_financial/app.py`
- `src/rag_and_riches_financial/config.py`
- `src/rag_and_riches_financial/models/`
- `src/rag_and_riches_financial/data/`
- `src/rag_and_riches_financial/ingestion/`
- `src/rag_and_riches_financial/preprocessing/`
- `src/rag_and_riches_financial/chunking/`
- `src/rag_and_riches_financial/embeddings/`
- `src/rag_and_riches_financial/vectorstore/`
- `src/rag_and_riches_financial/retrieval/`
- `src/rag_and_riches_financial/rerank/`
- `src/rag_and_riches_financial/generation/`
- `src/rag_and_riches_financial/evaluation/`
- `src/rag_and_riches_financial/ui/`
- `tests/`
- `docs/`

## Key Interfaces

### Ingestion

- Input: raw text files or synthetic document fixtures
- Output: normalized document records

### Chunking

- Input: normalized documents
- Output: chunk records tagged with strategy and metadata

### Retrieval

- Input: user query, namespace, retrieval mode
- Output: ranked chunk candidates

### Generation

- Input: question plus top-ranked chunks
- Output: grounded answer with citations

### Evaluation

- Input: questions, retrieved chunks, generated answers, expected evidence
- Output: metric scores for each strategy and mode

## Risks and Mitigations

- Risk: the semantic chunking implementation may outperform fixed-size chunking only because of better metadata or larger chunks.
  - Mitigation: keep embedding and retrieval settings constant where possible.
- Risk: LLM reranking may hide retrieval weaknesses.
  - Mitigation: evaluate raw retrieval candidates separately from reranked outputs.
- Risk: synthetic data may not fully reflect real financial documents.
  - Mitigation: include document-style variation across all four corpora and keep the benchmark broad.
- Risk: faithfulness and relevance targets may be too aggressive for a small MVP.
  - Mitigation: report measured values honestly and use the targets as acceptance goals rather than guaranteed outcomes.

## Acceptance Criteria

- The chat UI can answer questions using both chunking strategies.
- Pinecone stores vectors in separate `fixed` and `semantic` namespaces.
- Nebius Token Factory is used for at least one retrieval-related LLM call.
- The app can demo both query rewrite and rerank retrieval modes.
- Sample data exists for SEC filings, earnings transcripts, insurance claims, and loan documents.
- Evaluation outputs compare the two chunking strategies on the same question set.

## Next Step

After this spec is reviewed and approved, move to implementation planning and then build the modular pipeline in small testable pieces.
