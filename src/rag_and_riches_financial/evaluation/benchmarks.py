from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BenchmarkQuestion:
    label: str
    question: str
    focus: str


def build_benchmark_question_groups() -> dict[str, list[BenchmarkQuestion]]:
    return {
        "retrieval": [
            BenchmarkQuestion(
                label="Liquidity",
                question="What liquidity risks and funding pressure are disclosed across the SEC filings and loan documents?",
                focus="Cross-document retrieval",
            ),
            BenchmarkQuestion(
                label="Margins",
                question="What do the earnings calls say about margin pressure, pricing, and guidance?",
                focus="Earnings + filings",
            ),
            BenchmarkQuestion(
                label="Claims",
                question="How are claim reserves, settlement timing, and litigation exposure changing?",
                focus="Insurance claims",
            ),
            BenchmarkQuestion(
                label="Covenants",
                question="What covenant obligations and default triggers should we watch in the loan documents?",
                focus="Loan terms",
            ),
        ],
        "hybrid": [
            BenchmarkQuestion(
                label="Liquidity vs controls",
                question="Which documents mention both liquidity pressure and compliance or control remediation?",
                focus="Hybrid retrieval / RRF",
            ),
            BenchmarkQuestion(
                label="Reserve story",
                question="How do the claim reserve notes compare with the earnings commentary on loss trends?",
                focus="Cross-document synthesis",
            ),
            BenchmarkQuestion(
                label="Capital story",
                question="What is the outlook for capital management, and how does it relate to leverage and debt covenants?",
                focus="Multiple families",
            ),
        ],
        "rerank": [
            BenchmarkQuestion(
                label="Relevance sort",
                question="Which source best explains the company's liquidity and reserve pressure?",
                focus="Optional rerank",
            ),
            BenchmarkQuestion(
                label="Evidence focus",
                question="What is the clearest evidence of underwriting pressure and settlement activity?",
                focus="Optional rerank",
            ),
        ],
        "exact_id": [
            BenchmarkQuestion(
                label="Claim 1048",
                question="What happened with claim CLM-2026-1048?",
                focus="Exact claim ID",
            ),
            BenchmarkQuestion(
                label="Claim 1057",
                question="What is the status of claim CLM-2026-1057?",
                focus="Exact claim ID",
            ),
        ],
    }


def build_benchmark_questions() -> list[str]:
    questions: list[str] = []
    for group in build_benchmark_question_groups().values():
        questions.extend(item.question for item in group)
    return questions
