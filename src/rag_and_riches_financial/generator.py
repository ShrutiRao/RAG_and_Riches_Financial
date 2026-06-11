from typing import List


class FinancialGenerator:
    """A simple generator stub for financial answers."""

    def generate_answer(self, query: str, documents: List[dict]) -> str:
        summary_lines = [f"- {doc['title']}: {doc['content']}" for doc in documents]
        return (
            "Based on retrieved financial documents, here is a summary:\n"
            + "\n".join(summary_lines)
            + "\n\nPlease replace this generator with an LLM call for production use."
        )
