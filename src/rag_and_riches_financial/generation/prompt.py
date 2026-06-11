from __future__ import annotations


def build_system_prompt() -> str:
    return (
        "You're a helpful financial analyst chatbot. Talk like you're speaking on the phone: calm, natural, and concise.\n"
        "Use short sentences, avoid jargon when plain language works, and include only the most relevant details.\n"
        "If the answer is uncertain or incomplete, say that briefly instead of filling in gaps.\n"
        "Final reminder of Refusal Rule: If the user's question is not grounded in the provided financial documents, "
        "say you can only answer questions grounded in the provided financial documents, and do not guess or invent details."
    )


def build_prompt(query: str, chunks, retrieval_mode: str, chunking_strategy: str) -> str:
    lines = [
        "System:",
        build_system_prompt(),
        f"Question: {query}",
        f"Retrieval mode: {retrieval_mode}",
        f"Chunking strategy: {chunking_strategy}",
        "Use only the supplied context.",
    ]
    for chunk in chunks:
        lines.append(f"[{chunk.chunk_id}] {chunk.text}")
    return "\n".join(lines)
