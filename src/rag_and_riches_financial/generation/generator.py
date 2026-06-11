from __future__ import annotations

import re


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "do",
    "does",
    "for",
    "from",
    "get",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "our",
    "please",
    "show",
    "tell",
    "that",
    "the",
    "their",
    "them",
    "this",
    "to",
    "us",
    "was",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "would",
    "you",
}


def _normalize_term(term: str) -> str:
    cleaned = term.strip(".,?!:;()[]{}'\"").lower()
    if len(cleaned) > 4 and cleaned.endswith("s") and not cleaned.endswith("ss"):
        return cleaned[:-1]
    return cleaned


def _extract_identifiers(text: str) -> set[str]:
    return {match.upper() for match in re.findall(r"\b[A-Z]{2,}-\d{4}-\d{4}\b", text.upper())}


def _query_terms(query: str) -> set[str]:
    return {
        _normalize_term(term)
        for term in query.split()
        if term.strip() and _normalize_term(term) not in STOPWORDS
    }


def _text_terms(text: str) -> set[str]:
    return {
        _normalize_term(term)
        for term in text.split()
        if term.strip() and _normalize_term(term) not in STOPWORDS
    }


def _chunk_search_blob(chunk) -> str:
    metadata_parts = []
    for value in chunk.metadata.values():
        if isinstance(value, str):
            metadata_parts.append(value)
        elif isinstance(value, list):
            metadata_parts.extend(str(item) for item in value)
    return " ".join([chunk.chunk_id, chunk.doc_id, chunk.section, chunk.text, *metadata_parts]).lower()


FINANCE_INTENT_TERMS = {
    "financial",
    "finance",
    "revenue",
    "margin",
    "liquidity",
    "covenant",
    "claim",
    "claims",
    "reserve",
    "sec",
    "filing",
    "filings",
    "earnings",
    "loan",
    "loans",
    "risk",
    "risks",
    "compliance",
    "debt",
    "funding",
    "underwriting",
    "loss",
    "capital",
    "balance",
    "coverage",
    "delinquency",
    "portfolio",
    "investor",
    "investment",
    "stock",
    "share",
    "shares",
    "dividend",
    "buyback",
    "cash",
    "flow",
    "guidance",
}


def _has_financial_intent(query: str) -> bool:
    return bool(_query_terms(query).intersection(FINANCE_INTENT_TERMS))


def _should_refuse(query: str, chunks) -> bool:
    if not chunks:
        return True

    return not _has_financial_intent(query)


def _build_refusal_answer(query: str) -> str:
    return (
        "I can only answer questions grounded in the provided financial documents. "
        "Please ask about the SEC filings, earnings call transcripts, insurance claims, or loan documents."
    )


def _first_sentence(text: str) -> str:
    for separator in (". ", "? ", "! "):
        if separator in text:
            return text.split(separator, 1)[0].strip() + separator[0]
    return text.strip()


def _chunk_relevance_score(query_terms: set[str], chunk) -> int:
    chunk_terms = _text_terms(chunk.text)
    return len(query_terms.intersection(chunk_terms))


def _select_grounded_chunks(query: str, chunks, max_chunks: int = 2):
    if not chunks:
        return []

    query_identifiers = _extract_identifiers(query)
    if query_identifiers:
        exact_matches = [
            chunk
            for chunk in chunks
            if query_identifiers.intersection(_extract_identifiers(_chunk_search_blob(chunk)))
        ]
        if exact_matches:
            return exact_matches[:max_chunks]
        return []

    query_terms = _query_terms(query)
    ranked = sorted(
        ((_chunk_relevance_score(query_terms, chunk), index, chunk) for index, chunk in enumerate(chunks)),
        key=lambda item: (item[0], -item[1]),
        reverse=True,
    )
    grounded_chunks = [chunk for score, _, chunk in ranked if score > 0]
    return grounded_chunks[:max_chunks]


def _select_supporting_chunks(query: str, chunks, max_chunks: int = 3):
    if not chunks:
        return []

    query_identifiers = _extract_identifiers(query)
    if query_identifiers:
        exact_matches = [
            chunk
            for chunk in chunks
            if query_identifiers.intersection(_extract_identifiers(_chunk_search_blob(chunk)))
        ]
        return exact_matches[:max_chunks]

    grounded_chunks = _select_grounded_chunks(query, chunks, max_chunks=max_chunks)
    supporting_chunks = list(grounded_chunks)
    seen_ids = {chunk.chunk_id for chunk in supporting_chunks}

    for chunk in chunks:
        if chunk.chunk_id in seen_ids:
            continue
        supporting_chunks.append(chunk)
        seen_ids.add(chunk.chunk_id)
        if len(supporting_chunks) >= max_chunks:
            break

    return supporting_chunks


def _summarize_chunks(chunks, max_sentences: int = 2) -> str:
    if not chunks:
        return ""

    sentences = []
    for chunk in chunks:
        sentence = _first_sentence(chunk.text)
        if sentence and sentence not in sentences:
            sentences.append(sentence)
        if len(sentences) >= max_sentences:
            break

    if not sentences:
        return ""

    if len(sentences) == 1:
        return sentences[0]

    return f"{sentences[0]} {sentences[1]}"


def _chunk_source_label(chunk) -> str:
    source_name = chunk.metadata.get("source_name") or chunk.doc_id or "unknown source"
    section = chunk.section or chunk.metadata.get("section") or "General"
    return f"{source_name} | {section}"


def _chunk_excerpt(chunk, max_chars: int = 120) -> str:
    excerpt = _first_sentence(chunk.text).strip()
    if len(excerpt) <= max_chars:
        return excerpt
    return excerpt[: max_chars - 1].rstrip() + "..."


def _extract_signal_phrases(chunks) -> list[str]:
    text = " ".join(
        " ".join(
            [
                chunk.text,
                chunk.section or "",
                str(chunk.metadata.get("doc_type", "")),
                str(chunk.metadata.get("source_name", "")),
                str(chunk.metadata.get("title", "")),
            ]
        )
        for chunk in chunks
    ).lower()

    phrase_patterns = [
        ("liquidity risk", "liquidity risk"),
        ("covenant", "covenant pressure"),
        ("loan delinquen", "loan delinquencies"),
        ("capital ratio", "capital ratios"),
        ("margin", "margin pressure"),
        ("revenue", "revenue momentum"),
        ("earnings", "earnings"),
        ("reserve", "reserve movement"),
        ("settlement", "settlement timing"),
        ("claim", "claim activity"),
        ("underwriting", "underwriting"),
        ("compliance", "compliance"),
        ("disclosure", "disclosure risk"),
        ("sec", "SEC filing"),
        ("guidance", "guidance"),
    ]

    phrases: list[str] = []
    for needle, phrase in phrase_patterns:
        if needle in text and phrase not in phrases:
            phrases.append(phrase)

    identifiers = sorted(_extract_identifiers(text))
    for identifier in identifiers:
        if identifier not in phrases:
            phrases.append(identifier)

    if not phrases:
        phrases.append("the items below")

    return phrases[:4]


def _build_analyst_note(query: str, chunks) -> str:
    signal_phrases = _extract_signal_phrases(chunks)
    if len(signal_phrases) == 1:
        bottom_line = f"The retrieved passages point to {signal_phrases[0]}."
    else:
        bottom_line = "The retrieved passages point to " + ", ".join(signal_phrases[:-1]) + f", and {signal_phrases[-1]}."

    evidence_lines = []
    for chunk in chunks[:3]:
        evidence_lines.append(f"- {_chunk_source_label(chunk)}: {_chunk_excerpt(chunk)}")

    return "\n".join(
        [
            f"Bottom line: {bottom_line}",
            "Evidence:",
            *evidence_lines,
        ]
    )


def generate_answer(query: str, chunks, retrieval_mode: str, chunking_strategy: str) -> str:
    grounded_chunks = _select_grounded_chunks(query, chunks)
    if _should_refuse(query, grounded_chunks):
        return _build_refusal_answer(query)
    supporting_chunks = _select_supporting_chunks(query, chunks)
    note = _build_analyst_note(query, supporting_chunks)
    if note:
        return note

    summary = _summarize_chunks(grounded_chunks)
    if summary:
        return summary

    return "The retrieved documents do not provide enough detail for a concise answer."
