from pathlib import Path

from rag_and_riches_financial.ingestion.pdf_parser import parse_pdf, parse_pdf_text


def test_pdf_parser_returns_non_empty_text_for_pdf():
    path = Path("src/rag_and_riches_financial/data/SEC Filing 10-K Excerpt.pdf")

    text = parse_pdf_text(path)

    assert text.strip()
    assert len(text) > 100


def test_pdf_parser_falls_back_without_llamaparse():
    path = Path("src/rag_and_riches_financial/data/Earnings Call Transcript Q1 2026.pdf")

    text = parse_pdf(path, use_llamaparse=False)

    assert text.strip()
