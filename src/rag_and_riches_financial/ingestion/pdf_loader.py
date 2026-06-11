from __future__ import annotations

import json
from pathlib import Path

from rag_and_riches_financial.ingestion.pdf_parser import parse_pdf
from rag_and_riches_financial.models.documents import FinancialDocument


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "pdf_manifest.json"


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _load_manifest() -> dict:
    return json.loads(_manifest_path().read_text(encoding="utf-8"))


def load_pdf_documents() -> list[FinancialDocument]:
    manifest = _load_manifest()
    documents: list[FinancialDocument] = []
    for entry in manifest.get("documents", []):
        pdf_path = _data_dir() / entry["filename"]
        parsed_text = parse_pdf(pdf_path)
        documents.append(
            FinancialDocument(
                doc_id=entry["doc_id"],
                doc_type=entry["doc_type"],
                source_name=entry["source_name"],
                company=entry["company"],
                date=entry["date"],
                section=entry["section"],
                title=entry["title"],
                text=parsed_text,
                tags=entry.get("tags", []),
            )
        )
    return documents
