from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.models.documents import FinancialDocument


def chunk_fixed_size(
    document: FinancialDocument,
    cleaned_text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[ChunkRecord]:
    words = cleaned_text.split()
    if not words:
        return []

    chunks: list[ChunkRecord] = []
    start = 0
    index = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        chunk_text = " ".join(words[start:end])
        chunks.append(
            ChunkRecord(
                chunk_id=f"{document.doc_id}-fixed-{index}",
                doc_id=document.doc_id,
                chunk_index=index,
                chunking_strategy="fixed",
                section=document.section,
                text=chunk_text,
                metadata={
                    "doc_type": document.doc_type,
                    "source_name": document.source_name,
                    "date": document.date,
                    "title": document.title,
                    "tags": document.tags,
                },
            )
        )
        if end == len(words):
            break
        start = max(0, end - overlap)
        index += 1
    return chunks
