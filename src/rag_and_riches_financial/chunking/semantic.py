from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.models.documents import FinancialDocument


def chunk_semantic(document: FinancialDocument, cleaned_text: str) -> list[ChunkRecord]:
    sentences = [sentence.strip() for sentence in cleaned_text.split(".") if sentence.strip()]
    if not sentences:
        return []

    return [
        ChunkRecord(
            chunk_id=f"{document.doc_id}-semantic-{index}",
            doc_id=document.doc_id,
            chunk_index=index,
            chunking_strategy="semantic",
            section=document.section,
            text=f"{sentence}.",
            metadata={
                "doc_type": document.doc_type,
                "source_name": document.source_name,
                "date": document.date,
                "title": document.title,
                "tags": document.tags,
            },
        )
        for index, sentence in enumerate(sentences)
    ]
