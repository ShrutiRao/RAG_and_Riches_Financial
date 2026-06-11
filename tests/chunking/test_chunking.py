from rag_and_riches_financial.data.sample_documents import build_sample_corpus
from rag_and_riches_financial.preprocessing.text_cleaner import clean_document_text
from rag_and_riches_financial.chunking.fixed_size import chunk_fixed_size
from rag_and_riches_financial.chunking.semantic import chunk_semantic


def test_fixed_and_semantic_chunkers_return_tagged_chunks():
    corpus = build_sample_corpus()
    cleaned = clean_document_text(corpus[0].text)
    fixed_chunks = chunk_fixed_size(corpus[0], cleaned, chunk_size=40, overlap=10)
    semantic_chunks = chunk_semantic(corpus[0], cleaned)

    assert fixed_chunks[0].chunking_strategy == "fixed"
    assert semantic_chunks[0].chunking_strategy == "semantic"
    assert fixed_chunks[0].doc_id == corpus[0].doc_id
