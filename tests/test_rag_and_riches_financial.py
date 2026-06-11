from rag_and_riches_financial.embeddings import FinancialEmbedder
from rag_and_riches_financial.retriever import FinancialRetriever
from rag_and_riches_financial.generator import FinancialGenerator


def test_embedder_returns_vector():
    embedder = FinancialEmbedder()
    vector = embedder.embed("Financial growth")
    assert len(vector) == 128
    assert vector[0] == 9


def test_retriever_returns_documents():
    embedder = FinancialEmbedder()
    retriever = FinancialRetriever(embedder)
    docs = retriever.search("revenue growth")
    assert len(docs) == 2
    assert "title" in docs[0]


def test_generator_returns_summary():
    generator = FinancialGenerator()
    docs = [{"title": "Sample", "content": "Sample content."}]
    answer = generator.generate_answer("Test query", docs)
    assert "Sample" in answer
