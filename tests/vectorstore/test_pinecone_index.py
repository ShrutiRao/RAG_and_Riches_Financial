from rag_and_riches_financial.models.chunks import ChunkRecord
from rag_and_riches_financial.vectorstore.pinecone_index import PineconeIndex


class FakePineconeIndex:
    def __init__(self):
        self.upsert_calls = []
        self.search_calls = []

    def upsert_records(self, namespace, records):
        self.upsert_calls.append((namespace, records))

    def search(self, namespace, query, fields=None, rerank=None):
        self.search_calls.append((namespace, query, fields, rerank))
        return {
            "result": {
                "hits": [
                    {
                        "_id": "sec-001-fixed-0",
                        "_score": 0.97,
                        "fields": {
                            "text": "Liquidity risk increased.",
                            "doc_id": "sec-001",
                            "doc_type": "sec_filing",
                            "source_name": "10-K",
                            "date": "2025-12-31",
                            "section": "Risk Factors",
                            "title": "Liquidity and Credit Risk",
                            "chunk_index": 0,
                            "chunking_strategy": "fixed",
                            "tags": ["risk", "liquidity"],
                        },
                    }
                ]
            }
        }


def test_index_adapter_routes_chunks_by_namespace():
    index = PineconeIndex(index=FakePineconeIndex())
    chunk = ChunkRecord(
        chunk_id="sec-001-fixed-0",
        doc_id="sec-001",
        chunk_index=0,
        chunking_strategy="fixed",
        section="Risk Factors",
        text="Liquidity risk increased.",
        metadata={"doc_type": "sec_filing", "source_name": "10-K", "date": "2025-12-31", "title": "Liquidity and Credit Risk", "tags": ["risk", "liquidity"]},
    )

    index.upsert_chunk(chunk)

    namespace, records = index.index.upsert_calls[0]
    assert namespace == "fixed"
    assert records[0]["_id"] == "sec-001-fixed-0"
    assert records[0]["text"] == "Liquidity risk increased."
    assert records[0]["chunk_text"] == "Liquidity risk increased."


def test_index_adapter_searches_with_text_and_reconstructs_chunks():
    fake_index = FakePineconeIndex()
    index = PineconeIndex(index=fake_index)

    results = index.search("liquidity risk", namespace="fixed", top_k=1)

    assert fake_index.search_calls[0][1] == {"inputs": {"text": "liquidity risk"}, "top_k": 1}
    assert results[0].chunk_id == "sec-001-fixed-0"
    assert results[0].text == "Liquidity risk increased."
    assert results[0].metadata["doc_type"] == "sec_filing"
