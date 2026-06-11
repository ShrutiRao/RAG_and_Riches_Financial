from typing import List

from .embeddings import FinancialEmbedder


class FinancialRetriever:
    """A placeholder retriever for financial documents."""

    def __init__(self, embedder: FinancialEmbedder):
        self.embedder = embedder
        self.documents = [
            {
                "title": "Q1 Revenue Performance",
                "content": "Revenue growth accelerated for the mid-cap technology sector in Q1, led by cloud services and recurring software subscriptions.",
            },
            {
                "title": "Market Sentiment Summary",
                "content": "Investors are bullish on earnings stability and are monitoring margin expansion in SaaS companies.",
            },
        ]

    def search(self, query: str, top_k: int = 2) -> List[dict]:
        query_vector = self.embedder.embed(query)
        results = []

        for doc in self.documents:
            score = self._score(query_vector, self.embedder.embed(doc["content"]))
            results.append({"score": score, "document": doc})

        results.sort(key=lambda item: item["score"], reverse=True)
        return [item["document"] for item in results[:top_k]]

    def _score(self, query_vector, doc_vector) -> float:
        if not any(query_vector) or not any(doc_vector):
            return 0.0
        return float(sum(q * d for q, d in zip(query_vector, doc_vector)))
