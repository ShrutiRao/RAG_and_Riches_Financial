from rag_and_riches_financial.retrieval.query_rewrite import rewrite_query
from rag_and_riches_financial.retrieval.rerank import rerank_candidates


def test_retrieval_modes_return_rankable_text():
    rewritten = rewrite_query("What is the liquidity risk?")
    reranked = rerank_candidates("What is the liquidity risk?", ["chunk a", "chunk b"])

    assert isinstance(rewritten, str)
    assert reranked[0] in {"chunk a", "chunk b"}
