from rag_and_riches_financial.embeddings import FinancialEmbedder, get_embedder


def test_embedding_provider_factory_returns_openai_and_bge():
    openai_embedder = get_embedder("openai")
    bge_embedder = get_embedder("bge")

    assert openai_embedder.provider_name == "openai"
    assert bge_embedder.provider_name == "bge"
    assert len(openai_embedder.embed_query("risk disclosure")) == 128
    assert len(bge_embedder.embed_query("risk disclosure")) == 128


def test_bge_and_openai_embeddings_are_not_identical():
    openai_embedder = get_embedder("openai")
    bge_embedder = get_embedder("bge")

    openai_vector = openai_embedder.embed_query("liquidity risk and covenant pressure")
    bge_vector = bge_embedder.embed_query("liquidity risk and covenant pressure")

    assert openai_vector != bge_vector


def test_financial_embedder_keeps_existing_behavior():
    embedder = FinancialEmbedder()
    vector = embedder.embed("Financial growth")

    assert len(vector) == 128
    assert vector[0] == 9
