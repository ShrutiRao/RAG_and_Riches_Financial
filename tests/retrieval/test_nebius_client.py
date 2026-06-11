from rag_and_riches_financial.retrieval.nebius_client import NebiusChatClient, rerank_candidates, rewrite_query


class FakeCompletions:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)

        class Message:
            def __init__(self, content):
                self.content = content

        class Choice:
            def __init__(self, content):
                self.message = Message(content)

        class Response:
            def __init__(self, content):
                self.choices = [Choice(content)]

        return Response(self.content)


class FakeClient:
    def __init__(self, content):
        self.chat = type("Chat", (), {"completions": FakeCompletions(content)})()


def test_rewrite_query_uses_nebius_client():
    client = FakeClient("rephrase about liquidity, covenant pressure, and loan risk")

    rewritten = rewrite_query("What is the liquidity risk?", client=NebiusChatClient(client=client, model="test-model"))

    assert "liquidity" in rewritten


def test_rerank_candidates_can_run_without_nebius_and_preserve_ordering():
    ranked = rerank_candidates(
        "What is the liquidity risk?",
        ["loan covenants and liquidity pressure", "insurance claim reserve adjustment"],
        client=None,
    )

    assert ranked[0] == "loan covenants and liquidity pressure"
