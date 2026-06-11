from rag_and_riches_financial import config


def test_load_environment_invokes_dotenv_loader(monkeypatch):
    calls = []

    def fake_load_dotenv(*_args, **_kwargs):
        calls.append(True)
        return True

    monkeypatch.setattr(config, "load_dotenv", fake_load_dotenv)

    result = config.load_environment()

    assert result is True
    assert calls == [True]


def test_app_config_reads_current_environment(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "test-nebius-key")
    monkeypatch.setenv("NEBIUS_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    monkeypatch.setenv("LLAMA_CLOUD_API_KEY", "test-llama-key")

    app_config = config.AppConfig()

    assert app_config.nebius_api_key == "test-nebius-key"
    assert app_config.nebius_base_url == "https://example.test/v1"
    assert app_config.pinecone_api_key == "test-pinecone-key"
    assert app_config.llamaparse_api_key == "test-llama-key"
