
from rag_assistant.core.config import Settings


def test_settings_default_values() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_name == "Enterprise RAG Assistant"
    assert settings.retrieval_top_k == 5
    assert settings.chunk_size == 700
    assert settings.chunk_overlap == 100


def test_settings_reject_invalid_retrieval_top_k() -> None:
    try:
        Settings(_env_file=None, retrieval_top_k=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected invalid retrieval_top_k to raise ValueError")
