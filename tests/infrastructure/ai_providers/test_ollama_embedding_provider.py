import httpx
import pytest

from rag_assistant.core.exceptions import EmbeddingError
from rag_assistant.infrastructure.ai_providers.embeddings.ollama_embedding_provider import (
    OllamaEmbeddingProvider,
)


@pytest.fixture
def embedding_response() -> dict:
    return {
        "model": "nomic-embed-text",
        "embeddings": [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    }


@pytest.fixture
def provider_factory():
    clients: list[httpx.Client] = []

    def create(handler):
        client = httpx.Client(
            transport=httpx.MockTransport(handler)
        )
        clients.append(client)

        return OllamaEmbeddingProvider(
            base_url="http://localhost:11434",
            model="nomic-embed-text",
            client=client,
        )

    yield create

    for client in clients:
        client.close()


def test_embed_batch_returns_embeddings(
    provider_factory,
    embedding_response,
):
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/embed"

        payload = __import__("json").loads(request.content)
        assert payload["model"] == "nomic-embed-text"
        assert len(payload["input"]) == 2

        return httpx.Response(200, json=embedding_response)

    provider = provider_factory(handler)

    result = provider.embed_batch(("first text", "second text"))

    assert result == (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )


def test_embed_returns_single_embedding(provider_factory):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3]]},
        )

    provider = provider_factory(handler)

    result = provider.embed("test text")

    assert result == (0.1, 0.2, 0.3)


def test_embed_batch_rejects_empty_input(provider_factory):
    provider = provider_factory(
        lambda request: httpx.Response(200, json={})
    )

    with pytest.raises(ValueError, match="empty batch"):
        provider.embed_batch(())


def test_embed_batch_rejects_empty_text(provider_factory):
    provider = provider_factory(
        lambda request: httpx.Response(200, json={})
    )

    with pytest.raises(ValueError, match="Input texts"):
        provider.embed_batch(("valid text", " "))


def test_embed_batch_raises_on_http_error(provider_factory):
    provider = provider_factory(
        lambda request: httpx.Response(500)
    )

    with pytest.raises(EmbeddingError, match="communicate"):
        provider.embed_batch(("test text",))


def test_embed_batch_rejects_wrong_embedding_count(provider_factory):
    provider = provider_factory(
        lambda request: httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3]]},
        )
    )

    with pytest.raises(EmbeddingError, match="unexpected number"):
        provider.embed_batch(("first", "second"))


def test_embed_batch_rejects_invalid_vector(provider_factory):
    provider = provider_factory(
        lambda request: httpx.Response(
            200,
            content=b'{"embeddings": [[0.1, Infinity]]}',
            headers={"Content-Type": "application/json"},
        )
    )

    with pytest.raises(EmbeddingError, match="finite numbers"):
        provider.embed_batch(("test text",))


def test_embed_batch_rejects_inconsistent_dimensions(provider_factory):
    provider = provider_factory(
        lambda request: httpx.Response(
            200,
            json={
                "embeddings": [
                    [0.1, 0.2],
                    [0.3, 0.4, 0.5],
                ]
            },
        )
    )

    with pytest.raises(EmbeddingError, match="inconsistent dimensions"):
        provider.embed_batch(("first", "second"))