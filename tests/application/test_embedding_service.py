from collections.abc import Sequence
from uuid import uuid4

import pytest

from rag_assistant.application.services.embedding_service import (
    EmbeddingService,
)
from rag_assistant.core.exceptions import EmbeddingError
from rag_assistant.domain.entities.document_chunk import DocumentChunk
from rag_assistant.domain.entities.embedding_profile import EmbeddingProfile


class FakeEmbeddingProvider:
    """Deterministic embedding provider for unit tests."""

    def __init__(
        self,
        vectors: tuple[tuple[float, ...], ...],
    ) -> None:
        self._vectors = vectors
        self.received_texts: tuple[str, ...] | None = None
        self.call_count = 0

    def embed(self, text: str) -> tuple[float, ...]:
        raise NotImplementedError

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        self.call_count += 1
        self.received_texts = tuple(texts)
        return self._vectors


@pytest.fixture
def chunks() -> tuple[DocumentChunk, ...]:
    document_version_id = uuid4()

    return (
        DocumentChunk(
            document_version_id=document_version_id,
            chunk_index=0,
            content="Install Python.",
            token_count=3,
        ),
        DocumentChunk(
            document_version_id=document_version_id,
            chunk_index=1,
            content="Configure the environment.",
            token_count=4,
        ),
    )


@pytest.fixture
def profile() -> EmbeddingProfile:
    return EmbeddingProfile(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=3,
    )


def test_embed_chunks_returns_embeddings(
    chunks: tuple[DocumentChunk, ...],
    profile: EmbeddingProfile,
) -> None:
    vectors = (
        (0.1, 0.2, 0.3),
        (0.4, 0.5, 0.6),
    )
    provider = FakeEmbeddingProvider(vectors)
    service = EmbeddingService(provider=provider)

    embeddings = service.embed_chunks(chunks, profile)

    assert len(embeddings) == 2
    assert embeddings[0].document_chunk_id == chunks[0].id
    assert embeddings[1].document_chunk_id == chunks[1].id
    assert all(
        embedding.embedding_profile_id == profile.id
        for embedding in embeddings
    )
    assert embeddings[0].vector == vectors[0]
    assert embeddings[1].vector == vectors[1]


def test_embed_chunks_uses_batch_provider_once(
    chunks: tuple[DocumentChunk, ...],
    profile: EmbeddingProfile,
) -> None:
    provider = FakeEmbeddingProvider(
        vectors=((0.1, 0.2, 0.3), (0.4, 0.5, 0.6))
    )
    service = EmbeddingService(provider=provider)

    service.embed_chunks(chunks, profile)

    assert provider.call_count == 1
    assert provider.received_texts == (
        "Install Python.",
        "Configure the environment.",
    )


def test_embed_chunks_rejects_empty_batch(
    profile: EmbeddingProfile,
) -> None:
    provider = FakeEmbeddingProvider(vectors=())
    service = EmbeddingService(provider=provider)

    with pytest.raises(ValueError, match="empty batch"):
        service.embed_chunks((), profile)

    assert provider.call_count == 0


def test_embed_chunks_rejects_mismatched_vector_count(
    chunks: tuple[DocumentChunk, ...],
    profile: EmbeddingProfile,
) -> None:
    provider = FakeEmbeddingProvider(
        vectors=((0.1, 0.2, 0.3),)
    )
    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingError, match="number of vectors"):
        service.embed_chunks(chunks, profile)


def test_embed_chunks_rejects_incorrect_dimensions(
    chunks: tuple[DocumentChunk, ...],
    profile: EmbeddingProfile,
) -> None:
    provider = FakeEmbeddingProvider(
        vectors=(
            (0.1, 0.2),
            (0.4, 0.5, 0.6),
        )
    )
    service = EmbeddingService(provider=provider)

    with pytest.raises(EmbeddingError, match="dimensions"):
        service.embed_chunks(chunks, profile)