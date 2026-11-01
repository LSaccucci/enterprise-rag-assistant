
from datetime import datetime
from uuid import UUID

import pytest

from rag_assistant.domain.entities.embedding_profile import EmbeddingProfile


def test_embedding_profile_creation() -> None:
    profile = EmbeddingProfile(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=768,
    )

    assert isinstance(profile.id, UUID)
    assert profile.provider == "ollama"
    assert profile.model_name == "nomic-embed-text"
    assert profile.dimensions == 768


def test_embedding_profile_has_creation_timestamp() -> None:
    profile = EmbeddingProfile(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=768,
    )

    assert profile.created_at.tzinfo is not None


@pytest.mark.parametrize("provider", ["", "   "])
def test_embedding_profile_rejects_empty_provider(
    provider: str,
) -> None:
    with pytest.raises(ValueError, match="provider cannot be empty"):
        EmbeddingProfile(
            provider=provider,
            model_name="nomic-embed-text",
            dimensions=768,
        )


@pytest.mark.parametrize("model_name", ["", "   "])
def test_embedding_profile_rejects_empty_model_name(
    model_name: str,
) -> None:
    with pytest.raises(ValueError, match="model name cannot be empty"):
        EmbeddingProfile(
            provider="ollama",
            model_name=model_name,
            dimensions=768,
        )


@pytest.mark.parametrize("dimensions", [0, -1, 1.5, True])
def test_embedding_profile_rejects_invalid_dimensions(
    dimensions: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        EmbeddingProfile(
            provider="ollama",
            model_name="nomic-embed-text",
            dimensions=dimensions,
        )


def test_embedding_profile_is_immutable() -> None:
    profile = EmbeddingProfile(
        provider="ollama",
        model_name="nomic-embed-text",
        dimensions=768,
    )

    with pytest.raises(AttributeError):
        profile.model_name = "another-model"  # type: ignore[misc]


def test_embedding_profile_rejects_naive_creation_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="Creation timestamp must be timezone-aware",
    ):
        EmbeddingProfile(
            provider="ollama",
            model_name="nomic-embed-text",
            dimensions=768,
            created_at=datetime(2026, 11, 1, 10, 0),
        )
