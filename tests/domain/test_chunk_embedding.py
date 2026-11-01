from datetime import datetime
from uuid import UUID, uuid4

import pytest

from rag_assistant.domain.entities.chunk_embedding import ChunkEmbedding


def test_chunk_embedding_creation() -> None:
    document_chunk_id = uuid4()
    embedding_profile_id = uuid4()
    vector = (0.12, -0.34, 0.87)

    embedding = ChunkEmbedding(
        document_chunk_id=document_chunk_id,
        embedding_profile_id=embedding_profile_id,
        vector=vector,
    )

    assert isinstance(embedding.id, UUID)
    assert embedding.document_chunk_id == document_chunk_id
    assert embedding.embedding_profile_id == embedding_profile_id
    assert embedding.vector == vector


def test_chunk_embedding_has_creation_timestamp() -> None:
    embedding = ChunkEmbedding(
        document_chunk_id=uuid4(),
        embedding_profile_id=uuid4(),
        vector=(0.12, -0.34),
    )

    assert embedding.created_at.tzinfo is not None


def test_chunk_embedding_rejects_non_tuple_vector() -> None:
    with pytest.raises(ValueError, match="must be a tuple"):
        ChunkEmbedding(
            document_chunk_id=uuid4(),
            embedding_profile_id=uuid4(),
            vector=[0.12, -0.34],  # type: ignore[arg-type]
        )


def test_chunk_embedding_rejects_empty_vector() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ChunkEmbedding(
            document_chunk_id=uuid4(),
            embedding_profile_id=uuid4(),
            vector=(),
        )


@pytest.mark.parametrize(
    "vector",
    [
        (float("nan"),),
        (float("inf"),),
        (float("-inf"),),
        (True,),
        ("invalid",),
    ],
)
def test_chunk_embedding_rejects_invalid_vector_values(
    vector: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="finite numbers"):
        ChunkEmbedding(
            document_chunk_id=uuid4(),
            embedding_profile_id=uuid4(),
            vector=vector,
        )


def test_chunk_embedding_normalizes_values_to_float() -> None:
    embedding = ChunkEmbedding(
        document_chunk_id=uuid4(),
        embedding_profile_id=uuid4(),
        vector=(1, 2, 3),
    )

    assert embedding.vector == (1.0, 2.0, 3.0)
    assert all(isinstance(value, float) for value in embedding.vector)


def test_chunk_embedding_is_immutable() -> None:
    embedding = ChunkEmbedding(
        document_chunk_id=uuid4(),
        embedding_profile_id=uuid4(),
        vector=(0.12, -0.34),
    )

    with pytest.raises(AttributeError):
        embedding.vector = (0.5, 0.6)  # type: ignore[misc]


def test_chunk_embedding_rejects_naive_creation_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="Creation timestamp must be timezone-aware",
    ):
        ChunkEmbedding(
            document_chunk_id=uuid4(),
            embedding_profile_id=uuid4(),
            vector=(0.12, -0.34),
            created_at=datetime(2026, 11, 1, 10, 0),
        )