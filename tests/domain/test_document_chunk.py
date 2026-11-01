
from datetime import datetime
from uuid import UUID, uuid4

import pytest

from rag_assistant.domain.entities.document_chunk import DocumentChunk


def test_document_chunk_creation() -> None:
    document_version_id = uuid4()
    section_path = ("Python Library", "asyncio")

    chunk = DocumentChunk(
        document_version_id=document_version_id,
        chunk_index=0,
        content="Asyncio provides infrastructure for asynchronous programming.",
        token_count=12,
        section_path=section_path,
    )

    assert isinstance(chunk.id, UUID)
    assert chunk.document_version_id == document_version_id
    assert chunk.chunk_index == 0
    assert chunk.content == (
        "Asyncio provides infrastructure for asynchronous programming."
    )
    assert chunk.token_count == 12
    assert chunk.section_path == section_path


def test_document_chunk_has_creation_timestamp() -> None:
    chunk = DocumentChunk(
        document_version_id=uuid4(),
        chunk_index=0,
        content="Example content.",
        token_count=3,
    )

    assert chunk.created_at.tzinfo is not None


@pytest.mark.parametrize("chunk_index", [-1, 1.5, True])
def test_document_chunk_rejects_invalid_index(
    chunk_index: int,
) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        DocumentChunk(
            document_version_id=uuid4(),
            chunk_index=chunk_index,
            content="Example content.",
            token_count=3,
        )


@pytest.mark.parametrize("content", ["", "   ", "\n\t"])
def test_document_chunk_rejects_empty_content(content: str) -> None:
    with pytest.raises(ValueError, match="content cannot be empty"):
        DocumentChunk(
            document_version_id=uuid4(),
            chunk_index=0,
            content=content,
            token_count=1,
        )


@pytest.mark.parametrize("token_count", [0, -1, 1.5, True])
def test_document_chunk_rejects_invalid_token_count(
    token_count: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        DocumentChunk(
            document_version_id=uuid4(),
            chunk_index=0,
            content="Example content.",
            token_count=token_count,
        )


def test_document_chunk_rejects_invalid_section_path() -> None:
    with pytest.raises(ValueError, match="Section path must be a tuple"):
        DocumentChunk(
            document_version_id=uuid4(),
            chunk_index=0,
            content="Example content.",
            token_count=3,
            section_path=["Python", "asyncio"],  # type: ignore[arg-type]
        )


def test_document_chunk_rejects_empty_section() -> None:
    with pytest.raises(ValueError, match="empty sections"):
        DocumentChunk(
            document_version_id=uuid4(),
            chunk_index=0,
            content="Example content.",
            token_count=3,
            section_path=("Python", "   "),
        )


def test_document_chunk_is_immutable() -> None:
    chunk = DocumentChunk(
        document_version_id=uuid4(),
        chunk_index=0,
        content="Example content.",
        token_count=3,
    )

    with pytest.raises(AttributeError):
        chunk.content = "Modified content."  # type: ignore[misc]


def test_document_chunk_rejects_naive_creation_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="Creation timestamp must be timezone-aware",
    ):
        DocumentChunk(
            document_version_id=uuid4(),
            chunk_index=0,
            content="Example content.",
            token_count=3,
            created_at=datetime(2026, 11, 1, 10, 0),
        )
