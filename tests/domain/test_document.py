
from uuid import UUID, uuid4

import pytest

from rag_assistant.domain.entities.document import Document


def test_document_creation() -> None:
    source_id = uuid4()

    document = Document(
        source_id=source_id,
        title="Python Asyncio Documentation",
        canonical_url="https://docs.python.org/3/library/asyncio.html",
    )

    assert isinstance(document.id, UUID)
    assert document.source_id == source_id
    assert document.title == "Python Asyncio Documentation"
    assert document.canonical_url == (
        "https://docs.python.org/3/library/asyncio.html"
    )


def test_document_has_creation_timestamp() -> None:
    document = Document(
        source_id=uuid4(),
        title="FastAPI Documentation",
        canonical_url="https://fastapi.tiangolo.com/",
    )

    assert document.created_at.tzinfo is not None


def test_document_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="title cannot be empty"):
        Document(
            source_id=uuid4(),
            title="   ",
            canonical_url="https://docs.python.org/3/",
        )


def test_document_rejects_empty_url() -> None:
    with pytest.raises(ValueError, match="URL cannot be empty"):
        Document(
            source_id=uuid4(),
            title="Python Documentation",
            canonical_url="   ",
        )


def test_document_is_immutable() -> None:
    document = Document(
        source_id=uuid4(),
        title="Python Documentation",
        canonical_url="https://docs.python.org/3/",
    )

    with pytest.raises(AttributeError):
        document.title = "Modified title"  # type: ignore[misc]
