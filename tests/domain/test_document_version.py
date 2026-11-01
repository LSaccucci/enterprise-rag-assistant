
from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest

from rag_assistant.domain.entities.document_version import DocumentVersion


def test_document_version_creation() -> None:
    document_id = uuid4()
    acquired_at = datetime.now(UTC)
    content_hash = "a" * 64

    version = DocumentVersion(
        document_id=document_id,
        version_number=1,
        content_hash=content_hash,
        acquired_at=acquired_at,
    )

    assert isinstance(version.id, UUID)
    assert version.document_id == document_id
    assert version.version_number == 1
    assert version.content_hash == content_hash
    assert version.acquired_at == acquired_at


def test_document_version_rejects_invalid_version_number() -> None:
    with pytest.raises(
        ValueError,
        match="Version number must be a positive integer",
    ):
        DocumentVersion(
            document_id=uuid4(),
            version_number=0,
            content_hash="a" * 64,
            acquired_at=datetime.now(UTC),
        )


@pytest.mark.parametrize(
    "version_number",
    [1.5, True, False],
)
def test_document_version_rejects_non_integer_version_number(
    version_number: int,
) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        DocumentVersion(
            document_id=uuid4(),
            version_number=version_number,
            content_hash="a" * 64,
            acquired_at=datetime.now(UTC),
        )


@pytest.mark.parametrize(
    "content_hash",
    [
        "",
        "invalid",
        "a" * 63,
        "g" * 64,
    ],
)
def test_document_version_rejects_invalid_hash(
    content_hash: str,
) -> None:
    with pytest.raises(ValueError, match="valid SHA-256 hash"):
        DocumentVersion(
            document_id=uuid4(),
            version_number=1,
            content_hash=content_hash,
            acquired_at=datetime.now(UTC),
        )


def test_document_version_rejects_naive_acquisition_timestamp() -> None:
    with pytest.raises(
        ValueError,
        match="Acquisition timestamp must be timezone-aware",
    ):
        DocumentVersion(
            document_id=uuid4(),
            version_number=1,
            content_hash="a" * 64,
            acquired_at=datetime(2026, 11, 1, 10, 0),
        )


def test_document_version_is_immutable() -> None:
    version = DocumentVersion(
        document_id=uuid4(),
        version_number=1,
        content_hash="a" * 64,
        acquired_at=datetime.now(UTC),
    )

    with pytest.raises(AttributeError):
        version.version_number = 2  # type: ignore[misc]
