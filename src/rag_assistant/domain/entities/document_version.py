
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4

_SHA256_PATTERN = re.compile(r"^[a-fA-F0-9]{64}$")


@dataclass(frozen=True, slots=True)
class DocumentVersion:
    """Represents an immutable snapshot of a document."""

    document_id: UUID
    version_number: int
    content_hash: str
    acquired_at: datetime

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if (
                not isinstance(self.version_number, int)
                or isinstance(self.version_number, bool)
                or self.version_number < 1
        ):
            raise ValueError("Version number must be a positive integer.")

        if not _SHA256_PATTERN.fullmatch(self.content_hash):
            raise ValueError("Content hash must be a valid SHA-256 hash.")

        if self.acquired_at.utcoffset() is None:
            raise ValueError("Acquisition timestamp must be timezone-aware.")

        if self.created_at.utcoffset() is None:
            raise ValueError("Creation timestamp must be timezone-aware.")
