
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """Represents a text chunk extracted from a document version."""

    document_version_id: UUID
    chunk_index: int
    content: str
    token_count: int

    section_path: tuple[str, ...] = ()

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if (
            not isinstance(self.chunk_index, int)
            or isinstance(self.chunk_index, bool)
            or self.chunk_index < 0
        ):
            raise ValueError("Chunk index must be a non-negative integer.")

        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty.")

        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count < 1
        ):
            raise ValueError("Token count must be a positive integer.")

        if not isinstance(self.section_path, tuple):
            raise ValueError("Section path must be a tuple.")

        if any(
            not isinstance(section, str) or not section.strip()
            for section in self.section_path
        ):
            raise ValueError("Section path cannot contain empty sections.")

        if self.created_at.utcoffset() is None:
            raise ValueError("Creation timestamp must be timezone-aware.")
