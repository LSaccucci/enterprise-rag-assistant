
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class Document:
    """Represents a logical document in the knowledge base."""

    source_id: UUID
    title: str
    canonical_url: str

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Document title cannot be empty.")

        if not self.canonical_url.strip():
            raise ValueError("Document canonical URL cannot be empty.")
