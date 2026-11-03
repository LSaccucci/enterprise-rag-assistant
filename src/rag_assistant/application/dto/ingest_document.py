from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID


class IngestionOutcome(StrEnum):
    """Describe the outcome of a document ingestion operation."""

    CREATED = "created"
    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class IngestDocumentCommand:
    """Input data required to ingest a document."""

    source_id: UUID
    url: str


@dataclass(frozen=True, slots=True)
class IngestDocumentResult:
    """Result returned after ingesting a document."""

    document_id: UUID
    version_id: UUID
    outcome: IngestionOutcome
    chunk_count: int

    def __post_init__(self) -> None:
        if self.chunk_count < 0:
            raise ValueError("Chunk count cannot be negative.")