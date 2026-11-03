
from typing import Protocol, Sequence
from uuid import UUID

from rag_assistant.domain.entities.document_chunk import DocumentChunk


class DocumentChunkRepository(Protocol):
    """Port for accessing document chunks."""

    def save_many(self, chunks: Sequence[DocumentChunk]) -> None:
        """Persist multiple chunks belonging to a document version."""
        ...

    def get_by_document_version_id(
        self,
        document_version_id: UUID,
    ) -> tuple[DocumentChunk, ...]:
        """Return all chunks belonging to a document version."""
        ...
