from typing import Protocol
from uuid import UUID

from rag_assistant.domain.entities.document_source import DocumentSource


class DocumentSourceRepository(Protocol):
    """Port for accessing registered document sources."""

    def get_by_id(self, source_id: UUID) -> DocumentSource | None:
        """Return a source by its identifier, if it exists."""
        ...

    def get_by_base_url(self, base_url: str) -> DocumentSource | None:
        """Return a source by its base URL, if it exists."""
        ...