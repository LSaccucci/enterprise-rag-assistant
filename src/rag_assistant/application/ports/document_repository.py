from typing import Protocol
from uuid import UUID

from rag_assistant.domain.entities.document import Document


class DocumentRepository(Protocol):
    """Port for accessing logical documents."""

    def get_by_source_and_url(
        self,
        source_id: UUID,
        canonical_url: str,
    ) -> Document | None:
        """Find a document by source and canonical URL."""
        ...

    def save(self, document: Document) -> None:
        """Persist a document."""
        ...