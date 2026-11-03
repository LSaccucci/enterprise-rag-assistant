
from typing import Protocol
from uuid import UUID

from rag_assistant.domain.entities.document_version import DocumentVersion


class DocumentVersionRepository(Protocol):
    """Port for accessing document version history."""

    def get_latest_by_document_id(
        self,
        document_id: UUID,
    ) -> DocumentVersion | None:
        """Return the latest successfully persisted version of a document."""
        ...

    def save(self, version: DocumentVersion) -> None:
        """Persist a new document version."""
        ...
