from types import TracebackType
from typing import Protocol, Self

from rag_assistant.application.ports.chunk_embedding_repository import (
    ChunkEmbeddingRepository,
)
from rag_assistant.application.ports.document_chunk_repository import (
    DocumentChunkRepository,
)
from rag_assistant.application.ports.document_repository import (
    DocumentRepository,
)
from rag_assistant.application.ports.document_version_repository import (
    DocumentVersionRepository,
)


class UnitOfWork(Protocol):
    """Coordinate repositories within a single database transaction."""

    documents: DocumentRepository
    document_versions: DocumentVersionRepository
    document_chunks: DocumentChunkRepository
    chunk_embeddings: ChunkEmbeddingRepository

    def __enter__(self) -> Self:
        """Start or enter the unit of work."""
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Exit the unit of work and roll back if necessary."""
        ...

    def commit(self) -> None:
        """Persist all changes within the current transaction."""
        ...

    def rollback(self) -> None:
        """Discard all changes within the current transaction."""
        ...