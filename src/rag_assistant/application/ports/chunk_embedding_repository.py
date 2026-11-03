
from typing import Protocol, Sequence
from uuid import UUID

from rag_assistant.domain.entities.chunk_embedding import ChunkEmbedding


class ChunkEmbeddingRepository(Protocol):
    """Port for accessing chunk embeddings."""

    def save_many(self, embeddings: Sequence[ChunkEmbedding]) -> None:
        """Persist multiple chunk embeddings."""

        ...

    def get_by_chunk_and_profile(
        self,
        chunk_id: UUID,
        embedding_profile_id: UUID,
    ) -> ChunkEmbedding | None:
        """Return an embedding for a chunk and a specific profile."""

        ...
