from collections.abc import Sequence

from rag_assistant.application.ports.embedding_provider import EmbeddingProvider
from rag_assistant.core.exceptions import EmbeddingError
from rag_assistant.domain.entities.chunk_embedding import ChunkEmbedding
from rag_assistant.domain.entities.document_chunk import DocumentChunk
from rag_assistant.domain.entities.embedding_profile import EmbeddingProfile


class EmbeddingService:
    """Generate and associate embeddings with document chunks."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        profile: EmbeddingProfile,
    ) -> tuple[ChunkEmbedding, ...]:
        """Generate embeddings for a batch of document chunks."""

        if not chunks:
            raise ValueError("Cannot generate embeddings for an empty batch.")

        texts = tuple(chunk.content for chunk in chunks)
        vectors = self._provider.embed_batch(texts)

        if len(vectors) != len(chunks):
            raise EmbeddingError(
                "Embedding provider returned an unexpected number of vectors."
            )

        embeddings: list[ChunkEmbedding] = []

        for chunk, vector in zip(chunks, vectors, strict=True):
            if len(vector) != profile.dimensions:
                raise EmbeddingError(
                    "Embedding vector dimensions do not match the profile."
                )

            embeddings.append(
                ChunkEmbedding(
                    document_chunk_id=chunk.id,
                    embedding_profile_id=profile.id,
                    vector=vector,
                )
            )

        return tuple(embeddings)