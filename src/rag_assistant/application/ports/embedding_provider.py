
from typing import Protocol, Sequence


class EmbeddingProvider(Protocol):
    """Port for generating text embeddings."""

    def embed(self, text: str) -> tuple[float, ...]:
        """Generate an embedding for a single text."""
        ...

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Generate embeddings for multiple texts.

        The returned embeddings must preserve the input order.
        """
        ...
