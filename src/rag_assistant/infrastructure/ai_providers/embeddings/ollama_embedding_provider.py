import math
from collections.abc import Sequence
from typing import Any

import httpx

from rag_assistant.application.ports.embedding_provider import EmbeddingProvider
from rag_assistant.core.exceptions import EmbeddingError


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate text embeddings through the local Ollama API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not base_url.strip():
            raise ValueError("Ollama base URL cannot be empty.")

        if not model.strip():
            raise ValueError("Embedding model cannot be empty.")

        if timeout <= 0:
            raise ValueError("Timeout must be positive.")

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or httpx.Client(timeout=timeout)
        self._owns_client = client is None

    def embed(self, text: str) -> tuple[float, ...]:
        """Generate an embedding for a single text."""

        embeddings = self.embed_batch((text,))
        return embeddings[0]

    def embed_batch(
        self,
        texts: Sequence[str],
    ) -> tuple[tuple[float, ...], ...]:
        """Generate embeddings for a batch of texts."""

        if not texts:
            raise ValueError("Cannot generate embeddings for an empty batch.")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Input texts cannot be empty.")

        try:
            response = self._client.post(
                f"{self._base_url}/api/embed",
                json={
                    "model": self._model,
                    "input": list(texts),
                },
            )
            response.raise_for_status()
            payload = response.json()

        except httpx.HTTPError as exc:
            raise EmbeddingError(
                "Failed to communicate with the Ollama embedding API."
            ) from exc
        except ValueError as exc:
            raise EmbeddingError(
                "Ollama returned an invalid JSON response."
            ) from exc

        return self._parse_embeddings(payload, expected_count=len(texts))

    @staticmethod
    def _parse_embeddings(
        payload: Any,
        expected_count: int,
    ) -> tuple[tuple[float, ...], ...]:
        """Validate and convert the embedding response."""

        if not isinstance(payload, dict):
            raise EmbeddingError("Ollama returned an invalid response structure.")

        raw_embeddings = payload.get("embeddings")

        if not isinstance(raw_embeddings, list):
            raise EmbeddingError("Ollama response does not contain valid embeddings.")

        if len(raw_embeddings) != expected_count:
            raise EmbeddingError(
                "Ollama returned an unexpected number of embeddings."
            )

        embeddings: list[tuple[float, ...]] = []

        for raw_vector in raw_embeddings:
            if not isinstance(raw_vector, list) or not raw_vector:
                raise EmbeddingError("Ollama returned an invalid embedding vector.")

            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                for value in raw_vector
            ):
                raise EmbeddingError(
                    "Embedding vectors must contain only finite numbers."
                )

            embeddings.append(tuple(float(value) for value in raw_vector))

        dimensions = {len(vector) for vector in embeddings}

        if len(dimensions) != 1:
            raise EmbeddingError(
                "Ollama returned embeddings with inconsistent dimensions."
            )

        return tuple(embeddings)

    def close(self) -> None:
        """Close the HTTP client if this adapter created it."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "OllamaEmbeddingProvider":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()