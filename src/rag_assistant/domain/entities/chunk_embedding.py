import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class ChunkEmbedding:
    """Represents a vector embedding generated from a document chunk."""

    document_chunk_id: UUID
    embedding_profile_id: UUID
    vector: tuple[float, ...]

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not isinstance(self.vector, tuple):
            raise ValueError("Embedding vector must be a tuple.")

        if not self.vector:
            raise ValueError("Embedding vector cannot be empty.")

        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in self.vector
        ):
            raise ValueError(
                "Embedding vector must contain only finite numbers."
            )

        object.__setattr__(
            self,
            "vector",
            tuple(float(value) for value in self.vector),
        )

        if self.created_at.utcoffset() is None:
            raise ValueError("Creation timestamp must be timezone-aware.")