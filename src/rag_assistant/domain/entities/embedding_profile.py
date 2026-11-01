
from dataclasses import dataclass, field
from datetime import UTC, datetime
from uuid import UUID, uuid4


@dataclass(frozen=True, slots=True)
class EmbeddingProfile:
    """Describes a model configuration used to generate embeddings."""

    provider: str
    model_name: str
    dimensions: int

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if not self.provider.strip():
            raise ValueError("Embedding provider cannot be empty.")

        if not self.model_name.strip():
            raise ValueError("Embedding model name cannot be empty.")

        if (
            not isinstance(self.dimensions, int)
            or isinstance(self.dimensions, bool)
            or self.dimensions < 1
        ):
            raise ValueError("Embedding dimensions must be a positive integer.")

        if self.created_at.utcoffset() is None:
            raise ValueError("Creation timestamp must be timezone-aware.")
