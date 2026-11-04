from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Represents a text chunk produced by the chunking process."""

    content: str
    token_count: int
    section_path: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.content.strip():
            raise ValueError("Chunk content cannot be empty.")

        if (
            not isinstance(self.token_count, int)
            or isinstance(self.token_count, bool)
            or self.token_count < 1
        ):
            raise ValueError("Token count must be a positive integer.")

        if not isinstance(self.section_path, tuple):
            raise ValueError("Section path must be a tuple.")