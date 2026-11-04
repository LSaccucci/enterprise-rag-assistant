from typing import Protocol, Sequence


class Tokenizer(Protocol):
    """Port for encoding and decoding text tokens."""

    def encode(self, text: str) -> tuple[int, ...]:
        """Encode text into token IDs."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode token IDs into text."""
        ...