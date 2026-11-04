import pytest

from rag_assistant.application.dto.parsed_document import ParsedSection
from rag_assistant.application.services.text_chunker import TextChunker


class FakeTokenizer:
    """Deterministic tokenizer for unit tests."""

    def __init__(self) -> None:
        self._words: list[str] = []

    def encode(self, text: str) -> tuple[int, ...]:
        self._words = text.split()
        return tuple(range(len(self._words)))

    def decode(self, token_ids: tuple[int, ...]) -> str:
        return " ".join(self._words[token_id] for token_id in token_ids)


@pytest.fixture
def tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def chunker(tokenizer: FakeTokenizer) -> TextChunker:
    return TextChunker(
        tokenizer=tokenizer,
        chunk_size=5,
        chunk_overlap=2,
    )


def test_chunk_splits_text_with_overlap(
    chunker: TextChunker,
) -> None:
    section = ParsedSection(
        section_path=("Python", "Installation"),
        content="one two three four five six seven eight",
    )

    chunks = chunker.chunk(section)

    assert len(chunks) == 2
    assert chunks[0].content == "one two three four five"
    assert chunks[1].content == "four five six seven eight"
    assert chunks[0].token_count == 5
    assert chunks[1].token_count == 5


def test_chunk_preserves_section_path(
    chunker: TextChunker,
) -> None:
    section_path = ("Python", "Installation", "Windows")

    section = ParsedSection(
        section_path=section_path,
        content="one two three four five six seven",
    )

    chunks = chunker.chunk(section)

    assert all(chunk.section_path == section_path for chunk in chunks)


def test_chunk_returns_single_chunk_for_short_text(
    chunker: TextChunker,
) -> None:
    section = ParsedSection(
        section_path=("Python",),
        content="one two three",
    )

    chunks = chunker.chunk(section)

    assert len(chunks) == 1
    assert chunks[0].content == "one two three"
    assert chunks[0].token_count == 3


def test_chunk_handles_exact_chunk_size(
    chunker: TextChunker,
) -> None:
    section = ParsedSection(
        section_path=("Python",),
        content="one two three four five",
    )

    chunks = chunker.chunk(section)

    assert len(chunks) == 1
    assert chunks[0].token_count == 5


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (-1, 0),
        (5, -1),
        (5, 5),
        (5, 6),
    ],
)
def test_chunk_rejects_invalid_configuration(
    tokenizer: FakeTokenizer,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    with pytest.raises(ValueError):
        TextChunker(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )