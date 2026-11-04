from rag_assistant.application.dto.parsed_document import (
    ParsedDocument,
    ParsedSection,
)
from rag_assistant.application.services.document_chunking_service import (
    DocumentChunkingService,
)
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


def test_chunk_document_preserves_section_order() -> None:
    tokenizer = FakeTokenizer()
    text_chunker = TextChunker(
        tokenizer=tokenizer,
        chunk_size=3,
        chunk_overlap=1,
    )
    service = DocumentChunkingService(text_chunker=text_chunker)

    document = ParsedDocument(
        title="Python Guide",
        canonical_url="https://docs.example.com/python",
        sections=(
            ParsedSection(
                section_path=("Python", "Installation"),
                content="install python package",
            ),
            ParsedSection(
                section_path=("Python", "Configuration"),
                content="configure environment variables",
            ),
        ),
    )

    chunks = service.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].section_path == ("Python", "Installation")
    assert chunks[1].section_path == ("Python", "Configuration")


def test_chunk_document_does_not_mix_sections() -> None:
    tokenizer = FakeTokenizer()
    text_chunker = TextChunker(
        tokenizer=tokenizer,
        chunk_size=5,
        chunk_overlap=2,
    )
    service = DocumentChunkingService(text_chunker=text_chunker)

    document = ParsedDocument(
        title="Python Guide",
        canonical_url="https://docs.example.com/python",
        sections=(
            ParsedSection(
                section_path=("Python", "Installation"),
                content="alpha beta",
            ),
            ParsedSection(
                section_path=("Python", "Configuration"),
                content="gamma delta",
            ),
        ),
    )

    chunks = service.chunk_document(document)

    assert len(chunks) == 2
    assert chunks[0].content == "alpha beta"
    assert chunks[1].content == "gamma delta"
    assert chunks[0].section_path != chunks[1].section_path