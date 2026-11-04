from rag_assistant.application.dto.parsed_document import (
    ParsedDocument,
    ParsedSection,
)
from rag_assistant.application.dto.text_chunk import TextChunk
from rag_assistant.application.services.document_processing_service import (
    DocumentProcessingService,
)


class FakeHtmlParser:
    """Fake HTML parser for application service tests."""

    def __init__(self, document: ParsedDocument) -> None:
        self._document = document
        self.received_html: str | None = None
        self.received_url: str | None = None

    def parse(self, html: str, source_url: str) -> ParsedDocument:
        self.received_html = html
        self.received_url = source_url
        return self._document


class FakeDocumentChunkingService:
    """Fake document chunking service for application service tests."""

    def __init__(self, chunks: tuple[TextChunk, ...]) -> None:
        self._chunks = chunks
        self.received_document: ParsedDocument | None = None

    def chunk_document(
        self,
        document: ParsedDocument,
    ) -> tuple[TextChunk, ...]:
        self.received_document = document
        return self._chunks


def test_process_parses_html_and_chunks_document() -> None:
    document = ParsedDocument(
        title="Python Guide",
        canonical_url="https://docs.example.com/python",
        sections=(
            ParsedSection(
                section_path=("Python", "Installation"),
                content="Install Python on your system.",
            ),
        ),
    )

    chunks = (
        TextChunk(
            content="Install Python on your system.",
            token_count=6,
            section_path=("Python", "Installation"),
        ),
    )

    parser = FakeHtmlParser(document=document)
    chunking_service = FakeDocumentChunkingService(chunks=chunks)

    service = DocumentProcessingService(
        html_parser=parser,
        chunking_service=chunking_service,
    )

    html = "<html><body><h1>Python Guide</h1></body></html>"
    source_url = "https://docs.example.com/python"

    result = service.process(
        html=html,
        source_url=source_url,
    )

    assert parser.received_html == html
    assert parser.received_url == source_url
    assert chunking_service.received_document is document
    assert result.document is document
    assert result.chunks == chunks