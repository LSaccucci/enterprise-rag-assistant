from rag_assistant.application.dto.processed_document import ProcessedDocument
from rag_assistant.application.ports.html_parser import HtmlParser
from rag_assistant.application.services.document_chunking_service import (
    DocumentChunkingService,
)


class DocumentProcessingService:
    """Coordinate HTML parsing and document chunking."""

    def __init__(
        self,
        html_parser: HtmlParser,
        chunking_service: DocumentChunkingService,
    ) -> None:
        self._html_parser = html_parser
        self._chunking_service = chunking_service

    def process(
        self,
        html: str,
        source_url: str,
    ) -> ProcessedDocument:
        """Parse HTML and generate chunks from the resulting document."""

        document = self._html_parser.parse(
            html=html,
            source_url=source_url,
        )

        chunks = self._chunking_service.chunk_document(document)

        return ProcessedDocument(
            document=document,
            chunks=chunks,
        )