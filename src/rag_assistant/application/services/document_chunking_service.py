from rag_assistant.application.dto.parsed_document import ParsedDocument
from rag_assistant.application.dto.text_chunk import TextChunk
from rag_assistant.application.services.text_chunker import TextChunker


class DocumentChunkingService:
    """Coordinate chunking across all sections of a parsed document."""

    def __init__(self, text_chunker: TextChunker) -> None:
        self._text_chunker = text_chunker

    def chunk_document(
        self,
        document: ParsedDocument,
    ) -> tuple[TextChunk, ...]:
        """Chunk each document section without mixing section boundaries."""

        chunks: list[TextChunk] = []

        for section in document.sections:
            chunks.extend(self._text_chunker.chunk(section))

        return tuple(chunks)