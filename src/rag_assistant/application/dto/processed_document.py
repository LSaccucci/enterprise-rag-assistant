from dataclasses import dataclass

from rag_assistant.application.dto.parsed_document import ParsedDocument
from rag_assistant.application.dto.text_chunk import TextChunk


@dataclass(frozen=True, slots=True)
class ProcessedDocument:
    """Represent a parsed document and its generated chunks."""

    document: ParsedDocument
    chunks: tuple[TextChunk, ...]