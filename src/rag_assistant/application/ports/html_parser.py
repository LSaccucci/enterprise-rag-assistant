
from typing import Protocol

from rag_assistant.application.dto.parsed_document import ParsedDocument


class HtmlParser(Protocol):
    """Port for parsing HTML documents into structured content."""

    def parse(self, html: str, source_url: str) -> ParsedDocument:
        """Parse HTML and return its structured document representation."""
        ...
