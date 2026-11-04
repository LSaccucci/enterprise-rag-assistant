from rag_assistant.application.dto.parsed_document import ParsedSection
from rag_assistant.application.dto.text_chunk import TextChunk
from rag_assistant.application.ports.tokenizer import Tokenizer


class TextChunker:
    """Split document sections into overlapping text chunks."""

    def __init__(
        self,
        tokenizer: Tokenizer,
        chunk_size: int = 700,
        chunk_overlap: int = 100,
    ) -> None:
        if (
            not isinstance(chunk_size, int)
            or isinstance(chunk_size, bool)
            or chunk_size < 1
        ):
            raise ValueError("Chunk size must be a positive integer.")

        if (
            not isinstance(chunk_overlap, int)
            or isinstance(chunk_overlap, bool)
            or chunk_overlap < 0
        ):
            raise ValueError("Chunk overlap must be a non-negative integer.")

        if chunk_overlap >= chunk_size:
            raise ValueError("Chunk overlap must be smaller than chunk size.")

        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, section: ParsedSection) -> tuple[TextChunk, ...]:
        """Split a document section into overlapping chunks."""

        token_ids = self._tokenizer.encode(section.content)

        if not token_ids:
            return ()

        step = self._chunk_size - self._chunk_overlap
        chunks: list[TextChunk] = []

        for start in range(0, len(token_ids), step):
            end = min(start + self._chunk_size, len(token_ids))
            chunk_token_ids = token_ids[start:end]

            content = self._tokenizer.decode(chunk_token_ids).strip()

            if content:
                chunks.append(
                    TextChunk(
                        content=content,
                        token_count=len(chunk_token_ids),
                        section_path=section.section_path,
                    )
                )

            if end == len(token_ids):
                break

        return tuple(chunks)