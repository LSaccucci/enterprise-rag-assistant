
import pytest

from rag_assistant.core.exceptions import (
    DatabaseError,
    DocumentParsingError,
    EmbeddingError,
    IngestionError,
    RAGAssistantError,
)


def test_custom_exceptions_inherit_from_base_exception() -> None:
    assert issubclass(RAGAssistantError, Exception)
    assert issubclass(DatabaseError, RAGAssistantError)
    assert issubclass(IngestionError, RAGAssistantError)


def test_document_parsing_error_inherits_from_ingestion_error() -> None:
    assert issubclass(DocumentParsingError, IngestionError)


def test_embedding_error_inherits_from_ingestion_error() -> None:
    assert issubclass(EmbeddingError, IngestionError)


def test_ingestion_error_can_catch_document_parsing_error() -> None:
    with pytest.raises(IngestionError):
        raise DocumentParsingError("Invalid document format")
