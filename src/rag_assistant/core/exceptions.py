
class RAGAssistantError(Exception):
    """Base exception for the RAG Assistant application."""


class ConfigurationError(RAGAssistantError):
    """Raised when application configuration is invalid."""


class IngestionError(RAGAssistantError):
    """Raised when document ingestion fails."""


class DocumentParsingError(IngestionError):
    """Raised when a document cannot be parsed."""


class EmbeddingError(IngestionError):
    """Raised when embedding generation fails."""


class RetrievalError(RAGAssistantError):
    """Raised when document retrieval fails."""


class DatabaseError(RAGAssistantError):
    """Raised when a database operation fails."""


class LLMProviderError(RAGAssistantError):
    """Raised when an LLM provider operation fails."""
