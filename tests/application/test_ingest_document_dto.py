from uuid import uuid4

import pytest

from rag_assistant.application.dto.ingest_document import (
    IngestDocumentCommand,
    IngestDocumentResult,
    IngestionOutcome,
)


def test_ingest_document_command_can_be_created() -> None:
    source_id = uuid4()
    url = "https://docs.example.com/guide"

    command = IngestDocumentCommand(
        source_id=source_id,
        url=url,
    )

    assert command.source_id == source_id
    assert command.url == url


def test_ingest_document_result_can_be_created() -> None:
    document_id = uuid4()
    version_id = uuid4()

    result = IngestDocumentResult(
        document_id=document_id,
        version_id=version_id,
        outcome=IngestionOutcome.CREATED,
        chunk_count=5,
    )

    assert result.document_id == document_id
    assert result.version_id == version_id
    assert result.outcome is IngestionOutcome.CREATED
    assert result.chunk_count == 5


def test_ingest_document_result_allows_zero_chunks() -> None:
    result = IngestDocumentResult(
        document_id=uuid4(),
        version_id=uuid4(),
        outcome=IngestionOutcome.UNCHANGED,
        chunk_count=0,
    )

    assert result.chunk_count == 0


def test_ingest_document_result_rejects_negative_chunk_count() -> None:
    with pytest.raises(ValueError, match="Chunk count cannot be negative"):
        IngestDocumentResult(
            document_id=uuid4(),
            version_id=uuid4(),
            outcome=IngestionOutcome.CREATED,
            chunk_count=-1,
        )