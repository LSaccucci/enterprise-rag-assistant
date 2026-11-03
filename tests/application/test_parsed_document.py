import pytest

from rag_assistant.application.dto.parsed_document import (
    ParsedDocument,
    ParsedSection,
)


def test_parsed_section_can_be_created() -> None:
    section = ParsedSection(
        section_path=("Library Reference", "Coroutines"),
        content="Coroutines are a generalized form of subroutines.",
    )

    assert section.section_path == ("Library Reference", "Coroutines")
    assert section.content.startswith("Coroutines")


def test_parsed_section_allows_empty_section_path() -> None:
    section = ParsedSection(
        section_path=(),
        content="Introduction to asyncio.",
    )

    assert section.section_path == ()


def test_parsed_section_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="Section content cannot be empty"):
        ParsedSection(section_path=(), content=" ")


def test_parsed_document_can_be_created() -> None:
    section = ParsedSection(
        section_path=("Coroutines",),
        content="Coroutines are a generalized form of subroutines.",
    )

    document = ParsedDocument(
        title="Python asyncio",
        canonical_url="https://docs.python.org/3/library/asyncio.html",
        sections=(section,),
    )

    assert document.title == "Python asyncio"
    assert document.canonical_url.endswith("asyncio.html")
    assert len(document.sections) == 1


def test_parsed_document_rejects_empty_title() -> None:
    with pytest.raises(ValueError, match="Document title cannot be empty"):
        ParsedDocument(
            title=" ",
            canonical_url="https://docs.example.com",
            sections=(
                ParsedSection(section_path=(), content="Content"),
            ),
        )


def test_parsed_document_rejects_empty_sections() -> None:
    with pytest.raises(ValueError, match="at least one section"):
        ParsedDocument(
            title="Example",
            canonical_url="https://docs.example.com",
            sections=(),
        )


def test_parsed_document_rejects_invalid_section_type() -> None:
    with pytest.raises(ValueError, match="ParsedSection instances"):
        ParsedDocument(
            title="Example",
            canonical_url="https://docs.example.com",
            sections=("Invalid section",),
        )