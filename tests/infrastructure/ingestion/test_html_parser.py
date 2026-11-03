import pytest

from rag_assistant.core.exceptions import DocumentParsingError
from rag_assistant.infrastructure.ingestion.html_parser import (
    BeautifulSoupHtmlParser,
)


@pytest.fixture
def parser() -> BeautifulSoupHtmlParser:
    return BeautifulSoupHtmlParser()


def test_parse_extracts_document_title(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Python Documentation</title>
        </head>
        <body>
            <h1>Introduction</h1>
            <p>Welcome to Python documentation.</p>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://docs.python.org/",
    )

    assert result.title == "Python Documentation"


def test_parse_extracts_document_content(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Python Documentation</title>
        </head>
        <body>
            <p>Python is a programming language.</p>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://docs.python.org/",
    )

    assert len(result.sections) == 1
    assert result.sections[0].content == (
        "Python is a programming language."
    )


def test_parse_removes_non_content_elements(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Python Documentation</title>
            <style>.hidden { display: none; }</style>
        </head>
        <body>
            <script>alert("untrusted script");</script>
            <p>Relevant documentation.</p>
            <noscript>Fallback content.</noscript>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://docs.python.org/",
    )

    content = result.sections[0].content

    assert "Relevant documentation." in content
    assert "alert" not in content
    assert "Fallback content." not in content
    assert ".hidden" not in content


def test_parse_uses_h1_when_html_title_is_missing(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <body>
            <h1>FastAPI Tutorial</h1>
            <p>Getting started with FastAPI.</p>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://fastapi.tiangolo.com/",
    )

    assert result.title == "FastAPI Tutorial"


@pytest.mark.parametrize(
    ("html", "source_url"),
    [
        ("", "https://docs.python.org/"),
        ("   ", "https://docs.python.org/"),
        ("<html><body>Content</body></html>", ""),
    ],
)
def test_parse_rejects_invalid_input(
    parser: BeautifulSoupHtmlParser,
    html: str,
    source_url: str,
) -> None:
    with pytest.raises(DocumentParsingError):
        parser.parse(html=html, source_url=source_url)


def test_parse_rejects_html_without_title(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <body>
            <p>Documentation without a title.</p>
        </body>
    </html>
    """

    with pytest.raises(DocumentParsingError):
        parser.parse(
            html=html,
            source_url="https://docs.python.org/",
        )


def test_parse_rejects_html_without_extractable_content(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Python Documentation</title>
        </head>
        <body>
            <script>console.log("test");</script>
        </body>
    </html>
    """

    with pytest.raises(DocumentParsingError):
        parser.parse(
            html=html,
            source_url="https://docs.python.org/",
        )


def test_parse_preserves_source_url(
    parser: BeautifulSoupHtmlParser,
) -> None:
    source_url = "https://docs.python.org/3/"

    result = parser.parse(
        html="""
        <html>
            <head><title>Python Documentation</title></head>
            <body><p>Python documentation.</p></body>
        </html>
        """,
        source_url=source_url,
    )

    assert result.canonical_url == source_url


def test_parse_preserves_section_hierarchy(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Python Documentation</title>
        </head>
        <body>
            <h1>Python Documentation</h1>
            <p>Introduction to Python.</p>

            <h2>Getting Started</h2>
            <p>Install Python on your machine.</p>

            <h3>Windows</h3>
            <p>Download the Windows installer.</p>

            <h3>Linux</h3>
            <p>Use your distribution package manager.</p>

            <h2>Advanced Topics</h2>
            <p>Explore advanced Python features.</p>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://docs.python.org/",
    )

    assert len(result.sections) == 5

    assert result.sections[0].section_path == (
        "Python Documentation",
    )
    assert result.sections[0].content == "Introduction to Python."

    assert result.sections[1].section_path == (
        "Python Documentation",
        "Getting Started",
    )
    assert result.sections[1].content == (
        "Install Python on your machine."
    )

    assert result.sections[2].section_path == (
        "Python Documentation",
        "Getting Started",
        "Windows",
    )
    assert result.sections[2].content == (
        "Download the Windows installer."
    )

    assert result.sections[3].section_path == (
        "Python Documentation",
        "Getting Started",
        "Linux",
    )

    assert result.sections[4].section_path == (
        "Python Documentation",
        "Advanced Topics",
    )


def test_parse_handles_skipped_heading_levels(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Documentation</title>
        </head>
        <body>
            <h1>Documentation</h1>
            <p>Introduction.</p>

            <h3>Installation</h3>
            <p>Installation instructions.</p>

            <h2>Configuration</h2>
            <p>Configuration instructions.</p>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://example.com/docs",
    )

    assert result.sections[1].section_path == (
        "Documentation",
        "Installation",
    )

    assert result.sections[2].section_path == (
        "Documentation",
        "Configuration",
    )


def test_parse_preserves_content_before_first_heading(
    parser: BeautifulSoupHtmlParser,
) -> None:
    html = """
    <html>
        <head>
            <title>Documentation</title>
        </head>
        <body>
            <p>General introduction.</p>

            <h1>Installation</h1>
            <p>Installation instructions.</p>
        </body>
    </html>
    """

    result = parser.parse(
        html=html,
        source_url="https://example.com/docs",
    )

    assert result.sections[0].section_path == ()
    assert result.sections[0].content == "General introduction."

    assert result.sections[1].section_path == ("Installation",)
    assert result.sections[1].content == "Installation instructions."