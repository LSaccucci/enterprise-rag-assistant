from bs4 import BeautifulSoup
from bs4.element import NavigableString, PageElement, Tag

from rag_assistant.application.dto.parsed_document import (
    ParsedDocument,
    ParsedSection,
)
from rag_assistant.core.exceptions import DocumentParsingError


class BeautifulSoupHtmlParser:
    """Parse HTML documents using BeautifulSoup."""

    def parse(self, html: str, source_url: str) -> ParsedDocument:
        """Extract structured content from an HTML document."""

        if not isinstance(html, str) or not html.strip():
            raise DocumentParsingError("HTML content cannot be empty.")

        if not isinstance(source_url, str) or not source_url.strip():
            raise DocumentParsingError("Source URL cannot be empty.")

        soup = BeautifulSoup(html, "html.parser")

        title = self._extract_title(soup)

        for element in soup(["script", "style", "noscript"]):
            element.decompose()

        content_element = self._select_content_element(soup)
        self._remove_non_content_elements(content_element)

        sections = self._extract_sections(content_element)

        if not sections:
            raise DocumentParsingError(
                "HTML document does not contain extractable text."
            )

        return ParsedDocument(
            title=title,
            canonical_url=source_url,
            sections=sections,
        )

    @staticmethod
    def _extract_title(soup: BeautifulSoup) -> str:
        """Extract the document title from the HTML."""

        if soup.title:
            title = soup.title.get_text(" ", strip=True)
            if title:
                return title

        heading = soup.find("h1")
        if heading:
            title = heading.get_text(" ", strip=True)
            if title:
                return title

        raise DocumentParsingError(
            "HTML document does not contain a valid title."
        )

    @staticmethod
    def _extract_sections(
        content_element: Tag,
    ) -> tuple[ParsedSection, ...]:
        """Extract text sections while preserving heading hierarchy."""

        sections: list[ParsedSection] = []
        heading_stack: list[str] = []
        current_section_path: tuple[str, ...] = ()
        current_content: list[str] = []

        def flush_section() -> None:
            nonlocal current_content

            content = " ".join(current_content).strip()

            if content:
                sections.append(
                    ParsedSection(
                        section_path=current_section_path,
                        content=content,
                    )
                )

            current_content = []

        def visit(node: PageElement) -> None:
            nonlocal current_section_path

            if isinstance(node, NavigableString):
                text = " ".join(str(node).split())

                if text:
                    current_content.append(text)

                return

            if not isinstance(node, Tag):
                return

            if node.name in {"script", "style", "noscript"}:
                return

            if (
                node.name
                and len(node.name) == 2
                and node.name[0] == "h"
                and node.name[1] in "123456"
            ):
                heading_text = node.get_text(" ", strip=True)

                if not heading_text:
                    return

                flush_section()

                heading_level = int(node.name[1])

                del heading_stack[heading_level - 1:]
                heading_stack.append(heading_text)

                current_section_path = tuple(heading_stack)

                return

            if node.name == "li":
                item_text = node.get_text(" ", strip=True)

                if item_text:
                    current_content.append(f"- {item_text}")

                return

            for child in node.children:
                visit(child)

        visit(content_element)
        flush_section()

        return tuple(sections)

    @staticmethod
    def _select_content_element(soup: BeautifulSoup) -> Tag:
        """Select the most likely main content container."""

        main_element = soup.find("main")
        if main_element:
            return main_element

        article_element = soup.find("article")
        if article_element:
            return article_element

        return soup.body or soup

    @staticmethod
    def _remove_non_content_elements(content_element: Tag) -> None:
        """Remove common non-content elements from the selected container."""

        for element in content_element.find_all(
                ["nav", "footer", "aside"]
        ):
            element.decompose()