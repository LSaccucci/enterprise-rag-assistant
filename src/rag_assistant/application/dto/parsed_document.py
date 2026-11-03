from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParsedSection:
    """Represents a section extracted from an HTML document."""

    section_path: tuple[str, ...]
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.section_path, tuple):
            raise ValueError("Section path must be a tuple.")

        if any(
            not isinstance(section, str) or not section.strip()
            for section in self.section_path
        ):
            raise ValueError("Section path cannot contain empty sections.")

        if not self.content.strip():
            raise ValueError("Section content cannot be empty.")


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    """Represents the structured content extracted from an HTML document."""

    title: str
    canonical_url: str
    sections: tuple[ParsedSection, ...]

    def __post_init__(self) -> None:
        if not self.title.strip():
            raise ValueError("Document title cannot be empty.")

        if not self.canonical_url.strip():
            raise ValueError("Canonical URL cannot be empty.")

        if not isinstance(self.sections, tuple) or not self.sections:
            raise ValueError("Document must contain at least one section.")

        if any(
            not isinstance(section, ParsedSection)
            for section in self.sections
        ):
            raise ValueError("Sections must contain ParsedSection instances.")