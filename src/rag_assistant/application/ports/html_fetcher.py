from typing import Protocol


class HtmlFetcher(Protocol):
    """Port for securely fetching HTML documents."""

    def fetch(self, url: str) -> str:
        """Fetch HTML content from the given URL."""
        ...