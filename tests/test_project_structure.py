
from pathlib import Path


def test_application_package_exists() -> None:
    package_path = Path("src/rag_assistant")

    assert package_path.exists()
    assert (package_path / "__init__.py").exists()
