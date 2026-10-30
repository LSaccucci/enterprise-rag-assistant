
import logging

from rag_assistant.core.logging import setup_logging


def test_setup_logging_configures_info_level() -> None:
    setup_logging(debug=False)

    logger = logging.getLogger()

    assert logger.level == logging.INFO


def test_setup_logging_configures_debug_level() -> None:
    setup_logging(debug=True)

    logger = logging.getLogger()

    assert logger.level == logging.DEBUG
