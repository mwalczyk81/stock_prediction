import logging

from rich.console import Console
from rich.logging import RichHandler


def setup_logger() -> logging.Logger:
    handler = RichHandler(rich_tracebacks=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handler.setFormatter(formatter)

    logger = logging.getLogger("pr_analysis")

    if not logger.handlers:
        logger.addHandler(handler)

        logger.setLevel(logging.INFO)

        logger.propagate = False  # Prevent double logging.

    return logger


logger = setup_logger()

console = Console()
