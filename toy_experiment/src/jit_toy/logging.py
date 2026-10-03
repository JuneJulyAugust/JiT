"""Configure only at the CLI boundary, never as an import side effect."""

import sys
from pathlib import Path
from typing import Optional

from loguru import logger

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "{name}:{function} ({file.path}:{line}) | {message}"
)


def configure_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    logger.remove()
    logger.add(sys.stderr, level=level, format=LOG_FORMAT, colorize=False, diagnose=False)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logger.add(log_file, level=level, format=LOG_FORMAT, colorize=False, diagnose=False)
