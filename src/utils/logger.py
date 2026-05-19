"""统一日志配置模块。

使用方式:
    from src.utils.logger import setup_logging
    setup_logging()  # 在 main.py 启动时调用一次

各模块获取 logger:
    import logging
    logger = logging.getLogger("agent")
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from rich.logging import RichHandler

LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOG_DIR / "agent.log"

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | None = None, log_to_file: bool = True) -> None:
    """Initialize the 'agent' logger with console (Rich) and optional file handlers.

    Args:
        level: Log level string (DEBUG/INFO/WARNING/ERROR). Defaults to LOG_LEVEL env var.
        log_to_file: Whether to also write logs to logs/agent.log.
    """
    log_level = (level or LOG_LEVEL).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    logger = logging.getLogger("agent")
    logger.setLevel(numeric_level)
    logger.handlers.clear()
    logger.propagate = False

    # Console handler — Rich formatted
    console_handler = RichHandler(
        level=numeric_level,
        show_path=False,
        markup=False,
        rich_tracebacks=True,
    )
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # File handler — rotating, 5MB per file, keep 3 backups
    if log_to_file:
        LOG_DIR.mkdir(exist_ok=True)
        file_handler = RotatingFileHandler(
            LOG_FILE,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding="utf-8",
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))
        logger.addHandler(file_handler)
