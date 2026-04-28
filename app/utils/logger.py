"""
app/utils/logger.py
────────────────────
Configures application-wide structured logging.

Uses Python's standard `logging` module with a clean formatter that
produces consistent, readable output for both development (human-readable)
and production (JSON-friendly) environments.

Usage
-----
    from app.utils.logger import get_logger

    log = get_logger(__name__)
    log.info("Ticket received", extra={"ticket_id": "TKT-A1B2C"})
"""

import logging
import sys
from typing import Optional

from app.config import get_settings


class _ColourFormatter(logging.Formatter):
    """
    ANSI-coloured formatter for development terminals.

    Colours are applied only when the output stream is a real TTY,
    so log files and CI pipelines always receive plain text.
    """

    LEVEL_COLOURS = {
        logging.DEBUG:    "\033[36m",   # Cyan
        logging.INFO:     "\033[32m",   # Green
        logging.WARNING:  "\033[33m",   # Yellow
        logging.ERROR:    "\033[31m",   # Red
        logging.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD  = "\033[1m"

    def format(self, record: logging.LogRecord) -> str:
        colour = self.LEVEL_COLOURS.get(record.levelno, "")
        level  = f"{colour}{self.BOLD}{record.levelname:<8}{self.RESET}"
        name   = f"\033[2m{record.name}\033[0m"                # dim
        msg    = super().format(record)

        # Replace the default levelname in the formatted string with coloured one
        plain_level = f"{record.levelname:<8}"
        return msg.replace(plain_level, level, 1).replace(record.name, name, 1)


def configure_logging(level: Optional[str] = None) -> None:
    """
    Call once at application startup to set up the root logger.

    Parameters
    ----------
    level:
        Override the log level from settings (useful in tests).
    """
    settings = get_settings()
    log_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    handler = logging.StreamHandler(sys.stdout)

    if sys.stdout.isatty() and settings.is_development:
        formatter = _ColourFormatter(fmt=fmt, datefmt=datefmt)
    else:
        formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)

    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(log_level)

    # Silence noisy third-party loggers
    for noisy in ("httpcore", "httpx", "openai._base_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Parameters
    ----------
    name:
        Conventionally pass ``__name__`` so the logger path mirrors the
        module hierarchy (e.g. ``app.services.workflow``).
    """
    return logging.getLogger(name)
