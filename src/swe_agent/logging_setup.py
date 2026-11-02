"""Logging setup with automatic secret redaction."""

import logging
import re
import sys

# Patterns of common secret formats. If one appears in a log, we hide it.
_SECRET_PATTERNS = [
    re.compile(r"gsk_[A-Za-z0-9]{20,}"),          # Groq keys
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),    # GitHub tokens
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),  # GitHub fine-grained tokens
]


class SecretRedactingFilter(logging.Filter):
    """Replaces anything that looks like a secret with [REDACTED]."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        for pattern in _SECRET_PATTERNS:
            message = pattern.sub("[REDACTED]", message)
        record.msg = message
        record.args = ()  # message is already fully formatted
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once, at application start."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    )
    handler.addFilter(SecretRedactingFilter())

    root = logging.getLogger()
    root.handlers.clear()  # avoid duplicate lines if called twice
    root.addHandler(handler)
    root.setLevel(level.upper())