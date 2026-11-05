"""Logging setup with automatic secret redaction."""

import logging
import sys


def redact_secrets(text: str) -> str:
    """Replace anything that looks like a secret with [REDACTED]."""
    # Imported here so logging starts fast and no import cycle can form.
    from swe_agent.security.secrets import redact

    return redact(text)


class SecretRedactingFilter(logging.Filter):
    """Redacts secrets from every log line before it is printed."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact_secrets(record.getMessage())
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