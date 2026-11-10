import io
import logging

from swe_agent.logging_setup import SecretRedactingFilter, setup_logging


def test_groq_key_is_redacted():
    fake_key = "gsk_" + "a" * 30
    record = logging.LogRecord(
        "test", logging.INFO, "file.py", 1, f"calling api with {fake_key}", None, None
    )
    SecretRedactingFilter().filter(record)
    assert fake_key not in record.getMessage()
    assert "[REDACTED]" in record.getMessage()


def test_normal_message_unchanged():
    record = logging.LogRecord("test", logging.INFO, "file.py", 1, "hello world", None, None)
    SecretRedactingFilter().filter(record)
    assert record.getMessage() == "hello world"


def test_setup_logging_writes_to_the_given_stream_and_redacts():
    stream = io.StringIO()
    root = logging.getLogger()
    old_handlers, old_level = list(root.handlers), root.level
    try:
        setup_logging("INFO", stream=stream)
        logging.getLogger("t").info("hello %s", "gsk_" + "a" * 30)
        assert "hello" in stream.getvalue()
        assert "gsk_" not in stream.getvalue()
    finally:
        root.handlers[:] = old_handlers
        root.setLevel(old_level)