import logging

from swe_agent.logging_setup import SecretRedactingFilter


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