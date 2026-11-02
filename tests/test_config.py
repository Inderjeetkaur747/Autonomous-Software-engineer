from swe_agent.config import Settings


def test_defaults_are_safe():
    s = Settings(_env_file=None)  # ignore the real .env during tests
    assert s.max_repair_attempts == 3
    assert s.command_timeout_seconds == 120


def test_secret_is_hidden_when_printed():
    s = Settings(_env_file=None, groq_api_key="gsk_super_secret_value_123")
    assert "gsk_super_secret_value_123" not in repr(s)
    assert "gsk_super_secret_value_123" not in str(s)
    assert s.groq_api_key.get_secret_value() == "gsk_super_secret_value_123"