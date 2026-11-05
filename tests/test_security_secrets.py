from swe_agent.logging_setup import redact_secrets
from swe_agent.security import redact, scan_added_lines, scan_text, shannon_entropy

GROQ = "gsk_" + "a" * 30
GITHUB = "ghp_" + "b" * 30
AWS = "AKIA" + "IOSFODNN7EXAMPLE"
REAL_LOOKING = "aB3dE6gH9jK2mN5pQ8"


def test_entropy():
    assert shannon_entropy("aaaaaaaa") == 0.0
    assert shannon_entropy(REAL_LOOKING) > 4.0
    assert shannon_entropy("") == 0.0


def test_known_formats_are_found():
    text = f"a = '{GROQ}'\nb = {GITHUB}\nc = {AWS}\n-----BEGIN RSA PRIVATE KEY-----\n"
    kinds = {f.kind for f in scan_text(text)}
    assert {"groq_api_key", "github_token", "aws_access_key", "private_key"} <= kinds


def test_findings_never_contain_the_full_secret():
    finding = scan_text(f"key = {GROQ}")[0]
    assert GROQ not in finding.preview
    assert finding.preview.startswith("gsk_")
    assert finding.line == 1


def test_generic_rule_needs_a_secret_name_and_a_random_value():
    assert scan_text(f'db_password: "{REAL_LOOKING}"')[0].kind == "hardcoded_secret"
    assert scan_text('API_KEY = "your-api-key-goes-here"') == []  # placeholder
    assert scan_text('token = "aaaaaaaaaaaaaaaaaaaa"') == []  # not random
    assert scan_text(f'color = "{REAL_LOOKING}"') == []  # name is not secret-like


def test_a_known_format_is_not_reported_twice():
    findings = scan_text(f'token = "{GITHUB}"')
    assert [f.kind for f in findings] == ["github_token"]


def test_redact_keeps_names_and_removes_values():
    text = f'password = "{REAL_LOOKING}"\nkey = {GROQ}'
    out = redact(text)
    assert REAL_LOOKING not in out
    assert GROQ not in out
    assert out.startswith("password =")
    assert out.count("[REDACTED]") == 2


def test_redact_removes_whole_private_key_block():
    text = "x\n-----BEGIN PRIVATE KEY-----\nAAAA\nBBBB\n-----END PRIVATE KEY-----\ny"
    out = redact(text)
    assert "AAAA" not in out
    assert out == "x\n[REDACTED]\ny"


def test_only_added_lines_of_a_diff_are_scanned():
    diff = (
        "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n"
        f"-old = '{GROQ}'\n"
        f"+new = '{GITHUB}'\n"
    )
    findings = scan_added_lines(diff)
    assert [f.kind for f in findings] == ["github_token"]
    assert findings[0].line == 5


def test_logging_helper_uses_the_same_scanner():
    assert AWS not in redact_secrets(f"found {AWS} in config")