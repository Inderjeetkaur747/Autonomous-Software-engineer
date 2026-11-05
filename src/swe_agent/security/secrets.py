"""Find and hide secrets. One list of patterns for the whole project."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

PLACEHOLDER = "[REDACTED]"

# A whole private key block (used when hiding text).
_PEM_BLOCK = re.compile(
    r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----.*?"
    r"(?:-----END (?:[A-Z]+ )*PRIVATE KEY-----|\Z)",
    re.S,
)

_SPECIFIC = (
    ("private_key", re.compile(r"-----BEGIN (?:[A-Z]+ )*PRIVATE KEY-----")),
    ("groq_api_key", re.compile(r"gsk_[A-Za-z0-9]{20,}")),
    ("github_token", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    ("github_pat", re.compile(r"github_pat_[A-Za-z0-9_]{20,}")),
    ("aws_access_key", re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("slack_token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}")),
    ("stripe_key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{20,}")),
    ("openai_style_key", re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}")),
    (
        "jwt",
        re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
    ),
)

# name = "value"  where the name suggests a secret and the value looks random.
_GENERIC = re.compile(
    r"(?P<name>[A-Za-z0-9_.-]*(?:api[_-]?key|secret|token|passwd|password)[A-Za-z0-9_.-]*)"
    r"\s*[:=]\s*(?P<quote>[\"'])(?P<value>[^\"'\s]{12,})(?P=quote)",
    re.IGNORECASE,
)
_PLACEHOLDER_HINTS = (
    "example", "changeme", "your", "xxxx", "placeholder", "dummy", "sample",
    "todo", "redacted", "<", ">", "${", "{{", "%(",
)
MIN_ENTROPY = 3.0


@dataclass(frozen=True)
class SecretFinding:
    kind: str
    line: int  # 1-based line inside the scanned text
    preview: str  # masked: never the full secret


def shannon_entropy(value: str) -> float:
    """Bits of randomness per character. 'aaaa' = 0. A real key is above 3."""
    if not value:
        return 0.0
    total = len(value)
    return -sum((n / total) * math.log2(n / total) for n in Counter(value).values())


def _looks_real(value: str) -> bool:
    lowered = value.lower()
    if any(hint in lowered for hint in _PLACEHOLDER_HINTS):
        return False
    return shannon_entropy(value) >= MIN_ENTROPY


def _mask(value: str) -> str:
    return f"{value[:4]}**** ({len(value)} chars)"


def scan_text(text: str) -> list[SecretFinding]:
    """Find secrets. Findings never contain the full secret."""
    findings: list[SecretFinding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        spans: list[tuple[int, int]] = []
        for kind, pattern in _SPECIFIC:
            for match in pattern.finditer(line):
                spans.append(match.span())
                findings.append(SecretFinding(kind, number, _mask(match.group(0))))
        for match in _GENERIC.finditer(line):
            start, end = match.span("value")
            already_found = any(s <= start and end <= e for s, e in spans)
            if not already_found and _looks_real(match.group("value")):
                found = SecretFinding("hardcoded_secret", number, _mask(match.group("value")))
                findings.append(found)
    return findings


def scan_added_lines(diff_text: str) -> list[SecretFinding]:
    """Scan only the lines a patch ADDS. Removed lines are ignored."""
    findings: list[SecretFinding] = []
    for number, line in enumerate(diff_text.splitlines(), start=1):
        if line.startswith("+") and not line.startswith("+++"):
            for found in scan_text(line[1:]):
                findings.append(SecretFinding(found.kind, number, found.preview))
    return findings


def _redact_generic(match: re.Match[str]) -> str:
    value = match.group("value")
    if not _looks_real(value):
        return match.group(0)
    whole = match.group(0)
    start = match.start("value") - match.start(0)
    end = match.end("value") - match.start(0)
    return whole[:start] + PLACEHOLDER + whole[end:]


def redact(text: str) -> str:
    """Replace secrets with [REDACTED]. Variable names are kept, values are removed."""
    text = _PEM_BLOCK.sub(PLACEHOLDER, text)
    for _kind, pattern in _SPECIFIC:
        text = pattern.sub(PLACEHOLDER, text)
    return _GENERIC.sub(_redact_generic, text)