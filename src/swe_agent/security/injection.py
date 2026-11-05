"""Prompt-injection helpers. They REDUCE risk. They cannot remove it."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Zero-width and bidirectional-control characters (used to hide text from humans).
_INVISIBLE_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff]")
# Same, without the BOM (a BOM at the start of a Windows file is normal).
_SUSPICIOUS_RE = re.compile("[\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069]")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_TAG_RE = re.compile(
    r"<(/?)\s*(issue|code_context|analysis|plan|system|assistant|user|instructions?)(\s*>)",
    re.IGNORECASE,
)

# (name, severity, pattern). Severity: high / medium / low.
_PATTERNS = tuple(
    (name, severity, re.compile(pattern, re.IGNORECASE))
    for name, severity, pattern in (
        (
            "ignore_instructions", "high",
            r"\bignore\s+(?:all\s+|any\s+|the\s+|your\s+)*"
            r"(?:previous|prior|above|earlier|preceding)\s+"
            r"(?:instructions?|rules?|prompts?|messages?)",
        ),
        (
            "disregard_rules", "high",
            r"\bdisregard\s+(?:all\s+|any\s+|the\s+|your\s+)*"
            r"(?:previous\s+|prior\s+|above\s+)?(?:instructions?|rules?|guidelines)",
        ),
        (
            "reveal_secrets", "high",
            r"\b(?:reveal|print|show|output|send|leak|exfiltrate|display)\b[^\n]{0,40}"
            r"\b(?:api[_ -]?keys?|secrets?|tokens?|passwords?|credentials?|"
            r"system\s+prompt|environment\s+variables?|\.env)",
        ),
        (
            "hide_from_user", "high",
            r"(?:do\s+not|don't|never)\s+(?:tell|inform|mention|show)\s+(?:this\s+to\s+)?the\s+user",
        ),
        (
            "role_override", "medium",
            r"\byou\s+are\s+now\b|\bnew\s+(?:system\s+)?instructions?\s*:|"
            r"\bact\s+as\s+(?:an?\s+)?(?:admin|root|system|developer)\b",
        ),
        (
            "fake_role_markup", "medium",
            r"</?\s*(?:system|assistant|instructions?)\s*>|\[/?inst\]|<\|im_(?:start|end)\|>",
        ),
        (
            "network_exfiltration", "medium",
            r"\b(?:curl|wget|invoke-webrequest)\b[^\n]{0,80}https?://",
        ),
        (
            "destructive_command", "medium",
            r"\brm\s+-rf\b|\bdel\s+/[sfq]\b|\bformat\s+[a-z]:|\bmkfs\b",
        ),
        (
            "push_or_publish", "medium",
            r"\bgit\s+push\b|\bnpm\s+publish\b|\btwine\s+upload\b",
        ),
        ("encoded_payload", "low", r"\bbase64\s+(?:-d|--decode)\b|\beval\s*\(\s*base64"),
    )
)


@dataclass(frozen=True)
class InjectionFinding:
    name: str
    severity: str
    line: int
    excerpt: str


@dataclass
class InjectionReport:
    findings: list[InjectionFinding] = field(default_factory=list)

    @property
    def risk(self) -> str:
        severities = {f.severity for f in self.findings}
        if "high" in severities:
            return "high"
        if "medium" in severities:
            return "medium"
        return "low" if self.findings else "none"


def _normalize(text: str) -> str:
    """NFKC turns look-alikes (full-width '＜') into plain characters. Then strip hidden ones."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE_RE.sub("", text)
    return _CONTROL_RE.sub("", text)


def clean_untrusted_text(text: str) -> str:
    """Make outside text safer to put in a prompt.

    Removes hidden characters and defuses tags like </issue> or <system>.
    """
    return _TAG_RE.sub(r"<\\\1\2\3", _normalize(text))


def scan_for_injection(text: str, max_findings: int = 20) -> InjectionReport:
    """Flag suspicious phrases. We WARN the human. We do not silently delete text."""
    report = InjectionReport()
    for number, raw_line in enumerate(text.splitlines(), start=1):
        if _SUSPICIOUS_RE.search(raw_line):
            report.findings.append(
                InjectionFinding("hidden_characters", "medium", number, "(invisible characters)")
            )
        line = _normalize(raw_line)
        for name, severity, pattern in _PATTERNS:
            match = pattern.search(line)
            if match:
                excerpt = match.group(0)[:80]
                report.findings.append(InjectionFinding(name, severity, number, excerpt))
        if len(report.findings) >= max_findings:
            break
    return report