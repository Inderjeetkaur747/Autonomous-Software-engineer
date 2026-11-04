"""A tokenizer made for source code.

"get_user_by_email" -> ["get_user_by_email", "get", "user", "by", "email"]
"ZeroDivisionError" -> ["zerodivisionerror", "zero", "division", "error"]
"""

import re

_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
# Splits camelCase pieces: HTTPResponse -> HTTP, Response ; parseXml -> parse, Xml
_PART_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|[0-9]+")

# A few very common words that carry no meaning for ranking.
STOPWORDS = frozenset(
    {
        "the", "a", "an", "of", "to", "in", "is", "it", "and", "or", "for", "how",
        "do", "does", "we", "on", "with", "this", "that", "are", "be", "as", "at",
    }
)


def split_identifier(word: str) -> list[str]:
    """Split one identifier into lowercase parts (snake_case and camelCase)."""
    parts: list[str] = []
    for piece in word.split("_"):
        parts.extend(match.lower() for match in _PART_RE.findall(piece))
    return parts


def tokenize_code(text: str, *, drop_stopwords: bool = True) -> list[str]:
    tokens: list[str] = []
    for word in _WORD_RE.findall(text):
        if not word.strip("_"):
            continue  # a lone underscore carries no meaning
        parts = split_identifier(word)
        if parts != [word.lower()]:
            tokens.append(word.lower())  # keep the whole identifier too
        tokens.extend(parts)
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return tokens