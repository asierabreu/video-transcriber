"""Edit-ratio guard that stops the LLM from rewriting transcript content (§6.4 step 4)."""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def normalize_words(text: str) -> list[str]:
    """Lower-case word tokens with punctuation and apostrophes removed.

    Punctuation and casing fixes therefore cost nothing; only word changes count.
    """
    return _TOKEN_RE.findall(text.lower().replace("'", "").replace("’", ""))


def word_edit_distance(a: list[str], b: list[str]) -> int:
    """Levenshtein distance over word lists."""
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, wa in enumerate(a, start=1):
        current = [i]
        for j, wb in enumerate(b, start=1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (wa != wb)))
        previous = current
    return previous[-1]


def edit_ratio(original: str, cleaned: str) -> float:
    """Word edits needed to turn `original` into `cleaned`, relative to the original length."""
    a, b = normalize_words(original), normalize_words(cleaned)
    if not a:
        return 0.0 if not b else 1.0
    return word_edit_distance(a, b) / len(a)


def accept_edit(original: str, cleaned: str, max_ratio: float) -> bool:
    """True if the cleaned text is close enough to the original to be trusted."""
    if not cleaned.strip():
        return not original.strip()
    return edit_ratio(original, cleaned) <= max_ratio
