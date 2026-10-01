"""Post-processing helpers that turn raw model output into clean lyrics."""

from __future__ import annotations

import re

_PUNCTUATION = re.compile(r"[^a-z0-9'\s]")
_WHITESPACE = re.compile(r"\s+")


def clean_lyrics(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    lowered = text.lower().replace("|", " ")
    without_punctuation = _PUNCTUATION.sub(" ", lowered)
    return _WHITESPACE.sub(" ", without_punctuation).strip()


def merge_repeated_lines(lines: list[str]) -> list[str]:
    """Collapse immediately repeated lines (common with looped audio)."""
    merged: list[str] = []
    for line in lines:
        if not merged or merged[-1] != line:
            merged.append(line)
    return merged


def format_lyrics(raw_text: str, line_length: int = 60) -> str:
    """Wrap a raw transcript into readable lines without breaking words."""
    words = clean_lyrics(raw_text).split()
    lines: list[str] = []
    current: list[str] = []
    length = 0

    for word in words:
        addition = len(word) + (1 if current else 0)
        if length + addition > line_length and current:
            lines.append(" ".join(current))
            current = [word]
            length = len(word)
        else:
            current.append(word)
            length += addition

    if current:
        lines.append(" ".join(current))

    return "\n".join(merge_repeated_lines(lines))
