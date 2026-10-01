"""Evaluation metrics for lyric transcription."""

from __future__ import annotations


def _edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    """Levenshtein distance between two token sequences."""
    if not reference:
        return len(hypothesis)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + cost,  # substitution
                )
            )
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Compute WER = (substitutions + deletions + insertions) / reference length.

    Returns a fraction where 0.0 is a perfect transcript.
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0

    return _edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens)
