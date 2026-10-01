"""Dataset-level evaluation for transcription quality."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

from lyric_detector.metrics import word_error_rate
from lyric_detector.text import clean_lyrics

TranscribeFn = Callable[[str], str]


def evaluate_transcriptions(
    csv_path: str | Path,
    transcribe: TranscribeFn,
    limit: int | None = None,
) -> dict:
    """Compute mean WER over a CSV of ``audio,reference`` pairs.

    ``transcribe`` is any callable that maps an audio path to a transcript,
    which keeps this function testable without loading ML models.
    """
    with open(csv_path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if limit is not None:
        rows = rows[:limit]

    scores: list[float] = []
    for row in rows:
        hypothesis = clean_lyrics(transcribe(row["audio"]))
        reference = clean_lyrics(row["reference"])
        scores.append(word_error_rate(reference, hypothesis))

    mean = sum(scores) / len(scores) if scores else 0.0
    return {"files": len(scores), "mean_wer": mean}
