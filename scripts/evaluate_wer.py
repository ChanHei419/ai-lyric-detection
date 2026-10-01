"""Evaluate transcription quality with Word Error Rate (WER).

Input CSV columns: ``audio`` (file path) and ``reference`` (ground-truth lyrics).

Usage:
    python scripts/evaluate_wer.py data/test.csv --limit 50
"""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV with audio,reference columns")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only N files")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/eval"), help="Output folder"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from lyric_detector.config import PipelineConfig
    from lyric_detector.evaluation import evaluate_transcriptions
    from lyric_detector.pipeline import LyricPipeline

    pipeline = LyricPipeline(PipelineConfig())

    def transcribe(audio_path: str) -> str:
        result = pipeline.run(audio_path, output_dir=args.output_dir)
        return result.raw_text

    summary = evaluate_transcriptions(args.csv, transcribe, limit=args.limit)
    print(f"Files evaluated: {summary['files']}")
    print(f"Mean WER: {summary['mean_wer']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
