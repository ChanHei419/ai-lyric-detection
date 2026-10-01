"""Command-line interface for the lyric detection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from lyric_detector.config import ModelConfig, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lyric-detector",
        description="Extract and transcribe lyrics from a song using Demucs + Wav2Vec2.",
    )
    parser.add_argument("audio", type=Path, help="Input audio file (mp3, wav, ...)")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output directory"
    )
    parser.add_argument("--separator-model", default="htdemucs", help="Demucs model")
    parser.add_argument(
        "--asr-model",
        default="facebook/wav2vec2-base-960h",
        help="Hugging Face CTC checkpoint",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps"], help="Torch device"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    from lyric_detector.pipeline import LyricPipeline  # heavy imports happen here

    config = PipelineConfig(
        model=ModelConfig(
            separator_model=args.separator_model,
            asr_model=args.asr_model,
            device=args.device,
        )
    )
    pipeline = LyricPipeline(config)

    def progress(stage: str, percent: int) -> None:
        print(f"[{percent:3d}%] {stage}")

    result = pipeline.run(args.audio, args.output_dir, progress)

    print("\n--- Lyrics ---")
    print(result.lyrics)

    lyrics_file = Path(args.output_dir) / f"{args.audio.stem}_lyrics.txt"
    lyrics_file.write_text(result.lyrics, encoding="utf-8")
    print(f"\nSaved to {lyrics_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
