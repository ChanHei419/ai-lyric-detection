"""End-to-end pipeline: audio -> vocals -> transcript -> clean lyrics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from lyric_detector.config import PipelineConfig
from lyric_detector.text import format_lyrics

ProgressCallback = Callable[[str, int], None]


@dataclass
class TranscriptionResult:
    """Everything the pipeline produced for one input file."""

    source: str
    vocals_path: str
    raw_text: str
    lyrics: str


def _report(progress: ProgressCallback | None, stage: str, percent: int) -> None:
    if progress is not None:
        progress(stage, percent)


class LyricPipeline:
    """Coordinate vocal separation, transcription, and post-processing."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._separator = None
        self._transcriber = None

    @property
    def separator(self):
        if self._separator is None:
            from lyric_detector.separator import VocalSeparator

            model = self.config.model
            self._separator = VocalSeparator(
                model=model.separator_model, device=model.device
            )
        return self._separator

    @property
    def transcriber(self):
        if self._transcriber is None:
            from lyric_detector.transcriber import SpeechTranscriber

            model = self.config.model
            self._transcriber = SpeechTranscriber(
                model_name=model.asr_model, device=model.device
            )
        return self._transcriber

    def run(
        self,
        audio_path: str | Path,
        output_dir: str | Path = "outputs",
        progress: ProgressCallback | None = None,
    ) -> TranscriptionResult:
        """Run the full pipeline and return the transcription result."""
        audio = Path(audio_path)
        if not audio.exists():
            raise FileNotFoundError(f"audio file not found: {audio}")

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        _report(progress, "separating vocals", 5)
        vocals_path = self.separator.separate_to_file(audio, destination)

        _report(progress, "transcribing", 55)
        raw_text = self.transcriber.transcribe(
            vocals_path, sample_rate=self.config.model.sample_rate
        )

        _report(progress, "cleaning lyrics", 90)
        lyrics = format_lyrics(raw_text)

        _report(progress, "done", 100)
        return TranscriptionResult(
            source=str(audio),
            vocals_path=str(vocals_path),
            raw_text=raw_text,
            lyrics=lyrics,
        )
