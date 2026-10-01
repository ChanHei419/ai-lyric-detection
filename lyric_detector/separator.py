"""Vocal separation stage — Demucs wrapper with lazy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class VocalSeparator:
    """Extract the vocal stem from a mixed audio file using Demucs."""

    def __init__(self, model: str = "htdemucs", device: str = "cpu") -> None:
        self.model_name = model
        self.device = device
        self._separator: Any = None

    def _load(self) -> Any:
        if self._separator is None:
            from demucs.api import Separator  # heavy import, loaded on demand

            self._separator = Separator(model=self.model_name, device=self.device)
        return self._separator

    def separate(self, audio_path: str | Path) -> tuple[Any, int]:
        """Return ``(vocal_tensor, sample_rate)`` for an audio file."""
        separator = self._load()
        _origin, stems = separator.separate_audio_file(str(audio_path))
        return stems["vocals"], separator.samplerate

    def separate_to_file(
        self, audio_path: str | Path, output_dir: str | Path
    ) -> Path:
        """Separate vocals and save them as a WAV file; returns the path."""
        import torchaudio  # heavy import, loaded on demand

        vocals, sample_rate = self.separate(audio_path)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)

        output_path = destination / f"{Path(audio_path).stem}_vocals.wav"
        torchaudio.save(str(output_path), vocals.cpu(), sample_rate)
        return output_path
