"""Configuration objects for the lyric detection pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    """Model identifiers and audio settings."""

    separator_model: str = "htdemucs"
    asr_model: str = "facebook/wav2vec2-base-960h"
    sample_rate: int = 16000
    device: str = "cpu"


@dataclass
class PipelineConfig:
    """Runtime options for a transcription run."""

    keep_stems: bool = False
    min_segment_seconds: float = 0.5
    model: ModelConfig = field(default_factory=ModelConfig)
