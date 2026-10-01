"""AI lyric detection: separate vocals and transcribe lyrics end to end."""

__version__ = "1.0.0"

from lyric_detector.config import ModelConfig, PipelineConfig
from lyric_detector.evaluation import evaluate_transcriptions
from lyric_detector.metrics import word_error_rate
from lyric_detector.text import clean_lyrics, format_lyrics

__all__ = [
    "ModelConfig",
    "PipelineConfig",
    "evaluate_transcriptions",
    "word_error_rate",
    "clean_lyrics",
    "format_lyrics",
]
