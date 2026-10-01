"""Speech-to-text stage — Hugging Face CTC model with lazy imports."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class SpeechTranscriber:
    """Transcribe an audio file with a Hugging Face CTC model."""

    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base-960h",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _load(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            from transformers import (  # heavy import, loaded on demand
                Wav2Vec2ForCTC,
                Wav2Vec2Processor,
            )

            self._processor = Wav2Vec2Processor.from_pretrained(self.model_name)
            self._model = Wav2Vec2ForCTC.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model, self._processor

    def transcribe(self, audio_path: str | Path, sample_rate: int = 16000) -> str:
        """Transcribe an audio file into raw (uncleaned) text."""
        import torch  # heavy import, loaded on demand
        import torchaudio

        model, processor = self._load()

        waveform, file_rate = torchaudio.load(str(audio_path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if file_rate != sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, file_rate, sample_rate
            )

        inputs = processor(
            waveform.squeeze(0).numpy(),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(inputs.input_values.to(self.device)).logits

        predicted_ids = torch.argmax(logits, dim=-1)
        return processor.batch_decode(predicted_ids)[0].strip()
