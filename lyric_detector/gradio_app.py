"""Optional Gradio interface for interactive lyric transcription."""

from __future__ import annotations

from lyric_detector.config import PipelineConfig
from lyric_detector.pipeline import LyricPipeline


def build_interface(pipeline: LyricPipeline | None = None):
    """Create the Gradio app (imports Gradio at call time)."""
    import gradio as gr

    shared = pipeline or LyricPipeline(PipelineConfig())

    def transcribe(audio_path: str | None) -> str:
        if not audio_path:
            return "Please upload an audio file."
        result = shared.run(audio_path, output_dir="outputs/gradio")
        return result.lyrics

    return gr.Interface(
        fn=transcribe,
        inputs=gr.Audio(type="filepath", label="Song"),
        outputs=gr.Textbox(label="Detected lyrics", lines=12),
        title="AI Lyric Detection",
        description="Vocal separation with Demucs, transcription with a Wav2Vec2 CTC model.",
    )


if __name__ == "__main__":
    build_interface().launch()
