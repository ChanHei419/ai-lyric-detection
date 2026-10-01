"""FastAPI service exposing the pipeline with background jobs and progress."""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from lyric_detector.config import PipelineConfig
from lyric_detector.pipeline import LyricPipeline

# In-memory job store — swap for Redis/DB when deploying for real traffic.
JOBS: dict[str, dict] = {}


def create_app(pipeline: LyricPipeline | None = None):
    """Application factory (use with ``uvicorn --factory``)."""
    from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile

    app = FastAPI(title="AI Lyric Detection API", version="1.0.0")
    shared_pipeline = pipeline or LyricPipeline(PipelineConfig())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/transcribe")
    async def transcribe(
        background_tasks: BackgroundTasks, file: UploadFile = File(...)
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
        upload_dir = Path(tempfile.mkdtemp(prefix="lyrics_"))
        upload_path = upload_dir / f"input{suffix}"
        upload_path.write_bytes(await file.read())

        JOBS[job_id] = {
            "status": "queued",
            "stage": "queued",
            "progress": 0,
            "lyrics": None,
        }

        def progress(stage: str, percent: int) -> None:
            JOBS[job_id].update(stage=stage, progress=percent)

        def process() -> None:
            try:
                JOBS[job_id]["status"] = "running"
                result = shared_pipeline.run(
                    upload_path, upload_dir / "outputs", progress
                )
                JOBS[job_id].update(
                    status="completed",
                    stage="done",
                    progress=100,
                    lyrics=result.lyrics,
                    raw_text=result.raw_text,
                )
            except Exception as error:  # surface failures to the client
                JOBS[job_id].update(status="failed", error=str(error))

        background_tasks.add_task(process)
        return {"job_id": job_id, "status_url": f"/jobs/{job_id}"}

    @app.get("/jobs/{job_id}")
    def job_status(job_id: str) -> dict:
        if job_id not in JOBS:
            raise HTTPException(status_code=404, detail="job not found")
        return JOBS[job_id]

    return app
