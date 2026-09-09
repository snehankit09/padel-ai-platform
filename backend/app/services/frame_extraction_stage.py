"""
Glue between the Celery `validate` stage (app/workers/tasks.py) and the
pure frame-extraction logic in ml/common/frame_extraction.py — Part 5a.

Kept separate from ml/common on purpose (see that module's docstring):
this file is allowed to depend on the DB, Celery's payload shape, and
StorageService; ml/common isn't, so it stays usable outside this app.

The `ml.common.frame_extraction` import below is deferred to inside
run_frame_extraction() rather than done at module level. That matters
because this module is imported by app/workers/tasks.py, which is in turn
imported by the API route that dispatches the pipeline
(app/api/routes/videos.py: `from app.workers.tasks import process_video`)
— so it loads inside the `backend` (FastAPI) container too, not just
`worker`. docker-compose only mounts ./ml at /ml for the `worker` service
(README: ml/ is "imported by workers, not by the API directly"), so an
eager top-level `import ml...` here would break `backend` at startup. The
API process calls `.delay()` and never runs a stage body itself, so it
never needs ml/ to actually be importable — only to look like a normal
import to whoever's reading this file at the point it does matter.
"""

from __future__ import annotations

import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_frames_destination_dir

logger = logging.getLogger(__name__)


def run_frame_extraction(payload: dict) -> None:
    """
    Real body of the `validate` stage (pipeline stage 2: frame extraction).
    Looks up the video's stored file, extracts frames at
    settings.frame_sample_rate_fps into a video-specific frames directory,
    and records where they landed back onto `payload` so `detect_stage`
    (Part 5b+) can find them.

    Matches the stage_fn(payload) contract used by app/workers/tasks.py's
    run_stage: mutates `payload` in place, returns nothing.

    A video_id that isn't a valid UUID, or doesn't match any row, is
    treated the same way app/workers/tasks.py's own _set_video_progress
    treats it: logged and skipped rather than raised. That's not a normal
    operating case — it means the caller passed something odd — and
    treating it as a hard failure here would make it a *pipeline* failure
    (retries, FAILED status) for what's actually a caller bug, which is
    what _set_video_progress already decided isn't worth doing for the
    exact same situation elsewhere in this same chain.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[validate] video_id=%r is not a valid UUID; skipping frame extraction", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[validate] video_id=%s not found; skipping frame extraction", video_id)
            return
        source_key = video.file_path

    storage = get_storage_service()
    source_path = storage.get_local_path(source_key)
    frames_key = make_frames_destination_dir(video_uuid)
    frames_dir = storage.get_local_path(frames_key)

    logger.info(
        "[validate] video_id=%s extracting frames from %s -> %s (sample_fps=%s)",
        video_id, source_path, frames_dir, settings.frame_sample_rate_fps,
    )

    ensure_ml_importable()
    from ml.common.frame_extraction import extract_frames  # deferred — see module docstring

    result = extract_frames(
        video_path=source_path,
        output_dir=frames_dir,
        sample_fps=settings.frame_sample_rate_fps,
    )

    # Relative storage key, not the absolute frames_dir — same convention
    # as Video.file_path, so this stays meaningful regardless of which
    # container (or storage backend) reads it back later.
    payload["frames_dir"] = frames_key
    payload["frame_count"] = result.frame_count
    payload["frame_sample_fps"] = result.sample_fps

    logger.info(
        "[validate] video_id=%s extracted %d frames at %s fps",
        video_id, result.frame_count, result.sample_fps,
    )
