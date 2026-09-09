"""
Glue between the Celery `detect` stage (app/workers/tasks.py) and the pure
detection logic in ml/detection/{yolo_detector,player_ball_detection}.py —
Part 5d.

Same layering, and same deferred-import reasoning, as
app/services/frame_extraction_stage.py (Part 5a): this file is allowed to
depend on the DB, Celery's payload shape, and StorageService; the ml/
modules it wraps aren't. Its own `import ml...` (and the ultralytics model
load buried inside it) is deferred to inside run_player_ball_detection()
so the API container — which imports this module too, via
app/workers/tasks.py's `from app.workers.tasks import process_video`, and
never mounts ./ml at all — can still import it to call `.delay()` without
needing ml/ or ultralytics to actually be present.
"""

from __future__ import annotations

import json
import logging
import os
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_detections_destination_path

logger = logging.getLogger(__name__)


class DetectionStageError(Exception):
    """Raised when `detect` can't find something an earlier stage should already have produced."""


def run_player_ball_detection(payload: dict) -> None:
    """
    Real body of the `detect` stage (pipeline stage 3: player & ball
    detection). Reads the frame paths `validate` (Part 5a) recorded at
    `payload["frames_dir"]`, runs YOLO across every one of them — split
    into players vs. ball per frame, see ml/detection/player_ball_detection.py
    for why they're kept separate rather than detected identically — and
    writes the per-frame results to a detections JSON file for `track`
    (Part 6) to read back, the same way `validate` hands `track`'s
    predecessor a frames directory instead of re-decoding the source video.

    An invalid or missing video_id is treated the same way
    run_frame_extraction treats it (see that function's docstring): logged
    and skipped rather than raised, since it means the caller passed
    something odd, not that this stage itself has anything to report.

    A video that DOES exist but has no `frames_dir` on the payload is
    different: that can only happen if `validate` ran and didn't set it,
    or if this stage got invoked out of the chain's normal order — either
    way a broken pipeline contract, not a bad caller-supplied id — so this
    raises DetectionStageError (a real stage failure: retries, then FAILED
    if it never clears) rather than silently skipping.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[detect] video_id=%r is not a valid UUID; skipping detection", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[detect] video_id=%s not found; skipping detection", video_id)
        return

    frames_key = payload.get("frames_dir")
    if not frames_key:
        raise DetectionStageError(
            f"video_id={video_id}: payload has no 'frames_dir' — the validate stage "
            "should have set this before detect ever runs"
        )

    storage = get_storage_service()
    frames_dir = storage.get_local_path(frames_key)

    ensure_ml_importable()
    from ml.common.frame_extraction import list_frame_paths
    from ml.detection.player_ball_detection import (
        detect_players_and_ball_many,
        summarize_ball_detection_rate,
        to_serializable,
    )
    from ml.detection.yolo_detector import get_cached_model

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise DetectionStageError(
            f"video_id={video_id}: no frames found in {frames_dir!r} — validate should "
            "have populated this directory before detect ran"
        )

    logger.info(
        "[detect] video_id=%s running YOLO on %d frames (model=%s, device=%s)",
        video_id, len(frame_paths), settings.yolo_model_path, settings.yolo_device,
    )

    model = get_cached_model(settings.yolo_model_path, settings.yolo_device)
    results = detect_players_and_ball_many(
        model,
        frame_paths,
        player_confidence_threshold=settings.yolo_player_confidence_threshold,
        ball_confidence_threshold=settings.yolo_ball_confidence_threshold,
    )
    ball_detection_rate = summarize_ball_detection_rate(results)

    detections_key = make_detections_destination_path(video_uuid)
    detections_path = storage.get_local_path(detections_key)
    _write_detections_json(
        detections_path,
        {
            "model_path": settings.yolo_model_path,
            "player_confidence_threshold": settings.yolo_player_confidence_threshold,
            "ball_confidence_threshold": settings.yolo_ball_confidence_threshold,
            "frame_count": len(frame_paths),
            "ball_detection_rate": ball_detection_rate,
            "frames": to_serializable(results),
        },
    )

    # Relative storage key, not the absolute path — same convention as
    # payload["frames_dir"], so this stays meaningful regardless of which
    # container (or storage backend) reads it back later.
    payload["detections_path"] = detections_key
    payload["ball_detection_rate"] = ball_detection_rate

    logger.info(
        "[detect] video_id=%s done: %d frames, ball present in %.1f%% -> %s",
        video_id, len(frame_paths), ball_detection_rate * 100, detections_path,
    )
    if ball_detection_rate < 0.5:
        # Not a failure — a low rate can be a genuinely hard match (heavy
        # occlusion, poor lighting) rather than a bug — but PRD Section 13
        # names this as the platform's main risk, so it's worth surfacing
        # loudly rather than only in the JSON a human would have to go
        # looking for.
        logger.warning(
            "[detect] video_id=%s ball detected in only %.1f%% of frames — check footage "
            "quality / occlusion (PRD Section 13 risk: ball detection accuracy)",
            video_id, ball_detection_rate * 100,
        )


def _write_detections_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
