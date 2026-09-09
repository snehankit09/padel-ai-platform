"""
Glue between the Celery `detect` stage (app/workers/tasks.py) and the pure
court-detection logic in ml/detection/court_detector.py — Part 5e.

Same layering, and same deferred-import reasoning, as
app/services/detection_stage.py and app/services/frame_extraction_stage.py:
this file is allowed to depend on the DB, Celery's payload shape, and
StorageService; ml/detection/court_detector.py isn't.

Unlike player/ball detection (run once per frame, because players and the
ball actually move), a court's calibration is one constant for the whole
video — the camera doesn't move mid-match. So this doesn't run
detect_court on every sampled frame; it tries up to
settings.court_calibration_max_frame_attempts evenly-spaced candidate
frames and keeps the first one that calibrates cleanly, on the assumption
that most frames of real match footage look enough like every other frame
that only a genuinely bad one (a player standing on the boundary line
right when that frame was sampled, motion blur, a stray shadow) would make
detect_court raise CourtDetectionError — trying a few more candidates is
cheap insurance against picking exactly one unlucky frame, not a sign
anything is wrong with the approach.

If every attempted frame fails to calibrate, that's treated the same way
detection_stage.py treats a low ball-detection rate: logged loudly (with
a pointer to the specific PRD risk this maps to) rather than failed. A
video the platform can't calibrate the court for isn't a broken pipeline
run — Parts 6+ can still track pixel positions and derive most stats from
those; only the *court-relative* coordinates PRD 5.2/5.5 want are what's
missing, and that shouldn't block everything downstream of it.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_court_calibration_destination_path

logger = logging.getLogger(__name__)


class CourtDetectionStageError(Exception):
    """Raised when `detect` can't find something an earlier stage should already have produced."""


def _pick_candidate_frame_paths(frame_paths: list[str], max_attempts: int) -> list[str]:
    """
    Picks up to `max_attempts` frames, evenly spaced across the whole
    video rather than just the first N — spreads out the chance that
    whatever made one frame hard to calibrate (a player crossing a
    boundary line, a moment of motion blur) isn't also true a few seconds
    either side of it.
    """
    if len(frame_paths) <= max_attempts:
        return frame_paths
    step = len(frame_paths) / max_attempts
    return [frame_paths[int(i * step)] for i in range(max_attempts)]


def run_court_detection(payload: dict) -> None:
    """
    Real body of the court-calibration half of the `detect` stage
    (pipeline stage 3). Tries a handful of the frames `validate` (Part 5a)
    already extracted, keeps the first that calibrates, and writes the
    result to a calibration JSON file for Part 6+ to read back — same
    "write once, read back later, never re-derive" pattern as
    run_player_ball_detection's detections.json.

    An invalid/missing video_id, or a video with no `frames_dir` on the
    payload, are handled identically to run_player_ball_detection (see
    that function's docstring for why): the former is skipped quietly,
    the latter raises CourtDetectionStageError since it means the pipeline
    ran out of order, not that this stage has anything to report.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[detect] video_id=%r is not a valid UUID; skipping court detection", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[detect] video_id=%s not found; skipping court detection", video_id)
        return

    frames_key = payload.get("frames_dir")
    if not frames_key:
        raise CourtDetectionStageError(
            f"video_id={video_id}: payload has no 'frames_dir' — the validate stage "
            "should have set this before detect ever runs"
        )

    storage = get_storage_service()
    frames_dir = storage.get_local_path(frames_key)

    ensure_ml_importable()
    from ml.common.frame_extraction import list_frame_paths
    from ml.detection.court_detector import CourtDetectionError, detect_court

    frame_paths = list_frame_paths(frames_dir)
    if not frame_paths:
        raise CourtDetectionStageError(
            f"video_id={video_id}: no frames found in {frames_dir!r} — validate should "
            "have populated this directory before detect ran"
        )

    candidates = _pick_candidate_frame_paths(frame_paths, settings.court_calibration_max_frame_attempts)
    logger.info(
        "[detect] video_id=%s attempting court calibration on %d candidate frame(s)",
        video_id, len(candidates),
    )

    calibration = None
    last_error: Exception | None = None
    attempted_frame = None
    for frame_path in candidates:
        try:
            calibration = detect_court(
                frame_path,
                court_length_m=settings.court_length_m,
                court_width_m=settings.court_width_m,
            )
            attempted_frame = frame_path
            break
        except CourtDetectionError as exc:
            last_error = exc
            continue

    if calibration is None:
        # Not a stage failure — see module docstring. A genuinely
        # uncalibratable video is a real, expected outcome for hard
        # footage (PRD Section 13 risks: Lighting changes, Occlusion),
        # not a bug in this stage.
        logger.warning(
            "[detect] video_id=%s could not calibrate the court from any of %d candidate "
            "frame(s) — last error: %s (PRD Section 13 risks: Lighting changes, Occlusion). "
            "Court-relative coordinates will be unavailable for this video.",
            video_id, len(candidates), last_error,
        )
        payload["court_calibration_path"] = None
        return

    calibration_key = make_court_calibration_destination_path(video_uuid)
    calibration_path = storage.get_local_path(calibration_key)
    _write_calibration_json(
        calibration_path,
        {
            "source_frame": attempted_frame,
            "court_length_m": calibration.court_length_m,
            "court_width_m": calibration.court_width_m,
            "corners": {
                "top_left": calibration.corners.top_left,
                "top_right": calibration.corners.top_right,
                "bottom_right": calibration.corners.bottom_right,
                "bottom_left": calibration.corners.bottom_left,
            },
            "homography": calibration.homography.tolist(),
        },
    )

    payload["court_calibration_path"] = calibration_key

    logger.info(
        "[detect] video_id=%s court calibrated from %s -> %s",
        video_id, attempted_frame, calibration_path,
    )


def _write_calibration_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
