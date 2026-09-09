"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure serve-identification logic in ml/pipeline/serve_detection.py — Part
7b, the second piece of `analyze` to move past a no-op.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as app/services/rally_detection_stage.py. Reads back
three of `track`/`detect`/`analyze`'s earlier outputs —
payload["rally_segments_path"] (7a), payload["ball_tracks_path"] (6c),
payload["player_tracks_path"] (6b) — plus, optionally,
payload["court_calibration_path"] (5e) when it exists: see
ml/pipeline/serve_detection.py's module docstring for why calibration is
used when available and never required.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_serve_events_destination_path

logger = logging.getLogger(__name__)


class ServeDetectionStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_serve_detection(payload: dict) -> None:
    """
    Real body of the serve-identification half of the `analyze` stage.
    Reads rally segments (7a), ball tracks (6c), and player tracks (6b)
    already on the payload, plus a court calibration when Part 5e produced
    one for this video, and persists one ServeEvent per rally for later
    sub-parts (shot classification, point outcomes) and Part 9 to read
    back.

    Same not-found handling as rally_detection_stage.py: an invalid/
    missing video_id logs and returns; missing rally_segments_path,
    ball_tracks_path, or player_tracks_path each mean an earlier stage
    didn't run or didn't set it — a broken pipeline contract — so those
    raise ServeDetectionStageError (Part 4d's retry/fail path). A missing
    or None court_calibration_path is NOT an error: Part 5e already
    treats calibration failure as non-fatal for exactly this reason —
    every consumer downstream of it, this one included, has to work
    without it.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping serve detection", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[analyze] video_id=%s not found; skipping serve detection", video_id)
        return

    required = {
        "rally_segments_path": "the analyze stage's rally-detection half (Part 7a)",
        "ball_tracks_path": "the track stage (Part 6c)",
        "player_tracks_path": "the track stage (Part 6b)",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        details = "; ".join(f"'{key}' (should have been set by {required[key]})" for key in missing)
        raise ServeDetectionStageError(f"video_id={video_id}: payload is missing {details}")

    storage = get_storage_service()

    try:
        rallies_data = _read_json(storage.get_local_path(payload["rally_segments_path"]))
        ball_tracks_data = _read_json(storage.get_local_path(payload["ball_tracks_path"]))
        player_tracks_data = _read_json(storage.get_local_path(payload["player_tracks_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise ServeDetectionStageError(f"video_id={video_id}: could not read a required tracks/rally file: {exc}") from exc

    ensure_ml_importable()
    from ml.pipeline.rally_detection import RallySegment
    from ml.pipeline.serve_detection import (
        detect_serves,
        frames_from_tracks_json,
        summarize_serve_events,
        to_serializable,
    )

    rallies = [RallySegment(**r) for r in rallies_data["rallies"]]
    ball_frames = frames_from_tracks_json(ball_tracks_data)
    player_frames = frames_from_tracks_json(player_tracks_data)

    to_court_meters = _build_court_meters_converter(payload.get("court_calibration_path"), storage)

    logger.info(
        "[analyze] video_id=%s identifying servers for %d rally(s) (court_calibration=%s)",
        video_id, len(rallies), "yes" if to_court_meters is not None else "no",
    )

    events = detect_serves(
        rallies, ball_frames, player_frames,
        window_frames=settings.serve_detection_window_frames,
        max_distance_m=settings.serve_max_ball_player_distance_m,
        max_distance_px=settings.serve_max_ball_player_distance_px,
        to_court_meters=to_court_meters,
    )
    summary = summarize_serve_events(events)

    serves_key = make_serve_events_destination_path(video_uuid)
    serves_path = storage.get_local_path(serves_key)
    _write_json(
        serves_path,
        {
            "serve_detection_window_frames": settings.serve_detection_window_frames,
            "serve_max_ball_player_distance_m": settings.serve_max_ball_player_distance_m,
            "serve_max_ball_player_distance_px": settings.serve_max_ball_player_distance_px,
            **summary,
            "serves": to_serializable(events),
        },
    )

    payload["serve_events_path"] = serves_key
    payload["serve_identified_count"] = summary["identified_count"]

    logger.info(
        "[analyze] video_id=%s done: %d/%d serve(s) identified (%.0f%%, calibrated=%s) -> %s",
        video_id, summary["identified_count"], summary["serve_count"],
        summary["identified_rate"] * 100, summary["used_court_calibration"], serves_path,
    )
    if rallies and summary["identified_rate"] < 0.5:
        # Not a failure -- same reasoning as rally_detection_stage.py's
        # 0-rally warning, one level further down the pipeline: a real
        # match with most serves unidentified almost always traces back to
        # weak ball/player tracking a stage or two back, not to this
        # module's distance logic being wrong.
        logger.warning(
            "[analyze] video_id=%s identified fewer than half of %d serve(s) — check "
            "player_track_count/ball_coverage_rate from the track stage; weak tracking "
            "upstream is the most likely cause",
            video_id, summary["serve_count"],
        )


def _build_court_meters_converter(calibration_key, storage):
    """
    Returns a `(x, y) -> (x_m, y_m)` callable backed by the video's
    persisted court calibration (Part 5e), or None when no calibration
    exists for this video — the caller (detect_serves) treats None as
    "use raw pixel distance instead", never as an error.
    """
    if not calibration_key:
        return None

    calibration_path = storage.get_local_path(calibration_key)
    try:
        calibration_data = _read_json(calibration_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[analyze] could not read court calibration at %r (%s) — falling back to pixel distance",
            calibration_path, exc,
        )
        return None

    ensure_ml_importable()
    import numpy as np

    from ml.detection.court_detector import pixel_to_court_point

    homography = np.array(calibration_data["homography"], dtype=np.float64)
    return lambda point: pixel_to_court_point(homography, point)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
