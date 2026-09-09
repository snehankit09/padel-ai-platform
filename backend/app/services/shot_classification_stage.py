"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure shot-classification logic in ml/pipeline/shot_classification.py —
Part 7c, the third piece of `analyze` to move past a no-op.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as app/services/rally_detection_stage.py and
app/services/serve_detection_stage.py. Reads back rally segments (7a),
ball tracks (6c), and player tracks (6b) the same way serve detection
does, plus, optionally, a court calibration (5e) — same "used when
available, never required" posture as serve detection, for the same
reason (see ml/pipeline/shot_classification.py's module docstring, point
3: the volley/groundstroke distinction genuinely can't be made from
pixels alone, so it's reported as UNKNOWN rather than guessed at when no
calibration exists, not refused outright).
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_shots_destination_path

logger = logging.getLogger(__name__)


class ShotClassificationStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_shot_classification(payload: dict) -> None:
    """
    Real body of the shot-classification half of the `analyze` stage.
    Reads rally segments (7a), ball tracks (6c), and player tracks (6b)
    already on the payload, plus a court calibration when Part 5e
    produced one for this video, and persists a flat list of classified
    Shots for Part 8 (highlight clips — "show me every smash") and Part 9
    (shot-type stats) to read back.

    Same not-found handling as serve_detection_stage.py: an invalid/
    missing video_id logs and returns; missing rally_segments_path,
    ball_tracks_path, or player_tracks_path each mean an earlier stage
    didn't run or didn't set it — a broken pipeline contract — so those
    raise ShotClassificationStageError (Part 4d's retry/fail path). A
    missing or None court_calibration_path is NOT an error, for the same
    reason it isn't one for serve detection: Part 5e already treats
    calibration failure as non-fatal, and every consumer downstream of
    it has to work without it.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping shot classification", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[analyze] video_id=%s not found; skipping shot classification", video_id)
        return

    required = {
        "rally_segments_path": "the analyze stage's rally-detection half (Part 7a)",
        "ball_tracks_path": "the track stage (Part 6c)",
        "player_tracks_path": "the track stage (Part 6b)",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        details = "; ".join(f"'{key}' (should have been set by {required[key]})" for key in missing)
        raise ShotClassificationStageError(f"video_id={video_id}: payload is missing {details}")

    storage = get_storage_service()

    try:
        rallies_data = _read_json(storage.get_local_path(payload["rally_segments_path"]))
        ball_tracks_data = _read_json(storage.get_local_path(payload["ball_tracks_path"]))
        player_tracks_data = _read_json(storage.get_local_path(payload["player_tracks_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise ShotClassificationStageError(
            f"video_id={video_id}: could not read a required tracks/rally file: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.pipeline.rally_detection import RallySegment
    from ml.pipeline.shot_classification import (
        ball_points_from_tracks_json,
        detect_shots,
        player_frames_from_tracks_json,
        summarize_shots,
        to_serializable,
    )

    rallies = [RallySegment(**r) for r in rallies_data["rallies"]]
    ball_points = ball_points_from_tracks_json(ball_tracks_data)
    player_frames = player_frames_from_tracks_json(player_tracks_data)

    to_court_meters, court_length_m = _build_court_meters_converter(
        payload.get("court_calibration_path"), storage, settings.court_length_m
    )

    logger.info(
        "[analyze] video_id=%s classifying shots for %d rally(s) (court_calibration=%s)",
        video_id, len(rallies), "yes" if to_court_meters is not None else "no",
    )

    shots = detect_shots(
        rallies, ball_points, player_frames,
        max_distance_m=settings.shot_max_contact_player_distance_m,
        max_distance_px=settings.shot_max_contact_player_distance_px,
        smash_height_ratio=settings.shot_smash_height_ratio,
        lob_min_airborne_frames=settings.shot_lob_min_airborne_frames,
        net_proximity_m=settings.shot_net_proximity_m,
        court_length_m=court_length_m,
        to_court_meters=to_court_meters,
    )
    summary = summarize_shots(shots)

    shots_key = make_shots_destination_path(video_uuid)
    shots_path = storage.get_local_path(shots_key)
    _write_json(
        shots_path,
        {
            "shot_max_contact_player_distance_m": settings.shot_max_contact_player_distance_m,
            "shot_max_contact_player_distance_px": settings.shot_max_contact_player_distance_px,
            "shot_smash_height_ratio": settings.shot_smash_height_ratio,
            "shot_lob_min_airborne_frames": settings.shot_lob_min_airborne_frames,
            "shot_net_proximity_m": settings.shot_net_proximity_m,
            **summary,
            "shots": to_serializable(shots),
        },
    )

    payload["shots_path"] = shots_key
    payload["shot_count"] = summary["shot_count"]

    logger.info(
        "[analyze] video_id=%s done: %d shot(s) classified (%s) -> %s",
        video_id, summary["shot_count"], summary["shot_counts_by_type"], shots_path,
    )
    if shots and summary["unknown_rate"] > 0.5:
        # Not a failure -- same reasoning as serve_detection_stage.py's
        # low-identified-rate warning: most contacts landing in UNKNOWN
        # almost always means no court calibration was available for this
        # video (see module docstring point 3), not that this stage's
        # rule logic is wrong. check court_calibration_path upstream.
        logger.warning(
            "[analyze] video_id=%s more than half of %d shot(s) classified as unknown — "
            "check whether court_calibration_path is set for this video; the volley/"
            "groundstroke split needs it and reports unknown without it",
            video_id, summary["shot_count"],
        )


def _build_court_meters_converter(calibration_key, storage, default_court_length_m: float):
    """
    Returns (`(x, y) -> (x_m, y_m)` callable or None, court_length_m to
    use with it) — same "None means fall back, never an error" posture as
    serve_detection_stage.py's identically-named helper, plus the video's
    OWN calibrated court_length_m alongside the converter (not
    settings.court_length_m — see ml.pipeline.shot_classification.detect_shots'
    docstring for why the net-line math needs the same length the
    homography itself was built from, which is whatever Part 5e actually
    passed to compute_homography for this specific video, not necessarily
    today's default setting).
    """
    if not calibration_key:
        return None, default_court_length_m

    calibration_path = storage.get_local_path(calibration_key)
    try:
        calibration_data = _read_json(calibration_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[analyze] could not read court calibration at %r (%s) — falling back to no calibration",
            calibration_path, exc,
        )
        return None, default_court_length_m

    ensure_ml_importable()
    import numpy as np

    from ml.detection.court_detector import pixel_to_court_point

    homography = np.array(calibration_data["homography"], dtype=np.float64)
    court_length_m = calibration_data.get("court_length_m", default_court_length_m)
    return lambda point: pixel_to_court_point(homography, point), court_length_m


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
