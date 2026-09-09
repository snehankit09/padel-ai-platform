"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure point-outcome logic in ml/pipeline/point_outcome.py — Part 7d, the
fourth piece of `analyze` to move past a no-op.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as rally_detection_stage.py / serve_detection_stage.py /
shot_classification_stage.py. Reads back rally segments (7a) and ball
tracks (6c) the same way those do — but NOT player tracks or shots.json/
serves.json: point outcome only ever looks at where the ball ended up,
not who was near it (see ml/pipeline/point_outcome.py's module docstring
for why player position doesn't help distinguish a winner from an
unforced error anyway, which is the one place a shot/serve cross-
reference might otherwise have seemed useful).

Unlike serve/shot detection, a missing court calibration here is NOT
handled by falling back to a coarser pixel-based signal — see
point_outcome.py's module docstring for why "in/out of a rectangle" and
"at the net line" have no meaningful pixel-only version the way a
distance threshold does. A video with no calibration still gets exactly
one PointOutcome per rally (never silently dropped), just every one of
them OUTCOME_UNDETERMINED.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_point_outcomes_destination_path

logger = logging.getLogger(__name__)


class PointOutcomeStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_point_outcome_detection(payload: dict) -> None:
    """
    Real body of the point-outcome half of the `analyze` stage. Reads
    rally segments (7a) and ball tracks (6c) already on the payload, plus
    a court calibration when Part 5e produced one for this video, and
    persists one PointOutcome per rally for Part 9 (win/loss and
    error-rate stats) to read back.

    Same not-found handling as shot_classification_stage.py: an invalid/
    missing video_id logs and returns; missing rally_segments_path or
    ball_tracks_path means an earlier stage didn't run or didn't set it —
    a broken pipeline contract — so those raise PointOutcomeStageError
    (Part 4d's retry/fail path). A missing or None court_calibration_path
    is NOT an error, same reason it isn't for serve/shot detection —
    Part 5e already treats calibration failure as non-fatal — it just
    means every outcome for this video comes back OUTCOME_UNDETERMINED
    (see this stage's module docstring for why there's no coarser
    pixel-based fallback here the way there is for serve/shot detection).
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping point outcome detection", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[analyze] video_id=%s not found; skipping point outcome detection", video_id)
        return

    required = {
        "rally_segments_path": "the analyze stage's rally-detection half (Part 7a)",
        "ball_tracks_path": "the track stage (Part 6c)",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        details = "; ".join(f"'{key}' (should have been set by {required[key]})" for key in missing)
        raise PointOutcomeStageError(f"video_id={video_id}: payload is missing {details}")

    storage = get_storage_service()

    try:
        rallies_data = _read_json(storage.get_local_path(payload["rally_segments_path"]))
        ball_tracks_data = _read_json(storage.get_local_path(payload["ball_tracks_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise PointOutcomeStageError(f"video_id={video_id}: could not read a required tracks/rally file: {exc}") from exc

    ensure_ml_importable()
    from ml.pipeline.point_outcome import (
        ball_points_from_tracks_json,
        detect_point_outcomes,
        summarize_point_outcomes,
        to_serializable,
    )
    from ml.pipeline.rally_detection import RallySegment

    rallies = [RallySegment(**r) for r in rallies_data["rallies"]]
    ball_points = ball_points_from_tracks_json(ball_tracks_data)

    to_court_meters, court_length_m, court_width_m = _build_court_meters_converter(
        payload.get("court_calibration_path"), storage, settings.court_length_m, settings.court_width_m
    )

    logger.info(
        "[analyze] video_id=%s determining point outcomes for %d rally(s) (court_calibration=%s)",
        video_id, len(rallies), "yes" if to_court_meters is not None else "no",
    )

    outcomes = detect_point_outcomes(
        rallies, ball_points,
        out_of_bounds_margin_m=settings.point_outcome_out_of_bounds_margin_m,
        net_zone_m=settings.point_outcome_net_zone_m,
        net_deceleration_ratio=settings.point_outcome_net_deceleration_ratio,
        court_width_m=court_width_m,
        court_length_m=court_length_m,
        to_court_meters=to_court_meters,
    )
    summary = summarize_point_outcomes(outcomes)

    outcomes_key = make_point_outcomes_destination_path(video_uuid)
    outcomes_path = storage.get_local_path(outcomes_key)
    _write_json(
        outcomes_path,
        {
            "point_outcome_out_of_bounds_margin_m": settings.point_outcome_out_of_bounds_margin_m,
            "point_outcome_net_zone_m": settings.point_outcome_net_zone_m,
            "point_outcome_net_deceleration_ratio": settings.point_outcome_net_deceleration_ratio,
            **summary,
            "outcomes": to_serializable(outcomes),
        },
    )

    payload["point_outcomes_path"] = outcomes_key
    payload["point_outcome_undetermined_rate"] = summary["undetermined_rate"]

    logger.info(
        "[analyze] video_id=%s done: %s -> %s",
        video_id, summary["outcome_counts"], outcomes_path,
    )
    if outcomes and summary["undetermined_rate"] > 0.5:
        # Not a failure — same reasoning as serve_detection_stage.py's/
        # shot_classification_stage.py's own low-signal warnings: most
        # rallies coming back undetermined almost always traces back to
        # no court calibration for this video (check court_calibration_path
        # upstream) or weak ball tracking, not to this stage's threshold
        # logic being wrong.
        logger.warning(
            "[analyze] video_id=%s more than half of %d rally outcome(s) are undetermined — "
            "check court_calibration_path and ball_coverage_rate upstream",
            video_id, summary["outcome_count"],
        )


def _build_court_meters_converter(calibration_key, storage, default_court_length_m: float, default_court_width_m: float):
    """
    Returns (`(x, y) -> (x_m, y_m)` callable or None, court_length_m to
    use with it, court_width_m to use with it) — same "None means fall
    back, never an error" posture as serve_detection_stage.py's and
    shot_classification_stage.py's identically-shaped helpers, extended
    with court_width_m since (unlike those two) determining in/out of
    bounds needs BOTH court dimensions, not just the length needed for
    net-line math. Uses the video's OWN calibrated dimensions when
    present (whatever Part 5e actually passed to compute_homography for
    this specific video), falling back to today's default settings only
    when the calibration JSON doesn't have them.
    """
    if not calibration_key:
        return None, default_court_length_m, default_court_width_m

    calibration_path = storage.get_local_path(calibration_key)
    try:
        calibration_data = _read_json(calibration_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[analyze] could not read court calibration at %r (%s) — falling back to no calibration",
            calibration_path, exc,
        )
        return None, default_court_length_m, default_court_width_m

    ensure_ml_importable()
    import numpy as np

    from ml.detection.court_detector import pixel_to_court_point

    homography = np.array(calibration_data["homography"], dtype=np.float64)
    court_length_m = calibration_data.get("court_length_m", default_court_length_m)
    court_width_m = calibration_data.get("court_width_m", default_court_width_m)
    return lambda point: pixel_to_court_point(homography, point), court_length_m, court_width_m


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
