"""
Glue between the Celery `track` stage (app/workers/tasks.py) and the
ball-specific tracking logic in ml/tracking/{byte_tracker,ball_interpolation}.py
— Part 6c, the ball half of `track`. See app/services/player_tracking_stage.py
(Part 6b) for the player half and, in its module docstring, why the two run
as two entirely separate ByteTracker instances rather than one shared one.

Same "video existence check + broken-pipeline-contract check + deferred ml
import" shape as detection_stage.py / player_tracking_stage.py, plus one
step those don't need: a gap-interpolation pass. A raw ByteTracker result
for the ball is exactly as full of holes as the per-frame detections it
was built from, because ByteTracker.update() only ever returns tracks
matched *that* frame — every frame the ball was merely occluded or
motion-blurred is silently indistinguishable, in the raw output, from a
frame where the ball genuinely left the court entirely. Untangling those
two cases is ml/tracking/ball_interpolation.py's job (see its module
docstring); this file's only responsibility is running the ball tracker
and handing its output through that interpolation pass before persisting.
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
from app.services.storage import get_storage_service, make_ball_tracks_destination_path

logger = logging.getLogger(__name__)


class BallTrackingStageError(Exception):
    """Raised when `track` can't find something an earlier stage should already have produced."""


def run_ball_tracking(payload: dict) -> None:
    """
    Real body of the ball-tracking half of the `track` stage. Reads the
    per-frame detections `detect` (Part 5d/5e) already wrote to
    payload["detections_path"], keeps only each frame's `balls` entries
    (players are player_tracking_stage's job, from the same file), runs a
    dedicated ByteTracker across the whole sequence in frame order, bridges
    short gaps with ml/tracking/ball_interpolation.py, and persists the
    result for Part 7 (rally/event detection) to read back.

    Same not-found handling as player_tracking_stage.py: an invalid/missing
    video_id logs and returns; a video that exists but has no
    `detections_path` on the payload means `detect` either didn't run or
    didn't set it — a broken pipeline contract, not a bad caller-supplied
    id — so this raises BallTrackingStageError, a real stage failure Part
    4d's retry/fail path handles.

    Unlike run_player_tracking, this has no match-format sanity check to
    run (there's always exactly one ball on court, never a
    format-dependent expected count) — the equivalent "is this plausible"
    signal here is ball_coverage_rate, logged below.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[track] video_id=%r is not a valid UUID; skipping ball tracking", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[track] video_id=%s not found; skipping ball tracking", video_id)
        return

    detections_key = payload.get("detections_path")
    if not detections_key:
        raise BallTrackingStageError(
            f"video_id={video_id}: payload has no 'detections_path' — the detect stage "
            "should have set this before track ever runs"
        )

    storage = get_storage_service()
    detections_path = storage.get_local_path(detections_key)

    try:
        with open(detections_path) as f:
            detections_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise BallTrackingStageError(
            f"video_id={video_id}: could not read detections file at {detections_path!r}: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.detection.yolo_detector import FrameDetections
    from ml.tracking.ball_interpolation import interpolate_ball_gaps, summarize_ball_track_coverage
    from ml.tracking.byte_tracker import ByteTracker, to_serializable

    frames = [
        FrameDetections(
            frame_path=frame_entry["frame_path"],
            detections=[_dict_to_detection(d) for d in frame_entry["balls"]],
        )
        for frame_entry in detections_data["frames"]
    ]

    logger.info("[track] video_id=%s tracking ball across %d frames", video_id, len(frames))

    tracker = ByteTracker(
        track_thresh=settings.ball_track_thresh,
        match_thresh_low=settings.ball_track_match_thresh_low,
        iou_threshold=settings.ball_track_iou_threshold,
        max_age=settings.ball_track_max_age,
        min_hits=settings.ball_track_min_hits,
    )
    raw_results = tracker.update_many(frames)
    results = interpolate_ball_gaps(
        raw_results, max_gap_frames=settings.ball_track_max_interpolation_gap_frames
    )
    coverage = summarize_ball_track_coverage(results)

    tracks_key = make_ball_tracks_destination_path(video_uuid)
    tracks_path = storage.get_local_path(tracks_key)
    _write_tracks_json(
        tracks_path,
        {
            "ball_track_thresh": settings.ball_track_thresh,
            "ball_track_max_age": settings.ball_track_max_age,
            "ball_track_max_interpolation_gap_frames": settings.ball_track_max_interpolation_gap_frames,
            **coverage,
            "frames": to_serializable(results),
        },
    )

    payload["ball_tracks_path"] = tracks_key
    payload["ball_coverage_rate"] = coverage["ball_coverage_rate"]

    logger.info(
        "[track] video_id=%s done: ball tracked (observed+interpolated) in %.1f%% of frames "
        "(%d observed, %d interpolated) -> %s",
        video_id, coverage["ball_coverage_rate"] * 100,
        coverage["observed_ball_detections"], coverage["interpolated_ball_detections"], tracks_path,
    )
    if coverage["ball_coverage_rate"] < 0.5:
        # Not a failure — see detection_stage.py's ball_detection_rate
        # warning for the same reasoning, one stage downstream: even after
        # gap interpolation, a genuinely hard video (poor lighting, heavy
        # occlusion, a lot of lobs carrying the ball out of frame) can
        # legitimately land here without the pipeline having done
        # anything wrong.
        logger.warning(
            "[track] video_id=%s ball tracked in only %.1f%% of frames even after gap "
            "interpolation — check footage quality/occlusion, or whether the ball is "
            "leaving frame more than expected (PRD Section 13 risk: ball detection accuracy)",
            video_id, coverage["ball_coverage_rate"] * 100,
        )


def _dict_to_detection(d: dict):
    # Deferred import, same reason as run_ball_tracking's own — see
    # player_tracking_stage.py's identical helper.
    from ml.detection.yolo_detector import BoundingBox, Detection

    return Detection(
        class_id=d["class_id"],
        class_name=d["class_name"],
        confidence=d["confidence"],
        bbox=BoundingBox(d["bbox"]["x1"], d["bbox"]["y1"], d["bbox"]["x2"], d["bbox"]["y2"]),
    )


def _write_tracks_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
