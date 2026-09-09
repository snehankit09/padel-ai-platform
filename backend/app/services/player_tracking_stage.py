"""
Glue between the Celery `track` stage (app/workers/tasks.py) and the pure
tracking logic in ml/tracking/byte_tracker.py — Part 6b, players only.

Same layering, and same "video existence check + broken-pipeline-contract
check + deferred ml import" shape, as app/services/detection_stage.py and
app/services/court_detection_stage.py.

Players get their own ByteTracker instance, separate from the ball's
(Part 6c), rather than one shared tracker filtering by class after the
fact — two reasons, both from ByteTracker's own class docstring and
ml/tracking/byte_tracker.py's module docstring:
  1. Players and the ball move differently enough (slower, larger boxes
     vs. small and fast) that they warrant different track_thresh/
     max_age/min_hits tuning (see config.py's player_track_* settings) —
     one shared instance would force a single compromise setting for
     both.
  2. The module docstring flags a known limitation: bbox-IoU association
     struggles once per-frame displacement exceeds the object's own box
     size, which is a real risk for the ball specifically at this
     pipeline's frame_sample_rate_fps but not really a risk for players.
     Whatever Part 6c ends up doing about that (denser ball-specific
     sampling, a wider association gate) shouldn't have to touch player
     tracking's tuning at all — a shared instance would couple the two
     fixes together for no reason.
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
from app.services.storage import get_storage_service, make_player_tracks_destination_path

logger = logging.getLogger(__name__)


class PlayerTrackingStageError(Exception):
    """Raised when `track` can't find something an earlier stage should already have produced."""


def run_player_tracking(payload: dict) -> None:
    """
    Real body of the player-tracking half of the `track` stage. Reads the
    per-frame detections `detect` (Part 5d/5e) already wrote to
    payload["detections_path"], keeps only each frame's `players` entries
    (the ball is Part 6c's job, from the same file), runs a dedicated
    ByteTracker across the whole sequence in frame order, and persists the
    result for Part 7 (rally/event detection) to read back.

    Same not-found handling as detection_stage.py / court_detection_stage.py:
    an invalid/missing video_id logs and returns (not this stage's problem
    to report); a video that exists but has no `detections_path` on the
    payload means `detect` either didn't run or didn't set it, which is a
    broken pipeline contract, not a bad caller-supplied id — that raises
    PlayerTrackingStageError, a real stage failure Part 4d's retry/fail
    path handles.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[track] video_id=%r is not a valid UUID; skipping player tracking", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[track] video_id=%s not found; skipping player tracking", video_id)
            return
        # PRD-defined formats are "singles" (2 players) and "doubles" (4) —
        # used below only as a soft sanity check (a warning, not an
        # assertion), so an unrecognized/future format falls back to the
        # more common case rather than raising.
        expected_player_count = 2 if video.match is not None and video.match.format == "singles" else 4

    detections_key = payload.get("detections_path")
    if not detections_key:
        raise PlayerTrackingStageError(
            f"video_id={video_id}: payload has no 'detections_path' — the detect stage "
            "should have set this before track ever runs"
        )

    storage = get_storage_service()
    detections_path = storage.get_local_path(detections_key)

    try:
        with open(detections_path) as f:
            detections_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise PlayerTrackingStageError(
            f"video_id={video_id}: could not read detections file at {detections_path!r}: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.detection.yolo_detector import FrameDetections
    from ml.tracking.byte_tracker import ByteTracker, to_serializable

    frames = [
        FrameDetections(
            frame_path=frame_entry["frame_path"],
            detections=[_dict_to_detection(d) for d in frame_entry["players"]],
        )
        for frame_entry in detections_data["frames"]
    ]

    logger.info(
        "[track] video_id=%s tracking players across %d frames (expected_player_count=%d, format=%s)",
        video_id, len(frames), expected_player_count,
        video.match.format if video.match is not None else "unknown",
    )

    tracker = ByteTracker(
        track_thresh=settings.player_track_thresh,
        match_thresh_low=settings.player_track_match_thresh_low,
        iou_threshold=settings.player_track_iou_threshold,
        max_age=settings.player_track_max_age,
        min_hits=settings.player_track_min_hits,
    )
    results = tracker.update_many(frames)

    unique_track_ids = {t.track_id for frame_tracks in results for t in frame_tracks.tracks}
    frames_with_tracks = sum(1 for frame_tracks in results if len(frame_tracks) > 0)
    avg_players_per_frame = (
        sum(len(frame_tracks) for frame_tracks in results) / len(results) if results else 0.0
    )

    tracks_key = make_player_tracks_destination_path(video_uuid)
    tracks_path = storage.get_local_path(tracks_key)
    _write_tracks_json(
        tracks_path,
        {
            "expected_player_count": expected_player_count,
            "unique_track_count": len(unique_track_ids),
            "avg_players_per_frame": avg_players_per_frame,
            "frame_count": len(results),
            "frames_with_tracks": frames_with_tracks,
            "player_track_thresh": settings.player_track_thresh,
            "player_track_max_age": settings.player_track_max_age,
            "frames": to_serializable(results),
        },
    )

    payload["player_tracks_path"] = tracks_key
    payload["player_track_count"] = len(unique_track_ids)

    logger.info(
        "[track] video_id=%s done: %d unique player track(s), avg %.1f players/frame -> %s",
        video_id, len(unique_track_ids), avg_players_per_frame, tracks_path,
    )
    if len(unique_track_ids) != expected_player_count:
        # Not a failure — see detection_stage.py's ball_detection_rate
        # warning for the same reasoning. A real match can genuinely
        # produce more unique IDs than players on court (a player leaving
        # frame and re-entering fragments into a new track once max_age is
        # exceeded) or fewer (heavy occlusion keeping one player's track
        # from ever separately confirming) without the pipeline having
        # done anything wrong.
        logger.warning(
            "[track] video_id=%s found %d unique player track(s), expected %d for format=%s — "
            "could mean a player left/re-entered frame (ID fragmentation) or a false-positive "
            "person detection (PRD Section 13 risk: occlusion)",
            video_id, len(unique_track_ids), expected_player_count,
            video.match.format if video.match is not None else "unknown",
        )


def _dict_to_detection(d: dict):
    # Deferred import, same reason as run_player_tracking's own — this
    # helper is only ever called after ensure_ml_importable() has run, but
    # keeps its own import rather than relying on module-level names so it
    # stays correct even if called from a different entry point later.
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
