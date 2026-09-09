"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure rally-boundary logic in ml/pipeline/rally_detection.py — Part 7a,
the first piece of `analyze` to move past a no-op.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as app/services/ball_tracking_stage.py /
app/services/player_tracking_stage.py — this file is allowed to depend on
the DB, Celery's payload shape, and StorageService; ml/pipeline/rally_detection.py
isn't, so it stays usable from a one-off script or a test without any of
that.

Reads back `track`'s ball output (payload["ball_tracks_path"], Part 6c) —
not player_tracks_path: rally boundaries are derived purely from ball
activity (see ml/pipeline/rally_detection.py's module docstring for why),
so this stage has no need for player position data yet. A later sub-part
that wants to, say, use serve stance to sharpen a rally's exact start
frame would read player_tracks_path here too; nothing about this file's
shape would need to change to add that.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_rally_segments_destination_path

logger = logging.getLogger(__name__)


class RallyDetectionStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_rally_detection(payload: dict) -> None:
    """
    Real body of the rally-boundary half of the `analyze` stage. Reads
    the per-frame ball tracks `track` (Part 6c) already wrote to
    payload["ball_tracks_path"], turns them into the match's list of
    rally segments via ml/pipeline/rally_detection.py, and persists the
    result for Part 7b+ (shot/event classification within a rally) and
    Part 8/9 (highlight clips, stats) to read back.

    Same not-found handling as ball_tracking_stage.py: an invalid/missing
    video_id logs and returns; a video that exists but has no
    `ball_tracks_path` or `frame_sample_fps` on the payload means an
    earlier stage either didn't run or didn't set it — a broken pipeline
    contract, not a bad caller-supplied id — so this raises
    RallyDetectionStageError, a real stage failure Part 4d's retry/fail
    path handles. `frame_sample_fps` (Part 5a's validate stage) is needed
    here, not just a tracks file, because rally boundaries are reported
    in real-world timestamps (start_time_s/end_time_s), not just frame
    indices — Part 8's clip-extraction ffmpeg calls will want seconds,
    not sampled-frame numbers.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping rally detection", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[analyze] video_id=%s not found; skipping rally detection", video_id)
        return

    ball_tracks_key = payload.get("ball_tracks_path")
    if not ball_tracks_key:
        raise RallyDetectionStageError(
            f"video_id={video_id}: payload has no 'ball_tracks_path' — the track stage "
            "should have set this before analyze ever runs"
        )

    sample_fps = payload.get("frame_sample_fps")
    if not sample_fps:
        raise RallyDetectionStageError(
            f"video_id={video_id}: payload has no 'frame_sample_fps' — the validate stage "
            "should have set this before analyze ever runs"
        )

    storage = get_storage_service()
    ball_tracks_path = storage.get_local_path(ball_tracks_key)

    try:
        with open(ball_tracks_path) as f:
            ball_tracks_data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RallyDetectionStageError(
            f"video_id={video_id}: could not read ball tracks file at {ball_tracks_path!r}: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.pipeline.rally_detection import (
        detect_rally_segments,
        frames_from_ball_tracks_json,
        summarize_rally_segments,
        to_serializable,
    )

    signals = frames_from_ball_tracks_json(ball_tracks_data)

    logger.info(
        "[analyze] video_id=%s detecting rally boundaries across %d frames", video_id, len(signals)
    )

    segments = detect_rally_segments(
        signals,
        sample_fps=sample_fps,
        activity_gap_tolerance_frames=settings.rally_activity_gap_tolerance_frames,
        min_rally_duration_frames=settings.rally_min_duration_frames,
    )
    summary = summarize_rally_segments(segments, total_frame_count=len(signals))

    rallies_key = make_rally_segments_destination_path(video_uuid)
    rallies_path = storage.get_local_path(rallies_key)
    _write_rallies_json(
        rallies_path,
        {
            "rally_activity_gap_tolerance_frames": settings.rally_activity_gap_tolerance_frames,
            "rally_min_duration_frames": settings.rally_min_duration_frames,
            "frame_sample_fps": sample_fps,
            **summary,
            "rallies": to_serializable(segments),
        },
    )

    payload["rally_segments_path"] = rallies_key
    payload["rally_count"] = summary["rally_count"]

    logger.info(
        "[analyze] video_id=%s done: %d rally segment(s), %.1fs total rally time -> %s",
        video_id, summary["rally_count"], summary["total_rally_duration_s"], rallies_path,
    )
    if summary["rally_count"] == 0:
        # Not necessarily a failure — a video that's all warm-up/changeover
        # with no play captured is possible — but a real match with zero
        # detected rallies almost always means the ball-activity signal one
        # stage back is too weak to work with (check ball_coverage_rate),
        # not that the match genuinely had no points.
        logger.warning(
            "[analyze] video_id=%s found 0 rally segments — check ball_coverage_rate from the "
            "track stage; a weak ball-activity signal (poor tracking, heavy occlusion) is the "
            "most likely cause for a real match",
            video_id,
        )


def _write_rallies_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
