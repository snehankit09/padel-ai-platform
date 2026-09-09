"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure highlight-tagging logic in ml/pipeline/highlight_tagging.py — Part
7e, the fifth and final piece of `analyze` to move past a no-op.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as app/services/rally_detection_stage.py /
serve_detection_stage.py / shot_classification_stage.py /
point_outcome_stage.py. Reads back rally segments (7a) and shots (7c) —
the two inputs every tag_* rule in ml/pipeline/highlight_tagging.py
actually needs (see that module's docstring for why LONG_RALLY/
FAST_EXCHANGE/POWERFUL_SMASH/SPECTACULAR_SAVE are all derivable from just
those two, and why WINNING_SHOT/MATCH_POINT/BREAK_POINT aren't attempted
at all rather than degraded to a guess). Does NOT read serve_events_path
or point_outcomes_path — nothing this module tags depends on who served
or how the point ended, only on rally timing and in-rally shot sequence —
so, same as point_outcome_stage.py not needing shots.json/serves.json,
this stage doesn't require those two to exist even though they're
produced earlier in `analyze` for other consumers.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_highlights_destination_path

logger = logging.getLogger(__name__)


class HighlightTaggingStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_highlight_tagging(payload: dict) -> None:
    """
    Real body of the highlight-tagging half of the `analyze` stage. Reads
    rally segments (7a) and classified shots (7c) already on the payload,
    and persists a flat list of candidate HighlightEvents for Part 7f
    (app/services/analyze_persistence_stage.py, `analyze`'s own last
    sub-stage) to turn into `Highlight` DB rows, and later for Part 8
    (clip generation — trims each tagged window into an actual clip file,
    padded with pre/post-roll, and updates that row's clip_file_path) to
    read back.

    Same not-found handling as point_outcome_stage.py: an invalid/missing
    video_id logs and returns; missing rally_segments_path or shots_path
    means an earlier stage didn't run or didn't set it — a broken
    pipeline contract, not a bad caller-supplied id — so those raise
    HighlightTaggingStageError (Part 4d's retry/fail path). Unlike
    shot_classification_stage.py's own required set, there's no optional-
    with-graceful-fallback input here (no court_calibration_path
    equivalent): every rule this module runs works the same whether or
    not the video was calibrated, since none of them need court meters —
    see ml/pipeline/highlight_tagging.py's module docstring for exactly
    which fields each rule actually touches.

    `frame_sample_fps` (Part 5a's validate stage) is required for the
    same reason it's required by rally_detection_stage.py: shots.json
    stores frame indices, not seconds, and every tag_* rule that compares
    two contacts' timing (fast exchange, spectacular save) or windows a
    single shot (powerful smash) needs real seconds to apply its
    threshold against.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping highlight tagging", video_id)
        return

    with get_sync_db() as db:
        video_exists = db.get(Video, video_uuid) is not None
    if not video_exists:
        logger.warning("[analyze] video_id=%s not found; skipping highlight tagging", video_id)
        return

    required = {
        "rally_segments_path": "the analyze stage's rally-detection half (Part 7a)",
        "shots_path": "the analyze stage's shot-classification half (Part 7c)",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        details = "; ".join(f"'{key}' (should have been set by {required[key]})" for key in missing)
        raise HighlightTaggingStageError(f"video_id={video_id}: payload is missing {details}")

    sample_fps = payload.get("frame_sample_fps")
    if not sample_fps:
        raise HighlightTaggingStageError(
            f"video_id={video_id}: payload has no 'frame_sample_fps' — the validate stage "
            "should have set this before analyze ever runs"
        )

    storage = get_storage_service()

    try:
        rallies_data = _read_json(storage.get_local_path(payload["rally_segments_path"]))
        shots_data = _read_json(storage.get_local_path(payload["shots_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise HighlightTaggingStageError(
            f"video_id={video_id}: could not read a required rally/shots file: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.pipeline.highlight_tagging import (
        detect_highlights,
        summarize_highlights,
        to_serializable,
    )
    from ml.pipeline.rally_detection import RallySegment
    from ml.pipeline.shot_classification import Shot

    rallies = [RallySegment(**r) for r in rallies_data["rallies"]]
    shots = [Shot(**s) for s in shots_data["shots"]]

    logger.info(
        "[analyze] video_id=%s tagging highlight events across %d rally(s) and %d shot(s)",
        video_id, len(rallies), len(shots),
    )

    events = detect_highlights(
        rallies, shots,
        sample_fps=sample_fps,
        long_rally_min_duration_s=settings.highlight_long_rally_min_duration_s,
        long_rally_score_saturation_s=settings.highlight_long_rally_score_saturation_s,
        fast_exchange_max_interval_s=settings.highlight_fast_exchange_max_interval_s,
        fast_exchange_min_shot_count=settings.highlight_fast_exchange_min_shot_count,
        fast_exchange_score_saturation_count=settings.highlight_fast_exchange_score_saturation_count,
        smash_height_ratio=settings.shot_smash_height_ratio,
        powerful_smash_score_ceiling_ratio=settings.highlight_powerful_smash_score_ceiling_ratio,
        spectacular_save_max_response_s=settings.highlight_spectacular_save_max_response_s,
    )
    summary = summarize_highlights(events)

    highlights_key = make_highlights_destination_path(video_uuid)
    highlights_path = storage.get_local_path(highlights_key)
    _write_json(
        highlights_path,
        {
            "highlight_long_rally_min_duration_s": settings.highlight_long_rally_min_duration_s,
            "highlight_fast_exchange_max_interval_s": settings.highlight_fast_exchange_max_interval_s,
            "highlight_fast_exchange_min_shot_count": settings.highlight_fast_exchange_min_shot_count,
            "highlight_spectacular_save_max_response_s": settings.highlight_spectacular_save_max_response_s,
            "frame_sample_fps": sample_fps,
            **summary,
            "highlights": to_serializable(events),
        },
    )

    payload["highlights_path"] = highlights_key
    payload["highlight_count"] = summary["highlight_count"]

    logger.info(
        "[analyze] video_id=%s done: %d highlight event(s) tagged (%s) -> %s",
        video_id, summary["highlight_count"], summary["highlight_counts_by_type"], highlights_path,
    )


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
