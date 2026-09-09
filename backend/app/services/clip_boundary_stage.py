"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure clip-boundary logic in ml/pipeline/clip_boundaries.py — Part 8a, the
first piece of Part 8 (highlight clip generation) to move past not
existing at all.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as app/services/highlight_tagging_stage.py and every
other analyze sub-stage before it. Reads back Part 7e's tagged events
(highlights_path) — the one JSON input every padded boundary is derived
from — plus the video's own probed duration straight off the `Video` row
(Video.duration_seconds, set by the validate stage long before analyze
ever runs), which is the only reason this stage touches the DB at all;
same "one cheap existence-shaped query, then pure file I/O" split
highlight_tagging_stage.py uses.

What this does NOT do, and why: it does not touch `Highlight` rows.
Part 7f (app/services/analyze_persistence_stage.py) already persisted
one `Highlight` row per tagged event with *tight* start/end boundaries,
and its own module docstring is explicit that deciding the padding
window and writing a real, trimmed clip file are Part 8's job, done by
reading those rows back rather than this stage inserting a second row or
mutating them mid-`analyze`. This stage produces the padded boundaries
Part 8's later steps need for exactly that — clip_boundaries.json,
keyed by the same (rally_index, highlight_type, source_frame_index)
Part 7f's rows can be matched back against — without yet doing the
FFmpeg trim or the row update itself.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.storage import get_storage_service, make_clip_boundaries_destination_path

logger = logging.getLogger(__name__)


class ClipBoundaryStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_clip_boundary_calculation(payload: dict) -> None:
    """
    Real body of the clip-boundary half of the `analyze` stage. Reads
    Part 7e's tagged HighlightEvents (already on the payload) and the
    video's own probed duration, and persists a flat list of padded,
    clamped ClipBoundary objects for Part 8's later steps (FFmpeg
    trimming, updating each Highlight row's start/end + clip_file_path)
    to read back.

    Same not-found handling as highlight_tagging_stage.py: an
    invalid/missing video_id logs and returns; a video that exists but
    has no highlights_path on the payload means Part 7e didn't run or
    didn't set it — a broken pipeline contract, not a bad caller-supplied
    id — so that raises ClipBoundaryStageError (Part 4d's retry/fail
    path). A video row that exists but has no duration_seconds yet
    (validate — Part 5a — should always have set this before analyze
    ever runs) is the same kind of broken-contract failure, not a
    gracefully-degradable case: without a real duration there's nothing
    honest to clamp padding against.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping clip boundary calculation", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
    if video is None:
        logger.warning("[analyze] video_id=%s not found; skipping clip boundary calculation", video_id)
        return

    if not payload.get("highlights_path"):
        raise ClipBoundaryStageError(
            f"video_id={video_id}: payload is missing 'highlights_path' "
            "(should have been set by the analyze stage's highlight-tagging half, Part 7e)"
        )

    video_duration_s = video.duration_seconds
    if not video_duration_s or video_duration_s <= 0:
        raise ClipBoundaryStageError(
            f"video_id={video_id}: video has no positive duration_seconds — "
            "the validate stage should have set this before analyze ever runs"
        )

    storage = get_storage_service()

    try:
        highlights_data = _read_json(storage.get_local_path(payload["highlights_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipBoundaryStageError(
            f"video_id={video_id}: could not read the required highlights file: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.pipeline.clip_boundaries import compute_clip_boundaries, summarize_clip_boundaries, to_serializable
    from ml.pipeline.highlight_tagging import HighlightEvent

    events = [HighlightEvent(**e) for e in highlights_data.get("highlights", [])]

    logger.info(
        "[analyze] video_id=%s computing clip boundaries for %d highlight event(s)",
        video_id, len(events),
    )

    boundaries = compute_clip_boundaries(
        events,
        video_duration_s=video_duration_s,
        pre_roll_s=settings.clip_pre_roll_s,
        post_roll_s=settings.clip_post_roll_s,
        min_duration_s=settings.clip_min_duration_s,
    )
    summary = summarize_clip_boundaries(boundaries)

    clip_boundaries_key = make_clip_boundaries_destination_path(video_uuid)
    clip_boundaries_path = storage.get_local_path(clip_boundaries_key)
    _write_json(
        clip_boundaries_path,
        {
            "clip_pre_roll_s": settings.clip_pre_roll_s,
            "clip_post_roll_s": settings.clip_post_roll_s,
            "clip_min_duration_s": settings.clip_min_duration_s,
            "video_duration_s": video_duration_s,
            **summary,
            "clip_boundaries": to_serializable(boundaries),
        },
    )

    payload["clip_boundaries_path"] = clip_boundaries_key
    payload["clip_boundary_count"] = summary["clip_count"]

    logger.info(
        "[analyze] video_id=%s done: %d clip boundary(s) computed (avg %.1fs) -> %s",
        video_id, summary["clip_count"], summary["avg_clip_duration_s"], clip_boundaries_path,
    )


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
