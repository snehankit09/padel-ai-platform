"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and
Postgres — Part 7f, the sixth and final piece of `analyze`.

Parts 7a-7e (rally/serve/shot/outcome/highlight detection) each write
their own JSON artifact under storage/ and hand the next sub-stage a
path on the payload; nothing before this point has touched Postgres.
This module is what turns 7e's highlights.json into `Highlight` rows
and 7a's rallies.json + 7d's outcomes.json into the match-level slice
of `Statistic` rows (Part 2's schema) that ml.pipeline.stats_
aggregation (Part 9a) has already worked out how to compute honestly —
the two tables this pipeline can populate before Part 8 (clip
generation) and the rest of Part 9 (player-identity-keyed stats) exist.

What this does NOT do, and why:
  - `Highlight.clip_file_path` stays NULL here. A HighlightEvent from
    7e has tight event boundaries (ml/pipeline/highlight_tagging.py's
    own docstring: "NOT padded with the pre/post-roll a viewable clip
    will want"), not a real clip file — trimming one, and deciding the
    padding, is Part 8's job. Part 8 reads these rows back (by
    match_id) and UPDATEs clip_file_path once a clip actually exists,
    rather than this stage inserting a second row or guessing at a
    padding window it has no clip-generation config for.
  - Only the four StatTypes ml.pipeline.stats_aggregation.STAT_TYPE_
    STATUS marks STATUS_COMPUTABLE_MATCH_LEVEL (TOTAL_POINTS,
    RALLY_LENGTH_AVG, LONGEST_RALLY, ERRORS) get written to
    `statistics`, all match-level (player_id=NULL) — this stage calls
    compute_match_level_stats(rallies, outcomes) directly rather than
    re-deriving rally counts/durations itself, so the "which StatTypes
    are honestly computable today" decision lives in exactly one place.
    Every other StatType (serve_percentage, winners, smash_success_rate,
    distance_covered, ...) needs either a ByteTracker track_id ->
    app.models.player.Player identity mapping that doesn't exist
    anywhere in this codebase yet (7b/7c/7d all key players by
    track_id, never by Player.id) or a winner/unforced-error split
    7d's own module docstring already says position data can't
    provide — both real Part 9 work, not something to fake here just
    to make this table look more populated than the data supports.
    WINNERS specifically is NOT derivable from outcomes.json's
    OUTCOME_IN_BOUNDS_END bucket, which 7d's docstring is explicit
    covers both a winner and a receiving-side unforced error with no
    way to tell them apart — see stats_aggregation's own module
    docstring for why ERRORS (out_of_bounds/net only) is a safe,
    honest floor where WINNERS still isn't.

Idempotency: `analyze` retries re-run every sub-stage in `_analyze`
from scratch (see app/workers/tasks.py's module docstring), including
this one, on a video that may have already had rows written by an
earlier, failed attempt. Rather than trying to diff old vs new, this
stage deletes-then-reinserts everything IT owns for the match (every
Highlight row for the match; only the Statistic rows this stage
writes, matched by match_id + player_id IS NULL + stat_type IN
MATCH_LEVEL_STAT_TYPES) before writing fresh ones — cheap, correct on
retry, and never touches rows Part 8 (clip_file_path updates) or the
rest of Part 9 (player-identity-keyed Statistic rows) own.
"""

from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy import delete

from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.enums import HighlightType, StatType
from app.models.highlight import Highlight
from app.models.statistic import Statistic
from app.models.video import Video
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)

# The only StatTypes this stage is honest about deriving today --
# exactly the STATUS_COMPUTABLE_MATCH_LEVEL entries in
# ml.pipeline.stats_aggregation.STAT_TYPE_STATUS. Built from that
# module's own stat_key constants (StatType.value strings 1:1 -- see
# that module's docstring) rather than hand-listing the enum members a
# second time, so this tuple can't silently drift from Part 9a's own
# inventory of what's computable.
MATCH_LEVEL_STAT_TYPES: tuple[StatType, ...] = (
    StatType.TOTAL_POINTS,
    StatType.RALLY_LENGTH_AVG,
    StatType.LONGEST_RALLY,
    StatType.ERRORS,
)


class AnalyzePersistenceStageError(Exception):
    """Raised when `analyze` can't find something an earlier sub-stage should already have produced."""


def run_analyze_persistence(payload: dict) -> None:
    """
    Real body of the persistence half of the `analyze` stage. Reads
    7a's rally summary (payload["rally_segments_path"]), 7d's point
    outcomes (payload["point_outcomes_path"]), and 7e's tagged events
    (payload["highlights_path"]) -- all three already on the payload by
    the time this runs last in `_analyze` -- and turns them into
    Highlight/Statistic rows against the video's Match.

    Same not-found handling as every other analyze sub-stage: an
    invalid/missing video_id logs and returns; a video that exists but
    is missing rally_segments_path, point_outcomes_path, or
    highlights_path means an earlier sub-stage didn't run or didn't set
    it -- a broken pipeline contract, not a bad caller-supplied id --
    so this raises AnalyzePersistenceStageError, a real stage failure
    Part 4d's retry/fail path handles. The video-existence check and the
    actual writes use two separate `get_sync_db()` sessions (existence
    check, then file reads that need no DB at all, then a write
    session) rather than holding one session open across the file I/O
    in between -- same "don't hold a DB connection longer than the DB
    work needs" shape as every other stage in this pipeline.
    """
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping result persistence", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[analyze] video_id=%s not found; skipping result persistence", video_id)
            return
        match_id = video.match_id

    required = {
        "rally_segments_path": "the analyze stage's rally-detection half (Part 7a)",
        "point_outcomes_path": "the analyze stage's point-outcome half (Part 7d)",
        "highlights_path": "the analyze stage's highlight-tagging half (Part 7e)",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        details = "; ".join(f"'{key}' (should have been set by {required[key]})" for key in missing)
        raise AnalyzePersistenceStageError(f"video_id={video_id}: payload is missing {details}")

    storage = get_storage_service()
    try:
        rallies_data = _read_json(storage.get_local_path(payload["rally_segments_path"]))
        outcomes_data = _read_json(storage.get_local_path(payload["point_outcomes_path"]))
        highlights_data = _read_json(storage.get_local_path(payload["highlights_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalyzePersistenceStageError(
            f"video_id={video_id}: could not read a required rally/outcomes/highlights file: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.pipeline.point_outcome import PointOutcome
    from ml.pipeline.rally_detection import RallySegment
    from ml.pipeline.stats_aggregation import compute_match_level_stats

    rallies = [RallySegment(**r) for r in rallies_data["rallies"]]
    outcomes = [PointOutcome(**o) for o in outcomes_data["outcomes"]]
    stat_values = compute_match_level_stats(rallies, outcomes)

    with get_sync_db() as db:
        # Idempotency: clear only what this stage owns for this match
        # before reinserting (see module docstring).
        db.execute(delete(Highlight).where(Highlight.match_id == match_id))
        db.execute(
            delete(Statistic).where(
                Statistic.match_id == match_id,
                Statistic.player_id.is_(None),
                Statistic.stat_type.in_(MATCH_LEVEL_STAT_TYPES),
            )
        )

        highlight_count = 0
        for event in highlights_data.get("highlights", []):
            db.add(
                Highlight(
                    match_id=match_id,
                    event_type=HighlightType(event["highlight_type"]),
                    start_time_seconds=event["start_time_s"],
                    end_time_seconds=event["end_time_s"],
                    importance_score=event["importance_score"],
                    clip_file_path=None,
                )
            )
            highlight_count += 1

        # compute_match_level_stats already decides which of the four
        # match-level StatTypes actually apply (e.g. no RALLY_LENGTH_AVG/
        # LONGEST_RALLY when rally_count is 0) -- this stage just
        # translates each StatValue's plain stat_key string back to the
        # real StatType enum (1:1 by .value, same correspondence Part
        # 9a's own test_stats_aggregation.py cross-checks) and writes it.
        statistic_count = 0
        for stat_value in stat_values:
            db.add(
                Statistic(
                    match_id=match_id, player_id=None,
                    stat_type=StatType(stat_value.stat_key), value=stat_value.value,
                )
            )
            statistic_count += 1

        db.commit()

    payload["highlights_persisted_count"] = highlight_count
    payload["statistics_persisted_count"] = statistic_count

    logger.info(
        "[analyze] video_id=%s persisted %d highlight(s) and %d statistic(s) for match_id=%s",
        video_id, highlight_count, statistic_count, match_id,
    )


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
