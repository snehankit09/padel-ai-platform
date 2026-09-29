"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
label-resolution problem Tier 3a/3b created — Highlights Improvement
Roadmap Tier 3c, the actual final sub-stage of `analyze`.

**The ordering problem this stage exists to fix.** Tier 3b's
Highlight.player_track_id (a raw ByteTrack track_id) and Tier 2d's
team_sides/<video_id>/team_side_labels.json (the {track_id: label}
lookup that can turn one into a human-readable label) are each produced
by a different sub-stage, and — see app/workers/tasks.py's `_analyze`
call order — run_analyze_persistence (Part 7f, which inserts Highlight
rows with player_track_id already set) runs BEFORE run_team_side_labeling
(Tier 2d, which is what actually computes the lookup). Neither piece can
resolve a label on its own at the moment it runs: 7f doesn't have the
lookup yet, and 2d has no Highlight rows to update (it only ever touches
team_side_labels.json, never the highlights table — see its own module
docstring). This stage is what runs after BOTH exist, so it's the first
point in `_analyze` where resolution is even possible. Reordering
`_analyze` so team_side_labeling ran before persistence was considered
and rejected: 7f's own docstring already documents a specific
Highlight-row lifecycle (insert once with clip_file_path/thumbnail_file_path
NULL, Part 8b UPDATEs them later) that several other stages (8a's own
tight-window matching, 8b's UPDATE) already depend on running in the
current order; moving 7f would risk all of that just to save one small
UPDATE step at the end. A small final step that reads back what already
exists is the same "update a row an earlier sub-stage already inserted"
shape Part 8b already uses for clip_file_path/thumbnail_file_path — see
that module's own docstring for the precedent.

**What this does, in order:**
1. Reads payload["team_side_labels_path"] (Tier 2d) back off disk for
   its "track_labels" lookup. A MISSING path here is not an error — Tier
   2d's own docstring says a missing court calibration means it produces
   no artifact at all that run, and this stage inherits that same
   non-fatal skip rather than treating an upstream non-fatal skip as a
   fatal problem one stage later.
2. Reads back every Highlight row for this match with a non-NULL
   player_track_id (Tier 3b) — exactly the POWERFUL_SMASH/SPECTACULAR_SAVE
   rows Tier 3a ever attributes; see that module's own docstring.
3. For each, looks up str(player_track_id) in the lookup and UPDATEs
   player_label when found. A player_track_id with no entry in the
   lookup (Tier 1d skipped that track_id entirely — see
   ml.pipeline.team_side_positions.merge_track_fragments' docstring for
   when) is left with player_label=NULL rather than guessed at; this
   isn't a stage failure, just an honestly unresolved row, same posture
   Tier 2c's own build_track_label_lookup already takes for a track_id
   absent from its dict.

Idempotency: this stage only ever UPDATEs rows 7f freshly (re)inserted
earlier in the SAME analyze run (7f deletes-then-reinserts every
Highlight row for the match on every retry — see its own module
docstring), so a retried `analyze` always resolves fresh rows against a
freshly-recomputed lookup, never a stale mix of a previous attempt's
labels and this run's rows.
"""

from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy import select

from app.core.database import get_sync_db
from app.models.highlight import Highlight
from app.models.video import Video
from app.services.storage import get_storage_service

logger = logging.getLogger(__name__)


class TeamSideLabelResolutionStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_team_side_label_resolution(payload: dict) -> None:
    """
    Real body of the label-resolution half of the `analyze` stage. See
    module docstring for the full "why this exists and why it runs last"
    reasoning.

    A missing/invalid video_id logs and returns, same as every other
    analyze sub-stage. A missing team_side_labels_path is Tier 2d's own
    non-fatal skip propagating forward, not an error, so this stage logs
    and returns too — there's honestly nothing to resolve without a
    lookup. A present team_side_labels_path whose file can't be read, or
    whose JSON is missing "track_labels" entirely, IS a broken pipeline
    contract (2d always writes that key when it writes the file at all)
    and raises TeamSideLabelResolutionStageError.
    """
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping label resolution", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[analyze] video_id=%s not found; skipping label resolution", video_id)
            return
        match_id = video.match_id

    labels_path = payload.get("team_side_labels_path")
    if not labels_path:
        logger.info(
            "[analyze] video_id=%s has no team_side_labels_path (Tier 2d produced no artifact this "
            "run) — nothing to resolve, leaving every Highlight.player_label NULL",
            video_id,
        )
        return

    storage = get_storage_service()
    try:
        labels_data = _read_json(storage.get_local_path(labels_path))
        track_labels = labels_data["track_labels"]
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        raise TeamSideLabelResolutionStageError(
            f"video_id={video_id}: could not read team_side_labels_path or it's missing "
            f"'track_labels' (should have been written by Tier 2d): {exc}"
        ) from exc

    with get_sync_db() as db:
        rows = db.execute(
            select(Highlight).where(Highlight.match_id == match_id, Highlight.player_track_id.is_not(None))
        ).scalars().all()

        resolved_count = 0
        for row in rows:
            label = track_labels.get(str(row.player_track_id))
            if label is None:
                continue
            row.player_label = label
            resolved_count += 1

        db.commit()

    payload["team_side_labels_resolved_count"] = resolved_count

    logger.info(
        "[analyze] video_id=%s resolved %d/%d attributed highlight(s) to a player_label",
        video_id, resolved_count, len(rows),
    )


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
