"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure quadrant/label logic in ml/pipeline/team_side_positions.py —
Highlights Improvement Roadmap Tier 2d, closing out Tiers 1a-2c.

Same layering as every other analyze sub-stage (player_statistics_stage.py,
serve_detection_stage.py, ...): this file owns the DB lookup, the payload
contract, StorageService, and the deferred ml import; ml/pipeline/
team_side_positions.py knows none of those things and needs none of them
to be unit tested (see backend/tests/test_team_side_positions.py — every
Tier 1-2 test constructs tracks_data/homography directly, no DB, no
Celery, no real video).

**Reads two of `track`/`detect`'s earlier outputs**:
payload["player_tracks_path"] (Part 6b) and
payload["court_calibration_path"] (Part 5e). Unlike every other analyze
sub-stage that treats a missing court_calibration_path as "skip the
calibration-dependent parts, keep going" (serve_detection_stage.py,
player_statistics_stage.py), this stage has NOTHING to compute without
one — every Tier 1a-1d function needs a real homography to turn a
tracked pixel position into a court-relative one at all, there's no
pixel-only fallback path anywhere in team_side_positions.py. So a
missing calibration here means the whole stage logs and returns, same
non-fatal "this video just doesn't get this artifact" treatment
court_detection_stage.py's own calibration failure already gets one
stage earlier — not a reason to fail retryable `analyze`.

**What this stage does, in order:**
1. Reads player_tracks.json and calibration.json back off disk.
2. Calls ml.pipeline.team_side_positions.merge_track_fragments (Tier 1d)
   to turn every track_id in the match into inferred MergedPlayers.
3. Calls build_track_label_lookup (Tier 2c) to flatten those into the
   {track_id: label} dict Phase 3 will actually query.
4. Writes both, plus summarize_track_label_lookup's (Tier 2d) honest
   confidence diagnostic, to one JSON artifact
   (make_team_side_labels_destination_path) — the same "derived artifact
   a later stage reads back instead of recomputing" shape every other
   Part 7/8/9 stage already uses.

Idempotency: same destination key every run, so re-running `analyze`
for a video just overwrites this file with a fresh computation — no DB
rows owned by this stage, nothing to delete-then-reinsert.
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.core.database import get_sync_db
from app.services.storage import get_storage_service, make_team_side_labels_destination_path

logger = logging.getLogger(__name__)


class TeamSideLabelingStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_team_side_labeling(payload: dict) -> None:
    """
    Real body of the team-side-labeling half of the `analyze` stage. See
    module docstring for the full read -> compute -> persist shape.

    A missing/invalid video_id logs and returns, same as every other
    analyze sub-stage. A missing player_tracks_path is a broken pipeline
    contract (the track stage should already have set it) and raises
    TeamSideLabelingStageError. A missing or None court_calibration_path
    is NOT an error — see module docstring — it just means this run
    produces no team_side_labels_path at all.
    """
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping team-side labeling", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[analyze] video_id=%s not found; skipping team-side labeling", video_id)
            return

    player_tracks_path = payload.get("player_tracks_path")
    if not player_tracks_path:
        raise TeamSideLabelingStageError(
            f"video_id={video_id}: payload is missing 'player_tracks_path' "
            "(should have been set by the track stage, Part 6b)"
        )

    calibration_path = payload.get("court_calibration_path")
    if not calibration_path:
        logger.info(
            "[analyze] video_id=%s has no court_calibration_path — team-side labeling needs a real "
            "homography for every Tier 1a-1d computation, so this run produces no team_side_labels_path",
            video_id,
        )
        return

    storage = get_storage_service()

    try:
        player_tracks_data = _read_json(storage.get_local_path(player_tracks_path))
        calibration_data = _read_json(storage.get_local_path(calibration_path))
    except (OSError, json.JSONDecodeError) as exc:
        raise TeamSideLabelingStageError(
            f"video_id={video_id}: could not read player tracks or court calibration: {exc}"
        ) from exc

    ensure_ml_importable()
    import numpy as np

    from ml.pipeline.team_side_positions import (
        build_track_label_lookup,
        merge_track_fragments,
        summarize_track_label_lookup,
        to_serializable,
    )

    homography = np.array(calibration_data["homography"], dtype=np.float64)

    players = merge_track_fragments(player_tracks_data, homography)
    track_labels = build_track_label_lookup(players)

    logger.info(
        "[analyze] video_id=%s team-side labeling found %d merged player(s) across %d track_id(s)",
        video_id, len(players), len(track_labels),
    )

    labels_key = make_team_side_labels_destination_path(video_uuid)
    labels_path = storage.get_local_path(labels_key)
    _write_json(
        labels_path,
        {
            "players": to_serializable(players),
            # JSON object keys are always strings -- track_id (int) has
            # to be stringified here; Phase 3 converts back on read.
            "track_labels": {str(track_id): label for track_id, label in track_labels.items()},
            "summary": summarize_track_label_lookup(track_labels),
        },
    )
    payload["team_side_labels_path"] = labels_key
    payload["team_side_label_count"] = len(track_labels)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
