"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
track_id-keyed statistics logic in ml/pipeline/stats_aggregation.py (9a)
and ml/pipeline/player_identity.py (9b) — Part 9f, wiring the rest of
Part 9 into the pipeline for real.

Same "video existence check + broken-pipeline-contract check + deferred
ml import" shape as every other analyze sub-stage (rally_detection_stage.py,
serve_detection_stage.py, ...). Reads back four of `track`/`analyze`'s
earlier outputs — payload["rally_segments_path"] (7a), payload["shots_path"]
(7c), payload["point_outcomes_path"] (7d), payload["player_tracks_path"]
(6b) — plus, optionally, payload["court_calibration_path"] (5e), same
"used when available, never required" treatment serve_detection_stage.py
already gives it (see _build_court_meters_converter below, copied from
that module rather than imported, matching every other analyze sub-stage's
own local copy).

**What this stage does.** Three things, in order:

1. Calls ml.pipeline.stats_aggregation.player_level_stat_values (9a) to
   get every DISTANCE_COVERED/MOVEMENT_SPEED_AVG/REACTION_TIME_AVG/
   SMASH_SUCCESS_RATE/NET_SUCCESS_RATE StatValue this video's data
   honestly supports, still keyed by ByteTrack track_id.
2. Calls ml.pipeline.player_identity.assign_court_sides (9b) to get each
   track_id's SIDE_A/SIDE_B court-side grouping — real geometry, but
   NOT a Player.id (see that module's own docstring for exactly why
   grouping by side can't get all the way to individual player identity
   in this codebase).
3. Writes both, plus ml.pipeline.stats_aggregation.summarize_match_diagnostics'
   non-StatType breakdowns, to one JSON artifact
   (make_player_statistics_destination_path) — the same "derived artifact
   a later step reads back" shape every Part 7/8 stage already uses, and
   here specifically the shape a future human-in-the-loop identity-
   confirmation step would read to build a real track_id -> Player.id
   mapping (see that path helper's own docstring).

**What this stage deliberately does NOT do: guess a track_id -> Player.id
mapping.** ml.pipeline.player_identity.py's own module docstring is
explicit that nothing in this pipeline's data can tell two teammates on
the same court side apart, and app/services/player_statistics_persistence_stage.py
(9e) is built around taking that mapping as a caller-supplied argument
for exactly that reason. This stage is that caller for the automated
pipeline run, and the automated pipeline run has no such mapping to
supply — so it calls persist_player_statistics with an empty mapping,
which 9e's own docstring already establishes is not an error, just zero
rows written this call (the same "no signal, no row" posture
analyze_persistence_stage.py's ERRORS/WINNERS split and every
STATUS_NOT_COMPUTABLE verdict in stats_aggregation.py already take).
Doing anything else here — a coin-flip side-to-player_id assignment, a
"first MatchPlayer of that team_number" default — would be exactly the
kind of fabricated row this whole pipeline has refused to write at every
earlier stage, just moved one stage later.

Idempotency: delegated entirely to persist_player_statistics (9e), which
already deletes every PLAYER_LEVEL_STAT_TYPES row it owns for this match
before writing fresh ones. This stage adds nothing on top of that beyond
overwriting its own JSON artifact, which is naturally idempotent (same
destination key every run).
"""

from __future__ import annotations

import json
import logging
import uuid

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.video import Video
from app.services.player_statistics_persistence_stage import persist_player_statistics
from app.services.storage import get_storage_service, make_player_statistics_destination_path

logger = logging.getLogger(__name__)


class PlayerStatisticsStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_player_statistics_aggregation(payload: dict) -> None:
    """
    Real body of the player-statistics half of the `analyze` stage.
    Reads rally segments (7a), shots (7c), point outcomes (7d), and
    player tracks (6b) already on the payload, plus a court calibration
    when Part 5e produced one for this video, computes 9a/9b's
    track_id-keyed output, writes it to storage, and calls 9e with an
    empty identity mapping (see module docstring for why).

    Same not-found handling as every other analyze sub-stage: an
    invalid/missing video_id logs and returns; missing rally_segments_path,
    shots_path, point_outcomes_path, player_tracks_path, or
    frame_sample_fps each mean an earlier stage didn't run or didn't set
    it — a broken pipeline contract — so those raise
    PlayerStatisticsStageError (Part 4d's retry/fail path). A missing or
    None court_calibration_path is NOT an error, same reasoning as
    serve_detection_stage.py: DISTANCE_COVERED/MOVEMENT_SPEED_AVG and
    assign_court_sides simply aren't computed without it (see
    compute_distance_and_speed's and assign_court_sides' own docstrings),
    everything else still is.
    """
    settings = get_settings()
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping player statistics", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[analyze] video_id=%s not found; skipping player statistics", video_id)
            return
        match_id = video.match_id

    required = {
        "rally_segments_path": "the analyze stage's rally-detection half (Part 7a)",
        "shots_path": "the analyze stage's shot-classification half (Part 7c)",
        "point_outcomes_path": "the analyze stage's point-outcome half (Part 7d)",
        "player_tracks_path": "the track stage (Part 6b)",
    }
    missing = [key for key in required if not payload.get(key)]
    if missing:
        details = "; ".join(f"'{key}' (should have been set by {required[key]})" for key in missing)
        raise PlayerStatisticsStageError(f"video_id={video_id}: payload is missing {details}")

    sample_fps = payload.get("frame_sample_fps")
    if not sample_fps:
        raise PlayerStatisticsStageError(
            f"video_id={video_id}: payload has no 'frame_sample_fps' — the validate stage "
            "should have set this before analyze ever runs"
        )

    storage = get_storage_service()

    try:
        rallies_data = _read_json(storage.get_local_path(payload["rally_segments_path"]))
        shots_data = _read_json(storage.get_local_path(payload["shots_path"]))
        outcomes_data = _read_json(storage.get_local_path(payload["point_outcomes_path"]))
        player_tracks_data = _read_json(storage.get_local_path(payload["player_tracks_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlayerStatisticsStageError(
            f"video_id={video_id}: could not read a required rally/shots/outcomes/tracks file: {exc}"
        ) from exc

    ensure_ml_importable()
    from ml.pipeline.player_identity import PlayerIdentityError, assign_court_sides
    from ml.pipeline.player_identity import to_serializable as sides_to_serializable
    from ml.pipeline.player_identity import summarize_side_assignments
    from ml.pipeline.point_outcome import PointOutcome
    from ml.pipeline.rally_detection import RallySegment
    from ml.pipeline.serve_detection import frames_from_tracks_json
    from ml.pipeline.shot_classification import Shot
    from ml.pipeline.stats_aggregation import (
        player_level_stat_values,
        summarize_match_diagnostics,
        to_serializable as stats_to_serializable,
    )

    rallies = [RallySegment(**r) for r in rallies_data["rallies"]]
    shots = [Shot(**s) for s in shots_data["shots"]]
    outcomes = [PointOutcome(**o) for o in outcomes_data["outcomes"]]
    player_frames = frames_from_tracks_json(player_tracks_data)

    to_court_meters = _build_court_meters_converter(payload.get("court_calibration_path"), storage)

    logger.info(
        "[analyze] video_id=%s computing player-level statistics across %d rally(s), %d shot(s) "
        "(court_calibration=%s)",
        video_id, len(rallies), len(shots), "yes" if to_court_meters is not None else "no",
    )

    stat_values = player_level_stat_values(
        rallies, shots, outcomes, player_frames,
        sample_fps=sample_fps,
        to_court_meters=to_court_meters,
    )

    side_assignments = []
    if to_court_meters is not None:
        try:
            side_assignments = assign_court_sides(
                player_frames, to_court_meters=to_court_meters, court_length_m=settings.court_length_m
            )
        except PlayerIdentityError as exc:
            # Same non-fatal treatment court_detection_stage.py's own
            # calibration failure gets: a court-side grouping the pipeline
            # can't honestly compute yet doesn't mean the stat_values
            # above (which don't need it) should be thrown away too.
            logger.warning("[analyze] video_id=%s could not compute court-side assignments: %s", video_id, exc)

    # Load serves back too, purely for summarize_match_diagnostics' own
    # serve_counts_by_player/serve_identification breakdown -- optional,
    # same "diagnostic, not a StatType" treatment that function's own
    # docstring gives it. A missing serve_events_path (shouldn't happen
    # by this point in `_analyze`, but this stage doesn't own that
    # contract) just means an empty diagnostics section, not a stage
    # failure.
    serves = []
    serve_events_path = payload.get("serve_events_path")
    if serve_events_path:
        try:
            serves_data = _read_json(storage.get_local_path(serve_events_path))
            from ml.pipeline.serve_detection import ServeEvent

            serves = [ServeEvent(**s) for s in serves_data.get("serves", [])]
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(
                "[analyze] video_id=%s could not read serve_events_path for diagnostics: %s", video_id, exc
            )
    diagnostics = summarize_match_diagnostics(shots, serves)

    stats_key = make_player_statistics_destination_path(video_uuid)
    stats_path = storage.get_local_path(stats_key)
    _write_json(
        stats_path,
        {
            "frame_sample_fps": sample_fps,
            "used_court_calibration": to_court_meters is not None,
            "stat_values": stats_to_serializable(stat_values),
            "side_assignments": sides_to_serializable(side_assignments),
            "side_assignment_summary": summarize_side_assignments(side_assignments),
            "diagnostics": diagnostics,
        },
    )
    payload["player_stats_path"] = stats_key
    payload["player_stat_value_count"] = len(stat_values)

    # No track_id -> Player.id mapping exists for an automated pipeline
    # run (see module docstring) -- 9e already treats an empty mapping as
    # "zero rows written, not an error", which both keeps this call
    # honest and exercises the exact same idempotent delete-then-reinsert
    # path a future, real mapping will use.
    written = persist_player_statistics(match_id, stat_values, {})

    payload["player_statistics_persisted_count"] = written

    logger.info(
        "[analyze] video_id=%s computed %d player-level statistic(s) across %d track_id(s) "
        "(%d side assignment(s), %d persisted with no identity mapping yet) -> %s",
        video_id, len(stat_values), len({v.track_id for v in stat_values if v.track_id is not None}),
        len(side_assignments), written, stats_path,
    )


def _build_court_meters_converter(calibration_key, storage):
    """
    Returns a `(x, y) -> (x_m, y_m)` callable backed by the video's
    persisted court calibration (Part 5e), or None when no calibration
    exists for this video — every caller here treats None as "skip the
    calibration-dependent statistics instead of erroring", same
    "optional, never required" contract as serve_detection_stage.py's
    identically-named helper (kept as its own local copy rather than a
    shared import, matching every other analyze sub-stage's own copy of
    this same small helper).
    """
    if not calibration_key:
        return None

    calibration_path = storage.get_local_path(calibration_key)
    try:
        calibration_data = _read_json(calibration_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning(
            "[analyze] could not read court calibration at %r (%s) — skipping calibration-dependent "
            "player statistics",
            calibration_path, exc,
        )
        return None

    ensure_ml_importable()
    import numpy as np

    from ml.detection.court_detector import pixel_to_court_point

    homography = np.array(calibration_data["homography"], dtype=np.float64)
    return lambda point: pixel_to_court_point(homography, point)


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _write_json(path: str, data: dict) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)
