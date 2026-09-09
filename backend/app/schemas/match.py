"""
Match schemas — Part 12b's list-page shape (MatchListItem/MatchListResponse),
Part 12c's detail-page shape (MatchDetailResponse/HighlightItem), Part
12d's statistics shape (MatchStatisticsResponse/StatisticItem), and Part
12e's reel-player shape (ReelResponse).

Kept separate from app/schemas/video.py rather than reusing
VideoStatusResponse: this endpoint is Match-shaped (one row per match, not
per video) and only needs a handful of Video's fields folded in — not
the same list of callers, not the same shape.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import HighlightType, ReelStatus, StatType, VideoStatus


class MatchListItem(BaseModel):
    """
    One row of GET /matches. video_status/video_id are optional because
    Match.video is nullable at the DB level even though the only way to
    create a Match today (POST /videos/upload) always creates both in the
    same request — a future bulk-import or manual-entry path could still
    leave a match without one, and the list page should render that match
    (as "no video") rather than 500.

    thumbnail_url is always null for now: no stage in the pipeline
    generates a thumbnail image yet (Part 8's clips and Part 10's reel are
    the only derived video artifacts that exist today). The field stays on
    the contract now so the frontend's card layout — and its "thumbnail if
    available, placeholder otherwise" fallback — doesn't need a second
    change whenever that stage is added later.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    played_at: datetime
    venue: str | None
    format: str
    video_id: uuid.UUID | None
    video_status: VideoStatus | None
    thumbnail_url: str | None


class MatchListResponse(BaseModel):
    matches: list[MatchListItem]


class HighlightItem(BaseModel):
    """
    One row of GET /matches/{match_id}'s `highlights` list — one per
    `Highlight` DB row (app/models/highlight.py), Part 7f's persistence of
    Part 7e's tagging output, with Part 8b's clip fields folded in.

    importance_score is passed through exactly as stored: it's in [0, 1]
    and meaningful for comparing highlights of the *same* event_type, but
    — per ml/pipeline/highlight_tagging.py's own HighlightEvent docstring
    — not calibrated across types (a 0.9 long_rally and a 0.9
    powerful_smash aren't claimed to be equally exciting). This schema
    doesn't derive a cross-type rank from it, and the route below doesn't
    sort by it, for the same reason.

    clip_url is null whenever Highlight.clip_file_path is null — either
    Part 8b (clip extraction) hasn't run yet for this match, or this
    specific highlight's clip failed to extract (see
    app/services/clip_extraction_stage.py's own docstring: one bad clip
    doesn't fail the whole match's extraction, it just leaves that row's
    clip_file_path unset). The frontend's per-card fallback for a null
    clip_url is what a viewer actually sees for either case.
    """

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    event_type: HighlightType
    start_time_seconds: float
    end_time_seconds: float
    importance_score: float
    clip_url: str | None


class MatchDetailResponse(BaseModel):
    """GET /matches/{match_id} — match metadata plus its highlights, chronological (see route for why)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    played_at: datetime
    venue: str | None
    format: str
    video_id: uuid.UUID | None
    video_status: VideoStatus | None
    highlights: list[HighlightItem]


class StatisticItem(BaseModel):
    """
    One row of GET /matches/{match_id}/statistics — one per match-level
    `Statistic` DB row (player_id IS NULL). Only ever one of the four
    StatTypes Part 7f actually persists today (see the route's own
    docstring for exactly which, and why the other eight aren't here).
    """

    model_config = ConfigDict(from_attributes=True)

    stat_type: StatType
    value: float


class MatchStatisticsResponse(BaseModel):
    """
    GET /matches/{match_id}/statistics — Part 12d.

    `statistics` holds whatever match-level rows Part 7f has actually
    persisted for this match (empty if `analyze` hasn't reached that
    sub-stage yet, e.g. video_status is still queued/processing).

    `player_statistics_pending_reason` is a fixed, non-null string, not
    conditioned on this match's own data: every player-level StatType is
    blocked on the same still-unresolved track_id -> Player.id identity
    mapping (Part 9b), for every match, regardless of what did or didn't
    get computed here. Sending it as one static sentence — rather than
    the frontend inferring "stats are incomplete" from an empty list, or
    this endpoint omitting the field and leaving that gap unexplained —
    is what lets the UI say so plainly instead of just quietly not
    showing fields a viewer might otherwise expect and wonder about.
    """

    model_config = ConfigDict(from_attributes=True)

    match_id: uuid.UUID
    statistics: list[StatisticItem]
    player_statistics_pending_reason: str


class TrackStatValue(BaseModel):
    """
    One (stat_type, value) pair for a single tracked player within
    PlayerMovementStatsResponse — the same StatType vocabulary
    Statistic rows use, just not yet attached to a real Player.id (see
    PlayerMovementStatsResponse's own docstring). `sample_size` mirrors
    ml.pipeline.stats_aggregation.StatValue's own field: how many
    underlying observations (rallies, shots, frame-pairs) produced
    `value`, so the frontend can show a low-confidence badge on a rate
    computed from only 1-2 attempts instead of presenting every number
    with equal weight.
    """

    model_config = ConfigDict(from_attributes=True)

    stat_type: StatType
    value: float
    sample_size: int


class TrackMovementStats(BaseModel):
    """
    Every computed stat for one ByteTrack track_id, plus whatever 9b's
    court-side grouping could tell about it. `court_side` /
    `side_confidence` are None when no court calibration exists for the
    video (ml.pipeline.player_identity.assign_court_sides requires one —
    see that module's own docstring) — a track can still have movement
    stats without a resolved side in that case, so the two are surfaced
    independently rather than one implying the other.
    """

    model_config = ConfigDict(from_attributes=True)

    track_id: int
    court_side: str | None
    side_confidence: float | None
    stats: list[TrackStatValue]


class PlayerMovementStatsResponse(BaseModel):
    """
    GET /matches/{match_id}/player-movement-stats — Part 9g.

    Surfaces exactly what app/services/player_statistics_stage.py (9f)
    already computes and writes to storage (distance/speed/reaction-time/
    smash+net success rate, per ByteTrack track_id, via 9a/9b) but which
    `Statistic` has zero rows for today — persist_player_statistics (9e)
    is only ever called with an empty track_id -> Player.id mapping by
    the automated pipeline, since nothing in this codebase can tell two
    teammates on the same court side apart (see
    ml.pipeline.player_identity.py's own module docstring). Rather than
    make that real, already-computed data invisible until a human-in-
    the-loop identity step exists, this route reads 9f's own JSON
    artifact back and returns it keyed by track_id/court_side directly —
    an honest "these are the two/four players we tracked, not yet their
    names" view, same posture MatchStatisticsResponse's
    player_statistics_pending_reason already takes toward explaining the
    gap rather than hiding it.

    `tracks` is empty (not an error) whenever: the match has no video,
    `analyze` hasn't reached 9f yet for this video, or it reached 9f but
    every stat 9a could compute was skipped (shouldn't happen in
    practice, since REACTION_TIME_AVG needs no calibration, but not
    asserted here). `video_status` lets the frontend tell "still
    processing" apart from "processing finished but nothing to show"
    the same way ReelResponse's own `status`/video_status pairing
    already does.

    `used_court_calibration` is false whenever this video has no court
    calibration — in that case every track's `stats` list is limited to
    REACTION_TIME_AVG (the one 9a StatType that needs no calibration;
    see ml.pipeline.stats_aggregation's own STAT_TYPE_STATUS) and
    `court_side`/`side_confidence` are null for every track, not a
    pixel-based approximation of either — same "no meaningful
    pixel-only fallback for a physical quantity" reasoning
    ml.pipeline.stats_aggregation.compute_distance_and_speed's own
    docstring already gives.
    """

    model_config = ConfigDict(from_attributes=True)

    match_id: uuid.UUID
    video_status: VideoStatus | None
    used_court_calibration: bool
    tracks: list[TrackMovementStats]
    identity_pending_reason: str


class ReelResponse(BaseModel):
    """
    GET /matches/{match_id}/reel — Part 12e.

    `status` is null when no `Reel` row exists yet for this match at
    all — not a placeholder ReelStatus value, since PENDING/GENERATING/
    READY/FAILED are all real states app/services/reel_generation_stage.py's
    run_reel_generation writes to an actual row, and none of them
    honestly describes "the pipeline hasn't reached Part 10 for this
    match yet".

    Because run_reel_generation runs synchronously inside the `done`
    stage, and only that stage's own success flips Video.status to DONE
    (app/workers/tasks.py), a match whose video_status IS "done" will
    never observe status=GENERATING here — reel generation has already
    fully resolved to READY / FAILED / PENDING (PENDING meaning a real,
    empty, zero-clip reel — see app/services/reel_persistence_stage.py's
    own docstring on why that's not an error) by the time DONE is
    visible. GENERATING is only reachable, in principle, for a match
    whose video_status is still "processing" and whose `done` stage
    happens to be mid-assembly at the exact moment of a request — a real
    but narrow window, not something this route treats specially.

    `reel_url` is null whenever `status` isn't READY — either nothing's
    been assembled yet, or nothing ever will be for an empty or failed
    reel.

    `clip_count` is the number of ReelHighlight rows regardless of
    status — 0 for a genuinely empty reel (PENDING, `reel_max_clips=0`
    or no cuttable highlights) same as for "no Reel row yet" (status
    null). The frontend tells those two cases apart via `status`, not
    via this count, which is ambiguous between them on its own.
    """

    model_config = ConfigDict(from_attributes=True)

    match_id: uuid.UUID
    status: ReelStatus | None
    reel_url: str | None
    clip_count: int


