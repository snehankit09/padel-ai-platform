"""
Match routes — Part 12b's list endpoint (frontend/app/matches/page.tsx
reads from it), Part 12c's detail endpoint (frontend/app/matches/[id]/
page.tsx reads from it), Part 12d's statistics endpoint, and Part 12e's
reel endpoint (the same page's stats and reel-player sections read from
them, respectively). All live in this one router, same as videos.py
holds both of Part 3's endpoints.
"""

from __future__ import annotations

import json
import os
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.database import get_db
from app.models.enums import StatType
from app.models.match import Match
from app.models.reel import Reel
from app.models.statistic import Statistic
from app.schemas.match import (
    HighlightItem,
    MatchDetailResponse,
    MatchListItem,
    MatchListResponse,
    MatchStatisticsResponse,
    PlayerMovementStatsResponse,
    ReelResponse,
    StatisticItem,
    TrackMovementStats,
    TrackStatValue,
)
from app.services.storage import StorageService, get_storage_service, make_player_statistics_destination_path

router = APIRouter()

# Fixed, match-independent explanation for MatchStatisticsResponse's
# player_statistics_pending_reason — see that field's own docstring for
# why this is one static sentence rather than something derived per
# match. Every player-level StatType (distance_covered, movement_speed_
# avg, reaction_time_avg, net/smash_success_rate) is blocked on the same
# gap: ml.pipeline.stats_aggregation.STAT_TYPE_STATUS marks all of them
# STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY, and
# app/services/player_statistics_persistence_stage.py's own docstring is
# explicit that a real track_id -> Player.id mapping (Part 9b) is the one
# missing piece, not a per-match data quality issue this route could
# instead try to describe more specifically.
_PLAYER_STATISTICS_PENDING_REASON = (
    "Player-level stats (distance covered, movement speed, reaction time, smash/net "
    "success rate) aren't available yet. The pipeline computes them per tracked player "
    "today, but can't yet match a tracked player to a real Player profile — that mapping "
    "is still unresolved (Part 9b)."
)


@router.get("", response_model=MatchListResponse)
async def list_matches(db: AsyncSession = Depends(get_db)) -> MatchListResponse:
    """
    Most-recently-played first — that's the order a user checking on a
    match they just uploaded actually wants, and there's no filtering/
    pagination yet to complicate it (nothing in the PRD's Part 12 scope
    calls for either; add both here, not in the frontend, if that
    changes).

    selectinload(Match.video) avoids the N+1 that `for match in matches:
    match.video` would otherwise trigger one query per match for — a
    second query for all matches' videos, joined in Python by FK, instead
    of one query per row.
    """
    result = await db.execute(
        select(Match).options(selectinload(Match.video)).order_by(Match.played_at.desc())
    )
    matches = result.scalars().all()

    return MatchListResponse(
        matches=[
            MatchListItem(
                id=match.id,
                played_at=match.played_at,
                venue=match.venue,
                format=match.format,
                video_id=match.video.id if match.video else None,
                video_status=match.video.status if match.video else None,
                # No thumbnail-generation stage exists yet — see
                # MatchListItem's own docstring.
                thumbnail_url=None,
            )
            for match in matches
        ]
    )


@router.get("/{match_id}", response_model=MatchDetailResponse)
async def get_match(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> MatchDetailResponse:
    """
    Part 12c: match metadata plus its highlights, for the highlights &
    clips viewer.

    Highlights come back in chronological order (start_time_seconds
    ascending) — the order a viewer scrubbing through the actual match
    would encounter them in — rather than sorted by importance_score.
    HighlightItem's own docstring covers why: that score isn't calibrated
    *across* highlight_types, so a single global ranking by it would imply
    a precision the data doesn't have. A per-type ranking is something a
    future "top highlights" view could still add without this endpoint
    changing shape.

    selectinload for both Match.video and Match.highlights, same N+1
    reasoning as list_matches above — one extra query each instead of one
    per highlight for storage.get_url() below (which itself makes no
    query; it's a pure string operation on Highlight.clip_file_path).
    """
    result = await db.execute(
        select(Match)
        .where(Match.id == match_id)
        .options(selectinload(Match.video), selectinload(Match.highlights))
    )
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    highlights = sorted(match.highlights, key=lambda h: h.start_time_seconds)

    return MatchDetailResponse(
        id=match.id,
        played_at=match.played_at,
        venue=match.venue,
        format=match.format,
        video_id=match.video.id if match.video else None,
        video_status=match.video.status if match.video else None,
        highlights=[
            HighlightItem(
                id=highlight.id,
                event_type=highlight.event_type,
                start_time_seconds=highlight.start_time_seconds,
                end_time_seconds=highlight.end_time_seconds,
                importance_score=highlight.importance_score,
                clip_url=storage.get_url(highlight.clip_file_path) if highlight.clip_file_path else None,
            )
            for highlight in highlights
        ],
    )


@router.get("/{match_id}/statistics", response_model=MatchStatisticsResponse)
async def get_match_statistics(match_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> MatchStatisticsResponse:
    """
    Part 12d: whatever's actually been persisted from Part 9 for this
    match — today that's only the four match-level StatTypes
    app/services/analyze_persistence_stage.py's MATCH_LEVEL_STAT_TYPES
    writes (TOTAL_POINTS, RALLY_LENGTH_AVG, LONGEST_RALLY, ERRORS), all
    with player_id IS NULL. Every player-level StatType (the other eight)
    has no rows anywhere in this table yet, pipeline-wide — Part 9b is
    still unresolved — so this route doesn't query for player_id IS NOT
    NULL rows at all; player_statistics_pending_reason is what tells the
    frontend why, once, rather than the frontend having to infer it from
    an endpoint that just quietly never returns them.

    A match with a valid id but no Statistic rows yet (analyze hasn't
    reached Part 7f, or hasn't run at all) is a normal, non-error state —
    empty `statistics`, not a 404 — since the match itself does exist;
    only a missing match_id is a 404.
    """
    match_exists = await db.execute(select(Match.id).where(Match.id == match_id))
    if match_exists.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    result = await db.execute(
        select(Statistic).where(Statistic.match_id == match_id, Statistic.player_id.is_(None))
    )
    statistics = result.scalars().all()

    return MatchStatisticsResponse(
        match_id=match_id,
        statistics=[StatisticItem(stat_type=stat.stat_type, value=stat.value) for stat in statistics],
        player_statistics_pending_reason=_PLAYER_STATISTICS_PENDING_REASON,
    )


_TRACK_IDENTITY_PENDING_REASON = (
    "These are the players the pipeline actually tracked in the video, not yet matched to your "
    "saved Player profiles — no signal in this codebase can tell two teammates on the same side "
    "of the court apart yet (Part 9b), so track_id/court_side is the most specific label available."
)


@router.get("/{match_id}/player-movement-stats", response_model=PlayerMovementStatsResponse)
async def get_match_player_movement_stats(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> PlayerMovementStatsResponse:
    """
    Part 9g: read back Part 9f's own track_id-keyed JSON artifact
    (app/services/player_statistics_stage.py's `player_stats_path`) and
    return it directly, instead of leaving 9a-9f's already-computed
    numbers invisible until a real track_id -> Player.id mapping exists
    (see PlayerMovementStatsResponse's own docstring for why that gap
    exists and why this route doesn't try to close it).

    Same 404-only-on-missing-match, empty-is-normal posture as
    get_match_statistics and get_match_reel above: a match that exists
    but hasn't reached 9f yet (no video, still processing, or analyze
    hasn't run) gets `tracks=[]` plus whatever `video_status` explains
    why, not an error.
    """
    result = await db.execute(
        select(Match).where(Match.id == match_id).options(selectinload(Match.video))
    )
    match = result.scalar_one_or_none()
    if match is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    video = match.video
    if video is None:
        return PlayerMovementStatsResponse(
            match_id=match_id,
            video_status=None,
            used_court_calibration=False,
            tracks=[],
            identity_pending_reason=_TRACK_IDENTITY_PENDING_REASON,
        )

    stats_key = make_player_statistics_destination_path(video.id)
    stats_path = storage.get_local_path(stats_key)
    if not os.path.exists(stats_path):
        # `analyze` hasn't reached 9f for this video yet (still queued/
        # processing, or it failed before getting there) -- not this
        # route's job to distinguish further, same as get_match_reel's
        # own "no Reel row yet" case above.
        return PlayerMovementStatsResponse(
            match_id=match_id,
            video_status=video.status,
            used_court_calibration=False,
            tracks=[],
            identity_pending_reason=_TRACK_IDENTITY_PENDING_REASON,
        )

    with open(stats_path) as f:
        stats_data = json.load(f)

    side_by_track = {
        assignment["track_id"]: assignment for assignment in stats_data.get("side_assignments", [])
    }
    stats_by_track: dict[int, list[dict]] = {}
    for stat_value in stats_data.get("stat_values", []):
        track_id = stat_value.get("track_id")
        if track_id is None:
            # A match-level StatValue (shouldn't appear in 9f's output,
            # which only ever emits the five track_id-keyed StatTypes —
            # see player_statistics_persistence_stage.py's
            # PLAYER_LEVEL_STAT_TYPES) -- skipped defensively rather than
            # assumed impossible.
            continue
        stats_by_track.setdefault(track_id, []).append(stat_value)

    tracks = [
        TrackMovementStats(
            track_id=track_id,
            court_side=side_by_track.get(track_id, {}).get("court_side"),
            side_confidence=side_by_track.get(track_id, {}).get("side_confidence"),
            stats=[
                TrackStatValue(
                    stat_type=StatType(sv["stat_key"]),
                    value=sv["value"],
                    sample_size=sv["sample_size"],
                )
                for sv in track_stat_values
            ],
        )
        for track_id, track_stat_values in sorted(stats_by_track.items())
    ]

    return PlayerMovementStatsResponse(
        match_id=match_id,
        video_status=video.status,
        used_court_calibration=stats_data.get("used_court_calibration", False),
        tracks=tracks,
        identity_pending_reason=_TRACK_IDENTITY_PENDING_REASON,
    )


@router.get("/{match_id}/reel", response_model=ReelResponse)
async def get_match_reel(
    match_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    storage: StorageService = Depends(get_storage_service),
) -> ReelResponse:
    """
    Part 12e: the match's assembled reel, for the reel player.

    Its own endpoint, same reasoning as /statistics being separate from
    the match detail response (ReelResponse's own docstring / this
    router's module docstring): a Reel is a distinct artifact from
    Highlight rows, not something GET /matches/{id}'s existing
    `highlights` list should grow a nested field for.

    A match with a valid id but no Reel row yet is a normal, non-error
    state — `status=None` (see ReelResponse's own docstring on what that
    represents), not a 404. Only a missing match_id is a 404, same
    posture as get_match_statistics above.

    Ordered by created_at descending and takes the first row rather than
    scalar_one_or_none(): app/services/reel_persistence_stage.py's own
    idempotency section documents that a successful run leaves at most
    one Reel row per match (delete-then-reinsert), so in the steady
    state this is equivalent — but this route doesn't lean on that
    invariant holding under every possible interleaving to avoid a
    500 here specifically, since "which reel is current" has an obvious
    answer (the newest one) even if it briefly didn't.

    selectinload(Reel.reel_highlights) avoids an N+1 for `clip_count` —
    one extra query instead of a second implicit one when
    `.reel_highlights` is accessed below.
    """
    match_exists = await db.execute(select(Match.id).where(Match.id == match_id))
    if match_exists.scalar_one_or_none() is None:
        raise HTTPException(status_code=404, detail="Match not found.")

    result = await db.execute(
        select(Reel)
        .where(Reel.match_id == match_id)
        .options(selectinload(Reel.reel_highlights))
        .order_by(Reel.created_at.desc())
        .limit(1)
    )
    reel = result.scalars().first()

    if reel is None:
        return ReelResponse(match_id=match_id, status=None, reel_url=None, clip_count=0)

    return ReelResponse(
        match_id=match_id,
        status=reel.status,
        reel_url=storage.get_url(reel.file_path) if reel.file_path else None,
        clip_count=len(reel.reel_highlights),
    )
