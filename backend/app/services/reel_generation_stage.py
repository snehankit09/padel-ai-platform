"""
Reel generation — Part 10f, the glue between the Celery `done` stage
(app/workers/tasks.py) and every earlier piece of Part 10. Same role for
10a-10e as app/services/clip_extraction_stage.py has for Part 8's own
pure-logic modules: this is what turns already-persisted `Highlight` rows
(Part 7f, updated with real `clip_file_path`s by Part 8b) into one real,
playable reel file, by calling — in the exact order Part 10's own module
docstrings already describe:

    select_clips_for_reel   (10a, ml/pipeline/reel_selection.py)
      -> order_clips_for_reel + build_reel_timeline
                             (10b, ml/pipeline/reel_ordering.py)
      -> persist_reel        (10e, app/services/reel_persistence_stage.py)
      -> assemble_reel       (10c, ml/common/reel_assembly.py)
      -> UPDATE Reel.file_path/status once the file actually exists

This is the first Part 10 module to import anything from `app` at all —
10a/10b/10c are all pure-logic/pure-ffmpeg with no Postgres/Celery
dependency (see their own module docstrings), and 10e already does the
one narrow, mechanical job of turning an already-built timeline into
rows. This module is what builds that timeline from real data in the
first place, and is the reason 10e's `highlight_id` sanity check (see
that module's docstring) is not paranoia: every `ClipCandidate` built
below carries a real `Highlight.id`.

**Why only `Highlight` rows with a `clip_file_path` already set are
candidates at all.** `ClipCandidate.clip_file_path` is a required `str`,
not `str | None` (ml/pipeline/reel_selection.py) — a reel can only be
built from highlights Part 8b has actually cut a file for. A `Highlight`
row Part 8b hasn't reached yet (still mid-`analyze`, or one of the
per-clip failures app/services/clip_extraction_stage.py's own docstring
describes as real, partial, non-fatal failure) is excluded before
ranking rather than silently scored on some placeholder — same
"cuttable clips only" framing this codebase's own README already gives
10a's job.

**Why an empty candidate list — no cuttable highlights yet, or
`reel_max_clips=0` — still calls `persist_reel` with an empty timeline,
never `assemble_reel`.** 10e's own docstring is explicit that an empty
timeline is a real, empty `Reel` row, not an error (see that module's
docstring for the full "max_clips=0 honestly represented" reasoning).
`assemble_reel` itself refuses an empty `ordered_clip_paths`
(`ReelAssemblyError`) on purpose — concatenating zero files isn't
"generate a reel with nothing in it", it's not calling ffmpeg at all —
so this stage never calls it in that case; the `Reel` row's own
`status` (left at 10e's default, `ReelStatus.PENDING`) and `file_path`
(`NULL`) are the only signal a later `done` run, or a human, needs that
there was genuinely nothing to assemble.

**Why the `Reel` row is persisted (10e) BEFORE ffmpeg runs, at
`ReelStatus.GENERATING`, rather than only writing it once the file
exists.** Same reasoning `Video.current_stage` is written before each
pipeline stage runs (see app/workers/tasks.py's own module docstring on
Part 4c): a caller polling `Reel.status` mid-assembly (a real reel can
take a while to concatenate on a long match) should see `GENERATING`,
not nothing — the row existing IS the signal that Part 10 selection/
ordering already ran and produced a real composition, independent of
whether the FFmpeg step after it has finished yet. 10e's own
idempotency (delete-then-reinsert per match) already makes writing this
row first, then updating it after, safe on retry — a retried `done`
re-derives the same composition and 10e replaces the previous attempt's
row cleanly rather than leaving two.

**Why an ffmpeg failure updates the row to `ReelStatus.FAILED` and THEN
re-raises**, rather than swallowing it the way
app/services/clip_extraction_stage.py swallows a single clip's
extraction failure. That module's per-clip catch-and-continue exists
because one bad highlight out of a dozen shouldn't cost every other
highlight's already-successful clip (see its own docstring). There's no
equivalent partial-success shape here: `assemble_reel` produces exactly
one output file from the whole selected set in one ffmpeg run, so a
failure is total, not partial, and — same as clip_extraction_stage.py's
"every single boundary failing" case — worth `done`'s own retry/backoff
via `ReelGenerationStageError` rather than a silently-incomplete reel
nobody notices. The `FAILED` status write happens first so a `Reel` row
never sits at a stale `GENERATING` forever if every retry keeps failing
and `done` itself is eventually marked `VideoStatus.FAILED`.

**What this does NOT do, and why.** No music track gets attached here —
nothing in this codebase picks a music track today (`Reel.music_track`
stays whatever `persist_reel`'s own default leaves it, `None` — see PRD
Module 4's "swappable without regenerating the reel" framing already
noted in 10e's docstring: a real value is a human/product decision to
wire in later, not one to fabricate here just to make the reel look
more finished than the data supports). A title card (10d,
ml.common.reel_assembly.generate_title_card) DOES get attached, as of
the Reel Insta-Level Roadmap's Tier 1a — see _build_title_card_lines
below for where its text comes from (real Match/Player data, not
fabricated) — prepended as the first entry of ordered_clip_paths, same
"caller decides whether/what to prepend" composability 10d's own
docstring describes, this is just that caller now making that decision.
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.enums import ReelStatus
from app.models.highlight import Highlight
from app.models.match import Match
from app.models.match_player import MatchPlayer
from app.models.reel import Reel
from app.models.video import Video
from app.services.reel_persistence_stage import persist_reel
from app.services.storage import get_storage_service, make_reel_file_destination_path

logger = logging.getLogger(__name__)


class ReelGenerationStageError(Exception):
    """Raised when `done` can't find something an earlier stage should already have produced, or ffmpeg assembly itself fails."""


def _build_title_card_lines(match: Match) -> list[str]:
    """
    Reel Insta-Level Roadmap Tier 1a. Builds the 1-2 lines
    ml.common.reel_assembly.generate_title_card burns onto the reel's
    opening card, from real Match/MatchPlayer/Player data — never
    fabricated placeholder text (see this module's own "What this does
    NOT do" note for why that distinction matters here).

    Line 1 is a "Team A vs Team B" roster line, built from
    match.match_players grouped by team_number (not match.players — the
    association-proxy shortcut loses team_number, exactly the thing this
    line needs). Either side can legitimately be empty (a match created
    without full roster data) — this still produces a sensible line for
    whichever side actually has players, or skips the roster line
    entirely if neither does, rather than rendering "vs" with nothing on
    either side of it.

    Line 2 is venue + date. venue is nullable (Match.venue); played_at is
    NOT NULL, so a date is always available even when nothing else is —
    the one line this function is guaranteed to return.
    """
    by_team: dict[int, list[str]] = defaultdict(list)
    for match_player in match.match_players:
        by_team[match_player.team_number].append(match_player.player.full_name)

    lines: list[str] = []

    team_numbers = sorted(by_team.keys())
    team_strs = [", ".join(by_team[team_number]) for team_number in team_numbers if by_team[team_number]]
    if len(team_strs) >= 2:
        lines.append(f"{team_strs[0]} vs {team_strs[1]}")
    elif len(team_strs) == 1:
        lines.append(team_strs[0])
    # else: no roster data at all for this match -- no roster line, not a blank "vs".

    date_str = match.played_at.strftime("%b %d, %Y")
    lines.append(f"{match.venue} \u00b7 {date_str}" if match.venue else date_str)

    return lines


def run_reel_generation(payload: dict) -> None:
    """
    Real body of the `done` stage's reel-generation half. Reads every
    `clip_file_path`-populated `Highlight` row for the video's match,
    runs 10a selection / 10b ordering+pacing / 10e persistence / 10c
    ffmpeg assembly in that order (see module docstring), and leaves the
    persisted `Reel` row's `file_path`/`status` reflecting the outcome.

    Same not-found handling as every other pipeline sub-stage: an
    invalid/missing video_id logs and returns rather than raising —
    nothing for `done` to retry against a caller-supplied id that was
    never valid to begin with.

    Sets `payload["reel_id"]` (always, once a video/match is resolved)
    and, when a real file was assembled, `payload["reel_highlight_count"]`
    / `payload["reel_duration_s"]` — same "leave a record on the payload"
    convention every earlier `analyze` sub-stage already follows.
    """
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[done] video_id=%r is not a valid UUID; skipping reel generation", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[done] video_id=%s not found; skipping reel generation", video_id)
            return
        match_id = video.match_id

    # Own short, read-only session — Reel Insta-Level Roadmap Tier 1a.
    # Fetched here, once, rather than inside _build_title_card_lines
    # itself, so that function stays pure (takes a Match, returns
    # strings, no DB session of its own to manage) — same "keep the
    # DB-touching code and the pure logic in separate places" split this
    # module already draws around ml.common.reel_assembly and friends.
    with get_sync_db() as db:
        match = db.execute(
            select(Match)
            .where(Match.id == match_id)
            .options(selectinload(Match.match_players).selectinload(MatchPlayer.player))
        ).scalar_one()
        title_card_lines = _build_title_card_lines(match)

    ensure_ml_importable()
    from ml.common.reel_assembly import ReelAssemblyError, assemble_reel, generate_title_card, probe_clip_frame_size
    from ml.pipeline.reel_ordering import build_reel_timeline, order_clips_for_reel
    from ml.pipeline.reel_selection import ClipCandidate, select_clips_for_reel

    settings = get_settings()
    storage = get_storage_service()

    with get_sync_db() as db:
        cuttable_highlights = db.execute(
            select(Highlight).where(Highlight.match_id == match_id, Highlight.clip_file_path.is_not(None))
        ).scalars().all()

    candidates = [
        ClipCandidate(
            highlight_id=row.id,
            event_type=row.event_type.value,
            start_time_s=row.start_time_seconds,
            end_time_s=row.end_time_seconds,
            importance_score=row.importance_score,
            clip_file_path=row.clip_file_path,
        )
        for row in cuttable_highlights
    ]

    selected = select_clips_for_reel(
        candidates,
        max_clips=settings.reel_max_clips,
        target_duration_s=settings.reel_target_duration_s,
    )

    if not selected:
        reel_id = persist_reel(match_id, [])
        payload["reel_id"] = str(reel_id)
        payload["reel_highlight_count"] = 0
        logger.info(
            "[done] video_id=%s match_id=%s: %s -- persisted a real, empty Reel (id=%s), nothing to assemble",
            video_id, match_id,
            "no Highlight rows have a cut clip yet" if not candidates else "reel_max_clips/target_duration_s selected none",
            reel_id,
        )
        return

    ordered = order_clips_for_reel(selected, strategy=settings.reel_ordering_strategy)
    timeline = build_reel_timeline(ordered, transition_gap_s=settings.reel_transition_gap_s)

    # Persisted BEFORE ffmpeg runs, at GENERATING -- see module docstring.
    reel_id = persist_reel(match_id, timeline, status=ReelStatus.GENERATING)
    payload["reel_id"] = str(reel_id)

    # `ordered` and `timeline` are the same length, same order (10b
    # preserves position 1:1 -- see build_reel_timeline's own docstring),
    # so clip i's real file path lives on ordered[i] while its gap lives
    # on timeline[i]; ReelTimelineEntry itself carries no clip_file_path.
    ordered_clip_paths = [storage.get_local_path(candidate.clip_file_path) for candidate in ordered]
    gap_before_s = [entry.gap_before_s for entry in timeline]

    reel_key = make_reel_file_destination_path(video_uuid)
    reel_local_path = storage.get_local_path(reel_key)

    # Reel Insta-Level Roadmap Tier 1a. Sized to the reel's own real
    # clips (probe_clip_frame_size on the first one — same frame size
    # assemble_reel's own gap segments already probe for, see that
    # function's docstring) so the title card doesn't introduce a
    # resolution mismatch into a concat that otherwise assumes every
    # segment already matches. Written next to the reel's own output
    # rather than given a permanent storage key of its own — it's
    # consumed into the final reel by assemble_reel below and never
    # referenced again afterward, so it doesn't need one.
    title_card_width, title_card_height = probe_clip_frame_size(ordered_clip_paths[0])
    title_card_path = os.path.join(os.path.dirname(reel_local_path), "title_card.mp4")
    generate_title_card(title_card_path, title_card_lines, width=title_card_width, height=title_card_height)

    ordered_clip_paths = [title_card_path, *ordered_clip_paths]
    gap_before_s = [0.0, *gap_before_s]

    logger.info(
        "[done] video_id=%s match_id=%s: assembling Reel id=%s from %d clip(s) -> %s",
        video_id, match_id, reel_id, len(ordered_clip_paths), reel_local_path,
    )

    try:
        result = assemble_reel(ordered_clip_paths, gap_before_s, reel_local_path)
    except ReelAssemblyError as exc:
        with get_sync_db() as db:
            reel = db.get(Reel, reel_id)
            reel.status = ReelStatus.FAILED
            db.commit()
        raise ReelGenerationStageError(
            f"video_id={video_id} match_id={match_id}: ffmpeg failed to assemble Reel id={reel_id} from "
            f"{len(ordered_clip_paths)} clip(s): {exc}"
        ) from exc

    with get_sync_db() as db:
        reel = db.get(Reel, reel_id)
        reel.file_path = reel_key
        reel.status = ReelStatus.READY
        db.commit()

    payload["reel_highlight_count"] = len(ordered)
    payload["reel_duration_s"] = result.total_duration_s

    logger.info(
        "[done] video_id=%s match_id=%s: Reel id=%s ready -- %d clip(s), %.1fs, %d bytes at %s",
        video_id, match_id, reel_id, result.clip_count, result.total_duration_s, result.file_size_bytes, reel_key,
    )
