"""
Tests for app/services/reel_persistence_stage.py — Part 10e.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_player_statistics_persistence_stage.py: a real sqlite_session DB, no
mocking of the persistence logic itself. `ReelTimelineEntry` objects are
built via 10a's `ClipCandidate` + 10b's `build_reel_timeline` (ensure_ml_
importable() first, same as test_reel_ordering.py) with real `Highlight`
rows seeded first, since no glue stage wires this module into the
pipeline yet (see module docstring).

Run with: pytest backend/tests/test_reel_persistence_stage.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.reel_ordering import build_reel_timeline  # noqa: E402
from ml.pipeline.reel_selection import ClipCandidate  # noqa: E402

from app.models.enums import HighlightType, ReelStatus, UserRole
from app.models.highlight import Highlight
from app.models.match import Match
from app.models.reel import Reel
from app.models.reel_highlight import ReelHighlight
from app.models.user import User
from app.services.reel_persistence_stage import ReelPersistenceError, persist_reel


def _seed_match_with_highlights(session_factory, count: int = 3):
    """Match + `count` Highlight rows on it, tight (unpadded) 10s clips 30s apart."""
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()

        highlight_ids = []
        for i in range(count):
            highlight = Highlight(
                match_id=match.id,
                event_type=HighlightType.LONG_RALLY,
                start_time_seconds=float(i * 30),
                end_time_seconds=float(i * 30 + 10),
                importance_score=0.5 + i * 0.1,
            )
            s.add(highlight)
            s.flush()
            highlight_ids.append(highlight.id)
        s.commit()
        return match.id, highlight_ids


def _timeline_for(highlight_ids, *, importance_scores=None):
    """Builds real ReelTimelineEntry objects (10a ClipCandidate -> 10b build_reel_timeline) against real Highlight ids."""
    scores = importance_scores or [0.5 + i * 0.1 for i in range(len(highlight_ids))]
    candidates = [
        ClipCandidate(
            highlight_id=hid,
            event_type="long_rally",
            start_time_s=float(i * 30),
            end_time_s=float(i * 30 + 10),
            importance_score=scores[i],
            clip_file_path=f"clips/{hid}.mp4",
        )
        for i, hid in enumerate(highlight_ids)
    ]
    return build_reel_timeline(candidates)


def test_persists_reel_and_reel_highlights_with_correct_positions(sqlite_session):
    match_id, highlight_ids = _seed_match_with_highlights(sqlite_session)
    timeline = _timeline_for(highlight_ids)

    reel_id = persist_reel(match_id, timeline)

    with sqlite_session() as s:
        reel = s.get(Reel, reel_id)
        assert reel is not None
        assert reel.match_id == match_id
        assert reel.status == ReelStatus.PENDING
        assert reel.file_path is None

        rows = s.execute(
            select(ReelHighlight).where(ReelHighlight.reel_id == reel_id).order_by(ReelHighlight.position)
        ).scalars().all()
        assert [r.highlight_id for r in rows] == highlight_ids
        assert [r.position for r in rows] == [0, 1, 2]


def test_empty_timeline_writes_a_real_empty_reel_not_nothing(sqlite_session):
    match_id, _highlight_ids = _seed_match_with_highlights(sqlite_session)

    reel_id = persist_reel(match_id, [])

    with sqlite_session() as s:
        reel = s.get(Reel, reel_id)
        assert reel is not None
        assert reel.match_id == match_id

        rows = s.execute(select(ReelHighlight).where(ReelHighlight.reel_id == reel_id)).scalars().all()
        assert rows == []


def test_status_and_music_track_are_passed_through(sqlite_session):
    match_id, highlight_ids = _seed_match_with_highlights(sqlite_session, count=1)
    timeline = _timeline_for(highlight_ids)

    reel_id = persist_reel(match_id, timeline, status=ReelStatus.GENERATING, music_track="upbeat_01")

    with sqlite_session() as s:
        reel = s.get(Reel, reel_id)
        assert reel.status == ReelStatus.GENERATING
        assert reel.music_track == "upbeat_01"


def test_highlight_id_not_of_this_match_raises_and_writes_nothing(sqlite_session):
    match_id, highlight_ids = _seed_match_with_highlights(sqlite_session, count=1)
    _other_match_id, other_highlight_ids = _seed_match_with_highlights(sqlite_session, count=1)

    # Timeline built against the WRONG match's highlight id.
    timeline = _timeline_for(other_highlight_ids)

    with pytest.raises(ReelPersistenceError):
        persist_reel(match_id, timeline)

    with sqlite_session() as s:
        reels = s.execute(select(Reel).where(Reel.match_id == match_id)).scalars().all()
        assert reels == []  # the whole call refused, not a partial write


def test_fabricated_non_uuid_highlight_id_raises(sqlite_session):
    match_id, highlight_ids = _seed_match_with_highlights(sqlite_session, count=1)

    candidates = [
        ClipCandidate(
            highlight_id="not-a-real-uuid",
            event_type="long_rally",
            start_time_s=0.0,
            end_time_s=10.0,
            importance_score=0.9,
            clip_file_path="clips/x.mp4",
        )
    ]
    timeline = build_reel_timeline(candidates)

    with pytest.raises(ReelPersistenceError):
        persist_reel(match_id, timeline)


def test_rerun_does_not_duplicate_rows_and_replaces_stale_ordering(sqlite_session):
    match_id, highlight_ids = _seed_match_with_highlights(sqlite_session, count=3)
    first_timeline = _timeline_for(highlight_ids)

    first_reel_id = persist_reel(match_id, first_timeline)

    # Re-run with a different (reordered/trimmed) selection, as a retried
    # upstream stage might produce.
    second_timeline = _timeline_for(list(reversed(highlight_ids[:2])))
    second_reel_id = persist_reel(match_id, second_timeline)

    with sqlite_session() as s:
        reels = s.execute(select(Reel).where(Reel.match_id == match_id)).scalars().all()
        assert len(reels) == 1  # no duplicate Reel row left behind

        rows = s.execute(
            select(ReelHighlight).where(ReelHighlight.reel_id == second_reel_id).order_by(ReelHighlight.position)
        ).scalars().all()
        assert [r.highlight_id for r in rows] == list(reversed(highlight_ids[:2]))

        # The first attempt's ReelHighlight rows are gone, not orphaned.
        if first_reel_id != second_reel_id:
            stale_rows = s.execute(
                select(ReelHighlight).where(ReelHighlight.reel_id == first_reel_id)
            ).scalars().all()
            assert stale_rows == []


def test_reel_highlights_survive_alongside_other_matches_reel(sqlite_session):
    """Re-persisting one match's reel must never touch a different match's Reel/ReelHighlight rows."""
    match_a, highlights_a = _seed_match_with_highlights(sqlite_session, count=2)
    match_b, highlights_b = _seed_match_with_highlights(sqlite_session, count=2)

    persist_reel(match_a, _timeline_for(highlights_a))
    reel_b_id = persist_reel(match_b, _timeline_for(highlights_b))

    # Re-run match_a's persistence again.
    persist_reel(match_a, _timeline_for(highlights_a))

    with sqlite_session() as s:
        reel_b = s.get(Reel, reel_b_id)
        assert reel_b is not None
        assert reel_b.match_id == match_b

        rows_b = s.execute(select(ReelHighlight).where(ReelHighlight.reel_id == reel_b_id)).scalars().all()
        assert {r.highlight_id for r in rows_b} == set(highlights_b)
