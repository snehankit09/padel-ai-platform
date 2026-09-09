"""
Tests for app/services/player_statistics_persistence_stage.py — Part 9e.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_analyze_persistence_stage.py: a real sqlite_session DB, no mocking
of the persistence logic itself. StatValue objects are built directly
(ensure_ml_importable() first, same as test_stats_aggregation.py) rather
than routed through any Celery task, since no glue stage wires this
module into the pipeline yet (see module docstring).

Run with: pytest backend/tests/test_player_statistics_persistence_stage.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.stats_aggregation import (  # noqa: E402
    STAT_DISTANCE_COVERED,
    STAT_NET_SUCCESS_RATE,
    STAT_REACTION_TIME_AVG,
    STAT_SMASH_SUCCESS_RATE,
    STAT_TOTAL_POINTS,
    StatValue,
)

from app.models.enums import StatType, UserRole
from app.models.match import Match
from app.models.match_player import MatchPlayer
from app.models.player import Player
from app.models.statistic import Statistic
from app.models.user import User
from app.services.player_statistics_persistence_stage import (
    PlayerStatisticsPersistenceError,
    persist_player_statistics,
)


def _seed_match_with_players(session_factory, player_count: int = 2):
    """Match + `player_count` Players, each a MatchPlayer of that match (team_number alternating 1/2)."""
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()

        players = []
        for i in range(player_count):
            player = Player(full_name=f"Player {i}")
            s.add(player)
            s.flush()
            s.add(MatchPlayer(match_id=match.id, player_id=player.id, team_number=(i % 2) + 1))
            players.append(player.id)
        s.commit()
        return match.id, players


def test_persists_one_row_per_mapped_track_id_and_stat_type(sqlite_session):
    match_id, (player_a, player_b) = _seed_match_with_players(sqlite_session)

    stat_values = [
        StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=452.3, sample_size=1),
        StatValue(stat_key=STAT_REACTION_TIME_AVG, track_id=101, value=0.42, sample_size=6),
        StatValue(stat_key=STAT_SMASH_SUCCESS_RATE, track_id=202, value=0.75, sample_size=4),
        StatValue(stat_key=STAT_NET_SUCCESS_RATE, track_id=202, value=0.5, sample_size=2),
    ]
    mapping = {101: player_a, 202: player_b}

    written = persist_player_statistics(match_id, stat_values, mapping)
    assert written == 4

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        assert len(rows) == 4
        by_player = {}
        for row in rows:
            by_player.setdefault(row.player_id, {})[row.stat_type] = row.value

        assert by_player[player_a] == {
            StatType.DISTANCE_COVERED: 452.3,
            StatType.REACTION_TIME_AVG: 0.42,
        }
        assert by_player[player_b] == {
            StatType.SMASH_SUCCESS_RATE: 0.75,
            StatType.NET_SUCCESS_RATE: 0.5,
        }


def test_track_id_with_no_mapping_is_skipped_not_guessed(sqlite_session):
    match_id, (player_a,) = _seed_match_with_players(sqlite_session, player_count=1)

    stat_values = [
        StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=100.0, sample_size=1),
        # track_id 999 has no entry in the mapping below -- must be skipped, not written under a guess.
        StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=999, value=200.0, sample_size=1),
    ]
    mapping = {101: player_a}

    written = persist_player_statistics(match_id, stat_values, mapping)
    assert written == 1

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        assert len(rows) == 1
        assert rows[0].player_id == player_a
        assert rows[0].value == 100.0


def test_match_level_stat_values_are_not_this_stages_rows_to_write(sqlite_session):
    """A track_id=None (match-level) StatValue is Part 7f's job, never this stage's."""
    match_id, (player_a,) = _seed_match_with_players(sqlite_session, player_count=1)

    stat_values = [
        StatValue(stat_key=STAT_TOTAL_POINTS, track_id=None, value=12.0, sample_size=12),
        StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=100.0, sample_size=1),
    ]
    mapping = {101: player_a}

    written = persist_player_statistics(match_id, stat_values, mapping)
    assert written == 1

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        assert len(rows) == 1
        assert rows[0].stat_type == StatType.DISTANCE_COVERED


def test_empty_mapping_writes_nothing_and_is_not_an_error(sqlite_session):
    match_id, _players = _seed_match_with_players(sqlite_session)

    stat_values = [StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=100.0, sample_size=1)]

    written = persist_player_statistics(match_id, stat_values, {})
    assert written == 0

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        assert rows == []


def test_player_id_not_a_match_player_raises(sqlite_session):
    match_id, (player_a,) = _seed_match_with_players(sqlite_session, player_count=1)

    with sqlite_session() as s:
        stranger = Player(full_name="Not In This Match")
        s.add(stranger)
        s.commit()
        stranger_id = stranger.id

    stat_values = [StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=100.0, sample_size=1)]
    mapping = {101: stranger_id}

    with pytest.raises(PlayerStatisticsPersistenceError):
        persist_player_statistics(match_id, stat_values, mapping)

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        assert rows == []  # the whole call refused, not a partial write


def test_retry_does_not_duplicate_rows(sqlite_session):
    match_id, (player_a,) = _seed_match_with_players(sqlite_session, player_count=1)

    stat_values = [StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=100.0, sample_size=1)]
    mapping = {101: player_a}

    persist_player_statistics(match_id, stat_values, mapping)
    persist_player_statistics(match_id, stat_values, mapping)

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        assert len(rows) == 1


def test_rerun_never_touches_match_level_rows_from_part_7f(sqlite_session):
    """This stage's delete-before-reinsert must only ever touch player_id IS NOT NULL rows."""
    match_id, (player_a,) = _seed_match_with_players(sqlite_session, player_count=1)

    with sqlite_session() as s:
        s.add(Statistic(match_id=match_id, player_id=None, stat_type=StatType.TOTAL_POINTS, value=7.0))
        s.commit()

    stat_values = [StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=101, value=100.0, sample_size=1)]
    persist_player_statistics(match_id, stat_values, {101: player_a})

    with sqlite_session() as s:
        rows = s.execute(select(Statistic).where(Statistic.match_id == match_id)).scalars().all()
        stat_types = {row.stat_type for row in rows}
        assert StatType.TOTAL_POINTS in stat_types
        assert StatType.DISTANCE_COVERED in stat_types
        assert len(rows) == 2
