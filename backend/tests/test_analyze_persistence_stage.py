"""
Tests for app/services/analyze_persistence_stage.py — Part 7f.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_highlight_tagging_stage.py: a real LocalStorageService pointed at a
temp directory and a real sqlite_session DB, no mocking of the
persistence logic itself.

Run with: pytest backend/tests/test_analyze_persistence_stage.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.enums import HighlightType, StatType, UserRole
from app.models.highlight import Highlight
from app.models.match import Match
from app.models.statistic import Statistic
from app.models.user import User
from app.models.video import Video
from app.services.analyze_persistence_stage import (
    AnalyzePersistenceStageError,
    run_analyze_persistence,
)
from app.services.storage import LocalStorageService

RALLIES_KEY = "rallies/x/rallies.json"
OUTCOMES_KEY = "outcomes/x/outcomes.json"
HIGHLIGHTS_KEY = "highlights/x/highlights.json"

OUTCOME_IN_BOUNDS_END = "in_bounds_end"
OUTCOME_OUT_OF_BOUNDS = "out_of_bounds"
OUTCOME_NET = "net"

SAMPLE_FPS = 5.0


class _FakeSettings:
    local_storage_path: str


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.analyze_persistence_stage as stage_module

    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _rally_dict(rally_index: int, duration_s: float, *, start_frame: int = 0) -> dict:
    """One RallySegment's fields, exactly as rally_detection_stage.py persists them -- see ml.pipeline.rally_detection.RallySegment."""
    frame_count = max(int(duration_s * SAMPLE_FPS), 1)
    end_frame = start_frame + frame_count - 1
    return {
        "rally_index": rally_index, "start_frame": start_frame, "end_frame": end_frame,
        "start_time_s": start_frame / SAMPLE_FPS, "end_time_s": (end_frame + 1) / SAMPLE_FPS,
        "duration_s": duration_s, "frame_count": frame_count,
        "ball_active_frame_count": frame_count, "ball_observed_frame_count": frame_count,
        "mean_ball_confidence": 0.6,
    }


def _rallies_payload(rallies: list[dict]) -> dict:
    return {"rallies": rallies}


def _outcome_dict(rally_index: int, outcome: str, *, used_court_calibration: bool = True) -> dict:
    """One PointOutcome's fields, exactly as point_outcome_stage.py persists them -- see ml.pipeline.point_outcome.PointOutcome."""
    return {
        "rally_index": rally_index, "outcome": outcome, "reason": "test",
        "last_ball_frame_index": None, "last_ball_x_m": None, "last_ball_y_m": None,
        "speed_ratio": None, "used_court_calibration": used_court_calibration,
    }


def _outcomes_payload(outcomes: list[dict]) -> dict:
    return {"outcomes": outcomes}


def _highlight_event(
    highlight_type: str = "long_rally", start: float = 0.0, end: float = 20.0, score: float = 0.7
) -> dict:
    return {
        "rally_index": 1, "highlight_type": highlight_type,
        "start_time_s": start, "end_time_s": end,
        "importance_score": score, "reason": "test", "source_frame_index": None,
    }


def _seed_video_with_match(session_factory) -> tuple[str, str]:
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(match_id=match.id, file_path="uploads/test.mp4", original_filename="test.mp4")
        s.add(video)
        s.commit()
        return str(video.id), str(match.id)


def test_persists_highlights_and_match_level_statistics(sqlite_session, storage_env):
    video_id, match_id = _seed_video_with_match(sqlite_session)

    rallies = [_rally_dict(1, 10.0), _rally_dict(2, 20.0, start_frame=100), _rally_dict(3, 5.0, start_frame=300)]
    outcomes = [
        _outcome_dict(1, OUTCOME_IN_BOUNDS_END),
        _outcome_dict(2, OUTCOME_OUT_OF_BOUNDS),
        _outcome_dict(3, OUTCOME_NET),
    ]
    _write_json(storage_env, RALLIES_KEY, _rallies_payload(rallies))
    _write_json(storage_env, OUTCOMES_KEY, _outcomes_payload(outcomes))
    _write_json(
        storage_env, HIGHLIGHTS_KEY,
        {"highlights": [_highlight_event("long_rally"), _highlight_event("powerful_smash", score=0.9)]},
    )

    payload = {
        "video_id": video_id, "rally_segments_path": RALLIES_KEY,
        "point_outcomes_path": OUTCOMES_KEY, "highlights_path": HIGHLIGHTS_KEY,
    }
    run_analyze_persistence(payload)

    assert payload["highlights_persisted_count"] == 2
    assert payload["statistics_persisted_count"] == 4

    with sqlite_session() as s:
        highlights = s.execute(select(Highlight).where(Highlight.match_id == uuid.UUID(match_id))).scalars().all()
        assert {h.event_type for h in highlights} == {HighlightType.LONG_RALLY, HighlightType.POWERFUL_SMASH}
        assert all(h.clip_file_path is None for h in highlights)

        stats = {
            s_row.stat_type: s_row.value
            for s_row in s.execute(select(Statistic).where(Statistic.match_id == uuid.UUID(match_id))).scalars()
        }
        assert stats == {
            StatType.TOTAL_POINTS: 3.0,
            StatType.RALLY_LENGTH_AVG: (10.0 + 20.0 + 5.0) / 3,
            StatType.LONGEST_RALLY: 20.0,
            # errors = out_of_bounds + net, NOT in_bounds_end (winner/UE ambiguous) --
            # see ml.pipeline.stats_aggregation's module docstring.
            StatType.ERRORS: 2.0,
        }


def test_zero_rallies_skips_duration_stats_but_still_writes_total_points_and_errors(sqlite_session, storage_env):
    video_id, match_id = _seed_video_with_match(sqlite_session)

    _write_json(storage_env, RALLIES_KEY, _rallies_payload([]))
    _write_json(storage_env, OUTCOMES_KEY, _outcomes_payload([]))
    _write_json(storage_env, HIGHLIGHTS_KEY, {"highlights": []})

    payload = {
        "video_id": video_id, "rally_segments_path": RALLIES_KEY,
        "point_outcomes_path": OUTCOMES_KEY, "highlights_path": HIGHLIGHTS_KEY,
    }
    run_analyze_persistence(payload)

    assert payload["highlights_persisted_count"] == 0
    assert payload["statistics_persisted_count"] == 2

    with sqlite_session() as s:
        stats = {
            s_row.stat_type: s_row.value
            for s_row in s.execute(select(Statistic).where(Statistic.match_id == uuid.UUID(match_id))).scalars()
        }
        assert stats == {StatType.TOTAL_POINTS: 0.0, StatType.ERRORS: 0.0}


def test_retry_does_not_duplicate_rows(sqlite_session, storage_env):
    """Same stage re-run twice (analyze's own retry path) must not double-insert."""
    video_id, match_id = _seed_video_with_match(sqlite_session)

    _write_json(storage_env, RALLIES_KEY, _rallies_payload([_rally_dict(1, 10.0)]))
    _write_json(storage_env, OUTCOMES_KEY, _outcomes_payload([_outcome_dict(1, OUTCOME_NET)]))
    _write_json(storage_env, HIGHLIGHTS_KEY, {"highlights": [_highlight_event("long_rally")]})

    payload = {
        "video_id": video_id, "rally_segments_path": RALLIES_KEY,
        "point_outcomes_path": OUTCOMES_KEY, "highlights_path": HIGHLIGHTS_KEY,
    }
    run_analyze_persistence(payload)
    run_analyze_persistence(payload)

    with sqlite_session() as s:
        highlights = list(s.execute(select(Highlight).where(Highlight.match_id == uuid.UUID(match_id))).scalars())
        stats = list(s.execute(select(Statistic).where(Statistic.match_id == uuid.UUID(match_id))).scalars())
        assert len(highlights) == 1
        assert len(stats) == 4


def test_missing_highlights_path_raises(sqlite_session, storage_env):
    video_id, _ = _seed_video_with_match(sqlite_session)
    _write_json(storage_env, RALLIES_KEY, _rallies_payload([_rally_dict(1, 10.0)]))
    _write_json(storage_env, OUTCOMES_KEY, _outcomes_payload([]))

    payload = {"video_id": video_id, "rally_segments_path": RALLIES_KEY, "point_outcomes_path": OUTCOMES_KEY}
    with pytest.raises(AnalyzePersistenceStageError):
        run_analyze_persistence(payload)


def test_missing_point_outcomes_path_raises(sqlite_session, storage_env):
    video_id, _ = _seed_video_with_match(sqlite_session)
    _write_json(storage_env, RALLIES_KEY, _rallies_payload([_rally_dict(1, 10.0)]))
    _write_json(storage_env, HIGHLIGHTS_KEY, {"highlights": []})

    payload = {"video_id": video_id, "rally_segments_path": RALLIES_KEY, "highlights_path": HIGHLIGHTS_KEY}
    with pytest.raises(AnalyzePersistenceStageError):
        run_analyze_persistence(payload)


def test_missing_video_id_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": "not-a-uuid"}
    run_analyze_persistence(payload)  # must not raise
    assert "highlights_persisted_count" not in payload
