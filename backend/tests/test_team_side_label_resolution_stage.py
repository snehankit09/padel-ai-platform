"""
Tests for app/services/team_side_label_resolution_stage.py — Highlights
Improvement Roadmap Tier 3c.

Same "real logic, throwaway SQLite + LocalStorageService" approach as
test_team_side_labeling_stage.py and test_analyze_persistence_stage.py.

Run with: pytest backend/tests/test_team_side_label_resolution_stage.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import pytest
from sqlalchemy import select

from app.models.enums import HighlightType, UserRole
from app.models.highlight import Highlight
from app.models.match import Match
from app.models.user import User
from app.models.video import Video
from app.services.storage import LocalStorageService
from app.services.team_side_label_resolution_stage import (
    TeamSideLabelResolutionStageError,
    run_team_side_label_resolution,
)

LABELS_KEY = "team_sides/x/team_side_labels.json"


class _FakeSettings:
    local_storage_path: str


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.team_side_label_resolution_stage as stage_module

    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _seed_video_with_highlights(session_factory, highlights: list[dict]) -> tuple[str, str]:
    """highlights: list of {"event_type": HighlightType, "player_track_id": int | None}."""
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(match_id=match.id, file_path="uploads/test.mp4", original_filename="test.mp4")
        s.add(video)
        s.flush()
        for h in highlights:
            s.add(
                Highlight(
                    match_id=match.id,
                    event_type=h["event_type"],
                    start_time_seconds=0.0,
                    end_time_seconds=1.0,
                    importance_score=0.5,
                    player_track_id=h.get("player_track_id"),
                )
            )
        s.commit()
        return str(video.id), str(match.id)


def _base_payload(video_id: str, **overrides) -> dict:
    payload = {"video_id": video_id}
    payload.update(overrides)
    return payload


def test_resolves_attributed_highlights_to_their_label(sqlite_session, storage_env):
    video_id, match_id = _seed_video_with_highlights(
        sqlite_session,
        [
            {"event_type": HighlightType.POWERFUL_SMASH, "player_track_id": 1},
            {"event_type": HighlightType.SPECTACULAR_SAVE, "player_track_id": 2},
            {"event_type": HighlightType.LONG_RALLY, "player_track_id": None},
        ],
    )
    _write_json(
        storage_env, LABELS_KEY,
        {"players": [], "track_labels": {"1": "Player A", "2": "Player B"}, "summary": {}},
    )

    payload = _base_payload(video_id, team_side_labels_path=LABELS_KEY)
    run_team_side_label_resolution(payload)

    assert payload["team_side_labels_resolved_count"] == 2

    with sqlite_session() as s:
        by_type = {
            h.event_type: h.player_label
            for h in s.execute(select(Highlight).where(Highlight.match_id == uuid.UUID(match_id))).scalars()
        }
        assert by_type[HighlightType.POWERFUL_SMASH] == "Player A"
        assert by_type[HighlightType.SPECTACULAR_SAVE] == "Player B"
        assert by_type[HighlightType.LONG_RALLY] is None


def test_track_id_absent_from_lookup_leaves_label_null_not_guessed(sqlite_session, storage_env):
    """Tier 1d may have skipped a track_id entirely -- an honest miss, not a stage failure."""
    video_id, match_id = _seed_video_with_highlights(
        sqlite_session, [{"event_type": HighlightType.POWERFUL_SMASH, "player_track_id": 99}]
    )
    _write_json(storage_env, LABELS_KEY, {"players": [], "track_labels": {"1": "Player A"}, "summary": {}})

    payload = _base_payload(video_id, team_side_labels_path=LABELS_KEY)
    run_team_side_label_resolution(payload)  # must not raise

    assert payload["team_side_labels_resolved_count"] == 0
    with sqlite_session() as s:
        [highlight] = list(s.execute(select(Highlight).where(Highlight.match_id == uuid.UUID(match_id))).scalars())
        assert highlight.player_label is None


def test_no_team_side_labels_path_skips_without_error(sqlite_session, storage_env):
    """Tier 2d's own non-fatal skip (no calibration) propagates forward, not an error."""
    video_id, match_id = _seed_video_with_highlights(
        sqlite_session, [{"event_type": HighlightType.POWERFUL_SMASH, "player_track_id": 1}]
    )

    payload = _base_payload(video_id)  # no team_side_labels_path at all
    run_team_side_label_resolution(payload)  # must not raise

    assert "team_side_labels_resolved_count" not in payload
    with sqlite_session() as s:
        [highlight] = list(s.execute(select(Highlight).where(Highlight.match_id == uuid.UUID(match_id))).scalars())
        assert highlight.player_label is None


def test_malformed_labels_file_raises(sqlite_session, storage_env):
    video_id, _ = _seed_video_with_highlights(
        sqlite_session, [{"event_type": HighlightType.POWERFUL_SMASH, "player_track_id": 1}]
    )
    _write_json(storage_env, LABELS_KEY, {"players": []})  # missing "track_labels"

    payload = _base_payload(video_id, team_side_labels_path=LABELS_KEY)
    with pytest.raises(TeamSideLabelResolutionStageError, match="track_labels"):
        run_team_side_label_resolution(payload)


def test_no_attributed_highlights_resolves_nothing(sqlite_session, storage_env):
    video_id, match_id = _seed_video_with_highlights(
        sqlite_session, [{"event_type": HighlightType.LONG_RALLY, "player_track_id": None}]
    )
    _write_json(storage_env, LABELS_KEY, {"players": [], "track_labels": {"1": "Player A"}, "summary": {}})

    payload = _base_payload(video_id, team_side_labels_path=LABELS_KEY)
    run_team_side_label_resolution(payload)

    assert payload["team_side_labels_resolved_count"] == 0


def test_unknown_video_id_logs_and_returns_without_raising(sqlite_session, storage_env):
    payload = _base_payload(str(uuid.uuid4()), team_side_labels_path=LABELS_KEY)
    run_team_side_label_resolution(payload)  # must not raise
    assert "team_side_labels_resolved_count" not in payload


def test_invalid_video_id_logs_and_returns_without_raising(storage_env):
    payload = _base_payload("not-a-uuid", team_side_labels_path=LABELS_KEY)
    run_team_side_label_resolution(payload)  # must not raise
    assert "team_side_labels_resolved_count" not in payload
