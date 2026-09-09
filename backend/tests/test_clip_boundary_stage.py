"""
Tests for app/services/clip_boundary_stage.py — Part 8a.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_highlight_tagging_stage.py — real ml/pipeline/clip_boundaries.py
logic, and a real LocalStorageService pointed at a temp directory, no
mocking of the boundary math itself, only the storage/DB *location*.

Run with: pytest backend/tests/test_clip_boundary_stage.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import pytest

from app.core.ml_path import ensure_ml_importable
from app.models.enums import UserRole
from app.models.match import Match
from app.models.user import User
from app.models.video import Video
from app.services.clip_boundary_stage import (
    ClipBoundaryStageError,
    run_clip_boundary_calculation,
)
from app.services.storage import LocalStorageService

ensure_ml_importable()


class _FakeSettings:
    local_storage_path: str
    clip_pre_roll_s: float = 3.0
    clip_post_roll_s: float = 2.0
    clip_min_duration_s: float = 5.0


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Same seam as test_highlight_tagging_stage.py's storage_env fixture."""
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.clip_boundary_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _highlight_event_dict(
    rally_index: int,
    start_time_s: float,
    end_time_s: float,
    *,
    highlight_type: str = "long_rally",
    importance_score: float = 0.5,
    source_frame_index: int | None = None,
) -> dict:
    return {
        "rally_index": rally_index,
        "highlight_type": highlight_type,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "importance_score": importance_score,
        "reason": "test",
        "source_frame_index": source_frame_index,
    }


def _seed_video_with_match(session_factory, *, duration_seconds: float | None = 100.0) -> str:
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(
            match_id=match.id, file_path="uploads/test.mp4", original_filename="test.mp4",
            duration_seconds=duration_seconds,
        )
        s.add(video)
        s.commit()
        return str(video.id)


def test_computes_and_persists_padded_clip_boundaries(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, duration_seconds=100.0)

    _write_json(
        storage_env,
        "highlights/x/highlights.json",
        {"highlights": [_highlight_event_dict(1, 20.0, 30.0), _highlight_event_dict(2, 50.0, 50.0, highlight_type="powerful_smash")]},
    )

    payload = {"video_id": video_id, "highlights_path": "highlights/x/highlights.json"}
    run_clip_boundary_calculation(payload)

    assert payload["clip_boundary_count"] == 2
    with open(storage_env.get_local_path(payload["clip_boundaries_path"])) as f:
        written = json.load(f)

    assert written["clip_count"] == 2
    boundaries = {b["rally_index"]: b for b in written["clip_boundaries"]}
    assert boundaries[1]["start_time_s"] == 17.0
    assert boundaries[1]["end_time_s"] == 32.0
    # instantaneous event still gets padded out to at least the minimum
    assert (boundaries[2]["end_time_s"] - boundaries[2]["start_time_s"]) == 5.0
    assert written["video_duration_s"] == 100.0


def test_no_highlight_events_still_writes_an_empty_result(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, duration_seconds=60.0)
    _write_json(storage_env, "highlights/y/highlights.json", {"highlights": []})

    payload = {"video_id": video_id, "highlights_path": "highlights/y/highlights.json"}
    run_clip_boundary_calculation(payload)

    assert payload["clip_boundary_count"] == 0
    with open(storage_env.get_local_path(payload["clip_boundaries_path"])) as f:
        written = json.load(f)
    assert written["clip_boundaries"] == []


def test_missing_video_id_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": "not-a-uuid"}
    run_clip_boundary_calculation(payload)  # must not raise
    assert "clip_boundaries_path" not in payload


def test_video_not_found_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": str(uuid.uuid4())}
    run_clip_boundary_calculation(payload)  # must not raise
    assert "clip_boundaries_path" not in payload


def test_missing_highlights_path_raises(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, duration_seconds=100.0)
    payload = {"video_id": video_id}
    with pytest.raises(ClipBoundaryStageError, match="highlights_path"):
        run_clip_boundary_calculation(payload)


def test_missing_video_duration_raises(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, duration_seconds=None)
    _write_json(storage_env, "highlights/z/highlights.json", {"highlights": []})
    payload = {"video_id": video_id, "highlights_path": "highlights/z/highlights.json"}
    with pytest.raises(ClipBoundaryStageError, match="duration"):
        run_clip_boundary_calculation(payload)
