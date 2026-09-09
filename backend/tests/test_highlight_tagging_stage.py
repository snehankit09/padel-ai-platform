"""
Tests for app/services/highlight_tagging_stage.py — Part 7e.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_rally_detection_stage.py — real ml/pipeline/highlight_tagging.py
logic, and a real LocalStorageService pointed at a temp directory, no
mocking of the tagging logic itself, only the storage/DB *location*.

Run with: pytest backend/tests/test_highlight_tagging_stage.py -v
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
from app.services.highlight_tagging_stage import (
    HighlightTaggingStageError,
    run_highlight_tagging,
)
from app.services.storage import LocalStorageService

ensure_ml_importable()


class _FakeSettings:
    local_storage_path: str
    shot_smash_height_ratio: float = 1.05
    highlight_long_rally_min_duration_s: float = 15.0
    highlight_long_rally_score_saturation_s: float = 35.0
    highlight_fast_exchange_max_interval_s: float = 1.0
    highlight_fast_exchange_min_shot_count: int = 4
    highlight_fast_exchange_score_saturation_count: int = 8
    highlight_powerful_smash_score_ceiling_ratio: float = 1.6
    highlight_spectacular_save_max_response_s: float = 0.8


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Same seam as test_rally_detection_stage.py's storage_env fixture."""
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.highlight_tagging_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _rally_dict(rally_index: int, start_frame: int, end_frame: int, *, sample_fps: float = 5.0) -> dict:
    frame_count = end_frame - start_frame + 1
    return {
        "rally_index": rally_index, "start_frame": start_frame, "end_frame": end_frame,
        "start_time_s": start_frame / sample_fps, "end_time_s": (end_frame + 1) / sample_fps,
        "duration_s": frame_count / sample_fps, "frame_count": frame_count,
        "ball_active_frame_count": frame_count, "ball_observed_frame_count": frame_count,
        "mean_ball_confidence": 0.6,
    }


def _shot_dict(
    rally_index: int, frame_index: int, *, shot_type: str = "groundstroke",
    player_track_id: int | None = 1, contact_height_ratio: float | None = None,
    airborne_frames_after: int | None = 2,
) -> dict:
    return {
        "rally_index": rally_index, "frame_index": frame_index, "player_track_id": player_track_id,
        "shot_type": shot_type, "contact_height_ratio": contact_height_ratio,
        "airborne_frames_after": airborne_frames_after, "distance_from_net_m": None,
        "used_court_calibration": False,
    }


def _seed_video_with_match(session_factory) -> str:
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
        return str(video.id)


def test_tags_and_persists_highlights_end_to_end(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)

    _write_json(storage_env, "rallies/x/rallies.json", {"rallies": [_rally_dict(1, 0, 99)]})  # 20s rally
    _write_json(
        storage_env,
        "shots/x/shots.json",
        {
            "shots": [
                _shot_dict(1, 10, shot_type="smash", contact_height_ratio=1.3),
                _shot_dict(1, 12, player_track_id=2),
            ]
        },
    )

    payload = {
        "video_id": video_id,
        "rally_segments_path": "rallies/x/rallies.json",
        "shots_path": "shots/x/shots.json",
        "frame_sample_fps": 5.0,
    }
    run_highlight_tagging(payload)

    assert payload["highlight_count"] >= 2  # at least long_rally + powerful_smash
    with open(storage_env.get_local_path(payload["highlights_path"])) as f:
        written = json.load(f)
    assert written["highlight_count"] == payload["highlight_count"]
    types = {h["highlight_type"] for h in written["highlights"]}
    assert "long_rally" in types
    assert "powerful_smash" in types
    assert "spectacular_save" in types
    assert written["frame_sample_fps"] == 5.0


def test_no_qualifying_events_still_writes_an_empty_result(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)

    _write_json(storage_env, "rallies/y/rallies.json", {"rallies": [_rally_dict(1, 0, 9)]})  # 2s rally
    _write_json(storage_env, "shots/y/shots.json", {"shots": []})

    payload = {
        "video_id": video_id,
        "rally_segments_path": "rallies/y/rallies.json",
        "shots_path": "shots/y/shots.json",
        "frame_sample_fps": 5.0,
    }
    run_highlight_tagging(payload)

    assert payload["highlight_count"] == 0
    with open(storage_env.get_local_path(payload["highlights_path"])) as f:
        written = json.load(f)
    assert written["highlights"] == []


def test_missing_video_id_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": "not-a-uuid"}
    run_highlight_tagging(payload)  # must not raise
    assert "highlights_path" not in payload


def test_video_not_found_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": str(uuid.uuid4())}
    run_highlight_tagging(payload)  # must not raise
    assert "highlights_path" not in payload


def test_missing_rally_segments_path_raises(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    payload = {"video_id": video_id, "shots_path": "shots/x/shots.json", "frame_sample_fps": 5.0}
    with pytest.raises(HighlightTaggingStageError, match="rally_segments_path"):
        run_highlight_tagging(payload)


def test_missing_shots_path_raises(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    payload = {"video_id": video_id, "rally_segments_path": "rallies/x/rallies.json", "frame_sample_fps": 5.0}
    with pytest.raises(HighlightTaggingStageError, match="shots_path"):
        run_highlight_tagging(payload)


def test_missing_frame_sample_fps_raises(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    payload = {
        "video_id": video_id,
        "rally_segments_path": "rallies/x/rallies.json",
        "shots_path": "shots/x/shots.json",
    }
    with pytest.raises(HighlightTaggingStageError, match="frame_sample_fps"):
        run_highlight_tagging(payload)
