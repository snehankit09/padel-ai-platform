"""
Tests for app/services/rally_detection_stage.py — Part 7a.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_ball_tracking_stage.py — real ml/pipeline/rally_detection.py logic,
and a real LocalStorageService pointed at a temp directory, no mocking of
the detection logic itself, only the storage/DB *location*.

Run with: pytest backend/tests/test_rally_detection_stage.py -v
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
from app.services.rally_detection_stage import RallyDetectionStageError, run_rally_detection
from app.services.storage import LocalStorageService

ensure_ml_importable()


class _FakeSettings:
    local_storage_path: str
    rally_activity_gap_tolerance_frames: int = 2
    rally_min_duration_frames: int = 3


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """
    Points app.services.rally_detection_stage's get_storage_service and
    get_settings at a real LocalStorageService rooted in a temp dir — same
    seam as test_ball_tracking_stage.py's storage_env fixture.
    """
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.rally_detection_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_ball_tracks(storage: LocalStorageService, key: str, frames: list[dict]) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"frames": frames}, f)


def _frame_with_ball(observed: bool = True) -> dict:
    return {
        "frame_path": "f.jpg",
        "tracks": [
            {
                "track_id": 1, "class_id": 32, "class_name": "sports ball", "confidence": 0.6,
                "bbox": {"x1": 0, "y1": 0, "x2": 8, "y2": 8},
                "age": 1, "hits": 1, "time_since_update": 0 if observed else 3,
            }
        ],
    }


def _frame_without_ball() -> dict:
    return {"frame_path": "f.jpg", "tracks": []}


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


def test_two_rallies_separated_by_dead_time_end_to_end(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    # 5 frames of ball activity, 6 frames of nothing (well past the 2-frame
    # tolerance), 5 more frames of ball activity.
    frames = (
        [_frame_with_ball() for _ in range(5)]
        + [_frame_without_ball() for _ in range(6)]
        + [_frame_with_ball() for _ in range(5)]
    )
    _write_ball_tracks(storage_env, "tracks/x/ball_tracks.json", frames)

    payload = {"video_id": video_id, "ball_tracks_path": "tracks/x/ball_tracks.json", "frame_sample_fps": 5.0}
    run_rally_detection(payload)

    assert payload["rally_count"] == 2
    with open(storage_env.get_local_path(payload["rally_segments_path"])) as f:
        written = json.load(f)
    assert written["rally_count"] == 2
    assert len(written["rallies"]) == 2
    assert written["rallies"][0]["start_frame"] == 0
    assert written["rallies"][0]["end_frame"] == 4
    assert written["rallies"][1]["start_frame"] == 11
    assert written["rallies"][1]["end_frame"] == 15
    assert written["frame_sample_fps"] == 5.0


def test_short_gap_within_tolerance_keeps_one_rally(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    # 2-frame gap, tolerance is 2 -> bridged into a single rally.
    frames = (
        [_frame_with_ball() for _ in range(5)]
        + [_frame_without_ball() for _ in range(2)]
        + [_frame_with_ball() for _ in range(5)]
    )
    _write_ball_tracks(storage_env, "tracks/x/ball_tracks.json", frames)

    payload = {"video_id": video_id, "ball_tracks_path": "tracks/x/ball_tracks.json", "frame_sample_fps": 5.0}
    run_rally_detection(payload)

    assert payload["rally_count"] == 1


def test_no_ball_activity_at_all_produces_zero_rallies_and_does_not_raise(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    frames = [_frame_without_ball() for _ in range(10)]
    _write_ball_tracks(storage_env, "tracks/x/ball_tracks.json", frames)

    payload = {"video_id": video_id, "ball_tracks_path": "tracks/x/ball_tracks.json", "frame_sample_fps": 5.0}
    run_rally_detection(payload)  # must not raise

    assert payload["rally_count"] == 0


def test_missing_ball_tracks_path_raises_stage_error(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    with pytest.raises(RallyDetectionStageError, match="ball_tracks_path"):
        run_rally_detection({"video_id": video_id, "frame_sample_fps": 5.0})


def test_missing_frame_sample_fps_raises_stage_error(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    with pytest.raises(RallyDetectionStageError, match="frame_sample_fps"):
        run_rally_detection({"video_id": video_id, "ball_tracks_path": "tracks/x/ball_tracks.json"})


def test_unreadable_ball_tracks_file_raises_stage_error(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    payload = {
        "video_id": video_id,
        "ball_tracks_path": "tracks/does-not-exist.json",
        "frame_sample_fps": 5.0,
    }
    with pytest.raises(RallyDetectionStageError):
        run_rally_detection(payload)


def test_unknown_video_id_does_not_raise(sqlite_session, storage_env):
    run_rally_detection(
        {"video_id": str(uuid.uuid4()), "ball_tracks_path": "tracks/x/ball_tracks.json", "frame_sample_fps": 5.0}
    )


def test_malformed_video_id_does_not_raise(sqlite_session, storage_env):
    run_rally_detection(
        {"video_id": "not-a-uuid", "ball_tracks_path": "tracks/x/ball_tracks.json", "frame_sample_fps": 5.0}
    )
