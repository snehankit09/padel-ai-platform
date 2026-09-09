"""
Tests for app/services/ball_tracking_stage.py — Part 6c.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_player_tracking_stage.py — real ByteTracker (Part 6a), real
ml/tracking/ball_interpolation.py (Part 6c), and a real LocalStorageService
pointed at a temp directory, no mocking of the tracking/interpolation
logic itself, only the storage/DB *location*.

Run with: pytest backend/tests/test_ball_tracking_stage.py -v
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

import pytest

from app.core.ml_path import ensure_ml_importable
from app.models.enums import UserRole
from app.models.match import Match
from app.models.user import User
from app.models.video import Video
from app.services.ball_tracking_stage import BallTrackingStageError, run_ball_tracking
from app.services.storage import LocalStorageService

ensure_ml_importable()


class _FakeSettings:
    local_storage_path: str
    ball_track_thresh: float = 0.35
    ball_track_match_thresh_low: float = 0.05
    ball_track_iou_threshold: float = 0.15
    ball_track_max_age: int = 6
    ball_track_min_hits: int = 1
    ball_track_max_interpolation_gap_frames: int = 4


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """
    Points app.services.ball_tracking_stage's get_storage_service and
    get_settings at a real LocalStorageService rooted in a temp dir — same
    seam as test_player_tracking_stage.py's storage_env fixture.
    """
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.ball_tracking_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_fixture_detections(storage: LocalStorageService, key: str, frames: list[dict]) -> None:
    path = storage.get_local_path(key)
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"frames": frames}, f)


def _ball_det(x: float, y: float, conf: float = 0.5) -> dict:
    # Small step size relative to an 8px box on purpose — see
    # ml/tracking/byte_tracker.py's module docstring on why bbox-IoU
    # association loses identity once per-frame displacement exceeds the
    # object's own box size. These fixtures are testing gap
    # interpolation, not that known limitation, so motion stays inside
    # the range the tracker can actually follow (same range
    # test_byte_tracker.py's own realistic-motion tests use).
    return {
        "class_id": 32, "class_name": "sports ball", "confidence": conf,
        "bbox": {"x1": x, "y1": y, "x2": x + 8, "y2": y + 8},
    }


def _seed_video_with_match(session_factory, *, format: str = "doubles") -> str:
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format=format, uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(match_id=match.id, file_path="uploads/test.mp4", original_filename="test.mp4")
        s.add(video)
        s.commit()
        return str(video.id)


def test_short_occlusion_gap_is_interpolated_end_to_end(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    frames = []
    for i in range(6):
        balls = [] if i in (2, 3) else [_ball_det(i * 3, 100)]
        frames.append({"frame_path": f"f{i}.jpg", "players": [], "balls": balls})
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_ball_tracking(payload)

    assert payload["ball_coverage_rate"] == 1.0, "the occlusion gap should be fully bridged by interpolation"
    with open(storage_env.get_local_path(payload["ball_tracks_path"])) as f:
        written = json.load(f)
    assert written["interpolated_ball_detections"] == 2
    assert written["observed_ball_detections"] == 4
    # gap frames should be flagged as predicted (time_since_update > 0), not silently indistinguishable from real ones
    gap_frame = written["frames"][2]["tracks"][0]
    assert gap_frame["time_since_update"] > 0


def test_ball_leaving_frame_on_a_long_gap_is_not_fabricated(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    frames = []
    for i in range(12):
        # ball missing frames 2 through 9 -> 8-frame gap, well past the default cap
        balls = [] if 2 <= i <= 9 else [_ball_det(i * 3, 100)]
        frames.append({"frame_path": f"f{i}.jpg", "players": [], "balls": balls})
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_ball_tracking(payload)

    with open(storage_env.get_local_path(payload["ball_tracks_path"])) as f:
        written = json.load(f)
    for i in range(2, 10):
        assert written["frames"][i]["tracks"] == [], (
            f"frame {i} is inside a gap far longer than the interpolation cap and must stay empty"
        )
    assert payload["ball_coverage_rate"] < 1.0


def test_no_ball_detections_at_all_does_not_raise(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    frames = [{"frame_path": f"f{i}.jpg", "players": [], "balls": []} for i in range(5)]
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_ball_tracking(payload)  # must not raise

    assert payload["ball_coverage_rate"] == 0.0


def test_missing_detections_path_raises_stage_error(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session)
    with pytest.raises(BallTrackingStageError, match="detections_path"):
        run_ball_tracking({"video_id": video_id})


def test_unknown_video_id_does_not_raise(sqlite_session, storage_env):
    run_ball_tracking({"video_id": str(uuid.uuid4()), "detections_path": "detections/x.json"})


def test_malformed_video_id_does_not_raise(sqlite_session, storage_env):
    run_ball_tracking({"video_id": "not-a-uuid", "detections_path": "detections/x.json"})
