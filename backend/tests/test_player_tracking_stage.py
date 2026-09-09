"""
Tests for app/services/player_tracking_stage.py — Part 6b.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_pipeline_stages.py / test_pipeline_retry.py (fixtures in conftest.py)
— this is the first test of a detect/track-stage glue module against a
real DB session rather than a fully stubbed one, since Part 6b's job is
specifically about the DB round trip (match.format) and file round trip
(reading detect's detections.json, writing player_tracks.json), not just
the tracking math itself (that's ml/tracking's own test_byte_tracker.py).

Uses the real ByteTracker (Part 6a) and real LocalStorageService, pointed
at a temp directory — no mocking of the tracking logic itself, only the
storage/DB *location*.

Run with: pytest backend/tests/test_player_tracking_stage.py -v
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
from app.services.player_tracking_stage import PlayerTrackingStageError, run_player_tracking
from app.services.storage import LocalStorageService

ensure_ml_importable()


class _FakeSettings:
    local_storage_path: str
    player_track_thresh: float = 0.6
    player_track_match_thresh_low: float = 0.15
    player_track_iou_threshold: float = 0.3
    player_track_max_age: int = 5
    player_track_min_hits: int = 1  # 1 so short test sequences confirm on the first hit


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """
    Points app.services.player_tracking_stage's get_storage_service and
    get_settings at a real LocalStorageService rooted in a temp dir, so
    the stage glue's actual file read/write happens against real disk —
    only the location is test-controlled, not the storage abstraction
    itself.
    """
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.player_tracking_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_fixture_detections(storage: LocalStorageService, key: str, frames: list[dict]) -> None:
    path = storage.get_local_path(key)
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"frames": frames}, f)


def _player_det(x: float, y: float, conf: float = 0.9) -> dict:
    return {
        "class_id": 0, "class_name": "person", "confidence": conf,
        "bbox": {"x1": x, "y1": y, "x2": x + 30, "y2": y + 80},
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


def test_stable_track_id_across_smooth_motion(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, format="doubles")
    frames = [
        {"frame_path": f"f{i}.jpg", "players": [_player_det(i * 10, 100)], "balls": []}
        for i in range(10)
    ]
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_player_tracking(payload)

    assert payload["player_track_count"] == 1
    with open(storage_env.get_local_path(payload["player_tracks_path"])) as f:
        written = json.load(f)
    track_ids = {t["track_id"] for fr in written["frames"] for t in fr["tracks"]}
    assert track_ids == {track_ids.pop()}, "a single smoothly-moving player should keep one stable ID"


def test_track_survives_a_low_confidence_occlusion_dip(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, format="doubles")
    frames = []
    for i in range(8):
        conf = 0.2 if i in (3, 4) else 0.9  # dips below player_track_thresh, still above match_thresh_low
        frames.append({"frame_path": f"f{i}.jpg", "players": [_player_det(i * 10, 100, conf=conf)], "balls": []})
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_player_tracking(payload)

    with open(storage_env.get_local_path(payload["player_tracks_path"])) as f:
        written = json.load(f)
    ids_over_time = [fr["tracks"][0]["track_id"] for fr in written["frames"] if fr["tracks"]]
    assert len(set(ids_over_time)) == 1, f"expected one stable ID through the occlusion dip, got {ids_over_time}"


def test_two_players_get_two_distinct_stable_ids(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, format="doubles")
    frames = [
        {
            "frame_path": f"f{i}.jpg",
            "players": [_player_det(i * 10, 100), _player_det(300 - i * 10, 400)],
            "balls": [],
        }
        for i in range(6)
    ]
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_player_tracking(payload)
    assert payload["player_track_count"] == 2


def test_track_count_mismatch_with_match_format_does_not_raise(sqlite_session, storage_env, caplog):
    """Doubles expects 4 players; only providing 2 should warn, not fail the stage."""
    video_id = _seed_video_with_match(sqlite_session, format="doubles")
    frames = [{"frame_path": f"f{i}.jpg", "players": [_player_det(i * 10, 100)], "balls": []} for i in range(3)]
    _write_fixture_detections(storage_env, "detections/x.json", frames)

    payload = {"video_id": video_id, "detections_path": "detections/x.json"}
    run_player_tracking(payload)  # must not raise
    assert payload["player_track_count"] == 1


def test_missing_detections_path_raises_stage_error(sqlite_session, storage_env):
    video_id = _seed_video_with_match(sqlite_session, format="doubles")
    with pytest.raises(PlayerTrackingStageError, match="detections_path"):
        run_player_tracking({"video_id": video_id})


def test_unknown_video_id_does_not_raise(sqlite_session, storage_env):
    run_player_tracking({"video_id": str(uuid.uuid4()), "detections_path": "detections/x.json"})


def test_malformed_video_id_does_not_raise(sqlite_session, storage_env):
    run_player_tracking({"video_id": "not-a-uuid", "detections_path": "detections/x.json"})
