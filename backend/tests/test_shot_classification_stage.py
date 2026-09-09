"""
Tests for app/services/shot_classification_stage.py — Part 7c.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_rally_detection_stage.py / test_ball_tracking_stage.py — real
ml/pipeline/shot_classification.py logic and a real LocalStorageService
pointed at a temp directory, no mocking of the classification logic
itself, only the storage/DB *location*.

Run with: pytest backend/tests/test_shot_classification_stage.py -v
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
from app.services.shot_classification_stage import ShotClassificationStageError, run_shot_classification
from app.services.storage import LocalStorageService

ensure_ml_importable()


class _FakeSettings:
    local_storage_path: str
    court_length_m: float = 20.0
    court_width_m: float = 10.0
    shot_max_contact_player_distance_m: float = 2.5
    shot_max_contact_player_distance_px: float = 150.0
    shot_smash_height_ratio: float = 1.05
    shot_lob_min_airborne_frames: int = 6
    shot_net_proximity_m: float = 3.0


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Points the stage module's get_settings/get_storage_service at a real LocalStorageService rooted in a temp dir."""
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.shot_classification_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _ball_track_frame(x, y) -> dict:
    return {"tracks": [{"track_id": 1, "class_id": 32, "class_name": "sports ball", "confidence": 0.5,
                         "bbox": {"x1": x - 4, "y1": y - 4, "x2": x + 4, "y2": y + 4},
                         "age": 1, "hits": 1, "time_since_update": 0}]}


def _player_track_frame(x, y, top, bottom, track_id=1) -> dict:
    return {"tracks": [{"track_id": track_id, "class_id": 0, "class_name": "person", "confidence": 0.9,
                         "bbox": {"x1": x - 10, "y1": top, "x2": x + 10, "y2": bottom},
                         "age": 1, "hits": 1, "time_since_update": 0}]}


def _rally_dict(rally_index, start_frame, end_frame, n_frames) -> dict:
    return {
        "rally_index": rally_index, "start_frame": start_frame, "end_frame": end_frame,
        "start_time_s": start_frame / 5.0, "end_time_s": end_frame / 5.0, "duration_s": n_frames / 5.0,
        "frame_count": n_frames, "ball_active_frame_count": n_frames,
        "ball_observed_frame_count": n_frames, "mean_ball_confidence": 0.6,
    }


def _seed_video(session_factory) -> str:
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


def test_smash_classified_end_to_end_without_calibration(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)
    ys = [200, 150, 100, 70, 110, 160, 220]
    ball_frames = [_ball_track_frame(100, y) for y in ys]
    player_frames = [_player_track_frame(100, 150, 80, 190) for _ in ys]
    _write_json(storage_env, "tracks/ball.json", {"frames": ball_frames})
    _write_json(storage_env, "tracks/player.json", {"frames": player_frames})
    _write_json(storage_env, "rallies/r.json", {"rallies": [_rally_dict(1, 0, len(ys) - 1, len(ys))]})

    payload = {
        "video_id": video_id,
        "ball_tracks_path": "tracks/ball.json",
        "player_tracks_path": "tracks/player.json",
        "rally_segments_path": "rallies/r.json",
    }
    run_shot_classification(payload)

    assert payload["shot_count"] == 1
    with open(storage_env.get_local_path(payload["shots_path"])) as f:
        written = json.load(f)
    assert written["shots"][0]["shot_type"] == "smash"
    assert written["shots"][0]["used_court_calibration"] is False


def test_volley_uses_court_calibration_when_available(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)
    ys = [190, 185, 180, 175, 180, 185, 190]  # near-net contact
    ball_frames = [_ball_track_frame(100, y) for y in ys]
    player_frames = [_player_track_frame(100, 180, 150, 210) for _ in ys]
    _write_json(storage_env, "tracks/ball.json", {"frames": ball_frames})
    _write_json(storage_env, "tracks/player.json", {"frames": player_frames})
    _write_json(storage_env, "rallies/r.json", {"rallies": [_rally_dict(1, 0, len(ys) - 1, len(ys))]})

    # Identity-ish homography: scales pixels by 1/20 into meters (court_length_m=20, so net at y=10m == pixel y=200).
    import numpy as np
    homography = (np.eye(3) / 20.0).tolist()
    homography[2][2] = 1.0
    _write_json(storage_env, "calibration/c.json", {
        "court_length_m": 20.0, "court_width_m": 10.0, "homography": homography,
    })

    payload = {
        "video_id": video_id,
        "ball_tracks_path": "tracks/ball.json",
        "player_tracks_path": "tracks/player.json",
        "rally_segments_path": "rallies/r.json",
        "court_calibration_path": "calibration/c.json",
    }
    run_shot_classification(payload)

    with open(storage_env.get_local_path(payload["shots_path"])) as f:
        written = json.load(f)
    assert written["shots"][0]["shot_type"] == "volley"
    assert written["shots"][0]["used_court_calibration"] is True


def test_missing_court_calibration_path_is_not_an_error(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)
    ys = [200, 190, 180, 170]  # monotonic -- no contact, but must not raise either way
    ball_frames = [_ball_track_frame(100, y) for y in ys]
    player_frames = [_player_track_frame(100, 150, 80, 190) for _ in ys]
    _write_json(storage_env, "tracks/ball.json", {"frames": ball_frames})
    _write_json(storage_env, "tracks/player.json", {"frames": player_frames})
    _write_json(storage_env, "rallies/r.json", {"rallies": [_rally_dict(1, 0, len(ys) - 1, len(ys))]})

    payload = {
        "video_id": video_id,
        "ball_tracks_path": "tracks/ball.json",
        "player_tracks_path": "tracks/player.json",
        "rally_segments_path": "rallies/r.json",
        # no court_calibration_path at all
    }
    run_shot_classification(payload)  # must not raise
    assert payload["shot_count"] == 0


def test_missing_required_path_raises_stage_error(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)
    with pytest.raises(ShotClassificationStageError, match="ball_tracks_path"):
        run_shot_classification({
            "video_id": video_id,
            "rally_segments_path": "rallies/r.json",
            "player_tracks_path": "tracks/player.json",
        })


def test_unknown_video_id_does_not_raise(sqlite_session, storage_env):
    run_shot_classification({
        "video_id": str(uuid.uuid4()),
        "ball_tracks_path": "tracks/ball.json",
        "player_tracks_path": "tracks/player.json",
        "rally_segments_path": "rallies/r.json",
    })


def test_malformed_video_id_does_not_raise(sqlite_session, storage_env):
    run_shot_classification({
        "video_id": "not-a-uuid",
        "ball_tracks_path": "tracks/ball.json",
        "player_tracks_path": "tracks/player.json",
        "rally_segments_path": "rallies/r.json",
    })
