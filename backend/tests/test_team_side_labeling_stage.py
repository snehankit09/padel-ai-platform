"""
Tests for app/services/team_side_labeling_stage.py — Highlights
Improvement Roadmap Tier 2d.

Same "real logic, throwaway SQLite + LocalStorageService instead of
Postgres/S3" approach as test_player_statistics_stage.py: real
ml/pipeline/team_side_positions.py logic (Tiers 1d, 2c, 2d), no mocking
of any of it, only the storage/DB *location*.

Run with: pytest backend/tests/test_team_side_labeling_stage.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest

from app.core.ml_path import ensure_ml_importable
from app.models.enums import UserRole
from app.models.match import Match
from app.models.user import User
from app.models.video import Video
from app.services.storage import LocalStorageService
from app.services.team_side_labeling_stage import (
    TeamSideLabelingStageError,
    run_team_side_labeling,
)

ensure_ml_importable()

PLAYER_TRACKS_KEY = "tracks/x/player_tracks.json"
CALIBRATION_KEY = "courts/x/calibration.json"


class _FakeSettings:
    local_storage_path: str


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.team_side_labeling_stage as stage_module

    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


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


def _identity_homography() -> list[list[float]]:
    """Pixel coords == court-meter coords, same convention test_team_side_positions.py uses."""
    return np.eye(3, dtype=np.float64).tolist()


def _write_calibration(storage: LocalStorageService, key: str) -> None:
    _write_json(storage, key, {"court_length_m": 20.0, "court_width_m": 10.0, "homography": _identity_homography()})


def _track_frame(frame_index: int, entries: list[tuple[int, float, float]]) -> dict:
    """entries: list of (track_id, foot_x, foot_y) present in this frame."""
    return {
        "frame_path": f"/storage/frames/x/frame_{frame_index:06d}.jpg",
        "tracks": [
            {
                "track_id": track_id, "class_id": 0, "class_name": "person", "confidence": 0.9,
                "bbox": {"x1": x, "y1": y, "x2": x, "y2": y},
                "age": 1, "hits": 1, "time_since_update": 0,
            }
            for track_id, x, y in entries
        ],
    }


def _base_payload(video_id: str, **overrides) -> dict:
    payload = {"video_id": video_id, "player_tracks_path": PLAYER_TRACKS_KEY}
    payload.update(overrides)
    return payload


def test_writes_team_side_labels_artifact_for_a_real_doubles_match(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)

    # 4 quadrants, one clean track_id each, well inside each quadrant.
    frames = [
        _track_frame(i, [(1, 2.0, 3.0), (2, 8.0, 3.0), (3, 2.0, 15.0), (4, 8.0, 15.0)]) for i in range(5)
    ]
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": frames})
    _write_calibration(storage_env, CALIBRATION_KEY)

    payload = _base_payload(video_id, court_calibration_path=CALIBRATION_KEY)
    run_team_side_labeling(payload)

    assert payload["team_side_label_count"] == 4
    labels_path = storage_env.get_local_path(payload["team_side_labels_path"])
    with open(labels_path) as f:
        data = json.load(f)

    assert data["track_labels"] == {"1": "Player A", "2": "Player B", "3": "Player C", "4": "Player D"}
    assert data["summary"] == {"track_count": 4, "individual_label_count": 4, "team_fallback_count": 0}
    assert len(data["players"]) == 4
    assert {p["quadrant"] for p in data["players"]} == {"near_left", "near_right", "far_left", "far_right"}


def test_no_calibration_skips_without_error_and_writes_nothing(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": [_track_frame(0, [(1, 2.0, 3.0)])]})

    payload = _base_payload(video_id)  # no court_calibration_path at all
    run_team_side_labeling(payload)  # must not raise

    assert "team_side_labels_path" not in payload
    assert "team_side_label_count" not in payload


def test_missing_player_tracks_path_raises(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)
    payload = {"video_id": video_id, "court_calibration_path": CALIBRATION_KEY}
    with pytest.raises(TeamSideLabelingStageError, match="player_tracks_path"):
        run_team_side_labeling(payload)


def test_unknown_video_id_logs_and_returns_without_raising(sqlite_session, storage_env):
    payload = _base_payload(str(uuid.uuid4()), court_calibration_path=CALIBRATION_KEY)
    run_team_side_labeling(payload)  # must not raise
    assert "team_side_labels_path" not in payload


def test_invalid_video_id_logs_and_returns_without_raising(storage_env):
    payload = _base_payload("not-a-uuid", court_calibration_path=CALIBRATION_KEY)
    run_team_side_labeling(payload)  # must not raise
    assert "team_side_labels_path" not in payload


def test_low_confidence_quadrant_falls_back_to_team_label_in_the_persisted_artifact(sqlite_session, storage_env):
    video_id = _seed_video(sqlite_session)

    # track_id 1: near_left in 1 of 4 real frames, near_right in the other
    # 3 -- so its DOMINANT quadrant is near_right but at confidence 0.75,
    # well above 0.5; flip it closer to even to push confidence below 0.5.
    frames = [
        _track_frame(0, [(1, 8.0, 3.0)]),   # near_right
        _track_frame(1, [(1, 2.0, 3.0)]),   # near_left
        _track_frame(2, [(1, 2.0, 15.0)]),  # far_left
        _track_frame(3, [(1, 8.0, 15.0)]),  # far_right
        _track_frame(4, [(1, 8.0, 3.0)]),   # near_right again -> plurality winner, confidence 2/5 = 0.4
    ]
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": frames})
    _write_calibration(storage_env, CALIBRATION_KEY)

    payload = _base_payload(video_id, court_calibration_path=CALIBRATION_KEY)
    run_team_side_labeling(payload)

    labels_path = storage_env.get_local_path(payload["team_side_labels_path"])
    with open(labels_path) as f:
        data = json.load(f)

    assert data["track_labels"] == {"1": "Near Team"}
    assert data["summary"] == {"track_count": 1, "individual_label_count": 0, "team_fallback_count": 1}
