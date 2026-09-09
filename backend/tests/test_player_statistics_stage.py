"""
Tests for app/services/player_statistics_stage.py — Part 9f.

Same "real logic, throwaway SQLite + LocalStorageService instead of
Postgres/S3" approach as test_shot_classification_stage.py /
test_analyze_persistence_stage.py — real ml/pipeline logic (9a's
stats_aggregation, 9b's player_identity) and real 9e persistence, no
mocking of any of it, only the storage/DB *location*.

Run with: pytest backend/tests/test_player_statistics_stage.py -v
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pytest
from sqlalchemy import select

from app.core.ml_path import ensure_ml_importable
from app.models.enums import StatType, UserRole
from app.models.match import Match
from app.models.statistic import Statistic
from app.models.user import User
from app.models.video import Video
from app.services.player_statistics_stage import (
    PlayerStatisticsStageError,
    run_player_statistics_aggregation,
)
from app.services.storage import LocalStorageService

ensure_ml_importable()

SAMPLE_FPS = 5.0

RALLIES_KEY = "rallies/x/rallies.json"
SHOTS_KEY = "shots/x/shots.json"
OUTCOMES_KEY = "outcomes/x/outcomes.json"
PLAYER_TRACKS_KEY = "tracks/x/player_tracks.json"
CALIBRATION_KEY = "courts/x/calibration.json"


class _FakeSettings:
    local_storage_path: str
    court_length_m: float = 20.0
    court_width_m: float = 10.0


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.player_statistics_stage as stage_module

    monkeypatch.setattr(stage_module, "get_settings", lambda: settings)
    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


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


def _rally_dict(rally_index: int, duration_s: float, *, start_frame: int = 0) -> dict:
    frame_count = max(int(duration_s * SAMPLE_FPS), 1)
    end_frame = start_frame + frame_count - 1
    return {
        "rally_index": rally_index, "start_frame": start_frame, "end_frame": end_frame,
        "start_time_s": start_frame / SAMPLE_FPS, "end_time_s": (end_frame + 1) / SAMPLE_FPS,
        "duration_s": duration_s, "frame_count": frame_count,
        "ball_active_frame_count": frame_count, "ball_observed_frame_count": frame_count,
        "mean_ball_confidence": 0.6,
    }


def _shot_dict(
    rally_index: int, frame_index: int, player_track_id: int | None, shot_type: str,
    *, used_court_calibration: bool = True,
) -> dict:
    return {
        "rally_index": rally_index, "frame_index": frame_index, "player_track_id": player_track_id,
        "shot_type": shot_type, "contact_height_ratio": 1.1 if shot_type == "smash" else 0.9,
        "airborne_frames_after": None, "distance_from_net_m": 2.0,
        "used_court_calibration": used_court_calibration,
    }


def _outcome_dict(rally_index: int, outcome: str, *, used_court_calibration: bool = True) -> dict:
    return {
        "rally_index": rally_index, "outcome": outcome, "reason": "test",
        "last_ball_frame_index": None, "last_ball_x_m": None, "last_ball_y_m": None,
        "speed_ratio": None, "used_court_calibration": used_court_calibration,
    }


def _player_track_frame(entries: list[tuple[int, float, float]]) -> dict:
    """entries: list of (track_id, x, y) present in this frame."""
    return {
        "tracks": [
            {
                "track_id": track_id, "class_id": 0, "class_name": "person", "confidence": 0.9,
                "bbox": {"x1": x - 10, "y1": y - 10, "x2": x + 10, "y2": y + 10},
                "age": 1, "hits": 1, "time_since_update": 0,
            }
            for track_id, x, y in entries
        ]
    }


def _identity_homography() -> list[list[float]]:
    """Scales pixels by 1/20 into meters -- same convention test_shot_classification_stage.py uses."""
    homography = (np.eye(3) / 20.0).tolist()
    homography[2][2] = 1.0
    return homography


def _write_calibration(storage: LocalStorageService, key: str) -> None:
    _write_json(storage, key, {
        "court_length_m": 20.0, "court_width_m": 10.0, "homography": _identity_homography(),
    })


def _base_payload(video_id: str, **overrides) -> dict:
    payload = {
        "video_id": video_id,
        "rally_segments_path": RALLIES_KEY,
        "shots_path": SHOTS_KEY,
        "point_outcomes_path": OUTCOMES_KEY,
        "player_tracks_path": PLAYER_TRACKS_KEY,
        "frame_sample_fps": SAMPLE_FPS,
    }
    payload.update(overrides)
    return payload


def test_computes_and_writes_player_stats_without_calibration(sqlite_session, storage_env):
    """No court_calibration_path -> distance/speed/side-assignment are skipped, but reaction time and success rate still compute."""
    video_id, match_id = _seed_video_with_match(sqlite_session)

    rallies = [_rally_dict(1, 4.0), _rally_dict(2, 4.0, start_frame=50)]
    shots = [
        _shot_dict(1, 0, 1, "groundstroke"),
        _shot_dict(1, 5, 2, "smash"),  # rally 1's last shot -> track 2's smash attempt
        _shot_dict(2, 50, 2, "groundstroke"),
        _shot_dict(2, 55, 1, "volley"),  # rally 2's last shot -> track 1's volley attempt
    ]
    outcomes = [_outcome_dict(1, "out_of_bounds"), _outcome_dict(2, "in_bounds_end")]
    player_frames = [_player_track_frame([(1, 100, 150), (2, 300, 150)]) for _ in range(10)]

    _write_json(storage_env, RALLIES_KEY, {"rallies": rallies})
    _write_json(storage_env, SHOTS_KEY, {"shots": shots})
    _write_json(storage_env, OUTCOMES_KEY, {"outcomes": outcomes})
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": player_frames})

    payload = _base_payload(video_id)
    run_player_statistics_aggregation(payload)

    assert payload["player_stat_value_count"] > 0
    # No identity mapping exists yet for an automated run -- 0 rows, not an error.
    assert payload["player_statistics_persisted_count"] == 0

    with open(storage_env.get_local_path(payload["player_stats_path"])) as f:
        written = json.load(f)

    assert written["used_court_calibration"] is False
    stat_keys = {v["stat_key"] for v in written["stat_values"]}
    # Distance/speed need calibration -> absent, not zeroed.
    assert "distance_covered" not in stat_keys
    assert "movement_speed_avg" not in stat_keys
    # Reaction time and rally-ending shot success don't need calibration -> present.
    assert "reaction_time_avg" in stat_keys
    assert "smash_success_rate" in stat_keys
    assert "net_success_rate" in stat_keys
    # No calibration -> no side assignments either (assign_court_sides needs a real converter).
    assert written["side_assignments"] == []

    # No Statistic rows written -- an honest empty mapping, not a fabricated guess.
    with sqlite_session() as s:
        stats = list(s.execute(select(Statistic).where(Statistic.match_id == uuid.UUID(match_id))).scalars())
        assert stats == []


def test_computes_distance_speed_and_side_assignments_with_calibration(sqlite_session, storage_env):
    video_id, match_id = _seed_video_with_match(sqlite_session)

    rallies = [_rally_dict(1, 4.0)]
    shots = [_shot_dict(1, 0, 1, "groundstroke")]
    outcomes = [_outcome_dict(1, "in_bounds_end")]
    # Track 1 stays on the near side (small y), moves 100px between frames;
    # track 2 stays on the far side (large y) -- both should get real
    # distance/speed and a confident side assignment.
    player_frames = []
    for i in range(6):
        player_frames.append(_player_track_frame([(1, 100 + i * 20, 100), (2, 300, 350)]))
    _write_json(storage_env, RALLIES_KEY, {"rallies": rallies})
    _write_json(storage_env, SHOTS_KEY, {"shots": shots})
    _write_json(storage_env, OUTCOMES_KEY, {"outcomes": outcomes})
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": player_frames})
    _write_calibration(storage_env, CALIBRATION_KEY)

    payload = _base_payload(video_id, court_calibration_path=CALIBRATION_KEY)
    run_player_statistics_aggregation(payload)

    with open(storage_env.get_local_path(payload["player_stats_path"])) as f:
        written = json.load(f)

    assert written["used_court_calibration"] is True
    distance_values = {v["track_id"]: v["value"] for v in written["stat_values"] if v["stat_key"] == "distance_covered"}
    # Track 1 moves 20px/frame in x across 5 gaps = 100px total, /20 (homography scale) = 5m.
    assert distance_values[1] == pytest.approx(5.0, rel=1e-6)
    # Track 2 never moved -> 0m.
    assert distance_values[2] == pytest.approx(0.0, abs=1e-9)

    sides = {a["track_id"]: a["court_side"] for a in written["side_assignments"]}
    assert sides[1] == "side_a"  # y=100/20=5m < court_length_m/2=10m
    assert sides[2] == "side_b"  # y=350/20=17.5m >= 10m
    assert written["side_assignment_summary"]["track_count"] == 2

    # Still honest zero rows -- no track_id -> Player.id mapping exists yet.
    assert payload["player_statistics_persisted_count"] == 0
    with sqlite_session() as s:
        stats = list(s.execute(select(Statistic).where(Statistic.match_id == uuid.UUID(match_id))).scalars())
        assert stats == []


def test_retry_is_idempotent(sqlite_session, storage_env):
    """Running twice must not accumulate rows or blow up -- same idempotency contract as every other analyze sub-stage."""
    video_id, match_id = _seed_video_with_match(sqlite_session)

    rallies = [_rally_dict(1, 4.0)]
    shots = [_shot_dict(1, 0, 1, "groundstroke")]
    outcomes = [_outcome_dict(1, "in_bounds_end")]
    player_frames = [_player_track_frame([(1, 100, 150)]) for _ in range(4)]
    _write_json(storage_env, RALLIES_KEY, {"rallies": rallies})
    _write_json(storage_env, SHOTS_KEY, {"shots": shots})
    _write_json(storage_env, OUTCOMES_KEY, {"outcomes": outcomes})
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": player_frames})

    payload = _base_payload(video_id)
    run_player_statistics_aggregation(payload)
    run_player_statistics_aggregation(payload)

    with sqlite_session() as s:
        stats = list(s.execute(select(Statistic).where(Statistic.match_id == uuid.UUID(match_id))).scalars())
        assert stats == []  # still nothing fabricated, and no crash on the second pass


def test_missing_shots_path_raises(sqlite_session, storage_env):
    video_id, _ = _seed_video_with_match(sqlite_session)
    _write_json(storage_env, RALLIES_KEY, {"rallies": [_rally_dict(1, 4.0)]})
    _write_json(storage_env, OUTCOMES_KEY, {"outcomes": []})
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": []})

    payload = _base_payload(video_id)
    del payload["shots_path"]
    with pytest.raises(PlayerStatisticsStageError):
        run_player_statistics_aggregation(payload)


def test_missing_frame_sample_fps_raises(sqlite_session, storage_env):
    video_id, _ = _seed_video_with_match(sqlite_session)
    _write_json(storage_env, RALLIES_KEY, {"rallies": []})
    _write_json(storage_env, SHOTS_KEY, {"shots": []})
    _write_json(storage_env, OUTCOMES_KEY, {"outcomes": []})
    _write_json(storage_env, PLAYER_TRACKS_KEY, {"frames": []})

    payload = _base_payload(video_id)
    del payload["frame_sample_fps"]
    with pytest.raises(PlayerStatisticsStageError):
        run_player_statistics_aggregation(payload)


def test_missing_video_id_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": "not-a-uuid"}
    run_player_statistics_aggregation(payload)  # must not raise
    assert "player_stats_path" not in payload
