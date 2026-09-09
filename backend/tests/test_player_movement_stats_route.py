"""
End-to-end test for GET /matches/{match_id}/player-movement-stats — Part 9g.

Same real-stack convention as test_match_statistics.py: builds the match
via the actual upload route, then writes the 9f JSON artifact directly to
storage (rather than running the whole ML pipeline) so this test only
exercises the route's own read-back-and-reshape logic. Requires the
docker-compose stack running locally; written but not executed in this
sandbox. Run it yourself with:

    docker compose up -d db
    alembic upgrade head
    uvicorn app.main:app &
    pytest backend/tests/test_player_movement_stats_route.py -v
"""

import json
import subprocess
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.storage import get_storage_service, make_player_statistics_destination_path

client = TestClient(app)


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """A real, tiny, valid mp4 generated with ffmpeg — not a mocked file."""
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
            "-loglevel", "error",
        ],
        check=True,
    )
    return video_path


def _upload_match(sample_video_path: Path) -> tuple[str, str]:
    with open(sample_video_path, "rb") as f:
        response = client.post(
            "/videos/upload",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"venue": "Court 3", "format": "doubles"},
        )
    assert response.status_code == 201, response.text
    body = response.json()
    return body["match_id"], body["video_id"]


def _write_player_stats_artifact(video_id: str) -> None:
    """
    Writes exactly the shape app/services/player_statistics_stage.py (9f)
    itself writes — see that module's `_write_json` call — so this test
    exercises the route against a realistic artifact without running
    detection/tracking/analyze for real.
    """
    storage = get_storage_service()
    key = make_player_statistics_destination_path(uuid.UUID(video_id))
    path = storage.get_local_path(key)

    data = {
        "frame_sample_fps": 5,
        "used_court_calibration": True,
        "stat_values": [
            {"stat_key": "distance_covered", "track_id": 1, "value": 812.5, "sample_size": 1},
            {"stat_key": "movement_speed_avg", "track_id": 1, "value": 2.1, "sample_size": 1},
            {"stat_key": "reaction_time_avg", "track_id": 1, "value": 0.34, "sample_size": 6},
            {"stat_key": "distance_covered", "track_id": 2, "value": 734.0, "sample_size": 1},
        ],
        "side_assignments": [
            {"track_id": 1, "court_side": "side_a", "frames_observed": 120, "side_confidence": 0.92},
            {"track_id": 2, "court_side": "side_b", "frames_observed": 118, "side_confidence": 0.88},
        ],
        "side_assignment_summary": {
            "track_count": 2, "side_a_count": 1, "side_b_count": 1, "avg_side_confidence": 0.9,
        },
        "diagnostics": {},
    }
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def test_player_movement_stats_returns_track_keyed_data(sample_video_path: Path):
    match_id, video_id = _upload_match(sample_video_path)
    _write_player_stats_artifact(video_id)

    response = client.get(f"/matches/{match_id}/player-movement-stats")
    assert response.status_code == 200
    body = response.json()

    assert body["match_id"] == match_id
    assert body["used_court_calibration"] is True
    assert body["identity_pending_reason"]  # always a non-empty explanation

    tracks_by_id = {t["track_id"]: t for t in body["tracks"]}
    assert set(tracks_by_id) == {1, 2}
    assert tracks_by_id[1]["court_side"] == "side_a"
    assert tracks_by_id[2]["court_side"] == "side_b"
    stat_keys = {s["stat_type"] for s in tracks_by_id[1]["stats"]}
    assert stat_keys == {"distance_covered", "movement_speed_avg", "reaction_time_avg"}


def test_player_movement_stats_for_match_not_yet_analyzed_is_empty_not_error(sample_video_path: Path):
    match_id, _video_id = _upload_match(sample_video_path)

    response = client.get(f"/matches/{match_id}/player-movement-stats")
    assert response.status_code == 200
    body = response.json()
    assert body["tracks"] == []
    assert body["used_court_calibration"] is False
    assert body["identity_pending_reason"]


def test_player_movement_stats_for_unknown_match_returns_404():
    response = client.get("/matches/00000000-0000-0000-0000-000000000000/player-movement-stats")
    assert response.status_code == 404
