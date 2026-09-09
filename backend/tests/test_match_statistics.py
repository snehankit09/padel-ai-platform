"""
End-to-end test for GET /matches/{match_id}/statistics — Part 12d.

Same real-stack convention as test_match_detail.py: builds the match via
the actual upload route, then seeds match-level Statistic rows via
get_sync_db() the same way app/services/analyze_persistence_stage.py
(Part 7f) actually would. Requires the docker-compose stack running
locally; written but not executed in this sandbox. Run it yourself with:

    docker compose up -d db
    alembic upgrade head
    uvicorn app.main:app &
    pytest backend/tests/test_match_statistics.py -v
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_sync_db
from app.main import app
from app.models.enums import StatType
from app.models.player import Player
from app.models.statistic import Statistic

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


def _upload_match(sample_video_path: Path) -> str:
    with open(sample_video_path, "rb") as f:
        response = client.post(
            "/videos/upload",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"venue": "Court 3", "format": "singles"},
        )
    assert response.status_code == 201, response.text
    return response.json()["match_id"]


def test_statistics_returns_only_persisted_match_level_rows(sample_video_path: Path):
    match_id = _upload_match(sample_video_path)

    with get_sync_db() as db:
        db.add(Statistic(match_id=match_id, player_id=None, stat_type=StatType.TOTAL_POINTS, value=42))
        db.add(Statistic(match_id=match_id, player_id=None, stat_type=StatType.LONGEST_RALLY, value=18.5))
        db.commit()

    response = client.get(f"/matches/{match_id}/statistics")
    assert response.status_code == 200
    body = response.json()

    assert body["match_id"] == match_id
    stat_types = {s["stat_type"] for s in body["statistics"]}
    assert stat_types == {"total_points", "longest_rally"}
    assert body["player_statistics_pending_reason"]  # always a non-empty explanation


def test_statistics_for_match_with_none_persisted_yet_is_empty_not_error(sample_video_path: Path):
    match_id = _upload_match(sample_video_path)

    response = client.get(f"/matches/{match_id}/statistics")
    assert response.status_code == 200
    assert response.json()["statistics"] == []


def test_statistics_excludes_player_level_rows(sample_video_path: Path):
    """
    No player-level rows exist anywhere yet pipeline-wide (Part 9b), but
    this proves the query's own player_id IS NULL filter, not just the
    absence of such rows in practice — a future Part 9b landing shouldn't
    silently start leaking player rows into this match-level endpoint.
    """
    match_id = _upload_match(sample_video_path)

    with get_sync_db() as db:
        # A real Player row (Statistic.player_id has a FK to players.id) —
        # this test only cares that a player-scoped row is excluded, not
        # about the player's own identity.
        player = Player(full_name="Not Yet Identified")
        db.add(player)
        db.flush()
        db.add(
            Statistic(
                match_id=match_id,
                player_id=player.id,
                stat_type=StatType.DISTANCE_COVERED,
                value=120.0,
            )
        )
        db.add(Statistic(match_id=match_id, player_id=None, stat_type=StatType.ERRORS, value=3))
        db.commit()

    response = client.get(f"/matches/{match_id}/statistics")
    stat_types = {s["stat_type"] for s in response.json()["statistics"]}
    assert stat_types == {"errors"}


def test_statistics_for_unknown_match_returns_404():
    response = client.get("/matches/00000000-0000-0000-0000-000000000000/statistics")
    assert response.status_code == 404
