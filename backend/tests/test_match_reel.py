"""
End-to-end test for GET /matches/{match_id}/reel — Part 12e.

Same real-stack convention as test_match_statistics.py: builds the match
via the actual upload route, then seeds Reel/ReelHighlight rows via
get_sync_db() the same shape app/services/reel_persistence_stage.py
(Part 10e) actually writes. Requires the docker-compose stack running
locally; written but not executed in this sandbox. Run it yourself with:

    docker compose up -d db
    alembic upgrade head
    uvicorn app.main:app &
    pytest backend/tests/test_match_reel.py -v
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_sync_db
from app.main import app
from app.models.enums import HighlightType, ReelStatus
from app.models.highlight import Highlight
from app.models.reel import Reel
from app.models.reel_highlight import ReelHighlight

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


def test_reel_for_match_with_no_reel_row_yet_is_null_status_not_error(sample_video_path: Path):
    match_id = _upload_match(sample_video_path)

    response = client.get(f"/matches/{match_id}/reel")
    assert response.status_code == 200
    body = response.json()
    assert body["match_id"] == match_id
    assert body["status"] is None
    assert body["reel_url"] is None
    assert body["clip_count"] == 0


def test_reel_ready_returns_a_playable_url(sample_video_path: Path):
    match_id = _upload_match(sample_video_path)

    with get_sync_db() as db:
        highlight = Highlight(
            match_id=match_id,
            event_type=HighlightType.LONG_RALLY,
            start_time_seconds=10.0,
            end_time_seconds=25.0,
            importance_score=0.8,
            clip_file_path=f"clips/{match_id}/clip_000.mp4",
        )
        db.add(highlight)
        db.flush()

        reel = Reel(match_id=match_id, status=ReelStatus.READY, file_path=f"reels/{match_id}/reel.mp4")
        db.add(reel)
        db.flush()
        db.add(ReelHighlight(reel_id=reel.id, highlight_id=highlight.id, position=0))
        db.commit()

    response = client.get(f"/matches/{match_id}/reel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["reel_url"] == f"/media/reels/{match_id}/reel.mp4"
    assert body["clip_count"] == 1


def test_reel_pending_with_zero_clips_is_a_real_empty_reel_not_an_error(sample_video_path: Path):
    """
    ReelStatus.PENDING + file_path=None + no ReelHighlight rows is
    reel_max_clips=0's honest "no reel" case (see
    app/services/reel_persistence_stage.py's own docstring) — a real Reel
    row, not the same thing as no Reel row existing at all.
    """
    match_id = _upload_match(sample_video_path)

    with get_sync_db() as db:
        db.add(Reel(match_id=match_id, status=ReelStatus.PENDING, file_path=None))
        db.commit()

    response = client.get(f"/matches/{match_id}/reel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "pending"
    assert body["reel_url"] is None
    assert body["clip_count"] == 0


def test_reel_failed_has_no_url(sample_video_path: Path):
    match_id = _upload_match(sample_video_path)

    with get_sync_db() as db:
        db.add(Reel(match_id=match_id, status=ReelStatus.FAILED, file_path=None))
        db.commit()

    response = client.get(f"/matches/{match_id}/reel")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "failed"
    assert body["reel_url"] is None


def test_reel_for_unknown_match_returns_404():
    response = client.get("/matches/00000000-0000-0000-0000-000000000000/reel")
    assert response.status_code == 404
