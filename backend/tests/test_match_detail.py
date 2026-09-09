"""
End-to-end test for GET /matches/{match_id} — Part 12c.

Same real-stack convention as test_video_upload.py / test_match_list.py:
builds the match via the actual upload route, then seeds Highlight rows
via get_sync_db() the same way app/services/analyze_persistence_stage.py
(Part 7f) and app/services/clip_extraction_stage.py (Part 8b) actually
would — a sync session against the same Postgres database the async
TestClient path writes to, not a mock. Requires the docker-compose stack
running locally; written but not executed in this sandbox, same as those
two files' own docstrings explain. Run it yourself with:

    docker compose up -d db
    alembic upgrade head
    uvicorn app.main:app &
    pytest backend/tests/test_match_detail.py -v
"""

import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.database import get_sync_db
from app.main import app
from app.models.enums import HighlightType
from app.models.highlight import Highlight

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


def test_match_detail_includes_highlights_in_chronological_order(sample_video_path: Path):
    match_id = _upload_match(sample_video_path)

    with get_sync_db() as db:
        # Inserted out of chronological order on purpose, to prove the
        # route sorts by start_time_seconds rather than returning insert order.
        db.add(
            Highlight(
                match_id=match_id,
                event_type=HighlightType.POWERFUL_SMASH,
                start_time_seconds=40.0,
                end_time_seconds=42.0,
                importance_score=0.75,
                clip_file_path="clips/x/clip_001.mp4",
            )
        )
        db.add(
            Highlight(
                match_id=match_id,
                event_type=HighlightType.LONG_RALLY,
                start_time_seconds=10.0,
                end_time_seconds=25.0,
                importance_score=0.6,
                clip_file_path=None,  # extraction hasn't run / failed for this one
            )
        )
        db.commit()

    response = client.get(f"/matches/{match_id}")
    assert response.status_code == 200
    body = response.json()

    assert body["venue"] == "Court 3"
    assert body["format"] == "singles"

    highlights = body["highlights"]
    assert len(highlights) == 2
    assert [h["event_type"] for h in highlights] == ["long_rally", "powerful_smash"]

    long_rally, smash = highlights
    assert long_rally["clip_url"] is None  # no clip_file_path -> no URL
    assert smash["clip_url"] == "/media/clips/x/clip_001.mp4"
    assert 0.0 <= smash["importance_score"] <= 1.0


def test_match_detail_for_unknown_match_returns_404():
    response = client.get("/matches/00000000-0000-0000-0000-000000000000")
    assert response.status_code == 404
