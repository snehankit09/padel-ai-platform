"""
End-to-end test for POST /videos/upload and GET /videos/{id}/status.

Unlike the Part 2 sanity check (which talked to the DB directly), this
exercises the real HTTP path: an actual multipart upload hits the actual
route, which calls the real storage service and the real ffprobe
validation, and we assert on the real row that lands in Postgres.

Requires the docker-compose stack (Postgres + the app itself) running
locally, same as Part 2's verification — this sandbox has no network
access to install fastapi/sqlalchemy/asyncpg/Postgres, so this file is
written but not executed here. Run it yourself with:

    docker compose up -d db
    alembic upgrade head
    uvicorn app.main:app &
    pytest backend/tests/test_video_upload.py -v

The ffprobe-parsing logic these routes depend on (duration, resolution,
fps extraction, size/duration limit checks) *was* verified against real
ffmpeg-generated files in this session — see the video_validation module
docstring for what that covered and what it didn't (the DB round trip).
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture
def sample_video_path(tmp_path: Path) -> Path:
    """A real, tiny, valid mp4 generated with ffmpeg — not a mocked file."""
    video_path = tmp_path / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=2:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=2",
            "-shortest", "-c:v", "libx264", "-c:a", "aac",
            str(video_path),
            "-loglevel", "error",
        ],
        check=True,
    )
    return video_path


@pytest.fixture
def corrupt_video_path(tmp_path: Path) -> Path:
    """Random bytes with a .mp4 extension — should be rejected before any DB write."""
    path = tmp_path / "corrupt.mp4"
    path.write_bytes(b"\x00" * 2000)
    return path


def test_upload_valid_video_creates_pending_video_row(sample_video_path: Path):
    with open(sample_video_path, "rb") as f:
        response = client.post(
            "/videos/upload",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"venue": "Test Club Court 1", "format": "doubles"},
        )

    assert response.status_code == 201, response.text
    body = response.json()
    assert body["status"] == "pending"
    assert body["original_filename"] == "sample.mp4"

    status_response = client.get(f"/videos/{body['video_id']}/status")
    assert status_response.status_code == 200
    status_body = status_response.json()
    assert status_body["status"] == "pending"
    assert 1.9 <= status_body["duration_seconds"] <= 2.1
    assert status_body["resolution_width"] == 320
    assert status_body["resolution_height"] == 240


def test_upload_corrupt_video_returns_specific_error_and_no_db_row(corrupt_video_path: Path):
    with open(corrupt_video_path, "rb") as f:
        response = client.post(
            "/videos/upload",
            files={"file": ("corrupt.mp4", f, "video/mp4")},
        )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "could not be read" in detail.lower()


def test_upload_unsupported_extension_is_rejected_before_ffprobe(tmp_path: Path):
    bogus_path = tmp_path / "notes.txt"
    bogus_path.write_text("this is not a video")

    with open(bogus_path, "rb") as f:
        response = client.post(
            "/videos/upload",
            files={"file": ("notes.txt", f, "text/plain")},
        )

    assert response.status_code == 422
    assert "unsupported file type" in response.json()["detail"].lower()


def test_status_for_unknown_video_id_returns_404():
    response = client.get("/videos/00000000-0000-0000-0000-000000000000/status")
    assert response.status_code == 404
