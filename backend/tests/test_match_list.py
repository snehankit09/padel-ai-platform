"""
End-to-end test for GET /matches — Part 12b.

Same real-stack convention as test_video_upload.py: builds a match via
the actual upload route (not a direct DB insert) so the row this test
reads back is exactly what a real upload produces, then asserts on the
list response. Requires the docker-compose stack running locally, and is
written but not executed in this sandbox for the same reason
test_video_upload.py's own docstring gives — run it yourself with:

    docker compose up -d db
    alembic upgrade head
    uvicorn app.main:app &
    pytest backend/tests/test_match_list.py -v
"""

import subprocess
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
            "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
            "-loglevel", "error",
        ],
        check=True,
    )
    return video_path


def test_list_matches_includes_uploaded_match_with_its_video_status(sample_video_path: Path):
    with open(sample_video_path, "rb") as f:
        upload_response = client.post(
            "/videos/upload",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"venue": "Court 3", "format": "singles"},
        )
    assert upload_response.status_code == 201, upload_response.text
    uploaded = upload_response.json()

    list_response = client.get("/matches")
    assert list_response.status_code == 200

    matches = list_response.json()["matches"]
    match = next((m for m in matches if m["id"] == uploaded["match_id"]), None)
    assert match is not None, "uploaded match missing from /matches"

    assert match["venue"] == "Court 3"
    assert match["format"] == "singles"
    assert match["video_id"] == uploaded["video_id"]
    assert match["video_status"] in ("pending", "queued", "processing", "done", "failed")
    # No thumbnail-generation stage exists yet (see MatchListItem's docstring).
    assert match["thumbnail_url"] is None


def test_list_matches_orders_most_recently_played_first(sample_video_path: Path):
    match_ids_in_upload_order = []
    for venue in ("Court A", "Court B"):
        with open(sample_video_path, "rb") as f:
            response = client.post(
                "/videos/upload",
                files={"file": ("sample.mp4", f, "video/mp4")},
                data={"venue": venue},
            )
        assert response.status_code == 201, response.text
        match_ids_in_upload_order.append(response.json()["match_id"])

    matches = client.get("/matches").json()["matches"]
    positions = {m["id"]: i for i, m in enumerate(matches)}

    # played_at defaults to "now" for both (Part 3's route), so the second
    # upload's played_at is >= the first's -> it should sort at or before it.
    first_id, second_id = match_ids_in_upload_order
    assert positions[second_id] <= positions[first_id]
