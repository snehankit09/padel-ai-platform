"""
Tests for app/services/clip_extraction_stage.py — Part 8b.

Same "real logic, throwaway SQLite instead of Postgres" approach as
test_clip_boundary_stage.py: real ml/common/clip_extraction.py logic (real
ffmpeg, real files), a real LocalStorageService pointed at a temp
directory, no mocking of the cutting itself — only the storage/DB
*location*.

Run with: pytest backend/tests/test_clip_extraction_stage.py -v
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.models.enums import HighlightType, UserRole
from app.models.highlight import Highlight
from app.models.match import Match
from app.models.user import User
from app.models.video import Video
from app.services.clip_extraction_stage import (
    ClipExtractionStageError,
    _format_highlight_label,
    run_clip_extraction,
)
from app.services.storage import LocalStorageService


class _FakeSettings:
    local_storage_path: str


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Same seam as test_clip_boundary_stage.py's storage_env fixture."""
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.clip_extraction_stage as stage_module

    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


@pytest.fixture(scope="module")
def sample_video_bytes() -> bytes:
    """A real, ffmpeg-generated 10-second video, reused across this module's tests."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=10:size=64x64:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
                "-loglevel", "error",
            ],
            check=True,
        )
        return path.read_bytes()


def _write_json(storage: LocalStorageService, key: str, data: dict) -> None:
    path = storage.get_local_path(key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _clip_boundary_dict(
    rally_index: int,
    start_time_s: float,
    end_time_s: float,
    *,
    highlight_type: str = "long_rally",
    event_start_time_s: float | None = None,
    event_end_time_s: float | None = None,
    importance_score: float = 0.5,
) -> dict:
    return {
        "rally_index": rally_index,
        "highlight_type": highlight_type,
        "start_time_s": start_time_s,
        "end_time_s": end_time_s,
        "event_start_time_s": event_start_time_s if event_start_time_s is not None else start_time_s,
        "event_end_time_s": event_end_time_s if event_end_time_s is not None else end_time_s,
        "importance_score": importance_score,
        "reason": "test",
        "source_frame_index": None,
    }


def _seed_video_with_source_file(session_factory, storage: LocalStorageService, video_bytes: bytes) -> tuple[str, uuid.UUID]:
    """
    Seeds the User -> Match -> Video chain and writes a real sample video
    at Video.file_path's location — `run_clip_extraction` reads this file
    off disk (via source_path = storage.get_local_path(video.file_path)),
    same as seed_video() in conftest.py does for the pipeline chain tests.
    """
    source_key = f"videos/{uuid.uuid4()}/source.mp4"
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(match_id=match.id, file_path=source_key, original_filename="source.mp4", duration_seconds=10.0)
        s.add(video)
        s.commit()
        video_id = str(video.id)
        match_id = match.id

    dest = Path(storage.get_local_path(source_key))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(video_bytes)

    return video_id, match_id


def _seed_highlight(session_factory, match_id: uuid.UUID, *, event_type: HighlightType, start: float, end: float) -> None:
    with session_factory() as s:
        s.add(
            Highlight(
                match_id=match_id, event_type=event_type,
                start_time_seconds=start, end_time_seconds=end,
                importance_score=0.5, clip_file_path=None,
            )
        )
        s.commit()


# --- _format_highlight_label (Part 8d) --------------------------------------


def test_format_highlight_label_replaces_underscores_and_uppercases():
    assert _format_highlight_label("long_rally") == "LONG RALLY"
    assert _format_highlight_label("powerful_smash") == "POWERFUL SMASH"


def test_format_highlight_label_leaves_a_single_word_alone():
    assert _format_highlight_label("smash") == "SMASH"


def test_extracts_clips_and_updates_matching_highlight_rows(sqlite_session, storage_env, sample_video_bytes):
    video_id, match_id = _seed_video_with_source_file(sqlite_session, storage_env, sample_video_bytes)
    # Highlight rows as 7f would have inserted them: tight, un-padded times.
    _seed_highlight(sqlite_session, match_id, event_type=HighlightType.LONG_RALLY, start=2.0, end=3.0)
    _seed_highlight(sqlite_session, match_id, event_type=HighlightType.SPECTACULAR_SAVE, start=5.0, end=5.0)

    _write_json(
        storage_env,
        "clips/x/clip_boundaries.json",
        {
            "clip_boundaries": [
                _clip_boundary_dict(1, 1.0, 4.0, highlight_type="long_rally", event_start_time_s=2.0, event_end_time_s=3.0),
                _clip_boundary_dict(2, 3.0, 8.0, highlight_type="spectacular_save", event_start_time_s=5.0, event_end_time_s=5.0),
            ]
        },
    )

    payload = {"video_id": video_id, "clip_boundaries_path": "clips/x/clip_boundaries.json"}
    run_clip_extraction(payload)

    assert payload["clips_extracted_count"] == 2

    from sqlalchemy import select
    from app.core.database import SyncSessionLocal

    with SyncSessionLocal() as s:
        rows = {h.event_type: h for h in s.execute(select(Highlight).where(Highlight.match_id == match_id)).scalars()}

    long_rally = rows[HighlightType.LONG_RALLY]
    assert long_rally.start_time_seconds == 1.0
    assert long_rally.end_time_seconds == 4.0
    assert long_rally.clip_file_path is not None
    clip_path = Path(storage_env.get_local_path(long_rally.clip_file_path))
    assert clip_path.exists()
    assert clip_path.stat().st_size > 0

    save = rows[HighlightType.SPECTACULAR_SAVE]
    assert save.start_time_seconds == 3.0
    assert save.end_time_seconds == 8.0
    assert save.clip_file_path is not None
    assert save.clip_file_path != long_rally.clip_file_path


def test_no_clip_boundaries_extracts_nothing(sqlite_session, storage_env, sample_video_bytes):
    video_id, _match_id = _seed_video_with_source_file(sqlite_session, storage_env, sample_video_bytes)
    _write_json(storage_env, "clips/y/clip_boundaries.json", {"clip_boundaries": []})

    payload = {"video_id": video_id, "clip_boundaries_path": "clips/y/clip_boundaries.json"}
    run_clip_extraction(payload)

    assert payload["clips_extracted_count"] == 0


def test_missing_video_id_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": "not-a-uuid"}
    run_clip_extraction(payload)  # must not raise
    assert "clips_extracted_count" not in payload


def test_video_not_found_logs_and_returns(sqlite_session, storage_env):
    payload = {"video_id": str(uuid.uuid4())}
    run_clip_extraction(payload)  # must not raise
    assert "clips_extracted_count" not in payload


def test_missing_clip_boundaries_path_raises(sqlite_session, storage_env, sample_video_bytes):
    video_id, _match_id = _seed_video_with_source_file(sqlite_session, storage_env, sample_video_bytes)
    payload = {"video_id": video_id}
    with pytest.raises(ClipExtractionStageError, match="clip_boundaries_path"):
        run_clip_extraction(payload)


def test_boundary_with_no_matching_highlight_row_raises(sqlite_session, storage_env, sample_video_bytes):
    video_id, _match_id = _seed_video_with_source_file(sqlite_session, storage_env, sample_video_bytes)
    # No Highlight rows seeded at all — 7f "should have" inserted one, but didn't.
    _write_json(
        storage_env,
        "clips/z/clip_boundaries.json",
        {"clip_boundaries": [_clip_boundary_dict(1, 1.0, 4.0, event_start_time_s=2.0, event_end_time_s=3.0)]},
    )

    payload = {"video_id": video_id, "clip_boundaries_path": "clips/z/clip_boundaries.json"}
    with pytest.raises(ClipExtractionStageError, match="no Highlight row matches"):
        run_clip_extraction(payload)
