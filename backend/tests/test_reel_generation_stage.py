"""
Tests for app/services/reel_generation_stage.py — Part 10f.

Same "real logic, throwaway SQLite + real ffmpeg, no mocking of the work
itself" approach as test_clip_extraction_stage.py: a real LocalStorageService
pointed at a temp directory, real Highlight rows with real Part-8-style cut
clip files (via ml.common.clip_extraction.extract_clip, so their encoding
matches what ml.common.reel_assembly.assemble_reel expects to stream-copy),
and a real ffmpeg reel gets produced and probed at the end.

Run with: pytest backend/tests/test_reel_generation_stage.py -v
"""

from __future__ import annotations

import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import select

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()

from app.models.enums import HighlightType, ReelStatus, UserRole
from app.models.highlight import Highlight
from app.models.match import Match
from app.models.reel import Reel
from app.models.reel_highlight import ReelHighlight
from app.models.user import User
from app.models.video import Video
from app.services.reel_generation_stage import ReelGenerationStageError, run_reel_generation
from app.services.storage import LocalStorageService


class _FakeSettings:
    local_storage_path: str


@pytest.fixture
def storage_env(tmp_path, monkeypatch):
    """Same seam as test_clip_extraction_stage.py's storage_env fixture."""
    settings = _FakeSettings()
    settings.local_storage_path = str(tmp_path)
    storage = LocalStorageService(settings)

    import app.services.reel_generation_stage as stage_module

    monkeypatch.setattr(stage_module, "get_storage_service", lambda: storage)
    return storage


@pytest.fixture(scope="module")
def sample_video_bytes() -> bytes:
    """A real, ffmpeg-generated 20-second video with audio, reused across this module's tests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=20:size=64x64:rate=10",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                str(path),
                "-loglevel", "error",
            ],
            check=True,
        )
        return path.read_bytes()


def _seed_video(session_factory) -> tuple[str, uuid.UUID, uuid.UUID]:
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(match_id=match.id, file_path=f"videos/{uuid.uuid4()}/source.mp4", original_filename="source.mp4")
        s.add(video)
        s.commit()
        return str(video.id), video.id, match.id


def _seed_cuttable_highlight(
    session_factory,
    storage: LocalStorageService,
    match_id: uuid.UUID,
    source_path: str,
    *,
    event_type: HighlightType,
    start: float,
    end: float,
    importance_score: float,
    clip_key: str,
) -> uuid.UUID:
    """
    Seeds one Highlight row the way Part 7f + Part 8b would leave it: a
    real clip file already cut (via the real ml.common.clip_extraction
    logic, so it matches assemble_reel's stream-copy expectations) and
    clip_file_path pointing at it.
    """
    ensure_ml_importable()
    from ml.common.clip_extraction import extract_clip

    clip_path = storage.get_local_path(clip_key)
    extract_clip(source_path=source_path, output_path=clip_path, start_time_s=start, end_time_s=end)

    with session_factory() as s:
        highlight = Highlight(
            match_id=match_id, event_type=event_type,
            start_time_seconds=start, end_time_seconds=end,
            importance_score=importance_score, clip_file_path=clip_key,
        )
        s.add(highlight)
        s.commit()
        return highlight.id


def _probe(path: str) -> dict:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-show_entries",
         "format=duration", "-of", "json", path],
        capture_output=True, text=True,
    )
    import json
    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    return {
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "duration_s": float(data.get("format", {}).get("duration", 0.0)),
    }


def test_assembles_a_real_playable_reel_from_real_clips(sqlite_session, storage_env, sample_video_bytes):
    video_id, video_uuid, match_id = _seed_video(sqlite_session)
    source_key = f"videos/{video_uuid}/source.mp4"
    source_path = storage_env.get_local_path(source_key)
    Path(source_path).parent.mkdir(parents=True, exist_ok=True)
    Path(source_path).write_bytes(sample_video_bytes)

    h1 = _seed_cuttable_highlight(
        sqlite_session, storage_env, match_id, source_path,
        event_type=HighlightType.LONG_RALLY, start=1.0, end=3.0, importance_score=0.9,
        clip_key=f"clips/{video_uuid}/clip_000.mp4",
    )
    h2 = _seed_cuttable_highlight(
        sqlite_session, storage_env, match_id, source_path,
        event_type=HighlightType.POWERFUL_SMASH, start=10.0, end=12.0, importance_score=0.7,
        clip_key=f"clips/{video_uuid}/clip_001.mp4",
    )

    payload = {"video_id": video_id}
    run_reel_generation(payload)

    assert "reel_id" in payload
    assert payload["reel_highlight_count"] == 2
    assert payload["reel_duration_s"] > 0

    with sqlite_session() as s:
        reel = s.get(Reel, uuid.UUID(payload["reel_id"]))
        assert reel is not None
        assert reel.match_id == match_id
        assert reel.status == ReelStatus.READY
        assert reel.file_path is not None

        rows = s.execute(
            select(ReelHighlight).where(ReelHighlight.reel_id == reel.id).order_by(ReelHighlight.position)
        ).scalars().all()
        assert [r.highlight_id for r in rows] == [h1, h2]  # chronological default ordering
        assert [r.position for r in rows] == [0, 1]

    reel_local_path = storage_env.get_local_path(reel.file_path)
    assert Path(reel_local_path).exists()
    assert Path(reel_local_path).stat().st_size > 0

    probe = _probe(reel_local_path)
    assert probe["has_video"]
    assert probe["has_audio"]
    # Roughly the sum of both clip durations (2s + 2s) plus the default
    # reel_transition_gap_s (0.5s) between them -- a real, playable, non-trivial file.
    assert probe["duration_s"] > 3.5


def test_no_cuttable_highlights_persists_a_real_empty_reel_and_does_not_touch_ffmpeg(sqlite_session, storage_env):
    video_id, video_uuid, match_id = _seed_video(sqlite_session)

    # A Highlight row exists but has no cut clip yet (Part 8b hasn't reached it) -- not cuttable.
    with sqlite_session() as s:
        s.add(
            Highlight(
                match_id=match_id, event_type=HighlightType.LONG_RALLY,
                start_time_seconds=1.0, end_time_seconds=3.0,
                importance_score=0.5, clip_file_path=None,
            )
        )
        s.commit()

    payload = {"video_id": video_id}
    run_reel_generation(payload)

    assert payload["reel_highlight_count"] == 0

    with sqlite_session() as s:
        reel = s.get(Reel, uuid.UUID(payload["reel_id"]))
        assert reel is not None
        assert reel.file_path is None  # nothing was ever assembled

        rows = s.execute(select(ReelHighlight).where(ReelHighlight.reel_id == reel.id)).scalars().all()
        assert rows == []


def test_unknown_video_id_is_skipped_not_raised(sqlite_session, storage_env):
    payload = {"video_id": str(uuid.uuid4())}
    run_reel_generation(payload)  # must not raise
    assert "reel_id" not in payload


def test_invalid_video_id_is_skipped_not_raised(sqlite_session, storage_env):
    payload = {"video_id": "not-a-uuid"}
    run_reel_generation(payload)  # must not raise
    assert "reel_id" not in payload


def test_ffmpeg_assembly_failure_marks_reel_failed_and_raises(sqlite_session, storage_env, sample_video_bytes, monkeypatch):
    video_id, video_uuid, match_id = _seed_video(sqlite_session)
    source_key = f"videos/{video_uuid}/source.mp4"
    source_path = storage_env.get_local_path(source_key)
    Path(source_path).parent.mkdir(parents=True, exist_ok=True)
    Path(source_path).write_bytes(sample_video_bytes)

    _seed_cuttable_highlight(
        sqlite_session, storage_env, match_id, source_path,
        event_type=HighlightType.LONG_RALLY, start=1.0, end=3.0, importance_score=0.9,
        clip_key=f"clips/{video_uuid}/clip_000.mp4",
    )

    import app.services.reel_generation_stage as stage_module
    ensure_ml_importable()
    from ml.common.reel_assembly import ReelAssemblyError

    def _boom(*args, **kwargs):
        raise ReelAssemblyError("simulated ffmpeg failure")

    # Patch at the ml module level -- reel_generation_stage imports assemble_reel
    # locally (deferred import), so patch the source it imports from.
    import ml.common.reel_assembly as reel_assembly_module
    monkeypatch.setattr(reel_assembly_module, "assemble_reel", _boom)

    payload = {"video_id": video_id}
    with pytest.raises(ReelGenerationStageError):
        run_reel_generation(payload)

    with sqlite_session() as s:
        reel = s.get(Reel, uuid.UUID(payload["reel_id"]))
        assert reel is not None
        assert reel.status == ReelStatus.FAILED
        assert reel.file_path is None
