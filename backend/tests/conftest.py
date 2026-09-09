"""
Shared fixtures for the pipeline test suite (test_pipeline_stages.py,
test_pipeline_retry.py). Both need the same two things: Celery running
tasks synchronously in-process (no Redis/worker needed) and a throwaway
SQLite DB standing in for Postgres so app.core.database.get_sync_db's real
code path is exercised without the full docker-compose stack.
"""

import subprocess
import tempfile
import uuid
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import app.core.config as config_module
import app.core.database as db_module
import app.models  # noqa: F401 — registers every model on Base.metadata
from app.core.database import Base
from app.models.enums import UserRole, VideoStatus
from app.models.match import Match
from app.models.user import User
from app.models.video import Video
from app.workers.celery_app import celery_app

SEEDED_VIDEO_KEY = "uploads/test.mp4"


@pytest.fixture(autouse=True)
def eager_mode():
    """Run Celery tasks synchronously for the duration of each test."""
    original_eager = celery_app.conf.task_always_eager
    original_propagates = celery_app.conf.task_eager_propagates
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = True
    yield
    celery_app.conf.task_always_eager = original_eager
    celery_app.conf.task_eager_propagates = original_propagates


@pytest.fixture
def sqlite_session(tmp_path):
    """
    Points app.core.database's sync engine/sessionmaker at a fresh SQLite
    file for the duration of one test, so tasks.py's real `get_sync_db()`
    writes land somewhere inspectable. Restores the original
    (Postgres-pointed) engine afterward.

    Also points settings.local_storage_path at a fresh directory under the
    same tmp_path, for the same reason and on the same lifecycle: since
    Part 5a, the `validate` stage does real file I/O (extracting frames
    from whatever Video.file_path resolves to), not just DB writes, so
    pipeline tests need a real, writable, throwaway storage root the same
    way they need a real, throwaway DB. seed_video() below writes the
    matching sample video into this root.
    """
    db_path = tmp_path / "test_pipeline.db"
    test_engine = create_engine(f"sqlite:///{db_path}", future=True)
    Base.metadata.create_all(test_engine)
    test_session_factory = sessionmaker(bind=test_engine, expire_on_commit=False)

    original_engine = db_module.sync_engine
    original_factory = db_module.SyncSessionLocal
    db_module.sync_engine = test_engine
    db_module.SyncSessionLocal = test_session_factory

    settings = config_module.get_settings()
    original_storage_path = settings.local_storage_path
    storage_root = tmp_path / "storage"
    storage_root.mkdir(parents=True, exist_ok=True)
    settings.local_storage_path = str(storage_root)

    try:
        yield test_session_factory
    finally:
        db_module.sync_engine = original_engine
        db_module.SyncSessionLocal = original_factory
        settings.local_storage_path = original_storage_path


@pytest.fixture
def real_yolo_required():
    """
    Skips the test if a real YOLO model can't actually be loaded in this
    environment — ultralytics not installed, or yolov8n.pt can't be
    downloaded/read (see ml/detection/yolo_detector.ModelLoadError). Use
    this on any pipeline-chain test that doesn't override "detect" with a
    stand-in stage_fn, since the real `detect` stage (Part 5d) needs an
    actual model loaded to run at all — mirrors the same try-load-or-skip
    approach test_yolo_detector.py already uses for its own load_model
    integration tests, applied here at the pipeline-chain level instead of
    directly against ml/detection/yolo_detector.py.

    Uses get_cached_model (not load_model) so a successful check here also
    warms the same process-wide cache the real detect_stage looks up by
    (model_path, device) — the pipeline run inside the test then reuses
    this exact model instance instead of loading it a second time.
    """
    from app.core.ml_path import ensure_ml_importable

    ensure_ml_importable()
    from ml.detection.yolo_detector import ModelLoadError, get_cached_model

    try:
        get_cached_model()
    except ModelLoadError as exc:
        pytest.skip(f"real YOLO model unavailable in this environment: {exc}")


@lru_cache
def _sample_video_bytes() -> bytes:
    """
    A real, tiny, valid mp4, generated once via ffmpeg and reused (by copy)
    across every seed_video() call in the test session — matches the
    "real ffmpeg-generated file, not a mock" approach test_video_upload.py
    and test_pipeline_worker_e2e.py already use, just cached since here it
    needs to be written fresh per-test rather than posted once per test.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "sample.mp4"
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "testsrc=duration=1:size=64x64:rate=10",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(path),
                "-loglevel", "error",
            ],
            check=True,
        )
        return path.read_bytes()


def seed_video(session_factory, status=VideoStatus.QUEUED) -> str:
    """
    Creates the User -> Match -> Video chain the upload route would,
    minimally, AND writes a real sample video file at the location
    Video.file_path points to (under the current settings.local_storage_path
    — see sqlite_session, which every caller of this helper uses alongside
    it). A DB row alone used to be enough here, back when every pipeline
    stage was a no-op; since Part 5a, `validate` actually reads this file
    off disk, so callers need it to really exist.
    """
    with session_factory() as s:
        user = User(email=f"{uuid.uuid4()}@padel.ai", hashed_password="x", full_name="P", role=UserRole.CLUB_ADMIN)
        s.add(user)
        s.flush()
        match = Match(played_at=datetime.now(timezone.utc), venue="Test Court", format="doubles", uploaded_by=user.id)
        s.add(match)
        s.flush()
        video = Video(match_id=match.id, file_path=SEEDED_VIDEO_KEY, original_filename="test.mp4", status=status)
        s.add(video)
        s.commit()
        video_id = str(video.id)

    dest = Path(config_module.get_settings().local_storage_path) / SEEDED_VIDEO_KEY
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_sample_video_bytes())

    return video_id
