"""
Tests for per-stage retry + failure handling (Part 4d), in
app/workers/tasks.py.

Same eager-Celery + throwaway-SQLite approach as test_pipeline_stages.py
(fixtures in conftest.py) — real Celery retry mechanics, real
run_stage/compute_backoff_seconds code, real DB writes; no Redis or worker
process needed. What this can't verify is real countdown delays between
retries (eager mode retries immediately) or cross-process behavior — see
tests/test_pipeline_worker_e2e.py (Part 4e) for that.
"""

import uuid

import pytest

import app.workers.tasks as tasks_module
from app.models.enums import VideoStatus
from app.models.video import Video
from app.workers.tasks import compute_backoff_seconds, process_video
from conftest import seed_video


def test_compute_backoff_seconds_grows_exponentially_and_caps():
    settings = tasks_module.settings
    assert compute_backoff_seconds(0) == settings.pipeline_retry_backoff_seconds
    assert compute_backoff_seconds(1) == settings.pipeline_retry_backoff_seconds * 2
    assert compute_backoff_seconds(2) == settings.pipeline_retry_backoff_seconds * 4
    # Large attempt numbers must not overflow past the configured cap.
    assert compute_backoff_seconds(20) == settings.pipeline_retry_backoff_max_seconds


def test_stage_that_fails_then_succeeds_lets_video_finish_done(sqlite_session):
    """
    A stage that fails a couple of times but succeeds within the retry
    budget should NOT fail the video — the whole point of retrying. Proves
    retry isn't just "fail slower": a transient error the stage recovers
    from should be invisible to the end result.
    """
    video_id = seed_video(sqlite_session)
    attempts = {"count": 0}

    def fails_twice_then_succeeds(payload):
        attempts["count"] += 1
        if attempts["count"] <= 2:
            raise RuntimeError(f"transient error on attempt {attempts['count']}")
        return None

    tasks_module.STAGE_FN_OVERRIDES["detect"] = fails_twice_then_succeeds
    try:
        result = process_video.apply(args=[video_id])
        assert result.successful()
    finally:
        tasks_module.STAGE_FN_OVERRIDES.pop("detect", None)

    assert attempts["count"] == 3  # failed, retried, failed, retried, succeeded
    with sqlite_session() as s:
        video = s.get(Video, uuid.UUID(video_id))
        assert video.status == VideoStatus.DONE
        assert video.current_stage is None
        assert video.error_message is None  # never wrote a failure — it recovered


def test_stage_that_always_fails_exhausts_retries_and_fails_video(sqlite_session):
    """
    A stage that never recovers should retry exactly
    settings.pipeline_max_retries times (not indefinitely), then mark the
    video FAILED with an error message stating how many attempts were made.
    """
    video_id = seed_video(sqlite_session)
    attempts = {"count": 0}

    def always_fails(payload):
        attempts["count"] += 1
        raise RuntimeError("permanent failure")

    tasks_module.STAGE_FN_OVERRIDES["analyze"] = always_fails
    try:
        with pytest.raises(RuntimeError, match="permanent failure"):
            process_video.apply(args=[video_id])
    finally:
        tasks_module.STAGE_FN_OVERRIDES.pop("analyze", None)

    expected_attempts = tasks_module.settings.pipeline_max_retries + 1
    assert attempts["count"] == expected_attempts

    with sqlite_session() as s:
        video = s.get(Video, uuid.UUID(video_id))
        assert video.status == VideoStatus.FAILED
        assert video.current_stage == "analyze"
        assert f"failed after {expected_attempts} attempt(s)" in video.error_message
        assert "permanent failure" in video.error_message


def test_earlier_stages_are_not_rerun_when_a_later_stage_retries(sqlite_session):
    """
    The reason this is 5 separate tasks and not 1: retrying `track` must
    not re-run `detect`. Confirms detect only ever runs once even though
    track fails and retries several times.
    """
    video_id = seed_video(sqlite_session)
    detect_calls = {"count": 0}
    track_attempts = {"count": 0}

    def count_detect(payload):
        detect_calls["count"] += 1

    def flaky_track(payload):
        track_attempts["count"] += 1
        if track_attempts["count"] <= 1:
            raise RuntimeError("transient tracking hiccup")

    tasks_module.STAGE_FN_OVERRIDES["detect"] = count_detect
    tasks_module.STAGE_FN_OVERRIDES["track"] = flaky_track
    try:
        result = process_video.apply(args=[video_id])
        assert result.successful()
    finally:
        tasks_module.STAGE_FN_OVERRIDES.pop("detect", None)
        tasks_module.STAGE_FN_OVERRIDES.pop("track", None)

    assert detect_calls["count"] == 1, "detect should run exactly once, not re-run for track's retry"
    assert track_attempts["count"] == 2, "track should have retried exactly once before succeeding"
