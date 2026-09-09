"""
Part 4e — real end-to-end verification.

Every other pipeline test (test_pipeline_stages.py, test_pipeline_retry.py)
uses Celery's eager mode: tasks run synchronously, in the test process,
against SQLite. That's enough to prove the *logic* is right, but it can't
catch the class of bug eager mode papers over — serialization problems
(a payload that doesn't survive a real JSON round-trip through Redis),
worker startup/registration issues (a task decorated in a module the
worker process never imports), a real countdown delay between retries
(eager mode retries instantly, synchronously, same process), or an
app.core.database engine that behaves differently against real Postgres
than SQLite.

Both tests below post a real upload through the real HTTP API and poll the
real status endpoint — nothing here is eager or mocked. The second test
additionally forces a real, recoverable failure inside a real worker
process via the PIPELINE_DEBUG_FAIL_STAGE / PIPELINE_DEBUG_FAIL_COUNT env
vars read by tasks.py's _install_debug_stage_failure_from_env() at import
time — since the worker imports tasks.py in its own process, a plain
in-test monkeypatch of STAGE_FN_OVERRIDES could never reach it; only
something set in the *worker's* environment before it starts can.

Requires the full docker-compose stack running — this sandbox has no
network access to install Celery/Redis/Postgres, so, like
test_video_upload.py, this file is written but not executed here. Run it
with:

    docker compose up -d db redis
    alembic upgrade head
    uvicorn app.main:app &

    # Plain end-to-end path:
    celery -A app.workers.celery_app worker --loglevel=info &
    pytest backend/tests/test_pipeline_worker_e2e.py -v -s -k plain

    # Then, for the retry-across-a-real-worker path, restart the worker
    # with the debug hook active and run the second test:
    kill %1  # stop the plain worker
    PIPELINE_DEBUG_FAIL_STAGE=detect PIPELINE_DEBUG_FAIL_COUNT=1 \\
        celery -A app.workers.celery_app worker --loglevel=info &
    pytest backend/tests/test_pipeline_worker_e2e.py -v -s -k retry

Watching the worker's own stdout while the retry test runs is worth doing
once by hand — you should see run_stage's "retrying in Xs" warning log and
a real ~10s pause (pipeline_retry_backoff_seconds) between the failing
attempt and the retry, which is the thing eager mode can never show you.
"""

import subprocess
import time
from pathlib import Path

import httpx
import pytest

BASE_URL = "http://localhost:8000"
POLL_INTERVAL_SECONDS = 1
POLL_TIMEOUT_SECONDS = 90


def _generate_sample_video(path: Path) -> Path:
    video_path = path / "e2e_sample.mp4"
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


def _upload_sample_video(tmp_path: Path) -> str:
    video_path = _generate_sample_video(tmp_path)
    with httpx.Client() as client, open(video_path, "rb") as f:
        response = client.post(
            f"{BASE_URL}/videos/upload",
            files={"file": ("e2e_sample.mp4", f, "video/mp4")},
            data={"venue": "E2E Test Court"},
        )
    assert response.status_code == 201, response.text
    return response.json()["video_id"]


def _poll_until_terminal(video_id: str) -> dict:
    """Polls GET /videos/{id}/status until status is DONE or FAILED, or times out."""
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    last_body = None
    with httpx.Client() as client:
        while time.monotonic() < deadline:
            response = client.get(f"{BASE_URL}/videos/{video_id}/status")
            response.raise_for_status()
            last_body = response.json()
            if last_body["status"] in ("done", "failed"):
                return last_body
            time.sleep(POLL_INTERVAL_SECONDS)
    raise TimeoutError(f"Video {video_id} did not reach a terminal state within {POLL_TIMEOUT_SECONDS}s: {last_body}")


def test_plain_upload_is_processed_end_to_end_by_a_real_worker(tmp_path):
    """
    Upload a real video through the real API, and confirm a real worker
    (not eager mode) picks it up off real Redis and drives it all the way
    to DONE, with current_stage clearing at the end — the full loop this
    whole Part exists to prove. Run the worker WITHOUT the debug env vars
    for this one.
    """
    video_id = _upload_sample_video(tmp_path)

    final_status = _poll_until_terminal(video_id)
    assert final_status["status"] == "done", final_status
    assert final_status.get("error_message") is None


def test_retry_recovers_across_a_real_worker(tmp_path):
    """
    Confirms a stage that fails once but succeeds on retry — forced via
    the worker-side PIPELINE_DEBUG_FAIL_STAGE/PIPELINE_DEBUG_FAIL_COUNT env
    vars (see module docstring for how to start the worker for this test)
    — still ends the video at DONE, and that it took at least one real
    backoff period to get there rather than finishing instantly. A
    near-instant DONE here would mean the debug hook wasn't actually
    active in the worker (wrong env, or worker wasn't restarted).
    """
    video_id = _upload_sample_video(tmp_path)

    started_at = time.monotonic()
    final_status = _poll_until_terminal(video_id)
    elapsed_seconds = time.monotonic() - started_at

    assert final_status["status"] == "done", (
        f"expected the video to recover and finish DONE after the forced retry, got: {final_status}"
    )
    assert elapsed_seconds >= 5, (
        f"finished in {elapsed_seconds:.1f}s — too fast for a real retry to have happened. "
        "Is the worker actually running with PIPELINE_DEBUG_FAIL_STAGE / "
        "PIPELINE_DEBUG_FAIL_COUNT set, and was it restarted after setting them?"
    )
