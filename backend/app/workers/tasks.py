"""
Pipeline task chain — Parts 4b/4c/4d.

One Celery task per pipeline stage, not one giant `process_video` task, so
that a failure partway through (say, `track`) can retry just that stage
instead of re-running everything before it — including `detect`, which is
GPU-expensive. It's also what lets GPU-bound stages (detect, track) scale
on separate worker pools from CPU-bound ones (analyze) later, per PRD
Section 9/6 (Scalability, Reliability NFRs).

The PRD's system architecture (Section 9) defines 11 stages:
  1. Video Upload  2. Frame Extraction  3. Object Detection
  4. Player Tracking  5. Ball Tracking  6. Action Recognition
  7. Highlight Detection  8. Statistics Engine  9. Prediction Engine
  10. Reel Generator  11. Dashboard

Stage 1 already happened (Part 3, before this task ever runs). Stage 11 is
just the frontend reading finished data — not a pipeline task. The
remaining nine collapse into five Celery tasks for now, one per Part still
to come:

  validate  -> stage 2  (frame extraction / prep)              -> Part 5
  detect    -> stage 3  (court/player/ball detection)           -> Part 5
  track     -> stages 4-5 (player + ball tracking)              -> Part 6
  analyze   -> stages 6-9 (events, highlights, stats, predict.) -> Parts 7-9
  done      -> stage 10 (reel generation) + finalization        -> Part 10

Every stage below started as a no-op that logged that it ran and passed
its payload through unchanged. Real CV/ML logic replaces the body of each
function in its corresponding Part, without changing the chain shape or
the retry/status-writing wrapper around it — that's the whole point of
splitting them out now. `validate` (Part 5a: frame extraction), `detect`
(Part 5d: player & ball detection), `track` (Part 6: player + ball
tracking), `analyze` (Parts 7-9: events, highlights, stats), and `done`
(Part 10f: reel generation) are all filled in now — none of the five
remains a no-op.

Part 4c: every stage writes its progress back to Postgres before it runs,
so `Video.status` and `Video.current_stage` (read by GET /videos/{id}/status)
reflect what's actually happening instead of freezing at QUEUED the
instant the chain is dispatched.

Part 4d: per-stage retry. A stage failing once no longer fails the whole
video — it retries with exponential backoff up to
`settings.pipeline_max_retries` times *within that same stage*. Earlier
stages are never re-run just because a later one is retrying (that's the
whole reason this is 5 tasks and not 1). Only once a stage has exhausted
its own retries does the video actually get marked FAILED:

    QUEUED --[validate starts]--> PROCESSING (current_stage="validate")
       |                              |
       |                    (raises) -+-> retries < max? --yes--> retry
       |                              |         (same stage, backoff delay,
       |                              |          current_stage unchanged)
       |                              |
       |                              +--no (retries exhausted)--> FAILED
       |                                    (error_message = specific reason,
       |                                     current_stage = the stage that
       |                                     ultimately failed)
       v
    -> PROCESSING (current_stage="detect") -> ... -> "done" -> DONE

A stage that raises after exhausting retries stops the chain there —
Celery `chain` doesn't run later tasks after an earlier one fails — so
FAILED is always the last status a video lands in for that run, never
silently followed by DONE.
"""

from __future__ import annotations

import logging
import os
import uuid

from celery import chain
from celery.exceptions import MaxRetriesExceededError

from app.core.config import get_settings
from app.core.database import get_sync_db
from app.models.enums import VideoStatus
from app.models.video import Video
from app.services.analyze_persistence_stage import run_analyze_persistence
from app.services.ball_tracking_stage import run_ball_tracking
from app.services.clip_boundary_stage import run_clip_boundary_calculation
from app.services.clip_extraction_stage import run_clip_extraction
from app.services.court_detection_stage import run_court_detection
from app.services.detection_stage import run_player_ball_detection
from app.services.frame_extraction_stage import run_frame_extraction
from app.services.player_tracking_stage import run_player_tracking
from app.services.rally_detection_stage import run_rally_detection
from app.services.serve_detection_stage import run_serve_detection
from app.services.highlight_tagging_stage import run_highlight_tagging
from app.services.player_statistics_stage import run_player_statistics_aggregation
from app.services.point_outcome_stage import run_point_outcome_detection
from app.services.reel_generation_stage import run_reel_generation
from app.services.shot_classification_stage import run_shot_classification
from app.workers.celery_app import celery_app

logger = logging.getLogger(__name__)

settings = get_settings()

# Test-only seam. For stages that are still no-ops (track, analyze, done),
# this is what lets tests exercise real failure/retry behavior against a
# stage that actually raises, without touching the no-op production call
# sites below. For `validate` and `detect`, which now have real logic
# (Parts 5a/5d), setting an override here still takes priority over that
# real logic — see validate_stage/detect_stage — so tests can simulate a
# stage-specific failure without needing a real video file, extracted
# frames, or a loaded YOLO model on disk. Empty in normal operation.
STAGE_FN_OVERRIDES: dict[str, object] = {}


def _install_debug_stage_failure_from_env() -> None:
    """
    Debug-only hook for Part 4e's real end-to-end test. A REAL Celery
    worker process imports this module independently of whatever process
    dispatched the task, so a plain in-test monkeypatch of
    STAGE_FN_OVERRIDES never reaches it — there's no shared memory across
    processes. Setting PIPELINE_DEBUG_FAIL_STAGE / PIPELINE_DEBUG_FAIL_COUNT
    in the *worker's* environment before it starts lets a test make one
    real stage fail N times (then succeed) inside that real worker, so the
    retry path can be verified against a real countdown and a real second
    process — not eager mode's instant, same-process retry.

    Never set these outside of testing — see test_pipeline_worker_e2e.py
    for how they're used.
    """
    stage_name = os.environ.get("PIPELINE_DEBUG_FAIL_STAGE")
    fail_count_raw = os.environ.get("PIPELINE_DEBUG_FAIL_COUNT")
    if not stage_name or not fail_count_raw:
        return
    try:
        fail_count = int(fail_count_raw)
    except ValueError:
        logger.warning(
            "[pipeline] PIPELINE_DEBUG_FAIL_COUNT=%r is not an int; ignoring debug hook", fail_count_raw
        )
        return

    attempts = {"count": 0}

    def _debug_flaky_stage(payload: dict) -> None:
        attempts["count"] += 1
        if attempts["count"] <= fail_count:
            raise RuntimeError(
                f"[debug] simulated failure {attempts['count']}/{fail_count} for stage "
                f"'{stage_name}' (via PIPELINE_DEBUG_FAIL_STAGE)"
            )

    STAGE_FN_OVERRIDES[stage_name] = _debug_flaky_stage
    logger.warning(
        "[pipeline] DEBUG hook active: stage '%s' will fail %d time(s) before succeeding "
        "(PIPELINE_DEBUG_FAIL_STAGE/PIPELINE_DEBUG_FAIL_COUNT)",
        stage_name, fail_count,
    )


_install_debug_stage_failure_from_env()


def _set_video_progress(
    video_id: str,
    *,
    status: VideoStatus,
    current_stage: str | None,
    error_message: str | None = None,
) -> None:
    """
    Single place that writes pipeline progress to Postgres, used by every
    stage below. Swallows a missing video (logs + returns) rather than
    raising — a task shouldn't fail the whole stage just because the status
    write couldn't find a row; that's a data-integrity problem to notice
    via logs, not a reason to also fail the pipeline run.
    """
    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[pipeline] video_id=%r is not a valid UUID; skipping status update", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[pipeline] video_id=%s not found; skipping status update", video_id)
            return
        video.status = status
        video.current_stage = current_stage
        if error_message is not None:
            video.error_message = error_message
        db.commit()


def compute_backoff_seconds(attempt: int) -> int:
    """
    Exponential backoff for retry `attempt` (0-indexed: task.request.retries
    at the moment of failure, i.e. how many retries have already happened).
    Doubles each time starting from pipeline_retry_backoff_seconds, capped
    at pipeline_retry_backoff_max_seconds so a flaky stage can't end up
    waiting an hour between retries.

    Pulled out as its own pure function (no Celery/DB access) so the
    backoff math can be unit tested directly, without a broker or worker.
    """
    backoff = settings.pipeline_retry_backoff_seconds * (2 ** attempt)
    return min(backoff, settings.pipeline_retry_backoff_max_seconds)


def run_stage(
    task,
    stage_name: str,
    payload: dict,
    *,
    stage_fn=None,
    on_success_status: VideoStatus | None = None,
) -> dict:
    """
    Shared wrapper for every pipeline stage task.

    - Marks the video PROCESSING + this stage's name before running.
    - Runs `stage_fn(payload)` if given, else a no-op — `stage_fn` is the
      hook Parts 5-10 use to plug in real CV/ML logic without touching any
      of the retry/status-writing code below it. Left as None today, so
      every stage is still a no-op pass-through.
    - On success: if `on_success_status` is given (only `done` passes this,
      with VideoStatus.DONE), marks the video with that terminal status.
      Non-terminal stages leave the next stage's start-of-task write to
      update current_stage — nothing else needed on their success path.
    - On any exception: retries the stage (same task, exponential backoff)
      up to settings.pipeline_max_retries times. Only once retries are
      exhausted does it write FAILED with a specific error_message and
      let the exception propagate, which stops the Celery chain.

    `task` must be a bound task instance (`bind=True`), since retry count
    and `.retry()` both come from `task.request` / `task`.
    """
    video_id = payload["video_id"]
    _set_video_progress(video_id, status=VideoStatus.PROCESSING, current_stage=stage_name)
    logger.info(
        "[pipeline] video_id=%s stage=%s attempt=%d",
        video_id, stage_name, task.request.retries + 1,
    )

    try:
        if stage_fn is not None:
            stage_fn(payload)
    except Exception as exc:
        attempt = task.request.retries
        if attempt < settings.pipeline_max_retries:
            backoff = compute_backoff_seconds(attempt)
            logger.warning(
                "[pipeline] video_id=%s stage=%s failed (attempt %d/%d): %s — retrying in %ds",
                video_id, stage_name, attempt + 1, settings.pipeline_max_retries + 1, exc, backoff,
            )
            try:
                raise task.retry(exc=exc, countdown=backoff, max_retries=settings.pipeline_max_retries)
            except MaxRetriesExceededError:
                # Race between our own count check and Celery's internal one;
                # falls through to the same "exhausted" handling below.
                pass

        error_message = (
            f"Stage '{stage_name}' failed after {attempt + 1} attempt(s): {exc}"
        )[:1024]
        logger.error("[pipeline] video_id=%s stage=%s exhausted retries: %s", video_id, stage_name, exc)
        _set_video_progress(
            video_id,
            status=VideoStatus.FAILED,
            current_stage=stage_name,
            error_message=error_message,
        )
        raise

    if on_success_status is not None:
        _set_video_progress(video_id, status=on_success_status, current_stage=None)

    return payload


@celery_app.task(name="pipeline.validate", bind=True, max_retries=settings.pipeline_max_retries)
def validate_stage(self, payload: dict) -> dict:
    """
    Stage 2: frame extraction (Part 5a). Pulls sampled frames from the
    uploaded video into a video-specific frames directory for `detect`
    (Part 5d) to consume — see app/services/frame_extraction_stage.py for
    the real logic. Falls back to that real logic unless a test has set
    STAGE_FN_OVERRIDES["validate"], same pattern as every other stage.
    """
    stage_fn = STAGE_FN_OVERRIDES.get("validate", run_frame_extraction)
    return run_stage(self, "validate", payload, stage_fn=stage_fn)


@celery_app.task(name="pipeline.detect", bind=True, max_retries=settings.pipeline_max_retries)
def detect_stage(self, payload: dict) -> dict:
    """
    Stage 3: court calibration + player & ball detection (Parts 5d-5e).
    Runs YOLO (Part 5b) across every frame `validate` extracted (Part 5a),
    splitting each frame's results into players vs. ball — see
    app/services/detection_stage.py and ml/detection/player_ball_detection.py
    for why they're kept separate rather than detected identically.

    Court calibration (Part 5c) runs first, via
    app/services/court_detection_stage.py: unlike player/ball positions it
    only needs one representative frame, not every frame, so it's cheap to
    run ahead of the per-frame YOLO pass. It's non-fatal on its own (see
    that module's docstring for why a video the platform can't calibrate
    still gets player/ball detection run on it) — only
    run_player_ball_detection failing can retry/fail this stage, court
    detection failing just logs and moves on.

    Falls back to the real logic for both unless a test has set
    STAGE_FN_OVERRIDES["detect"], same pattern as validate_stage.
    """

    def _detect(p: dict) -> None:
        run_court_detection(p)
        run_player_ball_detection(p)

    stage_fn = STAGE_FN_OVERRIDES.get("detect", _detect)
    return run_stage(self, "detect", payload, stage_fn=stage_fn)


@celery_app.task(name="pipeline.track", bind=True, max_retries=settings.pipeline_max_retries)
def track_stage(self, payload: dict) -> dict:
    """
    Stages 4-5: player + ball tracking (Parts 6b/6c). Runs two independent
    ByteTracker instances — see app/services/player_tracking_stage.py's
    module docstring for why players and the ball aren't tracked by one
    shared tracker — each reading the same detections.json `detect` wrote,
    each filtered to its own class, each with its own tuning
    (settings.player_track_*/ball_track_*). The ball half additionally
    runs its result through ml/tracking/ball_interpolation.py to bridge
    short occlusion/motion-blur gaps before persisting — see
    app/services/ball_tracking_stage.py for why the player half doesn't
    need the same treatment.

    Falls back to the real logic for both unless a test has set
    STAGE_FN_OVERRIDES["track"], same pattern as detect_stage.
    """

    def _track(p: dict) -> None:
        run_player_tracking(p)
        run_ball_tracking(p)

    stage_fn = STAGE_FN_OVERRIDES.get("track", _track)
    return run_stage(self, "track", payload, stage_fn=stage_fn)


@celery_app.task(name="pipeline.analyze", bind=True, max_retries=settings.pipeline_max_retries)
def analyze_stage(self, payload: dict) -> dict:
    """
    Stages 6-9: action recognition, highlight detection, stats, predictions.
    Part 7a fills in the first piece of stage 6 — rally boundary detection
    (app/services/rally_detection_stage.py), segmenting the continuous
    match into discrete rally windows from `track`'s ball output. Part 7b
    fills in the next piece — serve identification
    (app/services/serve_detection_stage.py), reading 7a's rally segments
    back alongside `track`'s player/ball output (and, when available,
    `detect`'s court calibration — see that module's docstring for why
    it's optional) to identify which player served each rally. Part 7c
    fills in shot classification (app/services/shot_classification_stage.py),
    reading the same inputs as 7b plus 7a's rally segments to classify
    every in-rally contact as a smash/lob/volley/groundstroke — see that
    module's own docstring for why this stays a position-based proxy
    rather than reaching for a pose-estimation model that doesn't exist
    anywhere in this codebase yet. Part 7d fills in point-outcome detection
    (app/services/point_outcome_stage.py), reading 7a's rally segments and
    `track`'s ball output (not player tracks or 7b/7c's own output — see
    that module's docstring for why player position doesn't help here) to
    determine, per rally, whether the ball ended out of bounds, at the
    net, or in bounds (an honestly-scoped category covering both winners
    and unreturned/mishit balls, which this pipeline has no signal to
    tell apart — see that module's docstring). Part 7e fills in
    highlight-event tagging (app/services/highlight_tagging_stage.py),
    reading 7a's rally segments and 7c's classified shots back to apply
    rule-based thresholds for the HighlightType values this pipeline's
    data can actually support (long rally, fast exchange, powerful smash,
    spectacular save) — see that module's own docstring for why
    winning-shot/match-point/break-point are deliberately NOT tagged
    rather than guessed at. Part 7f closes out `analyze` itself
    (app/services/analyze_persistence_stage.py): it reads 7a's rally
    summary and 7e's tagged events back off the payload and writes them
    into Postgres, one `Highlight` row per tagged event (PRD Section 11
    / Part 2's schema) and a first, deliberately small slice of
    `Statistic` rows (match-level rally counts/durations only — see
    that module's own docstring for exactly which StatTypes still need
    Part 9's work first, and why). Every later sub-part (highlight clip
    generation — Part 8, the rest of the statistics engine — Part 9)
    reads these rows back instead of re-deriving them from the JSON
    artifacts itself. Part 8a fills in the first piece of Part 8 itself
    (app/services/clip_boundary_stage.py), reading 7e's tagged events back
    alongside the video's own probed duration to turn each event's tight
    boundaries into padded, clamped clip in/out points — see that
    module's own docstring for why this stays a separate JSON artifact
    (clip_boundaries.json) rather than mutating the `Highlight` rows 7f
    just wrote, and for exactly what Part 8's still-later steps (the
    actual FFmpeg trim, and updating clip_file_path) still need to do.
    Part 8b closes out Part 8 itself (app/services/clip_extraction_stage.py):
    it reads 8a's padded clip_boundaries.json back, actually invokes FFmpeg
    (ml/common/clip_extraction.py) once per boundary to cut a real,
    playable clip file, and UPDATEs the Highlight row 7f already inserted
    for that same event with the padded start/end times and the new clip
    file's storage key — see that module's own docstring for how a
    boundary is matched back to its row without a schema change.
    Part 9f closes out `analyze` with the rest of Part 9 itself
    (app/services/player_statistics_stage.py): it reads 7a's rally
    segments, 7c's classified shots, 7d's point outcomes, and `track`'s
    player positions back off the payload, runs 9a's track_id-keyed
    stat aggregation (ml/pipeline/stats_aggregation.py) and 9b's
    court-side grouping (ml/pipeline/player_identity.py), writes both to
    their own JSON artifact, and hands the result to 9e's persistence
    (app/services/player_statistics_persistence_stage.py) with an empty
    track_id -> Player.id mapping — see that module's own docstring for
    why this stage has no honest mapping to supply yet, and why an empty
    one is correct rather than a shortcut.

    Falls back to the real logic unless a test has set
    STAGE_FN_OVERRIDES["analyze"], same pattern as detect_stage/track_stage.
    """

    def _analyze(p: dict) -> None:
        run_rally_detection(p)
        run_serve_detection(p)
        run_shot_classification(p)
        run_point_outcome_detection(p)
        run_highlight_tagging(p)
        run_analyze_persistence(p)
        run_clip_boundary_calculation(p)
        run_clip_extraction(p)
        run_player_statistics_aggregation(p)

    stage_fn = STAGE_FN_OVERRIDES.get("analyze", _analyze)
    return run_stage(self, "analyze", payload, stage_fn=stage_fn)


@celery_app.task(name="pipeline.done", bind=True, max_retries=settings.pipeline_max_retries)
def done_stage(self, payload: dict) -> dict:
    """
    Stage 10: reel generation (Part 10f) + finalization. Reads back
    `analyze`'s already-persisted, already-cut `Highlight` rows (Part 7f
    + Part 8b) and turns them into one real, playable `Reel` file — 10a
    selection, 10b ordering/pacing, 10e persistence, 10c ffmpeg assembly,
    in that order — see app/services/reel_generation_stage.py's own
    module docstring for the full reasoning. Once that succeeds (or
    legitimately produces a real, empty `Reel` row — see that module's
    docstring on why an empty selection isn't a failure), this is still
    what actually flips the video to DONE, same as when this stage was a
    placeholder — every earlier stage having already succeeded (and
    retried its own way past any transient failures) is what `done`
    running at all means.

    Falls back to the real logic unless a test has set
    STAGE_FN_OVERRIDES["done"], same pattern as every other stage.
    """
    stage_fn = STAGE_FN_OVERRIDES.get("done", run_reel_generation)
    return run_stage(
        self, "done", payload,
        stage_fn=stage_fn,
        on_success_status=VideoStatus.DONE,
    )


@celery_app.task(name="process_video")
def process_video(video_id: str) -> dict:
    """
    Entry point dispatched by the upload route right after a video is
    queued (Part 4a) — `process_video.delay(video.id)`.

    Rather than doing the work itself, it builds the five-stage chain above
    and dispatches it. Using a Celery `chain` (not one stage calling the
    next stage's `.delay()` directly) means each stage only starts after
    the previous one *succeeds*, and Celery — not our code — owns retry and
    task-state tracking per stage.
    """
    payload = {"video_id": video_id}
    pipeline = chain(
        validate_stage.s(payload),
        detect_stage.s(),
        track_stage.s(),
        analyze_stage.s(),
        done_stage.s(),
    )
    pipeline.delay()
    return payload
