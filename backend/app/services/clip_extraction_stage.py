"""
Glue between the Celery `analyze` stage (app/workers/tasks.py) and the
pure FFmpeg cutting logic in ml/common/clip_extraction.py — Part 8b, the
second and final piece of Part 8 (highlight clip generation).

Same "video existence check + broken-pipeline-contract check + deferred ml
import" shape as every other analyze sub-stage before it, and specifically
mirrors app/services/clip_boundary_stage.py (Part 8a), whose
clip_boundaries.json output this stage exists to consume: Part 8a's own
docstring is explicit that it deliberately stops short of "doing the
FFmpeg trim or the row update itself" — this module is that remaining
work. Reads back Part 8a's padded, clamped ClipBoundary list plus the
video's own stored source file (Video.file_path, set by the upload route
in Part 3), which is the only reason this stage touches the DB for reads
at all; same "one cheap existence-shaped query, then real file I/O" split
every earlier analyze sub-stage uses.

**Matching a ClipBoundary back to its Highlight row.** Part 7f
(app/services/analyze_persistence_stage.py) already inserted one
`Highlight` row per HighlightEvent, with *tight*, un-padded start/end
times (clip_file_path left NULL — see that module's own docstring for
why). `Highlight` has no rally_index or source_frame_index column (see
app/models/highlight.py) for this stage to join a ClipBoundary back
against directly. What both rows *do* already agree on, unchanged from
the same HighlightEvent that produced them both, is (event_type,
tight-start, tight-end) — a ClipBoundary carries those tight values
forward in its own event_start_time_s/event_end_time_s fields
specifically so a later step wouldn't need a schema change just to find
its way back to the row it's about to UPDATE (see
ml/pipeline/clip_boundaries.py's ClipBoundary docstring). That match is
safe against a retried `analyze`: every sub-stage before this one,
including 7f, reruns from scratch on any retry (see app/workers/tasks.py's
module docstring), so by the time this stage runs the match's Highlight
rows are always freshly (re)inserted with tight times — never a mix of
this stage's own previous-attempt UPDATEs and the fresh rows 7f just
wrote moments earlier in the same run.

What this does NOT do, and why: it doesn't rank, dedupe, or cap the
number of clips extracted — every ClipBoundary Part 8a produced gets a
real file. Deciding which highlights actually make it into a shareable
"reel" (ordering, a per-match cap, filtering low importance_score) reads
`Highlight` rows back after they exist rather than needing to happen
before a clip is even cut, and belongs to whatever later part builds
reels/`Reel`+`ReelHighlight` rows (PRD Module 3 reel generation, this
codebase's Part 10 "done" stage territory), not to clip generation
itself.

**Part 8d — highlight-type label overlay.** Every clip gets its
HighlightType burned into the bottom-left corner via
ml/common/clip_extraction.py's drawtext support — `_format_highlight_label`
below is this stage's one piece of that work: turning a raw enum value
like "long_rally" into display text ("LONG RALLY"). That humanizing is a
domain decision (it needs to know what a HighlightType even is), so it
lives here rather than in the domain-agnostic ml module; see that
module's own docstring for why a player-name or score overlay isn't
attempted at all yet — this stage has no more access to that missing
data than the ml module does.
"""

from __future__ import annotations

import json
import logging
import uuid

from sqlalchemy import select

from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.core.config import get_settings
from app.models.highlight import Highlight
from app.models.video import Video
from app.services.storage import (
    get_storage_service,
    make_clip_file_destination_path,
    make_clip_thumbnail_destination_path,
)

logger = logging.getLogger(__name__)

_HIGHLIGHT_LABEL_SEPARATOR = " "

# Highlights Improvement Roadmap Tier 1b: only these two types have a
# real decisive contact frame (HighlightEvent.source_frame_index —
# ml/pipeline/highlight_tagging.py) to center a slow-motion window on.
# LONG_RALLY/FAST_EXCHANGES have no single such moment, so slow-motion
# isn't attempted for them regardless of settings.enable_highlight_slowmo.
_SLOWMO_ELIGIBLE_HIGHLIGHT_TYPES = {"powerful_smash", "spectacular_save"}


def _format_highlight_label(highlight_type: str) -> str:
    """
    "long_rally" -> "LONG RALLY". `highlight_type` here is always a
    ClipBoundary's own `highlight_type` field — a HighlightType enum's
    `.value` (see ml/pipeline/clip_boundaries.py's ClipBoundary docstring),
    i.e. already a lowercase, underscore-separated string by construction,
    not a value this stage needs to validate against the enum itself.
    """
    return highlight_type.replace("_", _HIGHLIGHT_LABEL_SEPARATOR).upper()


def _thumbnail_timestamp_s(boundary: dict, sample_fps: float | None) -> float:
    """
    Picks which moment within a clip's padded window to grab a poster
    frame from — Highlights Improvement Roadmap Tier 1a.

    Prefers `source_frame_index` (set on POWERFUL_SMASH and
    SPECTACULAR_SAVE ClipBoundaries — see ml/pipeline/highlight_tagging.py's
    HighlightEvent.source_frame_index) converted to an absolute video
    timestamp via `sample_fps`, the same frame-index -> seconds conversion
    ml/pipeline/highlight_tagging.py's own _frame_time_s uses. That's the
    actual decisive contact frame — a far better poster image than an
    arbitrary point in the clip. LONG_RALLY and FAST_EXCHANGES boundaries
    don't carry a source_frame_index (no single decisive frame exists for
    those types), so this falls back to the padded window's midpoint;
    `sample_fps` missing from the payload (shouldn't happen, but this
    stage has no reason to trust it blindly) falls back the same way.

    Clamped to [start_time_s, end_time_s) either way, with a small margin
    off the very end: a source_frame_index is always inside the *tight*
    window it came from, which is itself always inside the padded window
    this function receives, but landing an `-ss` seek exactly on or past
    a short clip's last frame is the kind of edge case worth a cheap
    guard rather than trusting the invariant to always hold.
    """
    start_s = boundary["start_time_s"]
    end_s = boundary["end_time_s"]

    source_frame_index = boundary.get("source_frame_index")
    if source_frame_index is not None and sample_fps:
        timestamp_s = source_frame_index / sample_fps
    else:
        timestamp_s = (start_s + end_s) / 2

    return min(max(timestamp_s, start_s), max(end_s - 0.01, start_s))


class ClipExtractionStageError(Exception):
    """Raised when `analyze` can't find something an earlier stage should already have produced."""


def run_clip_extraction(payload: dict) -> None:
    """
    Real body of the FFmpeg-trim half of Part 8. Reads Part 8a's
    clip_boundaries.json (already on the payload) and the video's own
    stored source file, cuts one real clip per ClipBoundary via
    ml/common/clip_extraction.py, and UPDATEs each matching Highlight
    row's start/end + clip_file_path — see module docstring for exactly
    how a boundary is matched back to its row.

    Same not-found handling as every other analyze sub-stage: an
    invalid/missing video_id logs and returns; a video that exists but
    has no clip_boundaries_path on the payload means Part 8a didn't run
    or didn't set it — a broken pipeline contract, not a bad
    caller-supplied id — so that raises ClipExtractionStageError (Part
    4d's retry/fail path). A ClipBoundary with no matching Highlight row
    is the same kind of broken-contract failure rather than something to
    skip past silently: Part 7f should have inserted one row per
    HighlightEvent Part 8a padded, in this same run, so a boundary with
    no row behind it means those two sub-stages disagreed about what
    highlights.json actually contained.

    **Part 8f — one clip's failure doesn't cost every other clip.** A
    single ClipExtractionError (a highlight whose padded window lands
    outside the source's real duration, a transient FFmpeg hiccup) used
    to abort this whole loop immediately, mid-transaction, before the
    single commit() at the end — which meant every clip already
    successfully extracted in that same run was discarded right along
    with the one that failed (their Highlight rows were never updated,
    though the clip files themselves were already written to disk,
    orphaned). Worse, since Part 4d retries the WHOLE `analyze` stage,
    not just this loop, a video with one persistently bad boundary would
    burn every retry attempt re-extracting every clip that already worked
    fine the first time, only to fail the entire stage regardless — one
    bad highlight taking down every other highlight for a match that
    might have a dozen good ones.

    Now: each boundary's extraction is caught individually and committed
    individually (so a crash or later failure can never roll back a
    clip that already succeeded), and only two outcomes end the stage in
    failure — a missing Highlight row (a structural bug, not a bad
    clip) or every single boundary failing to extract (systemic: almost
    certainly a corrupt/unreadable source file, not one bad timestamp).
    Any other mix — most clips fine, a few boundaries that individually
    fail — is real, partial success: payload["clips_failed_count"] and a
    warning log record it, but the stage does not fail over it, the same
    "some rallies come back undetermined, that's not a bug" posture
    point_outcome_stage.py already established for exactly this kind of
    per-item, content-dependent failure.
    """
    video_id = payload["video_id"]

    try:
        video_uuid = uuid.UUID(video_id)
    except (ValueError, TypeError):
        logger.warning("[analyze] video_id=%r is not a valid UUID; skipping clip extraction", video_id)
        return

    with get_sync_db() as db:
        video = db.get(Video, video_uuid)
        if video is None:
            logger.warning("[analyze] video_id=%s not found; skipping clip extraction", video_id)
            return
        source_key = video.file_path
        match_id = video.match_id

    if not payload.get("clip_boundaries_path"):
        raise ClipExtractionStageError(
            f"video_id={video_id}: payload is missing 'clip_boundaries_path' "
            "(should have been set by the analyze stage's clip-boundary half, Part 8a)"
        )

    storage = get_storage_service()
    source_path = storage.get_local_path(source_key)

    try:
        boundaries_data = _read_json(storage.get_local_path(payload["clip_boundaries_path"]))
    except (OSError, json.JSONDecodeError) as exc:
        raise ClipExtractionStageError(
            f"video_id={video_id}: could not read the required clip boundaries file: {exc}"
        ) from exc

    boundaries = boundaries_data.get("clip_boundaries", [])

    logger.info(
        "[analyze] video_id=%s extracting %d clip(s) from %s",
        video_id, len(boundaries), source_path,
    )

    ensure_ml_importable()
    from ml.common.clip_extraction import (  # deferred — see frame_extraction_stage.py
        ClipExtractionError,
        SlowMotionWindowError,
        extract_clip,
        extract_clip_with_slowmo,
        extract_thumbnail,
    )

    settings = get_settings()
    sample_fps = payload.get("frame_sample_fps")

    # One short, read-only session to build the tight-window -> highlight_id
    # lookup (same matching key as before — see module docstring), separate
    # from the per-boundary write sessions below. Keeping just the id (not
    # the ORM row itself) out of this session is deliberate: a row loaded
    # here would still be bound to *this* session, but by the time a later
    # boundary needs to mutate it, this session may already be closed —
    # db.get(Highlight, highlight_id) inside each boundary's own session
    # below re-fetches it fresh, bound to the session that will actually
    # commit it.
    with get_sync_db() as db:
        rows = db.execute(select(Highlight).where(Highlight.match_id == match_id)).scalars().all()
        highlight_id_by_tight_window = {
            (row.event_type.value, row.start_time_seconds, row.end_time_seconds): row.id for row in rows
        }

    extracted_count = 0
    failures: list[dict] = []

    for index, boundary in enumerate(boundaries):
        match_key = (
            boundary["highlight_type"],
            boundary["event_start_time_s"],
            boundary["event_end_time_s"],
        )
        highlight_id = highlight_id_by_tight_window.get(match_key)
        if highlight_id is None:
            raise ClipExtractionStageError(
                f"video_id={video_id}: no Highlight row matches clip boundary "
                f"rally_index={boundary.get('rally_index')} "
                f"highlight_type={boundary['highlight_type']!r} "
                f"(event_start_time_s={boundary['event_start_time_s']}, "
                f"event_end_time_s={boundary['event_end_time_s']}) — Part 7f should have "
                "inserted one Highlight row per HighlightEvent Part 8a padded, in this same "
                "analyze run"
            )

        clip_key = make_clip_file_destination_path(video_uuid, index)
        clip_path = storage.get_local_path(clip_key)
        label_text = _format_highlight_label(boundary["highlight_type"])

        # Highlights Improvement Roadmap Tier 1b: attempt a slow-motion
        # version first when this highlight type has a real decisive
        # frame, the feature is turned on, and we actually know
        # sample_fps (needed to convert source_frame_index into an
        # absolute video timestamp). Any reason it can't apply — feature
        # off, wrong type, no source_frame_index, or the window not
        # fitting this specific clip (SlowMotionWindowError) — falls
        # through to the exact same plain extract_clip() call as before,
        # not a failure: slow-motion is a bonus treatment, never a
        # requirement for a clip to count as extracted.
        slowmo_center_s = None
        if (
            settings.enable_highlight_slowmo
            and boundary["highlight_type"] in _SLOWMO_ELIGIBLE_HIGHLIGHT_TYPES
            and boundary.get("source_frame_index") is not None
            and sample_fps
        ):
            slowmo_center_s = boundary["source_frame_index"] / sample_fps

        try:
            if slowmo_center_s is not None:
                try:
                    extract_clip_with_slowmo(
                        source_path=source_path,
                        output_path=clip_path,
                        start_time_s=boundary["start_time_s"],
                        end_time_s=boundary["end_time_s"],
                        slowmo_center_s=slowmo_center_s,
                        slowmo_window_s=settings.highlight_slowmo_window_s,
                        slowmo_factor=settings.highlight_slowmo_factor,
                        label_text=label_text,
                    )
                except SlowMotionWindowError as exc:
                    logger.info(
                        "[analyze] video_id=%s clip %d/%d: slow-motion window didn't fit "
                        "(rally_index=%s highlight_type=%r), falling back to a plain clip: %s",
                        video_id, index + 1, len(boundaries), boundary.get("rally_index"),
                        boundary["highlight_type"], exc,
                    )
                    extract_clip(
                        source_path=source_path,
                        output_path=clip_path,
                        start_time_s=boundary["start_time_s"],
                        end_time_s=boundary["end_time_s"],
                        label_text=label_text,
                    )
            else:
                extract_clip(
                    source_path=source_path,
                    output_path=clip_path,
                    start_time_s=boundary["start_time_s"],
                    end_time_s=boundary["end_time_s"],
                    label_text=label_text,
                )
        except ClipExtractionError as exc:
            # (thumbnail extraction below only runs once the clip itself
            # has already succeeded — see that block's own comment for why
            # a thumbnail failure doesn't reach this except clause)
            logger.warning(
                "[analyze] video_id=%s clip %d/%d failed (rally_index=%s highlight_type=%r), "
                "leaving this Highlight row's clip_file_path unset and moving on: %s",
                video_id, index + 1, len(boundaries), boundary.get("rally_index"),
                boundary["highlight_type"], exc,
            )
            failures.append(
                {"rally_index": boundary.get("rally_index"), "highlight_type": boundary["highlight_type"], "reason": str(exc)}
            )
            continue

        # Thumbnail extraction — Highlights Improvement Roadmap Tier 1a.
        # Deliberately a soft failure: this only runs once the real clip
        # has already succeeded above, and a missing poster image is a
        # strictly smaller problem than a missing clip (the frontend's
        # existing placeholder fallback still applies), so it doesn't
        # belong in `failures` — that list drives this stage's own
        # retry/fail decision (see this function's docstring, "every
        # single boundary failing" is what makes this stage raise), and a
        # thumbnail-only failure shouldn't count toward that at all, let
        # alone cost a clip that already extracted fine its persistence.
        thumbnail_key: str | None = None
        try:
            thumbnail_key = make_clip_thumbnail_destination_path(video_uuid, index)
            extract_thumbnail(
                source_path=source_path,
                output_path=storage.get_local_path(thumbnail_key),
                timestamp_s=_thumbnail_timestamp_s(boundary, sample_fps),
            )
        except ClipExtractionError as exc:
            logger.warning(
                "[analyze] video_id=%s clip %d/%d: thumbnail extraction failed "
                "(rally_index=%s highlight_type=%r), leaving this Highlight row's "
                "thumbnail_file_path unset — clip itself still extracted fine: %s",
                video_id, index + 1, len(boundaries), boundary.get("rally_index"),
                boundary["highlight_type"], exc,
            )
            thumbnail_key = None

        # Its own session, separate from both the read above and every
        # other boundary's iteration — see this function's docstring for
        # why: a later boundary's failure (extraction or otherwise) can
        # then never roll back a clip that already succeeded and committed
        # here.
        with get_sync_db() as db:
            highlight = db.get(Highlight, highlight_id)
            highlight.start_time_seconds = boundary["start_time_s"]
            highlight.end_time_seconds = boundary["end_time_s"]
            highlight.clip_file_path = clip_key
            highlight.thumbnail_file_path = thumbnail_key
            db.commit()
        extracted_count += 1

    payload["clips_extracted_count"] = extracted_count
    payload["clips_failed_count"] = len(failures)

    logger.info(
        "[analyze] video_id=%s done: %d/%d clip(s) extracted and persisted (%d failed)",
        video_id, extracted_count, len(boundaries), len(failures),
    )

    if boundaries and extracted_count == 0:
        # Every single boundary failed -- almost certainly systemic (a
        # corrupt/unreadable source file, ffmpeg unavailable on this
        # worker), not one bad highlight. Worth Part 4d's retry/fail path,
        # unlike a partial failure mixed in with real successes.
        raise ClipExtractionStageError(
            f"video_id={video_id}: all {len(boundaries)} clip boundary(s) failed to extract — "
            f"first failure: {failures[0]['reason'] if failures else 'unknown'}"
        )
    if failures:
        logger.warning(
            "[analyze] video_id=%s %d/%d clip(s) failed to extract — see preceding per-clip "
            "warnings for reasons; %d clip(s) still extracted and persisted successfully",
            video_id, len(failures), len(boundaries), extracted_count,
        )


def _read_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
