"""
Rally boundary detection — Part 7a, the first piece of pipeline stage 6
(action recognition, per app/workers/tasks.py's stage-to-Part mapping).

Everything upstream of this module (Parts 5-6) answers "what's in this
frame, and where" and "which frame-to-frame boxes are the same object" —
but neither ever asks "is a point actually being played right now". A
90-minute match is mostly *not* rally: players resetting between points,
a server bouncing the ball, a let, a changeover. Part 8 (highlight clips)
and Part 9 (stats) both need the match cut into discrete rally windows
before they can do anything useful — a "longest rally" stat or a "long
rally" highlight (PRD Module 3/5) is meaningless without first knowing
where each rally starts and ends. That's this module's entire job:
per-frame ball tracks in -> a list of RallySegment(start_frame, end_frame)
out. Nothing here classifies *what kind* of rally it was, or scores it as
highlight-worthy — that's Part 7b+/Part 8's job, reading this module's
output back the same way Part 6 read Part 5's.

**Why ball activity, not a serve-motion/swing classifier.** The task this
module nominally serves — "detect a serve motion and the ball leaving the
racket" — describes what a human viewer would look for. But there's no
pose or swing-recognition model anywhere in this codebase yet (PRD
Section 13 lists this as future work, same "pretrained-first, fine-tune
later" posture as ml/detection/yolo_detector.py's COCO checkpoint), and
byte_tracker.py only ever gives us *positions*, not stroke identity. So
rather than pretend a capability that doesn't exist, this module uses the
best proxy actually available in Part 6's output: **the ball becoming
newly, sustainedly trackable is the earliest observable signal that play
has resumed**, and it becoming untrackable for a real stretch is the
earliest observable signal that it's stopped (ball gone dead/out/netted,
or a winner no one's chasing down anymore). That's a proxy, not the thing
itself — see detect_rally_segments's docstring for exactly where it can
be wrong (a very long, entirely-airborne lob with no ball detections
mid-flight could read as two rallies) — and a real swing/serve classifier
remains the more direct signal, whenever it exists (Part 7b+).

**Two knobs, not one, deliberately reusing but distinct from Part 6c's
gap concept.** ml/tracking/ball_interpolation.py already bridges short
occlusion/motion-blur gaps *within* what it already believes is one
continuous track, because fabricating a few frames of straight-line
position was worth it there. This module's `activity_gap_tolerance_frames`
is a second, coarser gap tolerance on top of that: it doesn't touch
position at all, only the yes/no question "is this still probably the
same rally", so it can afford to be more forgiving than interpolation
was (a real point can have a fast smash blur through several *sampled*
frames with literally zero ball detections, well past
ball_track_max_interpolation_gap_frames, and still obviously be
mid-rally). `min_rally_duration_frames` is unrelated to either gap
concept — it's a noise floor, dropping a stray one- or two-frame false
positive during a genuine break from ever being reported as a "rally" at
all.

Same "no Celery/DB/app dependency" layering as every other ml/ module
(ml/tracking/byte_tracker.py, ml/tracking/ball_interpolation.py): this
takes a plain sequence of BallFrameSignal in, and is fed by
app/services/rally_detection_stage.py (Part 7a's glue layer), which knows
how to turn a persisted ball_tracks.json back into that shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# See module docstring: a coarser, position-agnostic gap tolerance than
# ball_interpolation.py's own DEFAULT_MAX_INTERPOLATION_GAP_FRAMES (4),
# on top of whatever that module already bridged. At the default
# frame_sample_rate_fps of 5.0, 8 frames is ~1.6s of continuous
# no-ball-detected time that's still treated as "probably still the same
# rally" — long enough to survive a hard-to-track smash or a ball
# temporarily lost behind the net/a player, short enough that it rarely
# accidentally stitches two genuinely separate points together (real
# between-point dead time — walking back to serve, picking the ball up —
# tends to run several seconds, well past this).
DEFAULT_ACTIVITY_GAP_TOLERANCE_FRAMES = 8

# See module docstring: unrelated to the gap tolerance above — this drops
# a surviving activity run outright if it's shorter than this, so an
# isolated one- or two-frame false-positive ball detection during a real
# break never gets reported as a rally of its own. At 5fps, 5 frames is
# ~1s — short enough not to discard a genuinely quick point (an
# unreturned serve, a put-away smash), long enough that a single spurious
# detection can't clear it on its own.
DEFAULT_MIN_RALLY_DURATION_FRAMES = 5


class RallyDetectionError(Exception):
    """Raised when rally detection is given input it can't reasonably reason about."""


@dataclass(frozen=True)
class BallFrameSignal:
    """
    One sampled frame's ball-activity signal — the input shape this
    module's boundary logic actually needs, deliberately decoupled from
    ml/tracking/byte_tracker.TrackedDetection (no bbox, no track_id):
    rally boundaries only ever depend on *whether* the ball was
    trackable, never *where*, so this stays constructible from a plain
    parsed ball_tracks.json dict (see frames_from_ball_tracks_json)
    without importing anything tracking- or detection-specific.
    """

    frame_index: int
    frame_path: str | None
    # True if the ball had at least one track this frame — observed OR
    # interpolated (ml/tracking/ball_interpolation.py already decided an
    # interpolated frame is a trustworthy stand-in for a real detection;
    # this module has no reason to second-guess that here).
    ball_present: bool
    # True only if at least one of this frame's ball tracks was a real
    # detection (time_since_update == 0), not purely interpolated. Kept
    # separate from ball_present for RallySegment's summary stats — see
    # ball_observed_frame_count — not used by the boundary logic itself.
    observed: bool
    # Mean confidence across this frame's ball tracks, or None when
    # ball_present is False. Summary-only, same reasoning as `observed`.
    confidence: float | None = None


@dataclass(frozen=True)
class RallySegment:
    """
    One detected rally: a contiguous window of sampled frame indices
    (inclusive on both ends) where the ball was continuously — or
    near-continuously, within activity_gap_tolerance_frames — trackable,
    long enough to clear min_rally_duration_frames.

    `rally_index` is 1-based, in chronological order — the natural "rally
    #3 of the match" numbering Part 8/9 will want to display, not a
    frame-derived id that would mean nothing to a viewer.
    """

    rally_index: int
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    duration_s: float
    frame_count: int
    # Of frame_count, how many actually had ball_present True (as opposed
    # to being carried by activity_gap_tolerance_frames bridging). A rally
    # with a low ball_active_frame_count relative to frame_count is one
    # where this module's own gap-bridging did a lot of the work — worth
    # knowing, the same way ball_interpolation's summarize_ball_track_coverage
    # splits observed from interpolated.
    ball_active_frame_count: int
    ball_observed_frame_count: int
    mean_ball_confidence: float | None


def frames_from_ball_tracks_json(data: dict) -> list[BallFrameSignal]:
    """
    Turns a parsed ball_tracks.json (app/services/ball_tracking_stage.py's
    output shape: {"frames": [{"frame_path": ..., "tracks": [...]}, ...]})
    into the BallFrameSignal sequence detect_rally_segments needs.

    Pure dict-in, dataclass-out — no ml.detection/ml.tracking import
    needed, so this (and everything in this module) stays testable with
    plain fixture dicts, the same way test_ball_interpolation.py builds
    FrameTracks/TrackedDetection directly rather than round-tripping
    through JSON.

    frame_index is assigned by position (0-based, matching the frames
    list's own order) rather than read from the JSON, since ball_tracks.json
    doesn't carry an explicit index — its "frames" list is already in
    frame order by construction (ByteTracker.update_many iterates frames
    in order; see byte_tracker.py's module docstring on why order matters
    there), so position is a reliable index here too.
    """
    signals: list[BallFrameSignal] = []
    for frame_index, frame_entry in enumerate(data.get("frames", [])):
        tracks = frame_entry.get("tracks", [])
        ball_present = len(tracks) > 0
        observed = any(t.get("time_since_update", 0) == 0 for t in tracks)
        confidences = [t["confidence"] for t in tracks if "confidence" in t]
        mean_confidence = (sum(confidences) / len(confidences)) if confidences else None
        signals.append(
            BallFrameSignal(
                frame_index=frame_index,
                frame_path=frame_entry.get("frame_path"),
                ball_present=ball_present,
                observed=observed,
                confidence=mean_confidence,
            )
        )
    return signals


def _bridge_short_gaps(activity: Sequence[bool], max_gap: int) -> list[bool]:
    """
    Turns a False run into True when it's both no longer than `max_gap`
    AND sandwiched by True on both sides — never at the very start or end
    of the sequence, since there's no "before" or "after" real activity to
    connect it to there (mirrors ball_interpolation.interpolate_ball_gaps'
    identical refusal to extrapolate past its bracketing real detections,
    one layer up: gap-in-the-middle-of-known-activity vs.
    gap-at-the-edge-of-the-whole-recording are different claims, and only
    the first one is safe to fill in).

    Scans left to right, mutating a copy — a bridged gap becomes True in
    `result` immediately, so a later, adjacent gap's "was there activity
    immediately before this gap" check sees it correctly without needing
    a second pass.
    """
    n = len(activity)
    result = list(activity)
    i = 0
    while i < n:
        if result[i]:
            i += 1
            continue
        j = i
        while j < n and not result[j]:
            j += 1
        gap_len = j - i
        has_before = i > 0 and result[i - 1]
        has_after = j < n and result[j]
        if has_before and has_after and gap_len <= max_gap:
            for k in range(i, j):
                result[k] = True
        i = j
    return result


def _find_true_runs(activity: Sequence[bool]) -> list[tuple[int, int]]:
    """Contiguous (start_idx, end_idx) runs of True, both inclusive, in order."""
    runs: list[tuple[int, int]] = []
    n = len(activity)
    i = 0
    while i < n:
        if not activity[i]:
            i += 1
            continue
        j = i
        while j < n and activity[j]:
            j += 1
        runs.append((i, j - 1))
        i = j
    return runs


def detect_rally_segments(
    signals: Sequence[BallFrameSignal],
    *,
    sample_fps: float,
    activity_gap_tolerance_frames: int = DEFAULT_ACTIVITY_GAP_TOLERANCE_FRAMES,
    min_rally_duration_frames: int = DEFAULT_MIN_RALLY_DURATION_FRAMES,
) -> list[RallySegment]:
    """
    The whole algorithm, in three passes over the ball-activity signal:

      1. Bridge short gaps (`_bridge_short_gaps`) so a fast smash or a
         moment the ball briefly slipped detection doesn't fracture one
         real rally into several.
      2. Find the surviving contiguous True runs (`_find_true_runs`) —
         each one is a rally candidate.
      3. Drop any candidate shorter than `min_rally_duration_frames` as
         noise, and turn what's left into RallySegments with real-world
         timestamps.

    `signals` must be in frame order, one entry per sampled frame,
    including frames with no ball at all (ball_present=False) — exactly
    what frames_from_ball_tracks_json already produces, and exactly what
    every per-frame list elsewhere in this codebase (FrameDetections,
    FrameTracks) already looks like. Gaps are inferred from *adjacent
    list positions*, not from any timestamp field, so a caller that
    filtered frames out of the list first would silently corrupt the gap
    math — don't pre-filter.

    Frame-index-to-time conversion assumes sampled frames are evenly
    spaced at `sample_fps` (true by construction — see
    ml/common/frame_extraction.py's `fps` filter, Part 5a), so
    frame i's slice of real time is [i/sample_fps, (i+1)/sample_fps).
    start_time_s is the start of start_frame's slice; end_time_s is the
    end of end_frame's slice, i.e. duration_s == frame_count / sample_fps
    exactly.

    **Known limitation, same spirit as byte_tracker.py's module docstring
    flagging its own known limitation up front:** a single very long,
    fully-airborne shot (a deep lob the ball detector loses for its whole
    flight, longer than activity_gap_tolerance_frames) reads as the *end*
    of one rally and the *start* of the next, once the ball is picked up
    again — this module has no way to distinguish that from a genuine
    point ending and a new one beginning, since both look identical in
    the ball-activity signal alone. A real serve/swing classifier (Part
    7b+) is the more direct fix; until then, a suspiciously short
    inter-rally gap right at activity_gap_tolerance_frames's boundary is
    the observable symptom worth checking footage against.
    """
    if sample_fps <= 0:
        raise RallyDetectionError(f"sample_fps must be > 0, got {sample_fps!r}")

    raw_activity = [s.ball_present for s in signals]
    bridged_activity = _bridge_short_gaps(raw_activity, activity_gap_tolerance_frames)

    segments: list[RallySegment] = []
    rally_index = 1
    for start_frame, end_frame in _find_true_runs(bridged_activity):
        frame_count = end_frame - start_frame + 1
        if frame_count < min_rally_duration_frames:
            continue

        window = signals[start_frame : end_frame + 1]
        ball_active_frame_count = sum(1 for s in window if s.ball_present)
        ball_observed_frame_count = sum(1 for s in window if s.observed)
        confidences = [s.confidence for s in window if s.confidence is not None]
        mean_confidence = (sum(confidences) / len(confidences)) if confidences else None

        start_time_s = start_frame / sample_fps
        end_time_s = (end_frame + 1) / sample_fps

        segments.append(
            RallySegment(
                rally_index=rally_index,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                duration_s=end_time_s - start_time_s,
                frame_count=frame_count,
                ball_active_frame_count=ball_active_frame_count,
                ball_observed_frame_count=ball_observed_frame_count,
                mean_ball_confidence=mean_confidence,
            )
        )
        rally_index += 1

    return segments


def to_serializable(segments: Sequence[RallySegment]) -> list[dict]:
    """Plain-dict/JSON-safe form, mirroring byte_tracker.to_serializable / player_ball_detection.to_serializable."""
    return [
        {
            "rally_index": s.rally_index,
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "start_time_s": s.start_time_s,
            "end_time_s": s.end_time_s,
            "duration_s": s.duration_s,
            "frame_count": s.frame_count,
            "ball_active_frame_count": s.ball_active_frame_count,
            "ball_observed_frame_count": s.ball_observed_frame_count,
            "mean_ball_confidence": s.mean_ball_confidence,
        }
        for s in segments
    ]


def summarize_rally_segments(segments: Sequence[RallySegment], *, total_frame_count: int) -> dict:
    """
    Cheap, loggable match-level summary — the Part 7a analogue of
    ball_interpolation.summarize_ball_track_coverage /
    player_ball_detection.summarize_ball_detection_rate, one layer further
    downstream. `rally_frame_coverage_rate` (rally frames / whole match)
    is the sanity-check number worth watching: a padel match is mostly
    played, not idle, so a very low rate here more often means the ball
    activity signal itself is weak (check ball_coverage_rate one stage
    back) than that the match genuinely had almost no rallies.
    """
    rally_count = len(segments)
    total_rally_frames = sum(s.frame_count for s in segments)
    durations = [s.duration_s for s in segments]
    return {
        "rally_count": rally_count,
        "total_rally_frame_count": total_rally_frames,
        "total_rally_duration_s": sum(durations),
        "avg_rally_duration_s": (sum(durations) / rally_count) if rally_count else 0.0,
        "longest_rally_duration_s": max(durations) if durations else 0.0,
        "shortest_rally_duration_s": min(durations) if durations else 0.0,
        "rally_frame_coverage_rate": (total_rally_frames / total_frame_count) if total_frame_count else 0.0,
    }
