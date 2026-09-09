"""
Point outcome detection — Part 7d, the fourth piece of pipeline stage 6.

Parts 7a-7c answer when a rally happened, who served it, and what shots
were played during it. This module answers the question a stats page
actually wants next: how did the point end.

**Why this doesn't attempt a winner/unforced-error split.** The PRD's
natural vocabulary for a point's outcome is "winner", "unforced error",
"out", "net" — but the first two are indistinguishable from position data
alone. A winner (opponent simply can't reach a good shot) and an
unforced error on the RECEIVING side (opponent could have reached it but
mishandled the return, or didn't attempt one) look identical in this
pipeline's data: the rally's ball-activity signal ends with the ball
inside the court and nobody having hit it back. Telling those apart needs
a "did a player attempt and fail a shot" signal — something closer to
swing/pose analysis — which, same as every Part 7 module before this one,
does not exist anywhere in this codebase (PRD Section 13, again, lists
this as future work). Rather than fabricate that split from a signal that
can't support it, this module reports a single honest category for that
case (OUTCOME_IN_BOUNDS_END) and reserves confident, specific categories
for the two outcomes that genuinely ARE determinable from ball position
and motion alone: the ball ending outside the court boundary
(OUTCOME_OUT_OF_BOUNDS), and the ball's motion stopping abruptly right at
the net line (OUTCOME_NET) — a real kinematic signature (a hard stop at
a specific, known court location), not a guess.

**What this actually checks, concretely, and why it needs calibration.**
Unlike Parts 7b/7c, which degrade to a pixel-based fallback when a video
has no court calibration (Part 5e), point outcome has no meaningful
pixel-only fallback at all: "is this pixel position inside the court" and
"is this pixel position at the net line" are both *the same question*
7b/7c already flagged as needing calibrated meters to mean anything
consistent across different camera framings — there's no coarser version
of "in or out of a rectangle" that degrades gracefully the way a coarse
distance threshold does. So a video with no calibration gets
OUTCOME_UNDETERMINED for every rally, honestly, rather than a fabricated
pixel-based guess. When calibration exists, Part 5e's homography already
rectifies the court into a clean (0, 0) to (court_width_m, court_length_m)
rectangle (see ml.detection.court_detector.compute_homography's
convention — this module relies on that convention directly, the same
way ml.pipeline.shot_classification's net-proximity check already does),
which makes "in bounds" a plain axis-aligned range check and "at the net"
a plain distance-to-the-midline check — no polygon geometry needed.

**The net signal specifically** combines two things, deliberately, not
one: being near the net-line position (`near_net`) alone isn't enough —
a groundstroke rallied right past the net position is near it too,
without hitting it. What actually distinguishes "hit the net" is that the
ball's own motion abruptly stops there: this module compares the ball's
speed over its final tracked segment in the rally to its speed over the
segment just before that, and only calls OUTCOME_NET when both the
position AND a sharp deceleration line up. Either signal alone is
ambiguous; together they're a specific, physical claim about what just
happened, not a coincidence of geometry.

Same "no Celery/DB/app dependency" layering, and same "define this
module's own small per-frame position type rather than import one from a
sibling ml/pipeline module" posture, as ml.pipeline.shot_classification —
see that module's own BallPoint for the precedent. Fed by
app/services/point_outcome_stage.py (Part 7d's glue layer), which knows
how to turn persisted rallies.json/ball_tracks.json/court calibration
back into these shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

OUTCOME_OUT_OF_BOUNDS = "out_of_bounds"
OUTCOME_NET = "net"
OUTCOME_IN_BOUNDS_END = "in_bounds_end"
OUTCOME_UNDETERMINED = "undetermined"

# How far outside the calibrated court rectangle (in meters) a ball's
# last position can be before this still calls it "in bounds" — homography
# and detection both carry some real-world imprecision, and a ball landing
# right on or a few centimeters past the line shouldn't flip categories on
# noise alone. 0.3m is generous relative to a padel line's own width but
# small relative to the court's own 10m x 20m scale.
DEFAULT_OUT_OF_BOUNDS_MARGIN_M = 0.3

# How close to the net LINE (the court's midline, at court_length_m / 2)
# the ball's last position needs to be to even be considered for
# OUTCOME_NET — see module docstring for why this is combined with a
# deceleration check rather than used alone. 1.0m is deliberately tighter
# than ml.pipeline.shot_classification's DEFAULT_NET_PROXIMITY_M (3.0m,
# "close enough to be volleying"): that threshold answers "was this
# player standing at the net", this one answers "did the ball's flight
# stop AT the net", a much more specific physical claim.
DEFAULT_NET_ZONE_M = 1.0

# Speed of the ball's final tracked segment in a rally, as a fraction of
# its speed the segment before that. A ball that's still travelling
# normally when it goes out of view (out of bounds, or simply out of
# frame) keeps a ratio near 1.0; a ball that hits the net and drops
# collapses toward 0. 0.35 requires a real, sharp collapse — not just
# "slowed down somewhat", which normal flight deceleration under drag
# could plausibly produce on its own.
DEFAULT_NET_DECELERATION_RATIO = 0.35

# Real-world padel court dimensions, same defaults and same reasoning as
# ml.detection.court_detector.DEFAULT_COURT_LENGTH_M /
# ml.pipeline.shot_classification.DEFAULT_COURT_LENGTH_M — kept as local
# constants rather than importing either module, for the same
# "importable/testable with zero ml.detection dependency" posture every
# ml/pipeline module already commits to. A caller with a real per-video
# calibration passes its actual court_length_m/court_width_m through
# instead of relying on these defaults (see
# app/services/point_outcome_stage.py for where those come from).
DEFAULT_COURT_LENGTH_M = 20.0
DEFAULT_COURT_WIDTH_M = 10.0


class PointOutcomeError(Exception):
    """Raised when point outcome detection is given input it can't reasonably reason about."""


@dataclass(frozen=True)
class BallPoint:
    """Own copy of the same minimal per-frame ball position shape ml.pipeline.shot_classification.BallPoint defines — see that module's docstring for why each ml/pipeline module keeps its own rather than sharing one."""

    frame_index: int
    x: float
    y: float
    is_predicted: bool


@dataclass(frozen=True)
class PointOutcome:
    """
    One rally's outcome — always exactly one per rally (same "always
    emit, let the category carry the uncertainty" contract as
    ml.pipeline.serve_detection.ServeEvent), never silently dropped even
    when nothing about it could be determined.

    `reason` is always populated with a short, specific explanation — not
    just for OUTCOME_UNDETERMINED (though that's where it matters most,
    since "no calibration" and "no ball detections found" call for
    different follow-up) but for every category, so a caller inspecting
    one PointOutcome later doesn't have to re-derive why this module
    reached that conclusion.

    `last_ball_x_m`/`last_ball_y_m` follow
    ml.detection.court_detector.compute_homography's coordinate
    convention: x in [0, court_width_m], y in [0, court_length_m], net at
    y = court_length_m / 2 — the same convention
    ml.pipeline.shot_classification.ContactPoint.ball_court_y_m already
    uses. Both are None whenever used_court_calibration is False.
    """

    rally_index: int
    outcome: str
    reason: str
    last_ball_frame_index: int | None
    last_ball_x_m: float | None
    last_ball_y_m: float | None
    speed_ratio: float | None
    used_court_calibration: bool


def ball_points_from_tracks_json(data: dict) -> list[BallPoint]:
    """Same logic and same "ambiguous frame contributes nothing" posture as ml.pipeline.shot_classification.ball_points_from_tracks_json."""
    points: list[BallPoint] = []
    for frame_index, frame_entry in enumerate(data.get("frames", [])):
        tracks = frame_entry.get("tracks", [])
        if len(tracks) != 1:
            continue
        t = tracks[0]
        bbox = t["bbox"]
        points.append(
            BallPoint(
                frame_index=frame_index,
                x=(bbox["x1"] + bbox["x2"]) / 2,
                y=(bbox["y1"] + bbox["y2"]) / 2,
                is_predicted=t.get("time_since_update", 0) > 0,
            )
        )
    return points


def _real_points_in_window(ball_points: Sequence[BallPoint], start_frame: int, end_frame: int) -> list[BallPoint]:
    """
    Only REAL detections (is_predicted False), same reasoning
    find_contact_points already gives: an interpolated point is a
    straight line by construction (ml.tracking.ball_interpolation) and
    can't show a genuine deceleration or a genuine out-of-bounds
    landing — both are real physical events only a real detection can
    actually witness.
    """
    window = [p for p in ball_points if start_frame <= p.frame_index <= end_frame and not p.is_predicted]
    window.sort(key=lambda p: p.frame_index)
    return window


def determine_point_outcome(
    rally,  # ml.pipeline.rally_detection.RallySegment
    ball_points: Sequence[BallPoint],
    *,
    out_of_bounds_margin_m: float = DEFAULT_OUT_OF_BOUNDS_MARGIN_M,
    net_zone_m: float = DEFAULT_NET_ZONE_M,
    net_deceleration_ratio: float = DEFAULT_NET_DECELERATION_RATIO,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
    to_court_meters=None,  # Callable[[tuple[float, float]], tuple[float, float]] | None
) -> PointOutcome:
    """
    See module docstring for the full reasoning. Short version: no
    calibration -> OUTCOME_UNDETERMINED; no real ball detections in the
    rally's frame window -> OUTCOME_UNDETERMINED; otherwise, look at the
    LAST real detection in the window (the ball's position right as it
    stopped being actively rallied) and classify from there.
    """
    if to_court_meters is None:
        return PointOutcome(
            rally_index=rally.rally_index, outcome=OUTCOME_UNDETERMINED,
            reason="no court calibration available for this video",
            last_ball_frame_index=None, last_ball_x_m=None, last_ball_y_m=None,
            speed_ratio=None, used_court_calibration=False,
        )

    window = _real_points_in_window(ball_points, rally.start_frame, rally.end_frame)
    if not window:
        return PointOutcome(
            rally_index=rally.rally_index, outcome=OUTCOME_UNDETERMINED,
            reason="no real (non-interpolated) ball detections found in this rally's frame window",
            last_ball_frame_index=None, last_ball_x_m=None, last_ball_y_m=None,
            speed_ratio=None, used_court_calibration=True,
        )

    last_point = window[-1]
    last_x_m, last_y_m = to_court_meters((last_point.x, last_point.y))
    speed_ratio = _final_speed_ratio(window, to_court_meters)

    in_bounds = (
        -out_of_bounds_margin_m <= last_x_m <= court_width_m + out_of_bounds_margin_m
        and -out_of_bounds_margin_m <= last_y_m <= court_length_m + out_of_bounds_margin_m
    )
    near_net = abs(last_y_m - court_length_m / 2) <= net_zone_m

    if near_net and speed_ratio is not None and speed_ratio <= net_deceleration_ratio:
        outcome = OUTCOME_NET
        reason = (
            f"ball's final tracked motion stopped near the net line (within {net_zone_m}m) "
            f"with a sharp speed drop (final-segment speed ratio {speed_ratio:.2f})"
        )
    elif not in_bounds:
        outcome = OUTCOME_OUT_OF_BOUNDS
        reason = (
            f"ball's last tracked position ({last_x_m:.1f}m, {last_y_m:.1f}m) was outside "
            f"the {court_width_m:.1f}m x {court_length_m:.1f}m court boundary"
        )
    else:
        outcome = OUTCOME_IN_BOUNDS_END
        reason = (
            "rally ended with the ball's last tracked position inside the court -- cannot "
            "distinguish a clean winner from an unreturned/mishit ball without a swing-"
            "quality signal this pipeline doesn't have (see module docstring)"
        )

    return PointOutcome(
        rally_index=rally.rally_index, outcome=outcome, reason=reason,
        last_ball_frame_index=last_point.frame_index, last_ball_x_m=last_x_m, last_ball_y_m=last_y_m,
        speed_ratio=speed_ratio, used_court_calibration=True,
    )


def _final_speed_ratio(window: list[BallPoint], to_court_meters) -> float | None:
    """
    (speed of the final real segment) / (speed of the segment before it),
    both in court meters/frame. None when fewer than 3 real points exist
    in the window (need 2 segments to compare) or the earlier segment's
    speed is ~0 (nothing to meaningfully compare a ratio against — a ball
    that was already nearly stationary isn't decelerating into anything).
    """
    if len(window) < 3:
        return None
    p_a, p_b, p_c = window[-3], window[-2], window[-1]
    ax, ay = to_court_meters((p_a.x, p_a.y))
    bx, by = to_court_meters((p_b.x, p_b.y))
    cx, cy = to_court_meters((p_c.x, p_c.y))

    early_frames = max(1, p_b.frame_index - p_a.frame_index)
    late_frames = max(1, p_c.frame_index - p_b.frame_index)
    early_speed = math.hypot(bx - ax, by - ay) / early_frames
    late_speed = math.hypot(cx - bx, cy - by) / late_frames

    if early_speed < 1e-6:
        return None
    return late_speed / early_speed


def detect_point_outcomes(
    rallies: Sequence,  # Sequence[RallySegment]
    ball_points: Sequence[BallPoint],
    *,
    out_of_bounds_margin_m: float = DEFAULT_OUT_OF_BOUNDS_MARGIN_M,
    net_zone_m: float = DEFAULT_NET_ZONE_M,
    net_deceleration_ratio: float = DEFAULT_NET_DECELERATION_RATIO,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
    to_court_meters=None,
) -> list[PointOutcome]:
    """One PointOutcome per rally, in rally order — see PointOutcome's docstring for why every rally gets exactly one."""
    return [
        determine_point_outcome(
            rally, ball_points,
            out_of_bounds_margin_m=out_of_bounds_margin_m, net_zone_m=net_zone_m,
            net_deceleration_ratio=net_deceleration_ratio,
            court_width_m=court_width_m, court_length_m=court_length_m,
            to_court_meters=to_court_meters,
        )
        for rally in rallies
    ]


def to_serializable(outcomes: Sequence[PointOutcome]) -> list[dict]:
    return [
        {
            "rally_index": o.rally_index,
            "outcome": o.outcome,
            "reason": o.reason,
            "last_ball_frame_index": o.last_ball_frame_index,
            "last_ball_x_m": o.last_ball_x_m,
            "last_ball_y_m": o.last_ball_y_m,
            "speed_ratio": o.speed_ratio,
            "used_court_calibration": o.used_court_calibration,
        }
        for o in outcomes
    ]


def summarize_point_outcomes(outcomes: Sequence[PointOutcome]) -> dict:
    """Same "summary alongside raw data" pattern as every other Part 5-7 stage."""
    total = len(outcomes)
    by_outcome = {
        OUTCOME_OUT_OF_BOUNDS: 0, OUTCOME_NET: 0,
        OUTCOME_IN_BOUNDS_END: 0, OUTCOME_UNDETERMINED: 0,
    }
    for o in outcomes:
        by_outcome[o.outcome] = by_outcome.get(o.outcome, 0) + 1
    undetermined = by_outcome[OUTCOME_UNDETERMINED]
    return {
        "outcome_count": total,
        "outcome_counts": by_outcome,
        "undetermined_rate": (undetermined / total) if total else 0.0,
    }
