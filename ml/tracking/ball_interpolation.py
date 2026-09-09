"""
Ball-specific post-processing on top of the generic ByteTracker (Part 6a) —
Part 6c.

Separate module from byte_tracker.py on purpose, same "generic engine vs.
class-specific wrapper" split as ml/detection/player_ball_detection.py sits
above ml/detection/yolo_detector.py: nothing here is padel- or
ball-specific about the *matching* math (that's still ByteTracker's job),
only about what to do with the holes ByteTracker's own output leaves
behind.

Why the ball needs this and players don't: ByteTracker.update() only ever
returns tracks matched *this* frame (`time_since_update == 0` — see that
module's docstring). For a player that's exactly right: a frame where a
player wasn't matched usually means they're genuinely off-camera or
mid-occlusion behind another player, and Part 7 (event detection) is fine
treating that frame as "no player data" until they reappear. The ball is
different — PRD Section 13's named risks for it are occlusion (behind a
player, the net, the court's glass), motion blur, and leaving frame
entirely on a lob — and the first two are cases where the ball plainly
still existed and was moving on essentially the same line, just not
detected. Reporting those frames as "no ball" throws away reconstructible
trajectory data for no reason.

The fix is deliberately narrow: linear interpolation between the two real
detections bracketing a gap, and only up to a caller-supplied cap
(`max_gap_frames`, see settings.ball_track_max_interpolation_gap_frames).
Below the cap, a straight line between two nearby real points is a
reasonable stand-in for a frame or two of blur/occlusion. Above it, this
module does nothing — deliberately, not as an oversight — because a long
gap is much more likely to mean the ball left frame on a lob (PRD Section
13) or was genuinely lost, and a straight-line guess across that gap would
fabricate a trajectory the ball may never have taken, which is worse for
Part 7's event/rally detection than an honest hole. It's also physically
bounded in a way a short gap isn't: ByteTracker itself already drops a
track after `max_age` unmatched frames and starts a fresh track_id on
re-acquisition (see that module's docstring) — a gap this module is asked
to bridge only ever exists in the *input* because the track survived long
enough inside ByteTracker's own max_age to be matched again later, so
`max_gap_frames` only has anything to bridge at all when it's kept <=
ball_track_max_age.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

from ml.detection.yolo_detector import BoundingBox
from ml.tracking.byte_tracker import FrameTracks, TrackedDetection

# See module docstring — deliberately small. A padel ball sampled at
# frame_sample_rate_fps genuinely changes direction fast (off a wall, a
# volley), so even a "short" gap is trusting a straight line more than
# real ball physics strictly justifies; this default trades a bit of
# trajectory smoothness for not over-trusting the interpolation.
DEFAULT_MAX_INTERPOLATION_GAP_FRAMES = 4


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _interpolate_bbox(a: BoundingBox, b: BoundingBox, t: float) -> BoundingBox:
    """Straight-line interpolation of all four box edges — good enough for a small, roughly-rigid ball."""
    return BoundingBox(
        x1=_lerp(a.x1, b.x1, t),
        y1=_lerp(a.y1, b.y1, t),
        x2=_lerp(a.x2, b.x2, t),
        y2=_lerp(a.y2, b.y2, t),
    )


def interpolate_ball_gaps(
    frame_tracks: Sequence[FrameTracks],
    *,
    max_gap_frames: int = DEFAULT_MAX_INTERPOLATION_GAP_FRAMES,
) -> list[FrameTracks]:
    """
    Fills short gaps in a sequence of per-frame ball tracks with linear
    interpolation, per track_id independently, and leaves long gaps alone.

    `frame_tracks` is expected to be one ByteTracker instance's
    `update_many` output for the ball class only (see
    app/services/ball_tracking_stage.py — a dedicated ball-only tracker,
    same reasoning as the player half's dedicated instance), in frame
    order, one entry per frame including empty ones — exactly what
    `ByteTracker.update_many` already produces, so this is meant to be
    called directly on that result with no reshaping.

    Only ever fills a gap *between two real detections that share a
    track_id* — an interpolated frame is never used as an anchor for
    further interpolation, and a gap that spans two different track_ids
    (ByteTracker itself gave up and started a new identity) is never
    bridged at all, since connecting them would be asserting an identity
    ByteTracker's own matching explicitly did not find.

    Returns a new list (input is not mutated) the same length as
    `frame_tracks`, each frame's real tracks unchanged and any
    interpolated ones appended. An interpolated TrackedDetection is
    distinguishable from a real one via its existing `is_predicted`
    property (`time_since_update > 0`) — no new field needed, since
    "this frame's box isn't an observed detection" is exactly what that
    property already means for a Kalman-carried-forward player/ball box;
    interpolation is just a second way a box can end up in that state.
    """
    # track_id -> [(frame_idx, TrackedDetection), ...] for every REAL
    # (non-predicted) appearance, in frame order. Interpolated results are
    # collected separately below and never fed back in here, so a chain of
    # gaps can't compound.
    observed_by_track: dict[int, list[tuple[int, TrackedDetection]]] = {}
    for frame_idx, ft in enumerate(frame_tracks):
        for t in ft.tracks:
            observed_by_track.setdefault(t.track_id, []).append((frame_idx, t))

    # frame_idx -> extra TrackedDetection entries to add to that frame.
    injected: dict[int, list[TrackedDetection]] = {}

    for appearances in observed_by_track.values():
        appearances.sort(key=lambda pair: pair[0])
        for (idx_a, det_a), (idx_b, det_b) in zip(appearances, appearances[1:]):
            gap = idx_b - idx_a - 1
            if gap <= 0 or gap > max_gap_frames:
                continue
            for step in range(1, gap + 1):
                t = step / (gap + 1)
                interpolated = replace(
                    det_a,
                    bbox=_interpolate_bbox(det_a.bbox, det_b.bbox, t),
                    confidence=_lerp(det_a.confidence, det_b.confidence, t),
                    age=det_a.age + step,
                    hits=det_a.hits,
                    time_since_update=step,
                )
                injected.setdefault(idx_a + step, []).append(interpolated)

    result: list[FrameTracks] = []
    for frame_idx, ft in enumerate(frame_tracks):
        extra = injected.get(frame_idx)
        if not extra:
            result.append(ft)
        else:
            result.append(FrameTracks(frame_path=ft.frame_path, tracks=[*ft.tracks, *extra]))
    return result


def summarize_ball_track_coverage(frame_tracks: Sequence[FrameTracks]) -> dict:
    """
    Cheap, loggable summary of a (post-interpolation) ball track sequence —
    the Part 6c analogue of player_ball_detection.summarize_ball_detection_rate,
    one layer further downstream. Splits "frames with a ball" into observed
    vs. interpolated so a low `ball_coverage_rate` and a low
    `observed_ball_detections` share can be told apart: the former means
    interpolation genuinely couldn't help (gaps too long, or the ball
    detector missed it for good stretches); the latter alone, with coverage
    still high, just means gap-filling is doing a lot of the work and might
    be worth revisiting ball_track_max_interpolation_gap_frames.
    """
    total = len(frame_tracks)
    frames_with_ball = sum(1 for ft in frame_tracks if len(ft.tracks) > 0)
    observed = sum(1 for ft in frame_tracks for t in ft.tracks if not t.is_predicted)
    interpolated = sum(1 for ft in frame_tracks for t in ft.tracks if t.is_predicted)
    return {
        "frame_count": total,
        "frames_with_ball": frames_with_ball,
        "observed_ball_detections": observed,
        "interpolated_ball_detections": interpolated,
        "ball_coverage_rate": (frames_with_ball / total) if total else 0.0,
    }
