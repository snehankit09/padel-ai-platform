"""
Serve identification — Part 7b, the second piece of pipeline stage 6.

Part 7a answers "when did each rally happen" (RallySegment start/end).
This module answers the first *within-rally* question: "who served it".
Same posture as 7a's own module docstring on why this doesn't reach for a
swing/pose classifier that doesn't exist anywhere in this codebase yet
(PRD Section 13 lists this as future work): a serve is the one shot in a
rally with an unambiguous, purely spatial signature available from Parts
5-6's existing output, with no model needed —**the server is whichever
player is standing closest to the ball in the first moments the ball
becomes active at the start of a rally.** That's a proxy for "who swung
the racket that started the point", not the thing itself, and it can be
wrong the same ways any position-only proxy can (a partner standing
unusually close to the service line when their partner serves; the ball
detector's very first activated frame being slightly delayed relative to
the actual contact moment) — but it needs zero new model weights and
reuses exactly the position data Parts 5-6 already computed, which is the
same trade-off 7a made for rally boundaries themselves.

**Court-calibrated meters when available, raw pixels otherwise.** A fixed
pixel-distance threshold is resolution- and camera-distance-dependent —
150px means something completely different on a wide establishing shot
than a tight one. Part 5e already produces a homography (court_detector.py)
for videos where court calibration succeeded, so this module uses that to
convert both the ball's and each player's pixel position into real-world
court meters (ml.detection.court_detector.pixel_to_court_point) and
applies a meters-based threshold — resolution-independent, and a genuinely
meaningful physical distance ("within 2.5m of the ball" means the same
thing on every video). When no calibration exists for a video (Part 5e is
non-fatal on failure, by design), this falls back to the raw-pixel
threshold rather than refusing to identify a server at all — a weaker
signal is still a real one, and callers can tell which mode produced a
given ServeEvent from its `used_court_calibration` field rather than
being left to guess.

Same "no Celery/DB/app dependency" layering as ml/pipeline/rally_detection.py
and every ml/tracking module: plain per-frame position lists in, dataclasses
out. Fed by app/services/serve_detection_stage.py (Part 7b's glue layer),
which knows how to turn persisted ball_tracks.json/player_tracks.json/
court calibration JSON back into these shapes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

# How many sampled frames from a rally's start_frame to search for the
# closest ball-to-player approach. At the default frame_sample_rate_fps of
# 5.0, 3 frames is ~0.6s — long enough that the ball detector's first
# activated frame (which can lag the true contact instant by a frame or
# two; see ml/tracking/byte_tracker.py's own displacement-limitation
# caveat) is very likely inside the window, short enough that by the time
# it elapses the point is plausibly already in a rally rather than still
# at the serve.
DEFAULT_SERVE_WINDOW_FRAMES = 3

# Real-world distance, in meters, within which a player is considered
# plausibly "the one who just hit the ball" — used only when a court
# calibration is available for the video (see module docstring). A padel
# court is 10m wide; 2.5m is close enough to rule out the serving player's
# partner or an opponent already approaching the net, generous enough to
# tolerate the serve contact point being a stride or two from where a
# player's tracked bbox center actually sits.
DEFAULT_MAX_BALL_PLAYER_DISTANCE_M = 2.5

# Fallback pixel-distance threshold when no court calibration exists for
# the video. Deliberately generous and explicitly a weaker signal than the
# meters-based threshold above — see module docstring's third paragraph;
# this number means different things at different camera distances/
# resolutions, which is exactly why the calibrated path is preferred
# whenever it's available.
DEFAULT_MAX_BALL_PLAYER_DISTANCE_PX = 150.0


class ServeDetectionError(Exception):
    """Raised when serve detection is given input it can't reasonably reason about."""


@dataclass(frozen=True)
class TrackPoint:
    """
    One tracked object's position in one frame — deliberately decoupled
    from ml.tracking.byte_tracker.TrackedDetection (no bbox corners, no
    class fields) the same way rally_detection.BallFrameSignal is: this
    module only ever needs a center point and whether it was a real
    detection or an interpolated/predicted one, so it stays constructible
    from a plain parsed tracks JSON dict without importing anything
    tracking-specific.
    """

    track_id: int
    x: float
    y: float
    is_predicted: bool


@dataclass(frozen=True)
class ServeEvent:
    """
    One rally's identified (or not) serve. `identified` is False whenever
    no player was found within the configured distance threshold anywhere
    in the search window — including when the window has no usable ball
    or player position data at all (e.g. a rally starting right as the
    ball becomes trackable, with no player detections that frame). A
    ServeEvent is still emitted in that case (never silently dropped —
    every rally from Part 7a gets exactly one ServeEvent), with
    server_track_id=None so a caller can tell "no server identified" apart
    from "there was no rally to check" without cross-referencing rally
    counts.
    """

    rally_index: int
    frame_index: int | None
    server_track_id: int | None
    distance_m: float | None
    distance_px: float | None
    used_court_calibration: bool
    identified: bool


def frames_from_tracks_json(data: dict) -> list[list[TrackPoint]]:
    """
    Turns a parsed tracks JSON (the shared shape both
    app/services/player_tracking_stage.py's player_tracks.json and
    app/services/ball_tracking_stage.py's ball_tracks.json persist —
    ml.tracking.byte_tracker.to_serializable's output) into one list of
    TrackPoints per frame, indexed by position the same way
    rally_detection.frames_from_ball_tracks_json is — RallySegment's
    start_frame/end_frame are positions into this same list, since both
    tracks files share one underlying frame ordering (Part 5a's frame
    extraction).
    """
    frames: list[list[TrackPoint]] = []
    for frame_entry in data.get("frames", []):
        points = []
        for t in frame_entry.get("tracks", []):
            bbox = t["bbox"]
            center_x = (bbox["x1"] + bbox["x2"]) / 2
            center_y = (bbox["y1"] + bbox["y2"]) / 2
            points.append(
                TrackPoint(
                    track_id=t["track_id"],
                    x=center_x,
                    y=center_y,
                    is_predicted=t.get("time_since_update", 0) > 0,
                )
            )
        frames.append(points)
    return frames


def identify_server(
    rally,  # ml.pipeline.rally_detection.RallySegment -- not type-hinted directly to avoid a hard import cycle; see serve_detection_stage.py for the real usage
    ball_frames: Sequence[list[TrackPoint]],
    player_frames: Sequence[list[TrackPoint]],
    *,
    window_frames: int = DEFAULT_SERVE_WINDOW_FRAMES,
    max_distance_m: float = DEFAULT_MAX_BALL_PLAYER_DISTANCE_M,
    max_distance_px: float = DEFAULT_MAX_BALL_PLAYER_DISTANCE_PX,
    to_court_meters=None,  # Callable[[tuple[float, float]], tuple[float, float]] | None
) -> ServeEvent:
    """
    Searches frames [rally.start_frame, rally.start_frame + window_frames)
    for the closest ball-to-player approach, in court meters when
    `to_court_meters` is given (a homography-backed conversion function —
    see serve_detection_stage.py for how it's built from a persisted
    calibration), in raw pixels otherwise.

    Only ever considers a frame where the ball has EXACTLY one tracked
    point — same "an ambiguous frame teaches us nothing, don't guess"
    refusal ml.tracking.ball_interpolation.interpolate_ball_gaps already
    applies to its own gap-bridging boundaries. A frame with zero player
    points is simply skipped (nothing to compare against); a frame with
    multiple player points compares the ball against every one of them
    and keeps whichever is closest, since multiple simultaneously-tracked
    players in frame is the normal case, not an ambiguous one the way
    multiple ball candidates would be.

    Scans the WHOLE window rather than stopping at the first usable frame,
    keeping the single closest approach found anywhere in it — the exact
    contact frame isn't known in advance, and the closest approach across
    a short window is a better proxy for "the moment of contact" than
    whatever the first frame happens to show.
    """
    best_distance: float | None = None
    best_track_id: int | None = None
    best_frame_index: int | None = None
    best_distance_is_meters = to_court_meters is not None

    window_end = min(rally.start_frame + window_frames, len(ball_frames), len(player_frames))
    for frame_index in range(rally.start_frame, window_end):
        ball_points = ball_frames[frame_index]
        if len(ball_points) != 1:
            continue  # zero or ambiguous multiple ball candidates -- nothing trustworthy to compare against
        ball_point = ball_points[0]

        for player_point in player_frames[frame_index]:
            if to_court_meters is not None:
                bx, by = to_court_meters((ball_point.x, ball_point.y))
                px, py = to_court_meters((player_point.x, player_point.y))
            else:
                bx, by = ball_point.x, ball_point.y
                px, py = player_point.x, player_point.y
            distance = math.hypot(bx - px, by - py)

            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_track_id = player_point.track_id
                best_frame_index = frame_index

    threshold = max_distance_m if best_distance_is_meters else max_distance_px
    identified = best_distance is not None and best_distance <= threshold

    return ServeEvent(
        rally_index=rally.rally_index,
        frame_index=best_frame_index if identified else None,
        server_track_id=best_track_id if identified else None,
        distance_m=best_distance if (identified and best_distance_is_meters) else None,
        distance_px=best_distance if (identified and not best_distance_is_meters) else None,
        used_court_calibration=best_distance_is_meters,
        identified=identified,
    )


def detect_serves(
    rallies: Sequence,  # Sequence[RallySegment]
    ball_frames: Sequence[list[TrackPoint]],
    player_frames: Sequence[list[TrackPoint]],
    *,
    window_frames: int = DEFAULT_SERVE_WINDOW_FRAMES,
    max_distance_m: float = DEFAULT_MAX_BALL_PLAYER_DISTANCE_M,
    max_distance_px: float = DEFAULT_MAX_BALL_PLAYER_DISTANCE_PX,
    to_court_meters=None,
) -> list[ServeEvent]:
    """One ServeEvent per rally, in rally order — see ServeEvent's docstring for why every rally gets one, identified or not."""
    return [
        identify_server(
            rally, ball_frames, player_frames,
            window_frames=window_frames, max_distance_m=max_distance_m,
            max_distance_px=max_distance_px, to_court_meters=to_court_meters,
        )
        for rally in rallies
    ]


def to_serializable(events: Sequence[ServeEvent]) -> list[dict]:
    return [
        {
            "rally_index": e.rally_index,
            "frame_index": e.frame_index,
            "server_track_id": e.server_track_id,
            "distance_m": e.distance_m,
            "distance_px": e.distance_px,
            "used_court_calibration": e.used_court_calibration,
            "identified": e.identified,
        }
        for e in events
    ]


def summarize_serve_events(events: Sequence[ServeEvent]) -> dict:
    """Same "summary alongside raw data, don't make callers recompute it" pattern as every other Part 5-7 stage."""
    total = len(events)
    if total == 0:
        return {"serve_count": 0, "identified_count": 0, "identified_rate": 0.0, "used_court_calibration": False}
    identified = sum(1 for e in events if e.identified)
    return {
        "serve_count": total,
        "identified_count": identified,
        "identified_rate": identified / total,
        # True only if EVERY identified serve used calibration -- a mixed
        # result (some calibrated, some not) would be misleading to
        # collapse into one boolean, so this is deliberately conservative:
        # any() would overclaim, all() undersells a mostly-calibrated
        # video, so report the actual mix instead.
        "used_court_calibration": all(e.used_court_calibration for e in events if e.identified) if identified else False,
    }


def summarize_serve_events_by_track_id(events: Sequence[ServeEvent]) -> dict[int, dict]:
    """
    Per-`server_track_id` breakdown of `summarize_serve_events`' match-level
    numbers -- how many rallies each identified player served, keyed by the
    same ByteTrack track_id every other pending-player-identity stat in
    ml.pipeline.stats_aggregation uses (that module's own docstring covers
    why a track_id, not yet a `Player.id`, is still a legitimate thing to
    key a stat by today).

    Deliberately NOT an "identification rate per player": identification
    is a property of a RALLY (could this module tell who served it), not
    of a player -- a serve nobody could be identified for
    (`server_track_id is None`) has no track_id to attribute a rate to, so
    there is no "serves this player attempted" denominator to divide a
    per-player rate out of. That's the exact same "no denominator exists"
    gap ml.pipeline.stats_aggregation's module docstring already gives for
    why SERVE_PERCENTAGE itself isn't computable, one level down: this
    function doesn't invent one just because the aggregation is now
    per-player instead of match-level.

    What IS meaningful per player is the calibrated-vs-pixel-fallback mix
    behind their identified serves -- `calibrated_count`/`pixel_count`
    rather than a single collapsed boolean/rate, same "a mixed result
    shouldn't be flattened into one number" reasoning
    `summarize_serve_events`'s own `used_court_calibration` field comment
    already gives for the match-level version.

    Unidentified serves contribute nothing here (no track_id to key them
    by); a caller wanting the count of rallies with no identified server
    at all already has it from `summarize_serve_events`'s own
    `serve_count - identified_count`.
    """
    by_track: dict[int, dict] = {}
    for event in events:
        if not event.identified or event.server_track_id is None:
            continue
        entry = by_track.setdefault(
            event.server_track_id, {"serve_count": 0, "calibrated_count": 0, "pixel_count": 0}
        )
        entry["serve_count"] += 1
        if event.used_court_calibration:
            entry["calibrated_count"] += 1
        else:
            entry["pixel_count"] += 1
    return by_track
