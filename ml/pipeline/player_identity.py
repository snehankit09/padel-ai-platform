"""
Player identity grouping — Part 9b.

Every player-level StatValue in ml.pipeline.stats_aggregation.py (9a) is
keyed by ByteTrack `track_id`, not `app.models.player.Player.id` — see
that module's own StatValue docstring for why: a track_id is a tracker
artifact assigned arbitrarily as players are first detected in THIS
video, with no inherent connection to a real, named person. 9b's job is
to close as much of that gap as this pipeline's actual signals honestly
support — and to be explicit about the part that's left over, rather than
quietly gloss over it.

**What's actually solvable: which SIDE of the court a track played on.**
Padel is 2v2 (PRD default `format="doubles"`); each team occupies one
half of the court for the whole of a rally, split by the net line at
court_length_m / 2 (the same convention every Part 7 module already uses
— see ml.pipeline.point_outcome's module docstring). A track's own
tracked positions, majority-voted across every frame it appears in, tell
you which half it spent the match on — real geometry, needing only the
same court calibration (Part 5e) DISTANCE_COVERED/MOVEMENT_SPEED_AVG
already require, no new signal.

**What's NOT solvable here: which of a team's two players a track_id is.**
Two teammates share the same half of the court for the whole match —
there is no positional signal that tells two players on the same team
apart, and this codebase has no appearance-based signal either (no
jersey-number OCR, no face recognition, no re-identification model of any
kind). Telling teammate A's track from teammate B's would need one of
those to exist; grouping by court side genuinely can't get there no
matter how the thresholds are tuned. This is the same class of limit
ml.pipeline.point_outcome.py's module docstring already draws around
WINNERS (no signal exists to make that split, so it isn't attempted) —
not a gap to paper over with a coin-flip assignment.

**What this module does NOT attempt: mapping a side to a real team_number
or Player.id.** `app.models.match_player.MatchPlayer` stores team_number,
but nothing in this pipeline's data — or anywhere in the Match/Video
schema — records which physical court side a given team_number started
on for a given video. That correlation is genuinely external information
(the kind a human uploading the match would need to supply, e.g. "team 1
served from the near side"), not something derivable from tracked
positions no matter how the geometry is sliced. So this module's output
is a stable, video-scoped, arbitrary label (`SIDE_A`/`SIDE_B`) — real and
useful for team-level aggregation on its own (see summarize_side_assignments'
own team-level rollup use in 9c), but explicitly NOT wired to
MatchPlayer.team_number or Player.id. That wiring, and the individual-
teammate identity problem the previous paragraph describes, are both left
as an honest, open dependency for whatever surfaces that decision later
(most plausibly a human-in-the-loop UI step, not a pipeline stage) —
exactly the same "documented, not implemented, would need a real new
signal" posture ml.pipeline.stats_aggregation.py's own STAT_TYPE_STATUS
already takes for WINNERS/SERVE_PERCENTAGE/MOMENTUM_POSSESSION.

Same "no Celery/DB/app dependency" layering as every ml/pipeline module
before it. Fed by whatever 9c's glue stage becomes — this module has no
opinion on how its SIDE_A/SIDE_B labels get surfaced or stored.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from ml.pipeline.serve_detection import TrackPoint

SIDE_A = "side_a"
SIDE_B = "side_b"


class PlayerIdentityError(Exception):
    """Raised when side assignment is given input it can't reasonably reason about."""


@dataclass(frozen=True)
class TrackSideAssignment:
    """
    One track_id's court-side grouping across the whole video it was
    tracked in. `side_confidence` is the fraction of this track's own
    frames that landed on `court_side` — 1.0 means every single frame
    agreed, something meaningfully below 1.0 means the track spent real
    time on both halves (a player who legitimately crossed the net to
    retrieve a shot, a tracking ID that got swapped between two players
    mid-match — see module docstring on ByteTrack's own known
    limitations around fast, small objects; the same class of risk
    applies, more weakly, to player tracks too). Exposed rather than
    silently discarded so a caller can decide its own threshold for
    "trust this assignment" rather than this module picking one
    unilaterally.
    """

    track_id: int
    court_side: str
    frames_observed: int
    side_confidence: float


def assign_court_sides(
    player_tracks: Sequence[list[TrackPoint]],
    *,
    to_court_meters,  # Callable[[tuple[float, float]], tuple[float, float]]
    court_length_m: float,
) -> list[TrackSideAssignment]:
    """
    Majority-vote court-side assignment for every track_id that appears
    anywhere in `player_tracks` (the same frame-indexed
    list[list[TrackPoint]] shape
    ml.pipeline.stats_aggregation.player_positions_from_tracks_json
    already parses player_tracks.json into).

    Requires a real `to_court_meters` converter — same "no meaningful
    pixel-only fallback" reasoning ml.pipeline.point_outcome.py and
    ml.pipeline.stats_aggregation.py's DISTANCE_COVERED both already give
    for exactly this kind of court-relative geometry: raising
    PlayerIdentityError when it's None is the same honest refusal those
    modules make, not a new, stricter standard invented here.

    Only real (non-interpolated) positions count toward the majority
    vote — same "trust what was actually observed, not the tracker's own
    Kalman-predicted fill-in" posture ml.pipeline.point_outcome.py already
    applies to its own final-position/deceleration checks. A track with
    zero real positions anywhere is skipped entirely (nothing to vote
    with), not assigned an arbitrary default side.
    """
    if to_court_meters is None:
        raise PlayerIdentityError(
            "assign_court_sides requires a real court calibration (to_court_meters) — "
            "see module docstring for why there is no meaningful pixel-only fallback "
            "for court-side assignment"
        )

    side_a_frame_count: dict[int, int] = defaultdict(int)
    side_b_frame_count: dict[int, int] = defaultdict(int)

    for frame_points in player_tracks:
        for point in frame_points:
            if point.is_predicted:
                continue
            _, y_m = to_court_meters((point.x, point.y))
            if y_m < court_length_m / 2:
                side_a_frame_count[point.track_id] += 1
            else:
                side_b_frame_count[point.track_id] += 1

    all_track_ids = set(side_a_frame_count) | set(side_b_frame_count)
    assignments = []
    for track_id in sorted(all_track_ids):
        a_count = side_a_frame_count[track_id]
        b_count = side_b_frame_count[track_id]
        total = a_count + b_count
        if a_count >= b_count:
            side, majority_count = SIDE_A, a_count
        else:
            side, majority_count = SIDE_B, b_count
        assignments.append(
            TrackSideAssignment(
                track_id=track_id,
                court_side=side,
                frames_observed=total,
                side_confidence=majority_count / total,
            )
        )
    return assignments


def to_serializable(assignments: Sequence[TrackSideAssignment]) -> list[dict]:
    return [
        {
            "track_id": a.track_id,
            "court_side": a.court_side,
            "frames_observed": a.frames_observed,
            "side_confidence": a.side_confidence,
        }
        for a in assignments
    ]


def summarize_side_assignments(assignments: Sequence[TrackSideAssignment]) -> dict:
    """
    Same "summary alongside raw data" pattern as every Part 5-9 module —
    also doubles as the honest diagnostic for how trustworthy this
    video's side assignments are overall: a low avg_confidence across
    many tracks is a real signal something's off (frequent ID swaps, a
    camera angle where the net line isn't where calibration expects),
    worth surfacing the same way ball_coverage_rate/serve_identification_rate
    already are elsewhere.
    """
    total = len(assignments)
    if total == 0:
        return {"track_count": 0, "side_a_count": 0, "side_b_count": 0, "avg_side_confidence": 0.0}
    side_a_count = sum(1 for a in assignments if a.court_side == SIDE_A)
    return {
        "track_count": total,
        "side_a_count": side_a_count,
        "side_b_count": total - side_a_count,
        "avg_side_confidence": sum(a.side_confidence for a in assignments) / total,
    }
