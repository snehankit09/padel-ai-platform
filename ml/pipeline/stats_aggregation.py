"""
Statistics aggregation core — Part 9a (PRD Module 5).

Parts 7a-7e each answer one factual question about the match (when did a
rally happen, who served it, what shots were played, how did each point
end, which moments are highlight-worthy). Part 9's job is to turn that
already-structured output into the `app.models.enums.StatType` values
`Statistic` rows (Part 2) are shaped around. This module is the pure-logic
half of that: plain dataclasses in, `StatValue`s out, no Postgres, no
Celery, no `app` import — same layering every ml/pipeline module before
it already commits to (see e.g. ml/pipeline/highlight_tagging.py's module
docstring). 9b (track_id -> Player identity), 9c (glue stage reading
persisted JSON), and 9d (persistence + API) all build on top of what's
here; nothing in this module touches a database.

**The actual job of 9a: an honest inventory, not just scaffolding.**
`StatType` has 12 values (app/models/enums.py). Before writing a single
aggregation function it's worth being explicit about which of them this
pipeline's existing signals can actually support -- same posture Parts
7a-7e already took for their own scope (point_outcome.py's
OUTCOME_UNDETERMINED, shot_classification.py's SHOT_TYPE_UNKNOWN,
highlight_tagging.py's deliberately-unimplemented WINNING_SHOT/
MATCH_POINT/BREAK_POINT). Faking a number into every enum slot just to
make the `statistics` table look fuller than the data supports would be
the wrong tradeoff, exactly as analyze_persistence_stage.py's own module
docstring already says for the 3-stat stopgap it shipped instead of a
full Part 9. The verdict for each of the 12, and why, is recorded in
STAT_TYPE_STATUS at the bottom of the constants section below -- treat
that dict as the actual deliverable of this module's "decide which
stat_types to compute" half, not just documentation of it.

Short version of the reasoning:

  COMPUTABLE NOW, match-level, no player identity needed (implemented
  below):
    - TOTAL_POINTS        = rally count (7a)
    - RALLY_LENGTH_AVG     = mean rally duration (7a)
    - LONGEST_RALLY        = max rally duration (7a)
    - ERRORS               = count of rallies whose PointOutcome (7d) was
                              OUTCOME_OUT_OF_BOUNDS or OUTCOME_NET. This is
                              a genuine floor, not the full count: 7d's own
                              docstring is explicit that OUTCOME_IN_BOUNDS_END
                              conflates a winner with a receiving-side
                              unforced error, so any UEs hiding in that
                              bucket are NOT counted here. What IS counted
                              is unambiguous -- a rally that ends with the
                              ball out of bounds or netted is, by
                              definition, a hitting error by whoever made
                              that last contact, no winner/UE ambiguity
                              involved at all. Reported as an honest
                              undercount, not a guess.

  COMPUTABLE NOW, keyed by ByteTrack `track_id` rather than `Player.id`
  (implemented below -- 9b's whole job is relabeling these once a
  track_id -> Player mapping exists; the aggregation logic itself doesn't
  change):
    - REACTION_TIME_AVG    = for each in-rally shot (7c) thrown by a
                              DIFFERENT player than the immediately
                              preceding shot in that rally, the time
                              between the two contacts, averaged per
                              responding track_id. Direct generalization
                              of ml.pipeline.highlight_tagging.tag_
                              spectacular_saves' own response_s idea
                              (there restricted to "responded to a smash
                              within 0.8s and got flagged a highlight");
                              here it's every opponent-to-opponent
                              response in a rally, not just the fast ones.
    - SMASH_SUCCESS_RATE,
      NET_SUCCESS_RATE      = of a player's shots (7c) classified SMASH /
                              VOLLEY that also happen to be the LAST
                              contact in their rally, what fraction of
                              those rallies' outcomes (7d) were NOT a
                              hitting error (i.e. not OUT_OF_BOUNDS/NET).
                              This is a narrower claim than "won the
                              point with that shot" -- most smashes/
                              volleys aren't a rally's final contact, and
                              this only ever scores the ones that are --
                              but it's the one success/fail signal this
                              pipeline can attribute to a specific shot
                              without guessing at winner-vs-UE, for
                              exactly the same reason ERRORS above can
                              count OUT_OF_BOUNDS/NET but not attempt a
                              winners split.
    - DISTANCE_COVERED,
      MOVEMENT_SPEED_AVG    = sum of frame-to-frame player displacement
                              in court meters, and that distance over
                              elapsed time. ONLY computed when a court
                              calibration exists for the video -- same
                              "a physical quantity in raw pixels means a
                              different thing at every camera zoom/
                              distance, so there is no meaningful
                              pixel-only fallback" reasoning
                              ml.pipeline.point_outcome's own module
                              docstring already gives for why in/out-of-
                              court checks can't degrade gracefully
                              either. An uncalibrated video simply gets
                              no DISTANCE_COVERED/MOVEMENT_SPEED_AVG
                              StatValues, the same honest omission
                              OUTCOME_UNDETERMINED represents one layer
                              over.

  NOT COMPUTABLE with anything in this codebase today (documented, not
  implemented -- would need a real new signal, not a threshold tweak):
    - WINNERS              = exactly the split point_outcome.py's module
                              docstring already says position data alone
                              cannot make (a winner and a receiving-side
                              unforced error are literally indistinguishable
                              from where the ball ends up). Unlike ERRORS,
                              there is no unambiguous subset to fall back
                              to -- every winner lives inside the same
                              OUTCOME_IN_BOUNDS_END bucket a receiving UE
                              does, with no further signal in this
                              pipeline to split it.
    - SERVE_PERCENTAGE      = traditionally "% of first serves landed in
                              play". This pipeline's rally boundaries
                              (7a) are themselves defined by sustained
                              ball activity -- a faulted serve (into the
                              net, long) never starts sustained activity,
                              so it never produces a rally at all and is
                              structurally indistinguishable from ordinary
                              between-point dead time. There is no
                              denominator of "serves attempted" anywhere
                              in this pipeline's data, only "rallies that
                              happened" -- so this can't be computed, not
                              even approximately. `serve_identification_rate`
                              (summarize_serve_events' own diagnostic,
                              reused as-is below) answers a related but
                              genuinely different question -- "of the
                              rallies that did happen, how often could we
                              tell who served them" -- and is exposed as a
                              diagnostic alongside the real StatValues,
                              not smuggled in under the SERVE_PERCENTAGE
                              name. summarize_serve_events_by_track_id
                              (also reused as-is below) breaks that same
                              question down per identified server -- how
                              many rallies each track_id served -- but
                              deliberately stops there rather than
                              dividing out a per-player "rate": an
                              unidentified serve has no track_id to
                              attribute it to, so there's no "serves this
                              player attempted" denominator per player
                              any more than there's a match-level one.
    - MOMENTUM_POSSESSION  = needs live game/set/match score state to
                              mean anything, the same missing signal
                              highlight_tagging.py's own module docstring
                              already cites for why MATCH_POINT/
                              BREAK_POINT aren't tagged. No stage anywhere
                              in this codebase tracks a running score
                              (PRD Section 13 territory); momentum is a
                              derived concept one level further on top of
                              that same missing input, so it inherits the
                              same "not yet" verdict.

Everything this module DOES compute stays keyed by whatever identity it
actually has (None for match-level, a ByteTrack track_id for per-player)
and, for the track_id-keyed ones, is deliberately *not* yet written as
`Statistic` rows anywhere -- `Statistic.player_id` is a real FK to
`app.models.player.Player`, and a track_id is not one. 9b is the part
that builds that mapping; 9c/9d are what read this module's output back
and actually persist it. This module's contract ends at "here is a
correct, honestly-scoped StatValue per (stat_key, track_id-or-None)".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from ml.pipeline.point_outcome import OUTCOME_NET, OUTCOME_OUT_OF_BOUNDS, PointOutcome
from ml.pipeline.rally_detection import RallySegment, summarize_rally_segments
from ml.pipeline.serve_detection import (
    ServeEvent,
    TrackPoint,
    frames_from_tracks_json,
    summarize_serve_events,
    summarize_serve_events_by_track_id,
)
from ml.pipeline.shot_classification import SHOT_TYPE_SMASH, SHOT_TYPE_VOLLEY, Shot, summarize_shots

# Re-exported so 9c's glue layer can build player-position input the same
# way it already builds ball/player position input for 7b/7c, without a
# second copy of this exact (track_id, x, y, is_predicted) shape. Nothing
# about TrackPoint/frames_from_tracks_json is serve-specific -- unlike
# BallPoint/ContactPoint elsewhere in Parts 7b-7d, which each module
# deliberately re-defines its own copy of (see e.g.
# point_outcome.BallPoint's own docstring on why), there is no
# serve-detection-specific coupling to decouple from here, so reusing the
# existing type directly is the less-duplication choice, not an exception
# to the pattern.
__all__ = [
    "TrackPoint",
    "frames_from_tracks_json",
    "StatValue",
    "DistanceSpeedResult",
    "ReactionTimeResult",
    "ShotSuccessResult",
    "STAT_TOTAL_POINTS",
    "STAT_ERRORS",
    "STAT_RALLY_LENGTH_AVG",
    "STAT_LONGEST_RALLY",
    "STAT_DISTANCE_COVERED",
    "STAT_MOVEMENT_SPEED_AVG",
    "STAT_REACTION_TIME_AVG",
    "STAT_SMASH_SUCCESS_RATE",
    "STAT_NET_SUCCESS_RATE",
    "STAT_TYPE_STATUS",
    "STATUS_COMPUTABLE_MATCH_LEVEL",
    "STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY",
    "STATUS_NOT_COMPUTABLE",
    "compute_match_level_stats",
    "compute_distance_and_speed",
    "compute_reaction_times",
    "compute_rally_ending_shot_success_rate",
    "player_positions_from_tracks_json",
    "summarize_match_diagnostics",
    "to_serializable",
]

# --- stat_key constants ------------------------------------------------------
# Plain strings, deliberately matching app.models.enums.StatType's own
# .value strings 1:1 for anything this module actually computes, WITHOUT
# importing that enum -- same "no app/ dependency" posture every other
# ml/pipeline module already commits to (see e.g.
# ml.pipeline.shot_classification's SHOT_TYPE_* constants for the direct
# precedent: plain strings here, the real enum only ever gets built in
# app/services glue code). test_stats_aggregation.py cross-checks these
# against the real enum so the two can't silently drift apart.
STAT_TOTAL_POINTS = "total_points"
STAT_ERRORS = "errors"
STAT_RALLY_LENGTH_AVG = "rally_length_avg"
STAT_LONGEST_RALLY = "longest_rally"
STAT_DISTANCE_COVERED = "distance_covered"
STAT_MOVEMENT_SPEED_AVG = "movement_speed_avg"
STAT_REACTION_TIME_AVG = "reaction_time_avg"
STAT_SMASH_SUCCESS_RATE = "smash_success_rate"
STAT_NET_SUCCESS_RATE = "net_success_rate"

STATUS_COMPUTABLE_MATCH_LEVEL = "computable_match_level"
STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY = "computable_pending_player_identity"
STATUS_NOT_COMPUTABLE = "not_computable"

# The full 12-value inventory this module's docstring reasons through --
# see above for the full explanation behind each verdict. Every
# StatType.value from app/models/enums.py must appear here exactly once;
# test_stats_aggregation.py enforces that against the real enum so this
# can't quietly go stale as StatType evolves.
STAT_TYPE_STATUS: dict[str, dict[str, str]] = {
    STAT_TOTAL_POINTS: {
        "status": STATUS_COMPUTABLE_MATCH_LEVEL,
        "reason": "rally count, straight from Part 7a's rally summary.",
    },
    "winners": {
        "status": STATUS_NOT_COMPUTABLE,
        "reason": (
            "a winner and a receiving-side unforced error are indistinguishable from "
            "position data alone -- see ml.pipeline.point_outcome's module docstring. "
            "Every winner lives inside the same OUTCOME_IN_BOUNDS_END bucket a UE does, "
            "with no further signal anywhere in this pipeline to split it."
        ),
    },
    STAT_ERRORS: {
        "status": STATUS_COMPUTABLE_MATCH_LEVEL,
        "reason": (
            "count of rallies whose PointOutcome was OUTCOME_OUT_OF_BOUNDS or OUTCOME_NET -- "
            "an unambiguous hitting error, no winner/UE split needed. Deliberately an honest "
            "floor: unforced errors hiding inside OUTCOME_IN_BOUNDS_END are NOT counted."
        ),
    },
    "serve_percentage": {
        "status": STATUS_NOT_COMPUTABLE,
        "reason": (
            "a faulted serve never produces sustained ball activity, so it never produces a "
            "rally (Part 7a) at all -- it's structurally indistinguishable from ordinary "
            "between-point dead time. There is no 'serves attempted' denominator anywhere in "
            "this pipeline's data. serve_identification_rate (a diagnostic, not a StatType) "
            "answers a different question: of the rallies that DID happen, how often could "
            "Part 7b tell who served them -- and serve_counts_by_player breaks that same "
            "count down per identified server, without inventing a per-player rate that has "
            "the same missing-denominator problem one level down."
        ),
    },
    STAT_RALLY_LENGTH_AVG: {
        "status": STATUS_COMPUTABLE_MATCH_LEVEL,
        "reason": "mean rally duration, straight from Part 7a's rally summary.",
    },
    STAT_NET_SUCCESS_RATE: {
        "status": STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY,
        "reason": (
            "of a player's VOLLEY shots (Part 7c) that are also their rally's final contact, "
            "the fraction whose rally outcome (Part 7d) was not a hitting error. Keyed by "
            "track_id today -- 9b maps track_id to Player.id before this can become a "
            "Statistic row."
        ),
    },
    STAT_SMASH_SUCCESS_RATE: {
        "status": STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY,
        "reason": "same mechanism as net_success_rate, filtered to SHOT_TYPE_SMASH instead of SHOT_TYPE_VOLLEY.",
    },
    STAT_DISTANCE_COVERED: {
        "status": STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY,
        "reason": (
            "sum of frame-to-frame player displacement in court meters. Only computed when a "
            "court calibration exists for the video -- raw pixel distance means a different "
            "physical quantity at every camera zoom/distance, same reasoning "
            "ml.pipeline.point_outcome's module docstring already gives for why in/out-of-court "
            "checks have no meaningful pixel-only fallback. Keyed by track_id pending 9b."
        ),
    },
    STAT_MOVEMENT_SPEED_AVG: {
        "status": STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY,
        "reason": "distance_covered over elapsed time -- same calibration requirement, same track_id-pending-9b caveat.",
    },
    STAT_LONGEST_RALLY: {
        "status": STATUS_COMPUTABLE_MATCH_LEVEL,
        "reason": "max rally duration, straight from Part 7a's rally summary.",
    },
    STAT_REACTION_TIME_AVG: {
        "status": STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY,
        "reason": (
            "time between an opponent's in-rally contact (Part 7c) and this player's next "
            "contact, averaged per responding track_id -- a generalization of "
            "ml.pipeline.highlight_tagging.tag_spectacular_saves' own response_s idea to every "
            "opponent-to-opponent response, not just fast ones. Keyed by track_id pending 9b."
        ),
    },
    "momentum_possession": {
        "status": STATUS_NOT_COMPUTABLE,
        "reason": (
            "needs live game/set/match score state, which nothing in this codebase tracks yet "
            "-- the same missing signal ml.pipeline.highlight_tagging's module docstring already "
            "cites for why MATCH_POINT/BREAK_POINT aren't tagged. Momentum is a derived concept "
            "one level further on top of that same missing input."
        ),
    },
}


class StatsAggregationError(Exception):
    """Raised when stats aggregation is given input it can't reasonably reason about."""


# --- shared output shapes -----------------------------------------------------


@dataclass(frozen=True)
class StatValue:
    """
    One computed statistic, in the same (subject, stat_key, value) shape
    `Statistic` rows will eventually take -- but NOT a `Statistic` row
    itself (no match_id, no real player_id, no ORM/DB import here; that
    translation is 9c/9d's job).

    `track_id` is None for a match/team-level stat, or a ByteTrack
    track_id for a per-player one -- deliberately not named `player_id`,
    since it is NOT one: a track_id only becomes a real
    `app.models.player.Player` once 9b's identity mapping exists. Keeping
    the field named for what it actually is (a tracker artifact) rather
    than what it will eventually become avoids a caller silently treating
    it as an FK before 9b makes that legitimate.

    `sample_size` is how many underlying observations (rallies, shots, or
    frame-pairs, depending on stat_key) produced `value` -- a rate or
    average computed from a sample_size of 1 is real but shakier than one
    from 40, and callers ranking/displaying these should be able to tell
    the difference without recomputing it themselves.
    """

    stat_key: str
    track_id: int | None
    value: float
    sample_size: int


@dataclass(frozen=True)
class DistanceSpeedResult:
    """
    Per-track_id distance/speed summary. `used_court_calibration` is
    always True when this is returned at all -- compute_distance_and_speed
    returns an empty dict outright when no calibration is available (see
    its docstring), so a caller never has to check this field to know
    whether the numbers are trustworthy; its presence here is for
    completeness/debugging, not a live branch.

    `observed_frame_count`/`predicted_frame_count` mirror the same
    observed-vs-predicted transparency
    ml.pipeline.rally_detection.RallySegment already gives for ball
    coverage: a track whose displacement is mostly built from
    ByteTracker's own Kalman-predicted positions (time_since_update > 0,
    no real detection that frame) is a shakier distance/speed number than
    one built mostly from real detections, worth knowing even though both
    still contribute to `value`.
    """

    track_id: int
    total_distance_m: float
    avg_speed_mps: float
    elapsed_time_s: float
    observed_frame_count: int
    predicted_frame_count: int
    used_court_calibration: bool = True


@dataclass(frozen=True)
class ReactionTimeResult:
    """Per-track_id average response time (seconds) to an opponent's immediately preceding in-rally contact."""

    track_id: int
    avg_reaction_time_s: float
    response_count: int


@dataclass(frozen=True)
class ShotSuccessResult:
    """Per-track_id success rate for a specific shot_type when it's the last contact of its rally (see module docstring)."""

    track_id: int
    shot_type: str
    success_count: int
    attempt_count: int

    @property
    def success_rate(self) -> float:
        return self.success_count / self.attempt_count if self.attempt_count else 0.0


# --- player position parsing --------------------------------------------------


def player_positions_from_tracks_json(data: dict) -> list[list[TrackPoint]]:
    """
    Alias for `frames_from_tracks_json` (re-exported above from
    ml.pipeline.serve_detection) under a name that reads correctly at
    this module's own call sites -- `player_tracks.json` in, one
    TrackPoint list per frame out, same indexing convention every other
    Part 7 per-frame list already uses. Kept as a thin wrapper rather
    than just re-exporting the serve_detection name bare, so 9c's glue
    code importing from this module doesn't have to know the player-
    position parser happens to live in serve_detection.py's file for
    historical reasons.
    """
    return frames_from_tracks_json(data)


# --- match-level stats (StatType-ready today) ---------------------------------


def compute_match_level_stats(
    rallies: Sequence[RallySegment],
    outcomes: Sequence[PointOutcome],
) -> list[StatValue]:
    """
    TOTAL_POINTS, RALLY_LENGTH_AVG, LONGEST_RALLY (all straight off
    ml.pipeline.rally_detection.summarize_rally_segments) plus ERRORS
    (count of OUTCOME_OUT_OF_BOUNDS/OUTCOME_NET rallies) -- the four
    StatTypes STAT_TYPE_STATUS marks STATUS_COMPUTABLE_MATCH_LEVEL. Every
    other StatType needs either player identity (9b) or a signal this
    pipeline structurally doesn't have (see module docstring) and is
    deliberately absent from this function's output.

    `outcomes` need not cover every rally (a caller might only have
    outcomes for a subset, e.g. an uncalibrated video where every outcome
    is OUTCOME_UNDETERMINED) -- ERRORS' sample_size reflects how many
    outcomes were actually available to check, not len(rallies), so a
    caller can tell "zero errors observed" apart from "no outcome data to
    check in the first place".
    """
    summary = summarize_rally_segments(rallies, total_frame_count=0)
    rally_count = summary["rally_count"]

    values = [StatValue(stat_key=STAT_TOTAL_POINTS, track_id=None, value=float(rally_count), sample_size=rally_count)]

    if rally_count > 0:
        values.append(
            StatValue(
                stat_key=STAT_RALLY_LENGTH_AVG, track_id=None,
                value=summary["avg_rally_duration_s"], sample_size=rally_count,
            )
        )
        values.append(
            StatValue(
                stat_key=STAT_LONGEST_RALLY, track_id=None,
                value=summary["longest_rally_duration_s"], sample_size=rally_count,
            )
        )

    error_count = sum(1 for o in outcomes if o.outcome in (OUTCOME_OUT_OF_BOUNDS, OUTCOME_NET))
    values.append(
        StatValue(stat_key=STAT_ERRORS, track_id=None, value=float(error_count), sample_size=len(outcomes))
    )

    return values


# --- track_id-keyed stats (pending 9b's Player identity mapping) --------------


def compute_distance_and_speed(
    player_frames: Sequence[list[TrackPoint]],
    *,
    sample_fps: float,
    to_court_meters: Callable[[tuple[float, float]], tuple[float, float]] | None,
) -> dict[int, DistanceSpeedResult]:
    """
    Sums frame-to-frame displacement per track_id across the WHOLE match
    (not per-rally -- a player's between-point positioning/recovery is
    real movement too, and PRD Module 5's "distance covered" reads as a
    whole-match figure, the same way a real match stats overlay would
    report it).

    Returns an empty dict when `to_court_meters` is None -- see module
    docstring's DISTANCE_COVERED entry for why raw pixel distance has no
    meaningful fallback here, unlike Parts 7b/7c's threshold checks which
    can at least degrade to a resolution-dependent pixel version. There
    is no analogous "coarser but still meaningful" pixel version of a
    physical distance or speed.

    Only consecutive TEMPORAL entries for the same track_id contribute a
    displacement -- if a track_id drops out (occlusion, ByteTracker
    losing it, the object leaving frame) and either never returns or
    returns under a new track_id (a known ByteTracker limitation --see
    ml/tracking/byte_tracker.py's own module docstring), the gap simply
    contributes no distance rather than a straight-line guess across
    however long the player was untracked. That's a real undercount on a
    match with a lot of track churn, same shape as every other honest
    undercount in this module (ERRORS above) and pipeline (7a's ball
    activity gaps, 7d's OUTCOME_UNDETERMINED).
    """
    if to_court_meters is None:
        return {}

    # track_id -> ordered list of (frame_index, x_m, y_m, is_predicted)
    by_track: dict[int, list[tuple[int, float, float, bool]]] = {}
    for frame_index, points in enumerate(player_frames):
        for p in points:
            x_m, y_m = to_court_meters((p.x, p.y))
            by_track.setdefault(p.track_id, []).append((frame_index, x_m, y_m, p.is_predicted))

    results: dict[int, DistanceSpeedResult] = {}
    for track_id, entries in by_track.items():
        entries.sort(key=lambda e: e[0])
        total_distance_m = 0.0
        for (_, x1, y1, _), (_, x2, y2, _) in zip(entries, entries[1:]):
            total_distance_m += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        first_frame, last_frame = entries[0][0], entries[-1][0]
        elapsed_time_s = (last_frame - first_frame) / sample_fps if sample_fps > 0 else 0.0
        avg_speed_mps = (total_distance_m / elapsed_time_s) if elapsed_time_s > 0 else 0.0

        observed = sum(1 for e in entries if not e[3])
        predicted = sum(1 for e in entries if e[3])

        results[track_id] = DistanceSpeedResult(
            track_id=track_id,
            total_distance_m=total_distance_m,
            avg_speed_mps=avg_speed_mps,
            elapsed_time_s=elapsed_time_s,
            observed_frame_count=observed,
            predicted_frame_count=predicted,
        )

    return results


def compute_reaction_times(
    rallies: Sequence[RallySegment],
    shots: Sequence[Shot],
    *,
    sample_fps: float,
) -> dict[int, ReactionTimeResult]:
    """
    Per rally, walks its shots in frame order (7c's flat shot list,
    grouped and sorted here the same way
    ml.pipeline.highlight_tagging.detect_highlights already does before
    calling tag_fast_exchanges/tag_spectacular_saves) and, for each pair
    of consecutive shots hit by two DIFFERENT known track_ids, records
    the time gap as a reaction time credited to the RESPONDING player
    (the second shot's track_id). A shot with player_track_id=None (no
    player found within threshold -- see
    ml.pipeline.shot_classification.find_contact_points) can't be
    credited either way and is skipped, same "an unresolved contact isn't
    evidence" posture tag_spectacular_saves already takes for its own
    pairing.

    Deliberately whole-rally, not just smash responses like
    tag_spectacular_saves -- reaction time as a stat is about a player's
    general responsiveness across every exchange, not only the
    highlight-worthy fast ones.
    """
    shots_by_rally: dict[int, list[Shot]] = {}
    for shot in shots:
        shots_by_rally.setdefault(shot.rally_index, []).append(shot)
    for rally_shots in shots_by_rally.values():
        rally_shots.sort(key=lambda s: s.frame_index)

    totals: dict[int, list[float]] = {}
    for rally in rallies:
        rally_shots = shots_by_rally.get(rally.rally_index, [])
        for prev_shot, cur_shot in zip(rally_shots, rally_shots[1:]):
            if prev_shot.player_track_id is None or cur_shot.player_track_id is None:
                continue
            if cur_shot.player_track_id == prev_shot.player_track_id:
                continue
            gap_s = (cur_shot.frame_index - prev_shot.frame_index) / sample_fps if sample_fps > 0 else 0.0
            if gap_s < 0:
                continue
            totals.setdefault(cur_shot.player_track_id, []).append(gap_s)

    return {
        track_id: ReactionTimeResult(
            track_id=track_id,
            avg_reaction_time_s=sum(gaps) / len(gaps),
            response_count=len(gaps),
        )
        for track_id, gaps in totals.items()
    }


def compute_rally_ending_shot_success_rate(
    rallies: Sequence[RallySegment],
    shots: Sequence[Shot],
    outcomes: Sequence[PointOutcome],
    *,
    shot_type: str,
) -> dict[int, ShotSuccessResult]:
    """
    Backs both SMASH_SUCCESS_RATE (`shot_type=SHOT_TYPE_SMASH`) and
    NET_SUCCESS_RATE (`shot_type=SHOT_TYPE_VOLLEY`) -- one mechanism, see
    module docstring for why this particular success signal (rally-ending
    contact, not a hitting error) is the one this pipeline can actually
    attribute to a specific player's specific shot without reopening the
    winner/UE question ERRORS/WINNERS already settled above.

    Only rallies with a KNOWN outcome (`used_court_calibration=True` --
    an OUTCOME_UNDETERMINED result carries no evidence either way, same
    "an ambiguous signal isn't evidence" posture find_contact_points and
    identify_server both already take for their own unresolved cases) and
    a last shot of the requested `shot_type` with a known player_track_id
    contribute to either count. `attempt_count` is exactly how many such
    rally-ending shots were found, NOT how many shots of `shot_type` a
    player hit in total -- most smashes/volleys aren't a rally's final
    contact, and those never get a chance to contribute to this stat at
    all (see module docstring for the scope caveat).
    """
    shots_by_rally: dict[int, list[Shot]] = {}
    for shot in shots:
        shots_by_rally.setdefault(shot.rally_index, []).append(shot)
    for rally_shots in shots_by_rally.values():
        rally_shots.sort(key=lambda s: s.frame_index)

    outcome_by_rally: dict[int, PointOutcome] = {o.rally_index: o for o in outcomes}

    totals: dict[int, list[bool]] = {}
    for rally in rallies:
        rally_shots = shots_by_rally.get(rally.rally_index)
        if not rally_shots:
            continue
        last_shot = rally_shots[-1]
        if last_shot.shot_type != shot_type or last_shot.player_track_id is None:
            continue

        outcome = outcome_by_rally.get(rally.rally_index)
        if outcome is None or not outcome.used_court_calibration:
            continue

        was_success = outcome.outcome not in (OUTCOME_OUT_OF_BOUNDS, OUTCOME_NET)
        totals.setdefault(last_shot.player_track_id, []).append(was_success)

    return {
        track_id: ShotSuccessResult(
            track_id=track_id,
            shot_type=shot_type,
            success_count=sum(1 for s in successes if s),
            attempt_count=len(successes),
        )
        for track_id, successes in totals.items()
    }


def player_level_stat_values(
    rallies: Sequence[RallySegment],
    shots: Sequence[Shot],
    outcomes: Sequence[PointOutcome],
    player_frames: Sequence[list[TrackPoint]],
    *,
    sample_fps: float,
    to_court_meters: Callable[[tuple[float, float]], tuple[float, float]] | None,
) -> list[StatValue]:
    """
    Runs every track_id-keyed compute_* function above and flattens the
    result into one StatValue list, `track_id` populated throughout --
    the shape 9b will read back and relabel with real Player.id values.
    Distance/speed StatValues are simply absent when `to_court_meters` is
    None (see compute_distance_and_speed's own docstring), not zeroed
    out -- a missing StatValue and a genuine 0.0 mean different things,
    and this function never conflates the two.
    """
    values: list[StatValue] = []

    for track_id, result in compute_distance_and_speed(
        player_frames, sample_fps=sample_fps, to_court_meters=to_court_meters
    ).items():
        values.append(
            StatValue(stat_key=STAT_DISTANCE_COVERED, track_id=track_id, value=result.total_distance_m, sample_size=1)
        )
        values.append(
            StatValue(stat_key=STAT_MOVEMENT_SPEED_AVG, track_id=track_id, value=result.avg_speed_mps, sample_size=1)
        )

    for track_id, result in compute_reaction_times(rallies, shots, sample_fps=sample_fps).items():
        values.append(
            StatValue(
                stat_key=STAT_REACTION_TIME_AVG, track_id=track_id,
                value=result.avg_reaction_time_s, sample_size=result.response_count,
            )
        )

    for track_id, result in compute_rally_ending_shot_success_rate(
        rallies, shots, outcomes, shot_type=SHOT_TYPE_SMASH
    ).items():
        values.append(
            StatValue(
                stat_key=STAT_SMASH_SUCCESS_RATE, track_id=track_id,
                value=result.success_rate, sample_size=result.attempt_count,
            )
        )

    for track_id, result in compute_rally_ending_shot_success_rate(
        rallies, shots, outcomes, shot_type=SHOT_TYPE_VOLLEY
    ).items():
        values.append(
            StatValue(
                stat_key=STAT_NET_SUCCESS_RATE, track_id=track_id,
                value=result.success_rate, sample_size=result.attempt_count,
            )
        )

    return values


# --- diagnostics (not StatType-shaped -- explicitly requested alongside stat_types) ---


def summarize_match_diagnostics(
    shots: Sequence[Shot],
    serves: Sequence[ServeEvent],
) -> dict:
    """
    'Shot counts by type', 'serve identification rate', and 'serve counts
    by player' -- all called out explicitly as things Part 9 should
    compute, but none of them is a StatType the `statistics` table has a
    column for: shot_counts_by_type and serve_counts_by_player are both
    breakdowns (one number per shot_type / per track_id, not a single
    scalar), and serve_identification is a data-quality diagnostic about
    Part 7b's own confidence, not a fact about the match. Rather than
    force any of them into a mismatched StatValue, they're exposed here
    as their own dicts, straight off summarize_shots/summarize_serve_
    events/summarize_serve_events_by_track_id (Parts 7c/7b's own summary
    functions -- no new logic needed, just surfaced at this layer for
    whatever dashboard/report Module 5/7 build on top).

    serve_counts_by_player is keyed by track_id, same as every other
    STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY stat in this module -- 9b's
    identity mapping is what turns these track_ids into real Player.id
    values, same as it will for player_level_stat_values' output. It is
    deliberately a count, not a rate: see STAT_TYPE_STATUS['serve_
    percentage'] and summarize_serve_events_by_track_id's own docstring
    for why a per-player identification rate has no meaningful
    denominator to divide by.
    """
    return {
        "shot_counts_by_type": summarize_shots(shots),
        "serve_identification": summarize_serve_events(serves),
        "serve_counts_by_player": summarize_serve_events_by_track_id(serves),
    }


# --- serialization -------------------------------------------------------------


def to_serializable(values: Sequence[StatValue]) -> list[dict]:
    return [
        {
            "stat_key": v.stat_key,
            "track_id": v.track_id,
            "value": v.value,
            "sample_size": v.sample_size,
        }
        for v in values
    ]
