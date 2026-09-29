"""
Team-side attribution build, Tiers 1a-1d, 2a-2d.

Tier 1a — position-loading utility. Tier 1b — quadrant classification.
Tier 1c — per-track dominant quadrant. Tier 1d — track-fragment merging.
Tier 2a — quadrant-to-label mapping. Tier 2b — confidence-aware label
assignment. Tier 2c — track_id -> label lookup. Tier 2d — wiring this
into the real pipeline (app/services/team_side_labeling_stage.py); the
two small serialization helpers at the bottom of this file are 2d's
half of the pure-logic side, same split player_identity.py's own
to_serializable/summarize_side_assignments already established.

Highlights Improvement Roadmap Tier 3a
(backend/scripts/investigate_team_side_attribution.py) investigated
whether continuously converting a tracked player's foot position to
real-world court meters, via the Part 5c homography, is a reasonable
basis for team-side attribution. It was — but that script only ever
answered the question one rally window at a time, scanning tracks_data
inline, standalone, for a one-off diagnostic run.

This is the first piece of turning that investigation into a real,
reusable building block: given a track_id, return that track's full
sequence of real-world court positions across the ENTIRE match — every
frame it was tracked in, not just the frames inside one rally's
start_frame/end_frame window. Pure data plumbing, no side/flip
classification logic — that's a later tier's job, and it needs this
piece to be right first, since everything after it builds on top.

**Foot position, not bbox center.** Same reasoning Tier 3a's own module
docstring already spells out: a bounding box's bottom edge is where a
standing person's feet actually are, so it's the right proxy for "where
is this player standing on the court" — unlike
ml.pipeline.serve_detection.TrackPoint's bbox-center convention, which
exists for a different purpose (ball-to-player proximity, where the
center is an acceptable proxy for overall body position, not specifically
footing on the court). Deliberately NOT reusing that TrackPoint /
frames_from_tracks_json pair here for that reason: collapsing "the point
this module extracts" and "the point serve detection extracts" into one
shared helper would mean a future edit to one, for its own reasons,
silently changing the other's meaning.

**Real homography conversion, not Tier 3a's inlined stand-in.** Tier 3a's
own module docstring explains why IT inlines the perspective-transform
math: it's a standalone diagnostic script meant to run against nothing
but already-computed JSON on disk, with none of ensure_ml_importable()'s
dependency surface. This module has no such constraint — it's a real
ml/pipeline building block — so it calls
ml.detection.court_detector.pixel_to_court_point directly, the same
homography call point_outcome, serve_detection, and shot_classification
already all use for this exact conversion.

**Frame indexing.** frame_index is parsed from each frame's own
frame_path (frame_%06d.jpg — ml/common/frame_extraction.py), the same way
Tier 3a's own script derives it, rather than assumed from list position.
In practice frame extraction/tracking never drops or reorders frames, so
the two would agree here too — but that's an assumption this module
doesn't need to make just to get a frame number it can read directly off
the data instead.

Same "no Celery/DB/app dependency" layering as every ml/pipeline module:
a plain parsed tracks JSON dict and a homography array in, plain
dataclasses out.

**Tier 1b adds the other axis.** Tier 3a's own investigation only ever
split near/far — y against the net at court_length_m / 2. classify_quadrant
adds the left/right split Tier 3a never needed: x against the court's
width midpoint, court_width_m / 2. Deliberately kept as a small, isolated
function taking a bare (court_x, court_y) pair rather than a CourtPosition
— it has no reason to know about track_id, frame_index, or is_predicted,
and testing it needs nothing but coordinate pairs, no tracks JSON or
homography fixture at all. A later tier is what will call this once per
CourtPosition from load_track_positions' own output; this one only
answers "which quadrant is this point in", nothing about a whole track's
history across the match.

**Tier 1c is that later tier** — the direct 4-way extension of Tier 3a's
own 2-way "does this track stay on one side" check
(investigate_team_side_attribution.py's near_pct/far_pct/verdict logic),
now over all 4 quadrants instead of 2 halves. dominant_quadrant takes one
track's full position sequence (Tier 1a's own output), classifies every
point (Tier 1b), and reports the quadrant with the most frames plus what
fraction of frames actually agree with it — Tier 3a's own dominant_pct,
generalized from 2 buckets to 4.

**Predicted frames are excluded from the vote.** CourtPosition's own
docstring flagged this as the exact decision a later tier would need to
make; this is that tier making it, the same way
ml.pipeline.player_identity.assign_court_sides already excludes
ByteTrack's Kalman-predicted fill-in frames from its own near/far
majority vote — a predicted position is the tracker's best guess for a
frame with no real detection, not something actually observed, so it
shouldn't get a vote in "where did this track actually spend its time."

**Tier 1d turns quadrants into players.** ByteTrack mints a new track_id
whenever a player is lost (occlusion, leaves frame) and re-acquired after
`player_track_max_age` — see app/services/player_tracking_stage.py's own
`unique_track_count != expected_player_count` warning, which names this
exact failure mode ("a player leaving frame and re-entering fragments
into a new track"). One physical player can therefore own several
track_ids across a match. merge_track_fragments groups track_ids by their
Tier 1c dominant_quadrant — the same "each player owns one quadrant for
almost the whole match" assumption Tier 1c's own confidence field already
measures — on the theory that two fragments of the same player almost
always share a dominant quadrant, since neither fragment moved far enough
to change which quarter of the court it mostly occupied.

**Sharing a quadrant is necessary, not sufficient, for "same player."**
Two DIFFERENT players can share a dominant quadrant too — most plausibly
a mid-match side switch (real padel/tennis doubles switch ends between
sets), which this module has no signal to detect. So quadrant grouping
alone is a real but incomplete answer; merge_track_fragments adds one
more check before ever combining two track_ids: they must never be alive
at the same time. Two ByteTrack track_ids observed in overlapping frame
ranges are, by construction, two different physical bodies — one tracker
cannot emit two simultaneous IDs for one person — so an overlap is a hard
veto on merging, not a confidence penalty. This is the same "refuse
rather than guess" posture ml.pipeline.player_identity.py already takes
around teammate identity: where the signal genuinely can't distinguish
two players, this module leaves them distinct instead of picking one
arbitrarily.

Within a quadrant, grouping non-overlapping fragments into players is
exactly the classic interval-partitioning problem (the "minimum meeting
rooms" scheduling problem, run in reverse — here every "room" IS a
player, and the goal is the fewest rooms whose fragments still never
double-book). merge_track_fragments processes each quadrant's fragments
in start_frame order and, for each one, joins the eligible chain (one
whose last fragment ends before this one starts) with the LATEST
end_frame among the eligible chains, rather than the earliest-eligible
or a new chain — a fragment is more plausibly the same player who was
JUST in this quadrant than one who vanished from it much earlier and
never came back, so ties are broken toward temporal proximity, not
arbitrarily. A quadrant's fragments that never overlap end up in one
chain (one merged player); fragments that do overlap end up in separate
chains (an honest signal that this quadrant's grouping is ambiguous for
that pair — most likely a side switch, occasionally a genuine track_id
mixup — surfaced as more than one MergedPlayer sharing a quadrant rather
than silently forced together).

This module still doesn't decide whether the resulting count of merged
players is exactly `expected_player_count` (2 for singles, 4 for
doubles) — that comparison, and what to do about a mismatch, belongs to
whichever glue stage calls this, the same layering
player_tracking_stage.py already uses for its own raw track-count
warning.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

from ml.detection.court_detector import DEFAULT_COURT_LENGTH_M, DEFAULT_COURT_WIDTH_M, pixel_to_court_point

_FRAME_PATH_RE = re.compile(r"frame_(\d+)\.jpg$")

QUADRANT_NEAR_LEFT = "near_left"
QUADRANT_NEAR_RIGHT = "near_right"
QUADRANT_FAR_LEFT = "far_left"
QUADRANT_FAR_RIGHT = "far_right"

# Fixed iteration/tie-break order for dominant_quadrant: if two or more
# quadrants are exactly tied for the most real frames, the earliest one
# in this tuple wins, deterministically, the same way
# ml.pipeline.player_identity.assign_court_sides ties toward SIDE_A on
# a_count >= b_count rather than leaving a tie's outcome to dict/iteration
# order.
_QUADRANT_ORDER = (QUADRANT_NEAR_LEFT, QUADRANT_NEAR_RIGHT, QUADRANT_FAR_LEFT, QUADRANT_FAR_RIGHT)


class TrackPositionError(Exception):
    """Raised when a full-match position load is given input it can't reasonably work with."""


@dataclass(frozen=True)
class CourtPosition:
    """
    One track_id's real-world court position in one frame of the match.

    is_predicted mirrors ml.pipeline.serve_detection.TrackPoint's own
    field and the same time_since_update > 0 meaning (ByteTrack's
    Kalman-predicted fill-in for a frame with no real detection, not an
    actual observation). Exposed rather than filtered out here — "pure
    data plumbing" means a later tier decides whether predicted frames
    should count toward side/flip classification, the same way
    ml.pipeline.player_identity.assign_court_sides already makes that
    call explicitly rather than having it made for it upstream.
    """

    frame_index: int
    track_id: int
    court_x: float
    court_y: float
    is_predicted: bool


def _frame_index_from_path(frame_path: str) -> int:
    """Recovers the numeric frame index from a frame_%06d.jpg-style path (see ml/common/frame_extraction.py)."""
    match = _FRAME_PATH_RE.search(frame_path)
    if not match:
        raise TrackPositionError(f"could not parse a frame index out of {frame_path!r}")
    return int(match.group(1))


def load_track_positions(
    tracks_data: dict,
    homography,
    track_id: int,
) -> list[CourtPosition]:
    """
    Returns one track_id's full sequence of real-world court positions
    across the entire match — i.e. every frame in tracks_data["frames"]
    that track_id appears in, not just the frames inside one rally's
    start_frame/end_frame window. That's the one thing Tier 3a's own
    per-rally diagnostic loop never needed, and the reason this function
    exists rather than just reusing that loop as-is.

    `tracks_data` is a parsed player_tracks.json (the same shape
    ml.pipeline.serve_detection.frames_from_tracks_json and Tier 3a's own
    positions_by_frame both already parse). `homography` is Part 5c's 3x3
    pixel-to-court-meters matrix (calibration.json's "homography", already
    loaded into an ndarray by the caller — this function has no opinion
    on where that array came from).

    Frames the track_id doesn't appear in are simply skipped, not padded
    with a gap marker: a ByteTrack track_id can legitimately go
    unobserved for stretches (off frame, occluded), so this is a sequence
    of real (or Kalman-filled) observations, not a fixed-length timeline.

    Returned in ascending frame_index order regardless of the order
    tracks_data["frames"] lists them in.

    Raises TrackPositionError if track_id never appears anywhere in
    tracks_data — silently returning an empty list would be
    indistinguishable from "this track was tracked but never once
    observed", which can't actually happen (ByteTrack only ever creates a
    track_id off a real detection) and is far more likely to mean a
    typo'd or wrong track_id than a genuine zero-frame track.
    """
    positions: list[CourtPosition] = []

    for frame_entry in tracks_data.get("frames", []):
        frame_index = _frame_index_from_path(frame_entry["frame_path"])
        for t in frame_entry.get("tracks", []):
            if t["track_id"] != track_id:
                continue
            bbox = t["bbox"]
            foot_x = (bbox["x1"] + bbox["x2"]) / 2.0
            foot_y = bbox["y2"]  # bottom edge = feet, not bbox center — see module docstring
            court_x, court_y = pixel_to_court_point(homography, (foot_x, foot_y))
            positions.append(
                CourtPosition(
                    frame_index=frame_index,
                    track_id=track_id,
                    court_x=court_x,
                    court_y=court_y,
                    is_predicted=t.get("time_since_update", 0) > 0,
                )
            )

    if not positions:
        raise TrackPositionError(
            f"track_id={track_id} does not appear in any frame of this tracks data — "
            "check the track_id came from this same video's player_tracks.json"
        )

    positions.sort(key=lambda p: p.frame_index)
    return positions


def classify_quadrant(
    court_x: float,
    court_y: float,
    *,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
) -> str:
    """
    Classifies one real-world court point into one of 4 quadrants:
    near-left, near-right, far-left, far-right.

    "Near"/"far" is Tier 3a's own y-axis split, unchanged: y < net_y
    (court_length_m / 2) is near, everything else is far — see
    ml.detection.court_detector.compute_homography's own docstring for
    why y runs along the court's length with the net at the halfway
    point, not along x. "Left"/"right" is the new x-axis split this tier
    adds: x < court_width_m / 2 is left, everything else is right — the
    same halfway-point, half-open-interval convention, just on the other
    axis. "Left"/"right" here means whichever side of the width midpoint
    the calibrated homography's x=0 origin happens to be on (Part 5c's
    CourtCorners.top_left) — a real, stable court-relative axis, not a
    claim about camera-facing screen-left vs screen-right.

    A point exactly on either midpoint lands on the far/right side of
    that split, not because that's more "correct" than the other side,
    but so every point maps to exactly one quadrant with no gap and no
    double-count — the same half-open convention Tier 3a's own near/far
    split already uses (`"near" if y < net_y else "far"`).
    """
    net_y = court_length_m / 2.0
    mid_x = court_width_m / 2.0

    is_near = court_y < net_y
    is_left = court_x < mid_x

    if is_near and is_left:
        return QUADRANT_NEAR_LEFT
    if is_near and not is_left:
        return QUADRANT_NEAR_RIGHT
    if not is_near and is_left:
        return QUADRANT_FAR_LEFT
    return QUADRANT_FAR_RIGHT


@dataclass(frozen=True)
class DominantQuadrant:
    """
    One track_id's dominant quadrant across its full position sequence,
    plus how confidently it holds — the 4-way generalization of
    ml.pipeline.player_identity.TrackSideAssignment's own
    side/frames_observed/side_confidence shape, one level more granular.

    `confidence` is the fraction of this track's own REAL (non-predicted)
    frames that landed in `quadrant` — 1.0 means every real observation
    agreed, something meaningfully below 1.0 means the track spent real
    time in more than one quadrant (legitimate net play/poaching, or a
    track_id swapped between two players mid-match — see
    investigate_team_side_attribution.py's own module docstring on this
    same risk). Exposed rather than thresholded here so a caller decides
    its own "trust this" cutoff, same posture TrackSideAssignment already
    takes.
    """

    track_id: int
    quadrant: str
    frames_observed: int
    confidence: float


def dominant_quadrant(
    positions: Sequence[CourtPosition],
    *,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
) -> DominantQuadrant:
    """
    Classifies every position in one track's full sequence (Tier 1a's
    load_track_positions output) into a quadrant (Tier 1b's
    classify_quadrant), and reports whichever quadrant has the most real
    (non-predicted) frames, plus what fraction of real frames agree with
    it — the direct 4-way extension of Tier 3a's own 2-way near_pct/
    far_pct/verdict logic over a single track-rally pair, now over a
    track's entire match instead of one rally window, and over 4
    quadrants instead of 2 halves.

    `positions` must all share one track_id — mixing tracks here would
    silently answer a different, meaningless question ("which quadrant
    do these two unrelated players' positions land in together"), so
    this refuses rather than guessing which track_id the caller meant.

    Predicted frames (CourtPosition.is_predicted) are excluded from the
    vote entirely — see module docstring for why. If ties happen (two or
    more quadrants exactly tied for the most real frames), the earlier
    quadrant in near_left/near_right/far_left/far_right order wins,
    deterministically — an explicit, documented tie-break rather than
    leaving the outcome to dict ordering.

    Raises TrackPositionError if `positions` is empty, or if every
    position in it is predicted (zero real frames to vote with either
    way means there is nothing honest to report a dominant quadrant
    from).
    """
    if not positions:
        raise TrackPositionError("dominant_quadrant requires at least one position")

    track_ids = {p.track_id for p in positions}
    if len(track_ids) > 1:
        raise TrackPositionError(
            f"dominant_quadrant requires positions from a single track_id, got {sorted(track_ids)}"
        )
    track_id = positions[0].track_id

    counts = {quadrant: 0 for quadrant in _QUADRANT_ORDER}
    real_frame_count = 0
    for position in positions:
        if position.is_predicted:
            continue
        real_frame_count += 1
        quadrant = classify_quadrant(
            position.court_x, position.court_y, court_width_m=court_width_m, court_length_m=court_length_m
        )
        counts[quadrant] += 1

    if real_frame_count == 0:
        raise TrackPositionError(
            f"track_id={track_id} has no real (non-predicted) positions to compute a dominant quadrant from"
        )

    dominant = max(_QUADRANT_ORDER, key=lambda quadrant: counts[quadrant])
    return DominantQuadrant(
        track_id=track_id,
        quadrant=dominant,
        frames_observed=real_frame_count,
        confidence=counts[dominant] / real_frame_count,
    )


@dataclass(frozen=True)
class _TrackFragment:
    """
    Internal, module-private summary of one track_id's timeline — not a
    dataclass callers should build directly (they only ever get
    MergedPlayer back). Kept separate from CourtPosition/DominantQuadrant
    because merge_track_fragments needs one row per track_id, not one per
    frame or a raw per-track vote — this is the input to interval
    partitioning, nothing more.
    """

    track_id: int
    start_frame: int
    end_frame: int
    quadrant: str
    confidence: float
    frames_observed: int


@dataclass(frozen=True)
class MergedPlayer:
    """
    One inferred physical player — one or more ByteTrack track_ids that
    merge_track_fragments judged to be the same person's fragments, all
    sharing a dominant quadrant and never alive at the same time.

    `track_ids` is ordered by start_frame ascending (earliest fragment
    first). `confidence` is the MINIMUM of the merged fragments' own
    Tier 1c confidence — the weakest-link fragment, not an average, so
    one shaky fragment can't be hidden by averaging it against a
    confident one; a caller deciding whether to trust this player should
    see the worst case it's built on.
    """

    quadrant: str
    track_ids: tuple[int, ...]
    start_frame: int
    end_frame: int
    frames_observed: int
    confidence: float


def merge_track_fragments(
    tracks_data: dict,
    homography,
    *,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
) -> list[MergedPlayer]:
    """
    Groups every track_id in `tracks_data` into inferred physical
    players. See module docstring's Tier 1d section for the full
    reasoning; in short: group by Tier 1c dominant_quadrant, then within
    each quadrant merge fragments that never overlap in time, preferring
    to extend whichever eligible chain was most recently active.

    A track_id with no real (non-predicted) frames at all is skipped —
    dominant_quadrant has nothing honest to vote with for it (see its own
    docstring), and a fragment with zero real observations gives interval
    partitioning nothing meaningful to place either. This is a soft skip,
    same posture as player_tracking_stage.py's own count-mismatch
    warning: it's the caller's job to notice and warn if the resulting
    player count looks wrong, not this function's job to raise over it.

    Returns MergedPlayer entries sorted by start_frame ascending — NOT
    grouped by quadrant — so the first entry is always whichever player
    (of however many were found) first appears in the match.
    """
    track_ids = sorted(
        {
            t["track_id"]
            for frame_entry in tracks_data.get("frames", [])
            for t in frame_entry.get("tracks", [])
        }
    )

    fragments: list[_TrackFragment] = []
    for track_id in track_ids:
        positions = load_track_positions(tracks_data, homography, track_id)
        try:
            dq = dominant_quadrant(positions, court_width_m=court_width_m, court_length_m=court_length_m)
        except TrackPositionError:
            # Every position for this track_id was predicted -- nothing
            # real to classify or place in a timeline. See docstring.
            continue
        fragments.append(
            _TrackFragment(
                track_id=track_id,
                start_frame=positions[0].frame_index,
                end_frame=positions[-1].frame_index,
                quadrant=dq.quadrant,
                confidence=dq.confidence,
                frames_observed=dq.frames_observed,
            )
        )

    fragments_by_quadrant: dict[str, list[_TrackFragment]] = defaultdict(list)
    for fragment in fragments:
        fragments_by_quadrant[fragment.quadrant].append(fragment)

    merged: list[MergedPlayer] = []
    for quadrant, quadrant_fragments in fragments_by_quadrant.items():
        # Classic interval-partitioning greedy: process in start_frame
        # order, extend the latest-ending eligible chain, else open a new
        # one. See module docstring for why "latest eligible" over
        # "earliest eligible" or "always new".
        chains: list[list[_TrackFragment]] = []
        for fragment in sorted(quadrant_fragments, key=lambda f: f.start_frame):
            eligible = [chain for chain in chains if chain[-1].end_frame < fragment.start_frame]
            if eligible:
                best_chain = max(eligible, key=lambda chain: chain[-1].end_frame)
                best_chain.append(fragment)
            else:
                chains.append([fragment])

        for chain in chains:
            merged.append(
                MergedPlayer(
                    quadrant=quadrant,
                    track_ids=tuple(f.track_id for f in chain),
                    start_frame=chain[0].start_frame,
                    end_frame=chain[-1].end_frame,
                    frames_observed=sum(f.frames_observed for f in chain),
                    confidence=min(f.confidence for f in chain),
                )
            )

    merged.sort(key=lambda player: player.start_frame)
    return merged


# --- Tier 2a: quadrant -> label -----------------------------------------
#
# A fixed, documented convention, not a computed result -- reading order,
# near side first then far side, left before right within each, same as
# reading two lines of text: near_left=A, near_right=B, far_left=C,
# far_right=D. Any fixed order would work equally well; this one was
# picked because "near side first" matches which players a viewer looking
# at a single broadcast-style frame sees closest to camera first.
#
# This labels a QUADRANT, not a validated identity -- it says nothing
# about whether the MergedPlayer(s) occupying that quadrant are really
# one consistent physical player for the whole match (that's Tier 1d's
# job, and Tier 1d's own docstring already covers the case where a
# quadrant ends up with more than one MergedPlayer in it). A caller
# rendering "Player A" onto a clip is implicitly trusting Tier 1d's
# result for that quadrant, not just this mapping.
QUADRANT_LABELS: dict[str, str] = {
    QUADRANT_NEAR_LEFT: "Player A",
    QUADRANT_NEAR_RIGHT: "Player B",
    QUADRANT_FAR_LEFT: "Player C",
    QUADRANT_FAR_RIGHT: "Player D",
}


def quadrant_to_label(quadrant: str) -> str:
    """
    Maps one of the four QUADRANT_* constants to its fixed display label
    ("Player A"-"Player D"). See QUADRANT_LABELS above for the convention
    and its reasoning.

    Raises ValueError on anything not in QUADRANT_LABELS -- same "fail
    loud, don't guess" posture as classify_quadrant's own unreachable
    final-else (see its docstring): a typo'd or unrecognized quadrant
    string is a bug upstream, not a case worth silently mapping to
    nothing or to an arbitrary default label.
    """
    try:
        return QUADRANT_LABELS[quadrant]
    except KeyError:
        raise ValueError(
            f"unrecognized quadrant {quadrant!r}; expected one of {sorted(QUADRANT_LABELS)}"
        ) from None


# --- Tier 2b: confidence-aware label assignment -------------------------
#
# Tier 1c's confidence is "fraction of this track's real frames that
# agreed with its dominant quadrant" -- with 4 quadrants, the max() in
# dominant_quadrant only needs a PLURALITY to win, which can be as low as
# just over 0.25. A plurality winner at, say, 0.3 means over two-thirds
# of this track's observed time was spent somewhere else, split across
# the other three quadrants -- printing "Player A" on that is presenting
# a near-coin-flip as a fact. 0.5 is therefore the natural, principled
# line: it's the one threshold that means what "confident" should mean
# here -- the reported quadrant actually held an outright MAJORITY of
# real frames, not just an edge over its closest competitor.
#
# MergedPlayer.confidence (Tier 1d) is the MINIMUM confidence across a
# player's merged fragments -- so gating on it here also protects against
# a player whose fragments look fine on average but are actually carried
# by one shaky fragment; that fragment alone is enough to drop the whole
# player below the threshold.
DEFAULT_LABEL_CONFIDENCE_THRESHOLD = 0.5

NEAR_TEAM_LABEL = "Near Team"
FAR_TEAM_LABEL = "Far Team"

# Same two keys as QUADRANT_LABELS, deliberately -- assign_player_label
# depends on that (see its docstring) to validate both branches with one
# lookup instead of duplicating the "unrecognized quadrant" check.
QUADRANT_TEAM_LABELS: dict[str, str] = {
    QUADRANT_NEAR_LEFT: NEAR_TEAM_LABEL,
    QUADRANT_NEAR_RIGHT: NEAR_TEAM_LABEL,
    QUADRANT_FAR_LEFT: FAR_TEAM_LABEL,
    QUADRANT_FAR_RIGHT: FAR_TEAM_LABEL,
}


def assign_player_label(
    player: MergedPlayer,
    *,
    confidence_threshold: float = DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
) -> str:
    """
    The honest-degradation half of Tier 2: at or above
    confidence_threshold (inclusive -- exactly 0.5 counts as confident,
    since that IS an outright majority, not a near-miss), returns the
    individual Tier 2a label ("Player A"-"Player D"). Below it, falls
    back to a team-level label ("Near Team"/"Far Team") instead of
    printing an individual name the data doesn't actually support.

    quadrant_to_label does double duty here: it's both the "unrecognized
    quadrant" validation this function needs regardless of which branch
    it takes, AND the value the confident branch returns -- so an invalid
    player.quadrant fails loudly the same way on either branch, rather
    than only being checked on the path that happens to use it directly.

    This never fails "softly" by returning None or an empty label for a
    low-confidence player -- a team-level label is still a real, correct
    answer (near_left/near_right's shared team-level fact doesn't become
    less true just because which individual is which became uncertain),
    so there's no honest reason to return nothing at all.
    """
    individual_label = quadrant_to_label(player.quadrant)
    if player.confidence >= confidence_threshold:
        return individual_label
    return QUADRANT_TEAM_LABELS[player.quadrant]


# --- Tier 2c: track_id -> label lookup -----------------------------------
#
# Everything from Tier 1a through 2b builds a picture in terms of
# MergedPlayers -- but the thing every OTHER stage actually has on hand,
# frame by frame, is a raw ByteTrack track_id (see e.g.
# ml.pipeline.serve_detection.TrackPoint, or any tracks_data["frames"]
# entry). Phase 3 (attributing individual highlights) will be doing
# exactly that: "this clip's action involved track_id 17 -- whose
# highlight is this." A dict is that lookup; nothing about the question
# "what label does this track_id resolve to" needs more machinery than
# one hash lookup can already give it.
def build_track_label_lookup(
    players: Sequence[MergedPlayer],
    *,
    confidence_threshold: float = DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
) -> dict[int, str]:
    """
    Flattens a whole match's MergedPlayers (Tier 1d) into a single
    {track_id: label} dict, computing each player's label via Tier 2b's
    assign_player_label as it goes -- so a track_id belonging to a
    low-confidence player resolves to that player's team-level fallback,
    not an individual label, exactly as 2b would report it standalone.
    confidence_threshold passes straight through to assign_player_label;
    same default, same meaning.

    Every track_id in every player.track_ids ends up as one key. This
    should never collide -- merge_track_fragments (Tier 1d) partitions
    every real track_id into exactly one chain, so no track_id should
    legitimately belong to two MergedPlayers -- but this raises ValueError
    on a collision rather than letting the second player's label silently
    overwrite the first's. A silent overwrite here wouldn't fail loudly
    anywhere downstream; it would just make Phase 3 quietly attribute a
    highlight to the wrong player, which is exactly the failure mode this
    whole Tier exists to avoid.

    A track_id that never appears in `players` at all (Tier 1d's own
    skip case: a track_id with zero real, non-predicted positions -- see
    merge_track_fragments's docstring) is simply absent from the
    returned dict, not mapped to some placeholder value. Looking it up
    raises the ordinary KeyError; there's no honest label to hand back
    for a track this module never classified in the first place, so this
    doesn't invent one.
    """
    lookup: dict[int, str] = {}
    for player in players:
        label = assign_player_label(player, confidence_threshold=confidence_threshold)
        for track_id in player.track_ids:
            if track_id in lookup:
                raise ValueError(
                    f"track_id={track_id} appears in more than one MergedPlayer "
                    f"(already resolved to {lookup[track_id]!r}, now also in a player labeled {label!r})"
                )
            lookup[track_id] = label
    return lookup


# --- Tier 2d: serialization for persistence -----------------------------

def to_serializable(players: Sequence[MergedPlayer]) -> list[dict]:
    """
    Flattens MergedPlayers (Tier 1d) to plain dicts for JSON persistence —
    same "raw data alongside a lookup/summary" shape
    app/services/team_side_labeling_stage.py (Tier 2d) persists, same
    field-by-field style as player_identity.py's own to_serializable.

    Deliberately does NOT include a "label" field: a label depends on a
    confidence_threshold choice (Tier 2b), and baking one particular
    threshold's answer into the raw player record would hide that choice
    from whoever reads this back later. The persisted artifact's
    "track_labels" lookup (Tier 2c's build_track_label_lookup output) is
    where the label answer actually belongs.
    """
    return [
        {
            "quadrant": p.quadrant,
            "track_ids": list(p.track_ids),
            "start_frame": p.start_frame,
            "end_frame": p.end_frame,
            "frames_observed": p.frames_observed,
            "confidence": p.confidence,
        }
        for p in players
    ]


def summarize_track_label_lookup(lookup: dict) -> dict:
    """
    Same "honest diagnostic alongside the raw result" pattern as
    player_identity.py's summarize_side_assignments: a high
    team_fallback_count relative to track_count is a real signal this
    video's quadrant assignments were mostly low-confidence (frequent
    ID fragmentation Tier 1d couldn't cleanly merge, a camera angle
    quadrant classification struggles with) — worth surfacing the same
    way avg_side_confidence already is elsewhere, not just buried in
    individual MergedPlayer.confidence values a caller would have to go
    looking for.
    """
    total = len(lookup)
    if total == 0:
        return {"track_count": 0, "individual_label_count": 0, "team_fallback_count": 0}
    individual_count = sum(1 for label in lookup.values() if label in QUADRANT_LABELS.values())
    return {
        "track_count": total,
        "individual_label_count": individual_count,
        "team_fallback_count": total - individual_count,
    }
