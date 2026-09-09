"""
Shot/stroke classification — Part 7c, the third piece of pipeline stage 6
(action recognition).

Part 7a says *when* a point was being played (RallySegment). Part 7b says
*who* started it (ServeEvent). This module is the first to ask *what
happened during it*: which player hit the ball, when, and what kind of
shot it plausibly was (smash / volley / lob / groundstroke).

**Why this doesn't reach for pose estimation.** The task this module
nominally serves — "classify smash vs. volley vs. groundstroke from a
player's swing" — describes what a coach watching film would look at:
racket-arm angle, backswing, contact height *relative to the body*,
follow-through. A real version of that needs a pose-estimation model
feeding a swing classifier, and, same as rally boundary detection (Part
7a) and serve identification (Part 7b) before it, no such model exists
anywhere in this codebase yet (PRD Section 13 lists this as future work).
Bringing in a pose model for exactly one sub-part, with no shared
infrastructure and no labeled swing data to train a classifier on top of
it, would be a much larger and riskier addition than this module's job
justifies on its own — so, same "pretrained-first, best-available-proxy"
posture as every Part 5-7 module before it, this uses signals Parts 5-6
already computed: ball trajectory (byte_tracker.py + ball_interpolation.py)
and player bounding boxes (byte_tracker.py), nothing new.

**The proxy, concretely.** A "contact" is a frame where the ball's
vertical motion reverses direction near a player — the same closest-
approach idea Part 7b's identify_server uses for the serve specifically,
generalized to every such moment in a rally, not just its first one (see
find_contact_points). Each contact is then classified using two
bbox-derived signals, neither of which needs a single new model weight:

  1. **Contact height, relative to the contacting player's own bounding
     box** (not an absolute pixel/meter height, which would conflate a
     tall player's shoulder with a short player's overhead — see
     contact_height_ratio's docstring). This is the primary signal for
     "hit at or above head height" (smash) vs. everything else, and is
     unaffected by whether court calibration exists for the video.

  2. **How long the ball stays airborne before the next contact or the
     rally's end** (see _airborne_duration_frames). A lob's whole point
     is a long, high, slow arc deep into the opponent's court — it's
     airborne far longer than a fast exchange — so unusually long
     airtime is used as the lob signal, in place of the "high, slow arc"
     a pose/trajectory-shape model would otherwise need to recognize
     directly.

  3. **Distance from the net, in court meters — only when Part 5e
     produced a calibration for this video** (see module-level note on
     graceful degradation below). Distinguishes volley (contact made
     near the net) from groundstroke (contact made from further back),
     the padel/tennis textbook definition of the difference. Without
     calibration this distinction genuinely can't be made from pixels
     alone (a "close to the net" pixel distance means something
     different at every camera zoom/angle — same reasoning
     ml/pipeline/serve_detection.py's own module docstring gives for
     preferring calibrated meters), so a contact that isn't a smash or a
     lob and has no calibration to check is reported as UNKNOWN rather
     than guessing — same "an ambiguous signal teaches us nothing, don't
     force a classification" posture ServeEvent.identified=False and
     ball_interpolation.py's own gap cap both already take.

**Known limitation, stated plainly rather than hidden behind a false
positive rate:** padel has shots (bandeja, víbora, chiquita) with genuine
technique differences a real swing/pose classifier could tell apart, that
this proxy cannot — they'd all fall into whichever of the four buckets
above best matches their contact height/net distance/airtime, which is
not the same thing as being correctly identified. This module's SHOT_TYPE_*
buckets are a coarse, defensible first pass on top of the position data
that already exists, not a claim of stroke-technique accuracy — a real
pose-based classifier (PRD Section 13) remains the more direct fix,
exactly the same relationship Part 7a's docstring describes between its
own ball-activity proxy and a future swing/serve-motion classifier.

Same "no Celery/DB/app dependency" layering as every ml/ module before
it: plain per-frame position lists in, dataclasses out. Fed by
app/services/shot_classification_stage.py (Part 7c's glue layer), which
knows how to turn persisted ball_tracks.json/player_tracks.json/
rallies.json back into these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

SHOT_TYPE_SMASH = "smash"
SHOT_TYPE_LOB = "lob"
SHOT_TYPE_VOLLEY = "volley"
SHOT_TYPE_GROUNDSTROKE = "groundstroke"
SHOT_TYPE_UNKNOWN = "unknown"

# See module docstring point 1. contact_height_ratio = (player's bbox
# bottom - ball's y) / player's bbox height: ~0 means the ball was hit at
# the player's feet, ~1 near the top of their bbox (roughly head height),
# and above 1 means above their head. 1.05 is deliberately just above 1.0
# rather than exactly at it — a player's bbox top sits at the top of their
# head, and a genuine overhead contact happens with an extended arm above
# that, but bbox noise (a frame or two of imprecise player detection)
# means "exactly at the bbox top" is too strict a cutoff in practice.
DEFAULT_SMASH_HEIGHT_RATIO = 1.05

# See module docstring point 2. At the default frame_sample_rate_fps of
# 5.0, 6 frames is ~1.2s of continuous flight — long enough that a fast,
# flat rally exchange (typically airborne under half a second between
# contacts at this sample rate) won't cross it, short enough to still
# catch a genuine lob's much longer hang time before the point's next
# contact or the rally simply ends.
DEFAULT_LOB_MIN_AIRBORNE_FRAMES = 6

# See module docstring point 3. Same padel-court-width reasoning as
# ml/pipeline/serve_detection.py's DEFAULT_MAX_BALL_PLAYER_DISTANCE_M:
# a 20m-long padel court's net sits at the halfway line, and 3.0m of
# court depth on either side of it is comfortably "at the net" without
# reaching into what's clearly mid-court positioning.
DEFAULT_NET_PROXIMITY_M = 3.0

# Same reasoning, and same default, as ml/pipeline/serve_detection.py's
# DEFAULT_MAX_BALL_PLAYER_DISTANCE_M — a contact more than this far from
# any player isn't a trustworthy pairing, most likely a mis-tracked ball
# point or a player who isn't actually the one who made contact.
DEFAULT_MAX_CONTACT_PLAYER_DISTANCE_M = 2.5
DEFAULT_MAX_CONTACT_PLAYER_DISTANCE_PX = 150.0

# See module docstring point 3. Real-world court length, used only to
# locate the net line (at half the court's length) when converting a
# contact's ball position to court meters — same default as
# ml.detection.court_detector.DEFAULT_COURT_LENGTH_M, kept as a local
# constant rather than importing that module, since this stays
# importable/testable without any ml.detection dependency (same "no
# Celery/DB/app dependency" posture the module docstring already commits
# to). A caller with a real per-video calibration passes its actual
# court_length_m through instead of relying on this default.
DEFAULT_COURT_LENGTH_M = 20.0


class ShotClassificationError(Exception):
    """Raised when shot classification is given input it can't reasonably reason about."""


@dataclass(frozen=True)
class BallPoint:
    """
    One frame's ball center point plus enough context to find and reason
    about contacts: whether the point was a real detection or an
    interpolated one (ml/tracking/ball_interpolation.py — a genuine
    direction reversal is a real physical event, an interpolated straight
    line can't produce one, so contact-finding needs to tell the two
    apart) and the frame's global index, since a rally's window into
    these lists is expressed in the same frame numbering
    RallySegment.start_frame/end_frame already uses.
    """

    frame_index: int
    x: float
    y: float
    is_predicted: bool


@dataclass(frozen=True)
class PlayerBox:
    """
    One player's tracked bounding box in one frame — deliberately kept as
    a full box (unlike ml/pipeline/serve_detection.TrackPoint's
    center-only shape), because contact_height_ratio specifically needs
    the box's vertical extent, not just its center.
    """

    frame_index: int
    track_id: int
    x: float
    y: float
    top: float
    bottom: float


@dataclass(frozen=True)
class ContactPoint:
    """
    One candidate moment of racket-ball contact: a local reversal in the
    ball's vertical motion, paired with whichever tracked player was
    closest to it at that frame. `player_track_id`/`contact_height_ratio`
    are None when no player was found within the configured distance
    threshold — see find_contact_points's docstring for why such a
    contact is still kept (not discarded) at this stage.

    `ball_court_y_m` is the ball's own position along the court's length
    axis (0 at one baseline, court_length_m at the other — see
    ml.detection.court_detector.compute_homography's convention), only
    ever set when a calibration was available; it's deliberately NOT the
    same thing as distance_m (the ball-to-player pairing distance) —
    classify_shot uses ball_court_y_m to measure distance to the *net*,
    a court-position question, and distance_m only to judge whether the
    player pairing itself is trustworthy.
    """

    frame_index: int
    ball_x: float
    ball_y: float
    player_track_id: int | None
    contact_height_ratio: float | None
    distance_m: float | None
    distance_px: float | None
    used_court_calibration: bool
    ball_court_y_m: float | None = None


@dataclass(frozen=True)
class Shot:
    """
    One classified shot. `shot_type` is always one of the SHOT_TYPE_*
    constants, including SHOT_TYPE_UNKNOWN — same "always emit a result,
    let identified/shot_type carry the uncertainty" posture as
    ml.pipeline.serve_detection.ServeEvent, rather than silently dropping
    contacts a rule can't confidently resolve.
    """

    rally_index: int
    frame_index: int
    player_track_id: int | None
    shot_type: str
    contact_height_ratio: float | None
    airborne_frames_after: int | None
    distance_from_net_m: float | None
    used_court_calibration: bool


def contact_height_ratio(ball_y: float, player: PlayerBox) -> float | None:
    """
    (player.bottom - ball_y) / player's bbox height. None when the
    player's box has no usable height (a malformed/zero-height box —
    shouldn't happen from real detections, but this stays a None rather
    than raising or dividing by zero for a stray bad input).
    """
    height = player.bottom - player.top
    if height <= 0:
        return None
    return (player.bottom - ball_y) / height


def _distance(ax: float, ay: float, bx: float, by: float, to_court_meters) -> tuple[float, bool]:
    """Returns (distance, is_meters) — meters via to_court_meters when given, raw pixels otherwise."""
    if to_court_meters is not None:
        axm, aym = to_court_meters((ax, ay))
        bxm, bym = to_court_meters((bx, by))
        return ((axm - bxm) ** 2 + (aym - bym) ** 2) ** 0.5, True
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5, False


def _closest_player(
    ball: BallPoint,
    players: Sequence[PlayerBox],
    *,
    max_distance_m: float,
    max_distance_px: float,
    to_court_meters,
) -> tuple[PlayerBox | None, float | None, float | None, bool]:
    """Returns (player, distance_m, distance_px, used_court_calibration) for the closest player within threshold, or all-None/False if none qualifies."""
    best_player: PlayerBox | None = None
    best_distance: float | None = None
    best_is_meters = to_court_meters is not None

    for player in players:
        distance, is_meters = _distance(ball.x, ball.y, player.x, player.y, to_court_meters)
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_player = player

    if best_distance is None:
        return None, None, None, best_is_meters

    threshold = max_distance_m if best_is_meters else max_distance_px
    if best_distance > threshold:
        return None, None, None, best_is_meters

    return (
        best_player,
        best_distance if best_is_meters else None,
        best_distance if not best_is_meters else None,
        best_is_meters,
    )


def find_contact_points(
    rally,  # ml.pipeline.rally_detection.RallySegment -- not type-hinted directly, same reason as serve_detection.identify_server
    ball_points: Sequence[BallPoint],
    player_frames: Sequence[list[PlayerBox]],
    *,
    max_distance_m: float = DEFAULT_MAX_CONTACT_PLAYER_DISTANCE_M,
    max_distance_px: float = DEFAULT_MAX_CONTACT_PLAYER_DISTANCE_PX,
    to_court_meters=None,
) -> list[ContactPoint]:
    """
    Scans `rally`'s frame window in `ball_points` (a full-match, frame-
    indexed sequence — same indexing convention as
    ml.pipeline.rally_detection/serve_detection's frame lists) for local
    reversals in the ball's vertical direction: a frame where y was
    falling and starts rising, or rising and starts falling, immediately
    after. That reversal is the observable signature of *something*
    changing the ball's motion — a racket, the ground, a wall — and
    within an active rally window, a racket contact is overwhelmingly the
    likely cause.

    Only ever looks at REAL ball detections (is_predicted False) for the
    reversal test itself — an interpolated point is a straight line by
    construction (see ml.tracking.ball_interpolation.interpolate_ball_gaps)
    and can never show a direction reversal, so including them would only
    ever add noise, never a real contact. Consecutive real detections are
    compared directly, skipping over any interpolated points between them,
    so a contact that happens to fall inside a bridged gap is still
    findable from the real detections bracketing it.

    A found reversal frame with no player within the distance threshold
    is still returned as a ContactPoint (player_track_id=None, height/
    distance fields None) rather than dropped — same "always emit,
    unresolved fields carry the uncertainty" posture as this module's own
    Shot/ContactPoint types elsewhere; classify_shot then reports such a
    contact as SHOT_TYPE_UNKNOWN.
    """
    window = [p for p in ball_points if rally.start_frame <= p.frame_index <= rally.end_frame]
    real_points = [p for p in window if not p.is_predicted]
    real_points.sort(key=lambda p: p.frame_index)

    contacts: list[ContactPoint] = []
    for i in range(1, len(real_points) - 1):
        prev_p, cur_p, next_p = real_points[i - 1], real_points[i], real_points[i + 1]
        delta_before = cur_p.y - prev_p.y
        delta_after = next_p.y - cur_p.y
        if delta_before == 0 or delta_after == 0:
            continue
        reversed_direction = (delta_before > 0) != (delta_after > 0)
        if not reversed_direction:
            continue

        players_here = [pl for pl in player_frames[cur_p.frame_index]] if cur_p.frame_index < len(player_frames) else []
        player, dist_m, dist_px, used_calibration = _closest_player(
            cur_p, players_here,
            max_distance_m=max_distance_m, max_distance_px=max_distance_px,
            to_court_meters=to_court_meters,
        )
        height_ratio = contact_height_ratio(cur_p.y, player) if player is not None else None
        ball_court_y_m = to_court_meters((cur_p.x, cur_p.y))[1] if to_court_meters is not None else None

        contacts.append(
            ContactPoint(
                frame_index=cur_p.frame_index,
                ball_x=cur_p.x,
                ball_y=cur_p.y,
                player_track_id=player.track_id if player is not None else None,
                contact_height_ratio=height_ratio,
                distance_m=dist_m,
                distance_px=dist_px,
                used_court_calibration=used_calibration,
                ball_court_y_m=ball_court_y_m,
            )
        )
    return contacts


def _airborne_duration_frames(contact: ContactPoint, next_contact_frame_index: int | None, rally_end_frame: int) -> int:
    """Frames from this contact to whichever comes first: the next contact, or the rally's end."""
    end = next_contact_frame_index if next_contact_frame_index is not None else rally_end_frame
    return max(0, end - contact.frame_index)


def classify_shot(
    contact: ContactPoint,
    *,
    rally_index: int,
    airborne_frames: int,
    smash_height_ratio: float = DEFAULT_SMASH_HEIGHT_RATIO,
    lob_min_airborne_frames: int = DEFAULT_LOB_MIN_AIRBORNE_FRAMES,
    net_proximity_m: float = DEFAULT_NET_PROXIMITY_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
) -> Shot:
    """
    Applies the three-signal rule set from the module docstring, in
    priority order:

      1. contact_height_ratio >= smash_height_ratio -> SMASH. Checked
         first because an overhead contact is the least ambiguous signal
         this module has (unaffected by calibration availability), and a
         legitimate smash can also happen close to the net or with a long
         airborne follow-through, which would otherwise be misread as a
         volley or lob by the later rules.
      2. airborne_frames >= lob_min_airborne_frames -> LOB. Checked next:
         a lob's defining trait is time in the air, regardless of exactly
         where on court it was hit from.
      3. used_court_calibration and the ball's own distance to the net
         line (|ball_court_y_m - court_length_m / 2|, NOT the ball-to-
         player pairing distance — see ContactPoint's docstring) <=
         net_proximity_m -> VOLLEY.
      4. used_court_calibration (but not close enough to the net) ->
         GROUNDSTROKE.
      5. Otherwise (no player found at all, or no calibration and neither
         of the calibration-independent rules 1-2 fired) -> UNKNOWN,
         rather than guessing at the volley/groundstroke split with no
         reliable signal for it — see module docstring point 3.
    """
    height_ratio = contact.contact_height_ratio
    distance_from_net_m: float | None = None
    if contact.used_court_calibration and contact.ball_court_y_m is not None:
        distance_from_net_m = abs(contact.ball_court_y_m - court_length_m / 2)

    if height_ratio is not None and height_ratio >= smash_height_ratio:
        shot_type = SHOT_TYPE_SMASH
    elif airborne_frames >= lob_min_airborne_frames:
        shot_type = SHOT_TYPE_LOB
    elif distance_from_net_m is not None:
        shot_type = SHOT_TYPE_VOLLEY if distance_from_net_m <= net_proximity_m else SHOT_TYPE_GROUNDSTROKE
    else:
        shot_type = SHOT_TYPE_UNKNOWN

    return Shot(
        rally_index=rally_index,
        frame_index=contact.frame_index,
        player_track_id=contact.player_track_id,
        shot_type=shot_type,
        contact_height_ratio=height_ratio,
        airborne_frames_after=airborne_frames,
        distance_from_net_m=distance_from_net_m,
        used_court_calibration=contact.used_court_calibration,
    )


def detect_shots(
    rallies: Sequence,  # Sequence[RallySegment]
    ball_points: Sequence[BallPoint],
    player_frames: Sequence[list[PlayerBox]],
    *,
    max_distance_m: float = DEFAULT_MAX_CONTACT_PLAYER_DISTANCE_M,
    max_distance_px: float = DEFAULT_MAX_CONTACT_PLAYER_DISTANCE_PX,
    smash_height_ratio: float = DEFAULT_SMASH_HEIGHT_RATIO,
    lob_min_airborne_frames: int = DEFAULT_LOB_MIN_AIRBORNE_FRAMES,
    net_proximity_m: float = DEFAULT_NET_PROXIMITY_M,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
    to_court_meters=None,
) -> list[Shot]:
    """
    One Shot per contact point found in each rally, in chronological
    order across the whole match (not grouped/nested by rally — a flat
    list, same shape as detect_serves' flat list of one-per-rally
    ServeEvents, just with a variable count per rally instead of exactly
    one). A rally with zero found contacts (e.g. too short a real-
    detection run for any reversal to be findable) simply contributes no
    Shots — unlike ServeEvent, there's no fixed "one per rally" contract
    here to keep, so nothing is fabricated to fill a gap.

    `court_length_m` should be the SAME value the video's actual court
    calibration was computed with (see app/services/shot_classification_stage.py
    for how it reads that back from the persisted calibration JSON rather
    than assuming the default) — it's meaningless to net-proximity-check
    against half of a court length that doesn't match the homography
    `to_court_meters` itself was built from.
    """
    shots: list[Shot] = []
    for rally in rallies:
        contacts = find_contact_points(
            rally, ball_points, player_frames,
            max_distance_m=max_distance_m, max_distance_px=max_distance_px,
            to_court_meters=to_court_meters,
        )
        for i, contact in enumerate(contacts):
            next_frame = contacts[i + 1].frame_index if i + 1 < len(contacts) else None
            airborne = _airborne_duration_frames(contact, next_frame, rally.end_frame)
            shots.append(
                classify_shot(
                    contact,
                    rally_index=rally.rally_index,
                    airborne_frames=airborne,
                    smash_height_ratio=smash_height_ratio,
                    lob_min_airborne_frames=lob_min_airborne_frames,
                    net_proximity_m=net_proximity_m,
                    court_length_m=court_length_m,
                )
            )
    return shots


def ball_points_from_tracks_json(data: dict) -> list[BallPoint]:
    """
    Turns a parsed ball_tracks.json (ml.tracking.byte_tracker.to_serializable's
    output, as written by app/services/ball_tracking_stage.py) into one
    BallPoint per frame per ball track. Frames with zero or multiple ball
    tracks contribute nothing here — find_contact_points already only
    trusts a frame with exactly the ball activity it can reason about,
    same "ambiguous frame, skip it" posture ml.pipeline.serve_detection's
    identify_server takes for its own ball_points lookup.
    """
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


def player_frames_from_tracks_json(data: dict) -> list[list[PlayerBox]]:
    """
    Turns a parsed player_tracks.json into one list of PlayerBoxes per
    frame, indexed by position — the same indexing convention
    ball_points_from_tracks_json and rally_detection/serve_detection's
    equivalents already use, since all three tracks files share one
    underlying frame ordering (Part 5a's frame extraction).
    """
    frames: list[list[PlayerBox]] = []
    for frame_index, frame_entry in enumerate(data.get("frames", [])):
        boxes = []
        for t in frame_entry.get("tracks", []):
            bbox = t["bbox"]
            boxes.append(
                PlayerBox(
                    frame_index=frame_index,
                    track_id=t["track_id"],
                    x=(bbox["x1"] + bbox["x2"]) / 2,
                    y=(bbox["y1"] + bbox["y2"]) / 2,
                    top=bbox["y1"],
                    bottom=bbox["y2"],
                )
            )
        frames.append(boxes)
    return frames


def to_serializable(shots: Sequence[Shot]) -> list[dict]:
    return [
        {
            "rally_index": s.rally_index,
            "frame_index": s.frame_index,
            "player_track_id": s.player_track_id,
            "shot_type": s.shot_type,
            "contact_height_ratio": s.contact_height_ratio,
            "airborne_frames_after": s.airborne_frames_after,
            "distance_from_net_m": s.distance_from_net_m,
            "used_court_calibration": s.used_court_calibration,
        }
        for s in shots
    ]


def summarize_shots(shots: Sequence[Shot]) -> dict:
    """Same "summary alongside raw data" pattern as every other Part 5-7 stage."""
    total = len(shots)
    by_type = {
        SHOT_TYPE_SMASH: 0, SHOT_TYPE_LOB: 0, SHOT_TYPE_VOLLEY: 0,
        SHOT_TYPE_GROUNDSTROKE: 0, SHOT_TYPE_UNKNOWN: 0,
    }
    for s in shots:
        by_type[s.shot_type] = by_type.get(s.shot_type, 0) + 1
    unknown = by_type[SHOT_TYPE_UNKNOWN]
    return {
        "shot_count": total,
        "shot_counts_by_type": by_type,
        "unknown_rate": (unknown / total) if total else 0.0,
    }
