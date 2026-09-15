"""
Highlight-worthy event tagging — Part 7e, the fifth and final piece of
pipeline stage 6 (action recognition).

Parts 7a-7d each answer one factual question about the match (when did a
rally happen, who served it, what shots were played, how did it end).
This module asks a different kind of question: *is this moment worth
showing someone who didn't watch the whole match*. That's PRD Module 3
(Highlight Detection) — Part 8 will later turn a tagged event into an
actual trimmed clip file (FFmpeg) and a persisted `Highlight` row; this
module's only job is producing the *candidate* list of (type, time
window, importance score) tuples for Part 8 to work from, purely by
applying threshold rules to 7a-7d's already-computed output. No new
detection, tracking, or classification happens here — see the module
docstring of each Part 7a-7d module for why none of *them* reach for a
model that doesn't exist in this codebase yet; this module doesn't
either, for the same reason, and has even less need to: everything it
looks at is already a clean dataclass, not pixels.

**Why this is rules over 7a-7d's output, not a new model.** A highlight-
worthiness classifier trained on real footage is the "correct" long-term
answer (PRD Section 13, same future-work bucket as a swing/pose model),
but training one needs labeled highlight/not-highlight examples this
platform doesn't have yet. In the meantime, the six HighlightType values
below all correspond to a rule that's directly checkable against 7a-7d's
existing fields, with no missing signal:

  - LONG_RALLY:        rally.duration_s past a threshold (7a)
  - FAST_EXCHANGE:      a run of shots with short inter-contact gaps (7a+7c)
  - POWERFUL_SMASH:     any Shot classified SHOT_TYPE_SMASH (7c)
  - SPECTACULAR_SAVE:   a different player returning a smash in a hurry (7c)

**What's deliberately NOT tagged here, and why.** HighlightType (PRD
Module 3) also defines WINNING_SHOT, MATCH_POINT, and BREAK_POINT. None
of the three are emitted by this module:

  - WINNING_SHOT needs exactly the winner-vs-unforced-error distinction
    ml.pipeline.point_outcome's own module docstring already explains
    this pipeline cannot make from position data alone (a winner and a
    receiving-side unforced error look identical: the rally's ball
    activity ends with the ball in court and nobody having hit it back).
    Guessing here would mean silently overriding that module's own
    honesty about the same signal gap — not this module's call to make.
  - MATCH_POINT and BREAK_POINT are score-relative, not action-relative:
    "is this rally point/game/match point" needs live game/set/match
    score state (which side needs one more point to win the game, set,
    or match right now), and nothing in Parts 1-7 tracks a running score
    anywhere — there's no scoring engine yet (PRD Section 13/Module 5
    territory, same "future work" bucket every other Part 7 module's
    known-limitation section points to). Fabricating a score from rally
    counts alone would be a guess with no real signal behind it, the
    same kind of guess ml.pipeline.point_outcome's OUTCOME_UNDETERMINED
    and ml.pipeline.shot_classification's SHOT_TYPE_UNKNOWN both
    deliberately refuse to make.

Rather than omit these three silently, `detect_highlights` accepts an
optional `game_state_by_rally` argument (see its docstring) that's the
extension point for wiring real score context in once a scoring stage
exists — until then it defaults to None and MATCH_POINT/BREAK_POINT are
never emitted, which is the honest answer, not a placeholder guess.
WINNING_SHOT has no such extension point here on purpose: it isn't a
missing *input*, it's a signal this pipeline's shot/outcome data
structurally can't support, the same conclusion point_outcome.py already
reached for its own OUTCOME_IN_BOUNDS_END category.

Same "no Celery/DB/app dependency" layering as every ml/pipeline module
before it — plain RallySegment/Shot/PointOutcome/ServeEvent objects in
(the dataclasses 7a-7d already defined; this module reads their
*structured* output directly rather than redefining its own copies of
BallPoint/PlayerBox the way 7b/7c/7d do, since what this module needs
*is* their already-derived fields, not raw per-frame positions), a flat
list of HighlightEvent out. Fed by
app/services/highlight_tagging_stage.py (Part 7e's glue layer), which
knows how to read persisted rallies.json/shots.json (and, when present,
outcomes.json/serves.json) back into these shapes.

**Cross-type score comparability (Highlights Improvement Roadmap Tier 2).**
Every tag_* function's importance_score now follows the same convention:
0.0 at whatever threshold makes an event qualify for its HighlightType at
all, rising to 1.0 at that type's own saturation point. This wasn't
always true — tag_powerful_smash and tag_spectacular_saves used to
compress their scores into [0.5, 1.0] and [0.6, 1.0] respectively, so a
smash or save that only just barely qualified would still outscore a
genuinely impressive long rally or fast exchange, structurally, before
anyone even looked at how exciting either moment actually was. That
asymmetry was directly checkable by reading the four formulas side by
side — no real match footage needed to find or fix it, unlike the
HIGHLIGHT_TYPE_SCORE_WEIGHT question just below, which does.

Fixing the floor makes scores comparable *by construction*, not because
anyone has verified 0.7 always feels equally exciting whether it comes
from a rally or a smash — that's still an open, genuinely subjective
question. HIGHLIGHT_TYPE_SCORE_WEIGHT (applied centrally in
detect_highlights, not inside any individual tag_* function) is the
extension point for that: currently every weight is 1.0 — a deliberate
no-op, not a claim that no type deserves more or less weight than
another — until someone watches enough real reels to have an actual
opinion worth encoding. Bumping a weight without that review would trade
one unexamined bias (the old floor asymmetry) for a different one that's
merely harder to notice, not evidence-based just because it's a
different number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ml.pipeline.rally_detection import RallySegment
from ml.pipeline.shot_classification import SHOT_TYPE_SMASH, Shot

HIGHLIGHT_TYPE_LONG_RALLY = "long_rally"
HIGHLIGHT_TYPE_SPECTACULAR_SAVE = "spectacular_save"
HIGHLIGHT_TYPE_FAST_EXCHANGE = "fast_exchange"
HIGHLIGHT_TYPE_POWERFUL_SMASH = "powerful_smash"
# HIGHLIGHT_TYPE_WINNING_SHOT / _MATCH_POINT / _BREAK_POINT intentionally
# have no corresponding tag_* function — see module docstring.

# --- Long rally (7a) --------------------------------------------------------

# A rally at or beyond this duration is "long" enough to flag on its own
# merit — not a hard PRD number, just a reasonable padel-viewer's sense of
# "that point went on a while" (most points are under 10s; see
# ml.pipeline.rally_detection's own module docstring on how mundane
# between-point dead time compares to real rally length).
DEFAULT_LONG_RALLY_MIN_DURATION_S = 15.0

# Duration at which a long rally's importance_score saturates to 1.0 --
# a rally at exactly the minimum is barely "long" (score just above 0),
# one at or beyond this is an unambiguous marathon point.
DEFAULT_LONG_RALLY_SCORE_SATURATION_S = 35.0

# --- Fast exchange (7a + 7c) -------------------------------------------------

# Consecutive shots inside a rally with a gap this short (in seconds,
# derived from their frame_index and the video's sample_fps) count as part
# of one fast-exchange run -- short enough that it's plausibly a genuine
# rapid-fire net exchange, not just two ordinary groundstrokes.
DEFAULT_FAST_EXCHANGE_MAX_INTERVAL_S = 1.2

# A run of qualifying gaps needs at least this many shots in it before
# it's reported as its own highlight -- two fast shots in a row could
# just be tracking noise finding a spurious extra contact; several in a
# row is a real exchange.
DEFAULT_FAST_EXCHANGE_MIN_SHOT_COUNT = 4

# Shot count at which a fast-exchange run's importance_score saturates to
# 1.0 -- a run right at the minimum count is only just a "fast exchange",
# one twice as long is clearly the highlight-worthy version of one.
DEFAULT_FAST_EXCHANGE_SCORE_SATURATION_COUNT = 8

# --- Powerful smash (7c) -----------------------------------------------------

# contact_height_ratio at which a smash's importance_score saturates to
# 1.0. Shot classification's own DEFAULT_SMASH_HEIGHT_RATIO (1.05) is the
# floor every SHOT_TYPE_SMASH already cleared to be classified a smash at
# all; this is a materially higher ratio (well above head height, deep
# into "fully extended overhead contact" territory) so the score curve
# has real room in it rather than every smash landing near 1.0.
DEFAULT_POWERFUL_SMASH_SCORE_CEILING_RATIO = 1.6

# --- Spectacular save (7c) ---------------------------------------------------

# A return shot arriving this soon (seconds) after an opponent's smash --
# by a DIFFERENT player than the one who smashed -- counts as a save
# under real time pressure. 0.8s is short: at the default
# frame_sample_rate_fps of 5.0 that's about 4 sampled frames from smash
# contact to return contact.
DEFAULT_SPECTACULAR_SAVE_MAX_RESPONSE_S = 0.8

# --- Cross-type score calibration (Highlights Improvement Roadmap Tier 2) ---

# Applied to a HighlightEvent's importance_score in detect_highlights,
# AFTER every tag_* function has already scored its own event on its own
# [0, 1] "0 at qualifying threshold, 1 at saturation" scale (see module
# docstring's "Cross-type score comparability" section for why that scale
# is now shared). This is the extension point for a further, genuinely
# subjective adjustment -- "an impressive smash is worth more attention
# than an impressive long rally, even at the same normalized score" -- if
# and when real match footage review actually supports that claim.
#
# Every weight is 1.0 today: not because no type deserves more weight
# than another, but because nobody has watched enough real reels yet to
# say which ones and by how much. Change these only from that kind of
# review, not from a guess -- seed one bias for another otherwise.
HIGHLIGHT_TYPE_SCORE_WEIGHT: Mapping[str, float] = {
    HIGHLIGHT_TYPE_LONG_RALLY: 1.0,
    HIGHLIGHT_TYPE_FAST_EXCHANGE: 1.0,
    HIGHLIGHT_TYPE_POWERFUL_SMASH: 1.0,
    HIGHLIGHT_TYPE_SPECTACULAR_SAVE: 1.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class HighlightEvent:
    """
    One candidate highlight-worthy moment. `start_time_s`/`end_time_s` are
    the tight, actual boundaries of the event itself (a rally's own span
    for LONG_RALLY, a single contact's moment for POWERFUL_SMASH, etc.) --
    NOT padded with the pre/post-roll a viewable clip will want. That
    padding is deliberately Part 8's job (clip generation), the same
    layering ml.pipeline.rally_detection leaves "is this highlight-worthy"
    to Part 7e/8 rather than deciding it itself: this module answers
    "what happened and when", not "how should the clip around it be cut".

    `importance_score` is in [0, 1]. Every tag_* function scores its own
    events on the same "0.0 at the threshold that makes this event
    qualify at all, 1.0 at that type's own saturation point" scale (see
    module docstring's "Cross-type score comparability" section), so
    scores ARE comparable across HighlightTypes by construction -- a 0.7
    long_rally and a 0.7 powerful_smash both mean "moderately past this
    type's own qualifying bar," not two unrelated numbers. What that
    construction does NOT claim is that 0.7 *feels* equally exciting
    either way to an actual viewer -- HIGHLIGHT_TYPE_SCORE_WEIGHT is the
    deliberately-unused-for-now extension point for that further, more
    subjective adjustment, once real footage review actually justifies
    one. Part 8's ranking-across-types logic (ml.pipeline.reel_selection)
    sorts on this field directly for exactly this reason.

    `source_frame_index` is the single frame the event is anchored to when
    there is one (a shot's contact frame) -- None for whole-rally events
    (LONG_RALLY) that aren't anchored to one specific frame.
    """

    rally_index: int
    highlight_type: str
    start_time_s: float
    end_time_s: float
    importance_score: float
    reason: str
    source_frame_index: int | None = None


def tag_long_rally(
    rally: RallySegment,
    *,
    min_duration_s: float = DEFAULT_LONG_RALLY_MIN_DURATION_S,
    score_saturation_s: float = DEFAULT_LONG_RALLY_SCORE_SATURATION_S,
) -> HighlightEvent | None:
    """None when the rally doesn't clear min_duration_s; otherwise spans the whole rally."""
    if rally.duration_s < min_duration_s:
        return None

    span = max(score_saturation_s - min_duration_s, 1e-9)
    score = _clamp01((rally.duration_s - min_duration_s) / span)

    return HighlightEvent(
        rally_index=rally.rally_index,
        highlight_type=HIGHLIGHT_TYPE_LONG_RALLY,
        start_time_s=rally.start_time_s,
        end_time_s=rally.end_time_s,
        importance_score=score,
        reason=f"rally lasted {rally.duration_s:.1f}s (>= {min_duration_s:.1f}s threshold)",
    )


def _frame_time_s(frame_index: int, sample_fps: float) -> float:
    return frame_index / sample_fps


def tag_fast_exchanges(
    rally: RallySegment,
    shots_in_rally: Sequence[Shot],
    *,
    sample_fps: float,
    max_interval_s: float = DEFAULT_FAST_EXCHANGE_MAX_INTERVAL_S,
    min_shot_count: int = DEFAULT_FAST_EXCHANGE_MIN_SHOT_COUNT,
    score_saturation_count: int = DEFAULT_FAST_EXCHANGE_SCORE_SATURATION_COUNT,
) -> list[HighlightEvent]:
    """
    Finds maximal runs of consecutive shots (already ordered by
    frame_index -- see detect_highlights) whose inter-contact gap never
    exceeds max_interval_s, and reports each run of at least
    min_shot_count shots as its own FAST_EXCHANGE event. A rally can
    contribute more than one such run if a fast stretch is interrupted by
    a slower exchange and then picks back up again -- same "don't
    fabricate one continuous thing out of two genuinely separate ones"
    posture ml.pipeline.rally_detection's own gap-tolerance logic takes
    at the rally-boundary level, one layer further in.

    shots_in_rally must already belong to exactly this rally and be
    sorted by frame_index -- this function doesn't filter or sort itself
    (see detect_highlights, which does both once for every tag_* call
    that needs it, rather than each one repeating the same work).
    """
    if len(shots_in_rally) < min_shot_count:
        return []

    events: list[HighlightEvent] = []
    run_start_idx = 0
    for i in range(1, len(shots_in_rally) + 1):
        gap_ok = False
        if i < len(shots_in_rally):
            prev_t = _frame_time_s(shots_in_rally[i - 1].frame_index, sample_fps)
            cur_t = _frame_time_s(shots_in_rally[i].frame_index, sample_fps)
            gap_ok = (cur_t - prev_t) <= max_interval_s

        if not gap_ok:
            run = shots_in_rally[run_start_idx:i]
            if len(run) >= min_shot_count:
                start_t = _frame_time_s(run[0].frame_index, sample_fps)
                end_t = _frame_time_s(run[-1].frame_index, sample_fps)
                score = _clamp01((len(run) - min_shot_count) / max(score_saturation_count - min_shot_count, 1))
                events.append(
                    HighlightEvent(
                        rally_index=rally.rally_index,
                        highlight_type=HIGHLIGHT_TYPE_FAST_EXCHANGE,
                        start_time_s=start_t,
                        end_time_s=end_t,
                        importance_score=score,
                        reason=(
                            f"{len(run)} consecutive shots each <= {max_interval_s:.2f}s apart"
                        ),
                        source_frame_index=run[0].frame_index,
                    )
                )
            run_start_idx = i

    return events


def tag_powerful_smash(
    shot: Shot,
    rally_index: int,
    *,
    sample_fps: float,
    smash_height_ratio: float,
    score_ceiling_ratio: float = DEFAULT_POWERFUL_SMASH_SCORE_CEILING_RATIO,
) -> HighlightEvent | None:
    """
    None unless `shot` is already classified SHOT_TYPE_SMASH -- this
    never re-derives the smash/not-smash call itself (that's 7c's job,
    already done by the time this module sees the Shot), only scores and
    windows the ones 7c already flagged. `smash_height_ratio` should be
    the SAME threshold value 7c's classify_shot was actually run with for
    this video (see app/services/highlight_tagging_stage.py for where
    that comes from) -- it's the floor a score of 0.5 anchors to, so a
    caller passing a different value than what actually produced this
    Shot would make the score curve mean something inconsistent with how
    the shot was classified.

    Window is the contact frame through however long the ball stayed
    airborne afterward (Shot.airborne_frames_after, already computed by
    7c) -- the natural "moment of impact through the ball landing/being
    played again" span for a single-shot event, same idea
    tag_spectacular_saves uses for its own two-shot window.

    Score is 0.0 at smash_height_ratio itself (just barely cleared 7c's
    own smash threshold -- the least impressive shot 7c would still call
    a smash at all) rising linearly to 1.0 at score_ceiling_ratio, the
    SAME "0 at the qualifying threshold, 1 at saturation" convention
    tag_long_rally and tag_fast_exchanges already use -- see the module
    docstring's "Cross-type score comparability" section for why this
    matters and what changed here (Highlights Improvement Roadmap Tier 2).
    """
    if shot.shot_type != SHOT_TYPE_SMASH:
        return None

    start_t = _frame_time_s(shot.frame_index, sample_fps)
    airborne = shot.airborne_frames_after or 0
    end_t = _frame_time_s(shot.frame_index + airborne, sample_fps)

    if shot.contact_height_ratio is None:
        # A smash was classified without contact_height_ratio surviving
        # into this Shot -- shouldn't happen given classify_shot always
        # sets it for a SHOT_TYPE_SMASH, but score conservatively (a
        # neutral middle value, not a claim either way) rather than raise
        # over a scoring nicety. Not "the floor" under the current
        # (Tier 2) convention -- the floor is 0.0 now; 0.5 here is
        # deliberately a different, unrelated number that happens to
        # coincide with where the floor used to sit pre-Tier 2.
        score = 0.5
    else:
        span = max(score_ceiling_ratio - smash_height_ratio, 1e-9)
        score = _clamp01((shot.contact_height_ratio - smash_height_ratio) / span)

    return HighlightEvent(
        rally_index=rally_index,
        highlight_type=HIGHLIGHT_TYPE_POWERFUL_SMASH,
        start_time_s=start_t,
        end_time_s=max(end_t, start_t),
        importance_score=score,
        reason=(
            f"overhead contact (height_ratio={shot.contact_height_ratio}) classified as a smash"
        ),
        source_frame_index=shot.frame_index,
    )


def tag_spectacular_saves(
    rally: RallySegment,
    shots_in_rally: Sequence[Shot],
    *,
    sample_fps: float,
    max_response_s: float = DEFAULT_SPECTACULAR_SAVE_MAX_RESPONSE_S,
) -> list[HighlightEvent]:
    """
    A SPECTACULAR_SAVE is a shot that immediately follows an opponent's
    smash within the same rally: different player_track_id than the
    smasher, contact made within max_response_s of the smash's own
    contact frame. Both player_track_ids must be known (not None) --
    without knowing WHO returned it, "a different player" can't be
    checked, so an unresolved contact is skipped for this rule rather
    than guessed at, same "an ambiguous signal isn't evidence" posture
    every Part 7 module already takes for its own unresolved contacts
    (see ml.pipeline.shot_classification.classify_shot's SHOT_TYPE_UNKNOWN
    branch for the direct precedent).

    Deliberately doesn't require the save to itself be a good shot by any
    further criterion (height, placement, ...) -- getting a racket on a
    smash at all, fast enough, already is the "spectacular" part this
    proxy can check for; a real quality assessment of the return would
    need the same swing/pose signal every other Part 7 module's known
    limitation already points at.

    shots_in_rally must already belong to exactly this rally and be
    sorted by frame_index, same precondition as tag_fast_exchanges.

    Score is 0.0 at response_s == max_response_s (the slowest response
    that still counts as a save at all) rising linearly to 1.0 at an
    instant (response_s == 0) return -- same "0 at the qualifying
    threshold, 1 at saturation" convention every tag_* function in this
    module now uses; see the module docstring's "Cross-type score
    comparability" section (Highlights Improvement Roadmap Tier 2).
    """
    events: list[HighlightEvent] = []
    for prev_shot, cur_shot in zip(shots_in_rally, shots_in_rally[1:]):
        if prev_shot.shot_type != SHOT_TYPE_SMASH:
            continue
        if prev_shot.player_track_id is None or cur_shot.player_track_id is None:
            continue
        if cur_shot.player_track_id == prev_shot.player_track_id:
            continue

        response_s = _frame_time_s(cur_shot.frame_index, sample_fps) - _frame_time_s(
            prev_shot.frame_index, sample_fps
        )
        if response_s < 0 or response_s > max_response_s:
            continue

        score = _clamp01((max_response_s - response_s) / max_response_s)
        events.append(
            HighlightEvent(
                rally_index=rally.rally_index,
                highlight_type=HIGHLIGHT_TYPE_SPECTACULAR_SAVE,
                start_time_s=_frame_time_s(prev_shot.frame_index, sample_fps),
                end_time_s=_frame_time_s(cur_shot.frame_index, sample_fps),
                importance_score=score,
                reason=(
                    f"player {cur_shot.player_track_id} returned player {prev_shot.player_track_id}'s "
                    f"smash within {response_s:.2f}s"
                ),
                source_frame_index=cur_shot.frame_index,
            )
        )
    return events


def detect_highlights(
    rallies: Sequence[RallySegment],
    shots: Sequence[Shot],
    *,
    sample_fps: float,
    long_rally_min_duration_s: float = DEFAULT_LONG_RALLY_MIN_DURATION_S,
    long_rally_score_saturation_s: float = DEFAULT_LONG_RALLY_SCORE_SATURATION_S,
    fast_exchange_max_interval_s: float = DEFAULT_FAST_EXCHANGE_MAX_INTERVAL_S,
    fast_exchange_min_shot_count: int = DEFAULT_FAST_EXCHANGE_MIN_SHOT_COUNT,
    fast_exchange_score_saturation_count: int = DEFAULT_FAST_EXCHANGE_SCORE_SATURATION_COUNT,
    smash_height_ratio: float,
    powerful_smash_score_ceiling_ratio: float = DEFAULT_POWERFUL_SMASH_SCORE_CEILING_RATIO,
    spectacular_save_max_response_s: float = DEFAULT_SPECTACULAR_SAVE_MAX_RESPONSE_S,
    game_state_by_rally: Mapping[int, object] | None = None,
) -> list[HighlightEvent]:
    """
    Runs every tag_* rule across every rally and returns one flat list of
    HighlightEvents, in rally order and (within a rally) in the order
    each rule found them -- same "flat list, not nested by rally" shape
    ml.pipeline.shot_classification.detect_shots already uses, for the
    same reason: a rally can contribute zero, one, or several highlight
    events, so there's no fixed per-rally count to nest around the way
    ServeEvent/PointOutcome have.

    `smash_height_ratio` has no default here (unlike its DEFAULT_*
    counterpart in tag_powerful_smash) specifically so a caller can't
    forget it and silently drift from whatever value 7c's shot
    classification actually used for this video -- see
    tag_powerful_smash's docstring for why the two must match. Pass
    settings.shot_smash_height_ratio (or the video's own
    court-calibration-derived value, if 7c's stage ever grows one) here.

    `game_state_by_rally`: reserved extension point for a future scoring
    stage to plug live game/set/match score context in, keyed by
    rally_index. Always None today -- there's no scoring engine anywhere
    in this codebase yet (see module docstring) -- and MATCH_POINT/
    BREAK_POINT are never emitted regardless of what's passed here, since
    no tag_* function for them exists yet; accepting (and currently
    ignoring) the parameter now means adding that function later doesn't
    require changing this signature or every call site that already
    passes shots/rallies in.
    """
    shots_by_rally: dict[int, list[Shot]] = {}
    for shot in shots:
        shots_by_rally.setdefault(shot.rally_index, []).append(shot)
    for rally_shots in shots_by_rally.values():
        rally_shots.sort(key=lambda s: s.frame_index)

    events: list[HighlightEvent] = []
    for rally in rallies:
        rally_shots = shots_by_rally.get(rally.rally_index, [])

        long_rally_event = tag_long_rally(
            rally,
            min_duration_s=long_rally_min_duration_s,
            score_saturation_s=long_rally_score_saturation_s,
        )
        if long_rally_event is not None:
            events.append(long_rally_event)

        events.extend(
            tag_fast_exchanges(
                rally,
                rally_shots,
                sample_fps=sample_fps,
                max_interval_s=fast_exchange_max_interval_s,
                min_shot_count=fast_exchange_min_shot_count,
                score_saturation_count=fast_exchange_score_saturation_count,
            )
        )

        for shot in rally_shots:
            smash_event = tag_powerful_smash(
                shot,
                rally.rally_index,
                sample_fps=sample_fps,
                smash_height_ratio=smash_height_ratio,
                score_ceiling_ratio=powerful_smash_score_ceiling_ratio,
            )
            if smash_event is not None:
                events.append(smash_event)

        events.extend(
            tag_spectacular_saves(
                rally,
                rally_shots,
                sample_fps=sample_fps,
                max_response_s=spectacular_save_max_response_s,
            )
        )

    # Cross-type calibration (Tier 2) — applied once, centrally, here,
    # rather than inside each tag_* function, so every tag_* function's
    # own score stays a pure, type-local "0 at threshold, 1 at
    # saturation" number (useful on its own, e.g. for the docstring-level
    # tests that check one type's scoring in isolation) and this is the
    # one place that knows about cross-type weighting at all. A no-op
    # today since every weight is 1.0 — see HIGHLIGHT_TYPE_SCORE_WEIGHT's
    # own comment for why that's deliberate, not an oversight.
    if any(weight != 1.0 for weight in HIGHLIGHT_TYPE_SCORE_WEIGHT.values()):
        events = [
            HighlightEvent(
                rally_index=e.rally_index,
                highlight_type=e.highlight_type,
                start_time_s=e.start_time_s,
                end_time_s=e.end_time_s,
                importance_score=_clamp01(
                    e.importance_score * HIGHLIGHT_TYPE_SCORE_WEIGHT.get(e.highlight_type, 1.0)
                ),
                reason=e.reason,
                source_frame_index=e.source_frame_index,
            )
            for e in events
        ]

    return events


def to_serializable(events: Sequence[HighlightEvent]) -> list[dict]:
    return [
        {
            "rally_index": e.rally_index,
            "highlight_type": e.highlight_type,
            "start_time_s": e.start_time_s,
            "end_time_s": e.end_time_s,
            "importance_score": e.importance_score,
            "reason": e.reason,
            "source_frame_index": e.source_frame_index,
        }
        for e in events
    ]


def summarize_highlights(events: Sequence[HighlightEvent]) -> dict:
    """Same "summary alongside raw data" pattern as every other Part 5-7 stage."""
    total = len(events)
    by_type: dict[str, int] = {
        HIGHLIGHT_TYPE_LONG_RALLY: 0,
        HIGHLIGHT_TYPE_FAST_EXCHANGE: 0,
        HIGHLIGHT_TYPE_POWERFUL_SMASH: 0,
        HIGHLIGHT_TYPE_SPECTACULAR_SAVE: 0,
    }
    for e in events:
        by_type[e.highlight_type] = by_type.get(e.highlight_type, 0) + 1
    return {
        "highlight_count": total,
        "highlight_counts_by_type": by_type,
        "avg_importance_score": (sum(e.importance_score for e in events) / total) if total else 0.0,
    }
