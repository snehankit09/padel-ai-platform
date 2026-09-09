"""
Clip boundary calculation — Part 8a, the first piece of Part 8 (highlight
clip generation).

ml/pipeline/highlight_tagging.py (Part 7e) answers "what happened and
when", on purpose leaving its HighlightEvent.start_time_s/end_time_s as
the *tight*, actual boundaries of the event itself — a rally's own span
for LONG_RALLY, a single contact's moment (plus however long the ball
stayed airborne) for POWERFUL_SMASH, and so on. That module's own
docstring is explicit that padding a tight window into something a
person would actually enjoy watching is deliberately NOT its job. This
module is that job: turn each tight (start_time_s, end_time_s) into a
real clip in/out point — pre-roll before it, post-roll after it, clamped
to the video's own [0, duration] so a highlight near the very start or
end of the match doesn't request a timestamp that doesn't exist.

**Why uniform padding, not per-HighlightType padding.** It might seem
like a whole-rally LONG_RALLY event (already several seconds of real
action) needs less added lead-up than a single-instant POWERFUL_SMASH
contact does. But that difference is already baked into each event's own
tight boundaries by Part 7e — a LONG_RALLY's start_time_s is the rally's
own start (already includes the serve and the build-up), while a
POWERFUL_SMASH's start_time_s is just the contact frame. Applying the
*same* pre_roll_s/post_roll_s on top of each event's own type-appropriate
tight window already produces a type-appropriate final clip, without
this module needing to know anything about what each HighlightType means
— same layering split as Part 7e leaving "is this highlight-worthy" to
itself rather than duplicating Part 7a-7d's own logic.

**Why a minimum clip duration, separate from the padding itself.** Padded
duration is normally just tight_duration + pre_roll_s + post_roll_s, but
clamping against the video's [0, duration] boundary can shrink that —
a highlight starting at t=1.0s can't get a full pre_roll_s of lead-up out
of a video that doesn't have 1s of runway before it. min_duration_s is a
floor for a already-padded-and-clamped window, applied by taking back
room from whichever side of the clamp still has it (see
_expand_to_min_duration) rather than by moving both endpoints in from a
clamped boundary, which would just re-introduce the exact frame budget
that boundary already ran out of, or moving them out from where they'd
otherwise land, which would silently discard part of the pre/post-roll a
caller explicitly asked for. If the *video itself* is shorter than
min_duration_s, this still returns the widest clip the video can
actually offer (the full [0, video_duration_s] span) rather than raising
or producing a clip that requests time outside the source file — a
short-video edge case, not something a threshold tweak here could fix.

Same "no Celery/DB dependency" layering as every ml/pipeline module
before it — plain HighlightEvent in (Part 7e's dataclass, read back
unmodified), a flat list of ClipBoundary out. Fed by
app/services/clip_boundary_stage.py (Part 8a's glue layer), which knows
how to read persisted highlights.json back into HighlightEvent objects
and where the video's own duration_seconds lives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ml.pipeline.highlight_tagging import HighlightEvent

# Seconds of lead-up added before the event's own tight start_time_s —
# enough to let a viewer register "something is about to happen" (the
# opponent setting up, the ball already in flight) rather than starting
# exactly on the moment itself, which reads as an abrupt, context-free cut.
DEFAULT_CLIP_PRE_ROLL_S = 3.0

# Seconds of follow-through added after the event's own tight
# end_time_s — a shorter beat than the pre-roll: enough to see the point
# actually finish (ball land, players react) without the clip lingering
# on dead time once the highlight-worthy moment is over.
DEFAULT_CLIP_POST_ROLL_S = 2.0

# Floor on the final, padded-and-clamped clip duration. Set equal to
# DEFAULT_CLIP_PRE_ROLL_S + DEFAULT_CLIP_POST_ROLL_S: an event with an
# instantaneous tight window (e.g. a POWERFUL_SMASH with no airborne
# frames after contact) already reaches exactly this duration from
# padding alone whenever there's room on both sides, so this floor only
# ever has to do real work when clamping against the video boundary ate
# into one side — see _expand_to_min_duration.
DEFAULT_CLIP_MIN_DURATION_S = DEFAULT_CLIP_PRE_ROLL_S + DEFAULT_CLIP_POST_ROLL_S


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _expand_to_min_duration(
    start: float, end: float, *, min_duration_s: float, lower_bound: float, upper_bound: float
) -> tuple[float, float]:
    """
    Grows [start, end] outward (never past [lower_bound, upper_bound])
    until it's at least min_duration_s long, or until the bound itself
    runs out of room — whichever comes first. Splits the shortfall evenly
    between the two sides first, then hands any leftover a side couldn't
    absorb to the other side, so a clip pinned against one edge of the
    video still gets its full minimum duration out of the other edge
    whenever the video is long enough to provide it.
    """
    duration = end - start
    deficit = min_duration_s - duration
    if deficit <= 0:
        return start, end

    room_before = start - lower_bound
    room_after = upper_bound - end

    add_before = min(deficit / 2, room_before)
    add_after = min(deficit / 2, room_after)
    leftover = deficit - add_before - add_after

    if leftover > 0:
        extra_after = min(leftover, room_after - add_after)
        add_after += extra_after
        leftover -= extra_after
    if leftover > 0:
        extra_before = min(leftover, room_before - add_before)
        add_before += extra_before
        leftover -= extra_before
    # leftover > 0 here only if the video itself is shorter than
    # min_duration_s — nothing left to redistribute, both bounds maxed out.

    return start - add_before, end + add_after


@dataclass(frozen=True)
class ClipBoundary:
    """
    One highlight event's actual clip in/out points — the padded,
    clamped counterpart to a HighlightEvent's tight
    start_time_s/end_time_s. Everything from the source HighlightEvent
    that Part 8's later steps (clip trimming, ranking, persisting back
    onto a Highlight row) need to match a boundary back to the event it
    came from is carried through unchanged (rally_index, highlight_type,
    importance_score, reason, source_frame_index) rather than requiring
    a second lookup into highlights.json.

    `event_start_time_s`/`event_end_time_s` keep the original tight
    window alongside the padded one — useful for debugging ("why does
    this clip start at 12.4s") and for Part 8's later ranking/preview
    logic, which may care about the actual highlight-worthy moment within
    the clip, not just the clip's own outer edges.
    """

    rally_index: int
    highlight_type: str
    start_time_s: float
    end_time_s: float
    event_start_time_s: float
    event_end_time_s: float
    importance_score: float
    reason: str
    source_frame_index: int | None = None


def compute_clip_boundary(
    event: HighlightEvent,
    *,
    video_duration_s: float,
    pre_roll_s: float = DEFAULT_CLIP_PRE_ROLL_S,
    post_roll_s: float = DEFAULT_CLIP_POST_ROLL_S,
    min_duration_s: float = DEFAULT_CLIP_MIN_DURATION_S,
) -> ClipBoundary:
    """
    Pads `event`'s tight window by pre_roll_s/post_roll_s, clamps the
    result to [0, video_duration_s], then tops it up to min_duration_s if
    clamping left it short (see _expand_to_min_duration). video_duration_s
    is the video's own probed duration (Video.duration_seconds, set by
    the validate stage — Part 5a — before analyze ever runs), not
    anything derived from the events themselves: a highlight tagged right
    at the very end of the last rally still needs a real ceiling to clamp
    its post-roll against.
    """
    raw_start = event.start_time_s - pre_roll_s
    raw_end = event.end_time_s + post_roll_s

    start = _clamp(raw_start, 0.0, video_duration_s)
    end = _clamp(raw_end, 0.0, video_duration_s)
    end = max(end, start)  # defends against a pathological video_duration_s < 0

    start, end = _expand_to_min_duration(
        start, end, min_duration_s=min_duration_s, lower_bound=0.0, upper_bound=max(video_duration_s, 0.0)
    )

    return ClipBoundary(
        rally_index=event.rally_index,
        highlight_type=event.highlight_type,
        start_time_s=start,
        end_time_s=end,
        event_start_time_s=event.start_time_s,
        event_end_time_s=event.end_time_s,
        importance_score=event.importance_score,
        reason=event.reason,
        source_frame_index=event.source_frame_index,
    )


def compute_clip_boundaries(
    events: Sequence[HighlightEvent],
    *,
    video_duration_s: float,
    pre_roll_s: float = DEFAULT_CLIP_PRE_ROLL_S,
    post_roll_s: float = DEFAULT_CLIP_POST_ROLL_S,
    min_duration_s: float = DEFAULT_CLIP_MIN_DURATION_S,
) -> list[ClipBoundary]:
    """
    Maps compute_clip_boundary across every event, preserving `events`'
    own order — same flat, order-preserving shape
    ml.pipeline.highlight_tagging.detect_highlights already returns, so a
    caller zipping this output back against `events` (or against the
    Highlight rows Part 7f already persisted in that same order) doesn't
    need to re-sort or re-match on anything.
    """
    return [
        compute_clip_boundary(
            event,
            video_duration_s=video_duration_s,
            pre_roll_s=pre_roll_s,
            post_roll_s=post_roll_s,
            min_duration_s=min_duration_s,
        )
        for event in events
    ]


def to_serializable(boundaries: Sequence[ClipBoundary]) -> list[dict]:
    return [
        {
            "rally_index": b.rally_index,
            "highlight_type": b.highlight_type,
            "start_time_s": b.start_time_s,
            "end_time_s": b.end_time_s,
            "event_start_time_s": b.event_start_time_s,
            "event_end_time_s": b.event_end_time_s,
            "importance_score": b.importance_score,
            "reason": b.reason,
            "source_frame_index": b.source_frame_index,
        }
        for b in boundaries
    ]


def summarize_clip_boundaries(boundaries: Sequence[ClipBoundary]) -> dict:
    """Same "summary alongside raw data" pattern as summarize_highlights."""
    total = len(boundaries)
    durations = [b.end_time_s - b.start_time_s for b in boundaries]
    return {
        "clip_count": total,
        "total_clip_duration_s": sum(durations),
        "avg_clip_duration_s": (sum(durations) / total) if total else 0.0,
    }
