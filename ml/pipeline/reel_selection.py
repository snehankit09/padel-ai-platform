"""
Clip selection — Part 10a, the first piece of Part 10 (reel generator,
PRD Module 4).

Part 8 already cut one real, playable clip per tagged HighlightEvent —
for a long match that can be a lot of clips, more than anyone wants
sitting back-to-back in one auto-generated reel. This module decides
*which* of those already-cut clips make it into the reel. It doesn't
decide what order they play in or how much gap sits between them — see
ml/pipeline/reel_ordering.py (Part 10b) for that, deliberately kept as
its own module: selection ("is this clip good enough to include") and
ordering ("given the included clips, what order/pacing") are different
questions with different failure modes, and reel_ordering.py's own
`order_clips_for_reel` needs to work standalone against any caller's
already-selected list, not just this module's output.

**Why selection ranks by importance_score, never by chronological
position.** A reel is a highlights reel — the point is to show the best
moments, not a chronological digest. `HighlightEvent.importance_score`
(Part 7e) is already the one number this pipeline has that represents
"how highlight-worthy is this", so it's the only defensible ranking key
here. (Playback ORDER is a completely separate question — see
reel_ordering.py's own docstring on why *that* module defaults to
chronological instead. A reel can rank by importance to decide inclusion
and still play the included clips in match order — that's exactly what
this module + reel_ordering.py's default, used together, produce.)

**Why two caps, applied together, not one.** `reel_max_clips` (a hard
count) and `reel_target_duration_s` (a soft duration budget) fail
differently: a match with many short, punchy highlights could clear a
duration budget while producing a reel with far more clips than anyone
wants to sit through; a match with a few very long rallies could clear a
clip-count cap while producing a reel that runs long. `select_clips_for_reel`
enforces both at once — the same "don't trust a single number to capture
a multi-dimensional constraint" reasoning already present elsewhere in
this codebase (e.g. Part 7e's spectacular-save detection requiring BOTH a
timing window AND a different responding player, not either alone).

**Why `reel_max_clips=0` means a real, empty reel — not an error, and not
exempted.** A caller that explicitly configured "no reel" gets no reel,
full stop, even if there's a spectacular 5-star highlight sitting right
there — the same "explicit zero is unambiguous, not a sentinel that needs
special-casing" posture reel_ordering.py's `transition_gap_s=0.0` already
uses. Contrast this with the NEXT rule:

**Why the single highest-importance clip is always kept, even if it alone
exceeds `reel_target_duration_s`.** A 35-second LONG_RALLY highlight
against a 30-second target duration shouldn't produce a completely empty
reel just because the best moment in the match happens to run a few
seconds over budget — that's the target acting as a hard wall instead of
the soft budget it's meant to be. This exemption applies ONLY to the
single best clip, and only when `reel_max_clips` genuinely allows at
least one clip (the `reel_max_clips=0` case above is checked first and
always wins) — every clip after the first still has to fit the remaining
budget.

Same "no Celery/DB/app dependency" layering as every ml/pipeline module
before it. Fed by whatever 10a's own glue stage becomes — this module has
no opinion on how ClipCandidates get built from persisted Highlight rows
or where its output gets stored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

DEFAULT_REEL_MAX_CLIPS = 10
DEFAULT_REEL_TARGET_DURATION_S = 120.0


class ReelSelectionError(Exception):
    """Raised when clip selection is given input it can't reasonably use."""


@dataclass(frozen=True)
class ClipCandidate:
    """
    One already-cut Part 8 clip, as far as reel selection/ordering need
    to know about it. `highlight_id` is left untyped (`object`, not
    `uuid.UUID`) on purpose: a real caller supplies the actual
    `Highlight.id` once 10a's own glue stage exists to build these from
    persisted rows, but every pure-logic test in this module and in
    reel_ordering.py constructs ClipCandidates directly with a plain
    string id — this dataclass shouldn't force a database dependency onto
    code that never touches the database.
    """

    highlight_id: object
    event_type: str
    start_time_s: float
    end_time_s: float
    importance_score: float
    clip_file_path: str

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


def _rank_by_importance(candidates: Sequence[ClipCandidate]) -> list[ClipCandidate]:
    """Highest importance_score first; ties broken by original input position — see module docstring."""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (-pair[1].importance_score, pair[0]))
    return [candidate for _, candidate in indexed]


def select_top_n(candidates: Sequence[ClipCandidate], max_clips: int) -> list[ClipCandidate]:
    """
    Keeps the `max_clips` highest-importance candidates (ties broken by
    original position), with no duration awareness at all — the
    count-only half of select_clips_for_reel, exposed standalone for a
    caller that only cares about the count cap (e.g. a fixed-length
    "top 5 moments" feature, not the full reel pipeline).
    """
    if max_clips < 0:
        raise ReelSelectionError(f"max_clips must be >= 0, got {max_clips!r}")
    return _rank_by_importance(candidates)[:max_clips]


def select_by_target_duration(
    candidates: Sequence[ClipCandidate], *, target_duration_s: float
) -> list[ClipCandidate]:
    """
    Greedily keeps highest-importance candidates while their combined
    duration fits `target_duration_s`, with no count limit at all — the
    duration-only half of select_clips_for_reel. The single
    highest-importance candidate is always included first, even if its
    own duration alone exceeds target_duration_s (see module docstring);
    every candidate after it only gets added if there's real budget left.
    """
    if target_duration_s < 0:
        raise ReelSelectionError(f"target_duration_s must be >= 0, got {target_duration_s!r}")

    ranked = _rank_by_importance(candidates)
    if not ranked:
        return []

    selected = [ranked[0]]
    total_duration_s = ranked[0].duration_s
    for candidate in ranked[1:]:
        if total_duration_s + candidate.duration_s <= target_duration_s:
            selected.append(candidate)
            total_duration_s += candidate.duration_s
    return selected


def select_clips_for_reel(
    candidates: Sequence[ClipCandidate],
    *,
    max_clips: int = DEFAULT_REEL_MAX_CLIPS,
    target_duration_s: float = DEFAULT_REEL_TARGET_DURATION_S,
) -> list[ClipCandidate]:
    """
    The real entry point Part 10's glue stage calls: both caps enforced
    together (see module docstring for why one alone isn't enough).

    Order of operations, precisely: `max_clips == 0` short-circuits to an
    empty list immediately — that check always wins, even over the
    "always keep the single best clip" exemption below (see module
    docstring's two rules on this — the empty-reel case is checked
    first, deliberately, so it can never be silently overridden by the
    best-clip exemption). Otherwise, candidates are ranked by importance;
    the best one is always kept (even over target_duration_s alone) as
    long as max_clips allows at least one clip; every subsequent
    candidate is added only while BOTH the count is still under
    max_clips AND the running total still fits target_duration_s.
    """
    if max_clips < 0:
        raise ReelSelectionError(f"max_clips must be >= 0, got {max_clips!r}")
    if target_duration_s < 0:
        raise ReelSelectionError(f"target_duration_s must be >= 0, got {target_duration_s!r}")
    if max_clips == 0:
        return []

    ranked = _rank_by_importance(candidates)
    if not ranked:
        return []

    selected = [ranked[0]]
    total_duration_s = ranked[0].duration_s
    for candidate in ranked[1:]:
        if len(selected) >= max_clips:
            break
        if total_duration_s + candidate.duration_s <= target_duration_s:
            selected.append(candidate)
            total_duration_s += candidate.duration_s
    return selected


def to_serializable(candidates: Sequence[ClipCandidate]) -> list[dict]:
    return [
        {
            "highlight_id": str(c.highlight_id) if c.highlight_id is not None else None,
            "event_type": c.event_type,
            "start_time_s": c.start_time_s,
            "end_time_s": c.end_time_s,
            "duration_s": c.duration_s,
            "importance_score": c.importance_score,
            "clip_file_path": c.clip_file_path,
        }
        for c in candidates
    ]


def summarize_selection(candidates: Sequence[ClipCandidate]) -> dict:
    """Same "summary alongside raw data" pattern as every Part 5-10 module before it."""
    total_duration_s = sum(c.duration_s for c in candidates)
    return {
        "selected_count": len(candidates),
        "total_duration_s": total_duration_s,
        "avg_importance_score": (sum(c.importance_score for c in candidates) / len(candidates)) if candidates else 0.0,
    }
