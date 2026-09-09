"""
Clip ordering & pacing — Part 10b, the second piece of Part 10 (reel
generator, PRD Module 4).

Part 10a (`reel_selection.py`) already decided *which* already-cut clips
make the reel and handed them back in chronological order on purpose —
selection and ordering were deliberately kept as separate questions (see
that module's docstring). This module is where ordering actually
happens: given the selected `ClipCandidate`s, decide what order they
play in, then lay them out on the reel's own timeline with however much
gap sits between one clip and the next. It doesn't cut, re-encode, or
render a crossfade — that's FFmpeg assembly, later Part 10 work (see the
note on `transition_gap_s` below on why this module only reserves the
time rather than rendering the transition itself).

**Why chronological is the default ordering strategy, not
importance-sorted.** A padel rally builds: the exchange that sets up a
smash is what makes the smash worth watching. Reordering by
`importance_score` would routinely play the biggest putaway before the
rally that produced it, which reads as out of context rather than
exciting — the same "coherent, watchable" reasoning 10a already used to
justify returning chronological order by default. `order_clips_for_reel`
still exposes an explicit `"importance"` strategy for the opposite use
case — a short, greatest-hits-style teaser where narrative continuity
matters less than leading with the best moment — but a caller has to ask
for that; it's not the default.

**Why both strategies reuse the same tie-break rule as 10a.** Ties (two
clips at the same `start_time_s`, or two importance-tied clips) are
broken by original input position, same "earlier in the match wins"
rule `reel_selection._rank_by_importance` uses. Consistent tie-breaking
across both modules means a caller never sees clip order flip depending
on which stage last touched the list.

**Why `order_clips_for_reel` re-sorts chronologically instead of
trusting its input's order.** 10a's contract already returns selections
in chronological order, but this module doesn't assume its caller was
10a — sorting defensively by `start_time_s` (falling back to original
position on a tie) makes `order_clips_for_reel` correct standalone, not
just correct when fed 10a's output.

**Why pacing is a single flat `transition_gap_s`, not a per-event-type
or per-transition-style setting.** Nothing in the brief calls for
different transition treatments per highlight type, and a flat gap
keeps the timeline math simple and deterministic: `build_reel_timeline`
just reserves `transition_gap_s` seconds *between* consecutive clips
(never before the first clip or after the last — there's nothing to
transition from/to there) and lets a later FFmpeg assembly stage decide
*how* to fill that reserved time — a hard cut, a crossfade, a music
sting, whatever Module 4's export step wants. That's the same layering
split as clip_boundaries.py (Part 8a) deciding padding vs. clip_extraction
(Part 8b) actually cutting the file: this module answers "how long,"
assembly answers "how." `transition_gap_s=0.0` is the explicit "hard
cut, no reserved gap" case, same "explicit zero is unambiguous" pattern
`reel_max_clips=0` uses in 10a — not a sentinel that needs special-casing.

**Why `build_reel_timeline` is a separate function from
`order_clips_for_reel` rather than one combined step.** A caller that
only wants the play order (e.g. to build `ReelHighlight.position` rows)
shouldn't have to supply a gap value it doesn't care about, and a caller
who already has an ordered list from somewhere else (tests, a manual
override) shouldn't have to re-run ranking just to get timeline math.
Same "thin, composable pieces" reasoning as 10a's `select_top_n` /
`select_by_target_duration` wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

from ml.pipeline.reel_selection import ClipCandidate

OrderingStrategy = Literal["chronological", "importance"]


@dataclass(frozen=True)
class ReelTimelineEntry:
    """
    One clip's placement in the assembled reel: which source clip it is,
    where it sits in play order, and where that lands on the reel's own
    timeline once transition gaps are accounted for. `position` maps
    directly to `ReelHighlight.position` — a future stage layer writes
    these rows 1:1.
    """

    highlight_id: object
    position: int
    event_type: str
    source_start_s: float
    source_end_s: float
    clip_duration_s: float
    gap_before_s: float
    reel_start_s: float
    reel_end_s: float


def _sort_chronological(candidates: Sequence[ClipCandidate]) -> list[ClipCandidate]:
    """Earliest start_time_s first; ties broken by original input position."""
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (pair[1].start_time_s, pair[0]))
    return [candidate for _, candidate in indexed]


def _sort_by_importance(candidates: Sequence[ClipCandidate]) -> list[ClipCandidate]:
    """
    Highest importance_score first; ties broken by original input
    position — same rule as reel_selection._rank_by_importance, kept as
    its own copy here since 10a's version is a private module helper.
    """
    indexed = list(enumerate(candidates))
    indexed.sort(key=lambda pair: (-pair[1].importance_score, pair[0]))
    return [candidate for _, candidate in indexed]


def order_clips_for_reel(
    candidates: Sequence[ClipCandidate],
    *,
    strategy: OrderingStrategy = "chronological",
) -> list[ClipCandidate]:
    """
    Returns `candidates` resequenced for playback per `strategy`:

      - "chronological" (default) — earliest start_time_s first. The
        coherent, in-context viewing order — see module docstring.
      - "importance"              — highest importance_score first, for
        a greatest-hits-style teaser rather than a narrative reel.

    Does not select/filter anything — that's 10a's job. Feed it an
    already-selected list.
    """
    if strategy == "chronological":
        return _sort_chronological(candidates)
    if strategy == "importance":
        return _sort_by_importance(candidates)
    raise ValueError(f"Unknown ordering strategy: {strategy!r}")


def build_reel_timeline(
    ordered_clips: Sequence[ClipCandidate],
    *,
    transition_gap_s: float = 0.0,
) -> list[ReelTimelineEntry]:
    """
    Lays `ordered_clips` out on the reel's own timeline, inserting
    `transition_gap_s` seconds between consecutive clips (never before
    the first or after the last — see module docstring). Assumes
    `ordered_clips` is already in the play order the caller wants;
    call `order_clips_for_reel` first if it isn't.

    `transition_gap_s` is clamped to 0.0 if negative — a caller passing
    a negative gap almost certainly means "no gap," not "overlap the
    clips," and this module has no notion of overlapping playback.
    """
    gap_s = max(0.0, transition_gap_s)

    entries: list[ReelTimelineEntry] = []
    cursor_s = 0.0
    for position, clip in enumerate(ordered_clips):
        gap_before_s = gap_s if position > 0 else 0.0
        cursor_s += gap_before_s
        reel_start_s = cursor_s
        reel_end_s = reel_start_s + clip.duration_s
        entries.append(
            ReelTimelineEntry(
                highlight_id=clip.highlight_id,
                position=position,
                event_type=clip.event_type,
                source_start_s=clip.start_time_s,
                source_end_s=clip.end_time_s,
                clip_duration_s=clip.duration_s,
                gap_before_s=gap_before_s,
                reel_start_s=reel_start_s,
                reel_end_s=reel_end_s,
            )
        )
        cursor_s = reel_end_s

    return entries


def to_serializable(entries: Sequence[ReelTimelineEntry]) -> list[dict]:
    return [
        {
            "highlight_id": str(e.highlight_id) if e.highlight_id is not None else None,
            "position": e.position,
            "event_type": e.event_type,
            "source_start_s": e.source_start_s,
            "source_end_s": e.source_end_s,
            "clip_duration_s": e.clip_duration_s,
            "gap_before_s": e.gap_before_s,
            "reel_start_s": e.reel_start_s,
            "reel_end_s": e.reel_end_s,
        }
        for e in entries
    ]


def summarize_timeline(entries: Sequence[ReelTimelineEntry]) -> dict:
    """Same "summary alongside raw data" pattern as summarize_selection (10a)."""
    total_gap_s = sum(e.gap_before_s for e in entries)
    total_clip_s = sum(e.clip_duration_s for e in entries)
    return {
        "clip_count": len(entries),
        "total_clip_duration_s": total_clip_s,
        "total_gap_duration_s": total_gap_s,
        "total_reel_duration_s": entries[-1].reel_end_s if entries else 0.0,
    }
