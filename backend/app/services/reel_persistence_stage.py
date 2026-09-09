"""
Reel + ReelHighlight persistence — Part 10e.

Parts 10a (ml/pipeline/reel_selection.py) and 10b (ml/pipeline/reel_
ordering.py) are both pure-logic modules with no Postgres/Celery/`app`
import (same layering as every ml/pipeline module before them — see
their own module docstrings) that end in a list of `ReelTimelineEntry`,
whose own docstring already says the quiet part: "`position` maps
directly to `ReelHighlight.position` — a future stage layer writes these
rows 1:1." This module is that future stage layer: given a match and
10b's already-selected, already-ordered timeline, it writes one `Reel`
row and one `ReelHighlight` row per entry — Part 2's association-object
pattern (`ReelHighlight`, same reasoning as `MatchPlayer`: a bare
`secondary=` table can't carry the `position` column
`reel.highlights.append(highlight)` would need).

No glue stage wires this into app/workers/tasks.py yet — `done_stage` is
still Part 10's own placeholder (see that module's docstring), pending
whatever later Part assembles 10a's selection and 10b's ordering out of
real, persisted `Highlight` rows and hands the result here. Like Part 9e
(`persist_player_statistics`), this module's job is narrow and purely
mechanical: take a `timeline_entries` sequence a caller already built
and turn it into rows — it doesn't decide what belongs in a reel or what
order it plays in, that's 10a/10b's job.

**Why every `highlight_id` is checked against this match's own
`Highlight` rows before anything is written.**
`ReelTimelineEntry.highlight_id` (inherited from `ClipCandidate`) is
typed `object`, not `uuid.UUID` — 10a's own docstring is explicit that's
deliberate, so its pure-logic tests can build `ClipCandidate`s with a
plain string id without a database dependency. A real caller building
`ClipCandidate`s from persisted rows supplies the actual `Highlight.id`,
but this module doesn't take that on faith: every `highlight_id` must
already be a real `Highlight` row belonging to `match_id`, or the whole
call is refused with `ReelPersistenceError` and nothing is written —
the same "don't write a row the data doesn't support" posture 9e's own
`MatchPlayer` check already takes (see that module's docstring), here
because a wrong-match or fabricated `highlight_id` would otherwise
silently produce a `ReelHighlight` row pointing at a clip that was never
actually cut for this match.

**Why an empty `timeline_entries` still writes a real `Reel` row, not
nothing.** `reel_max_clips=0` is 10a's own explicit "no reel, no
exemption" case (see reel_selection.py's module docstring) — a real,
empty reel, not an error and not a caller mistake to special-case.
Refusing to write a `Reel` row at all here would silently turn that
honest zero-clip decision into "this stage never ran," indistinguishable
from a video that hasn't reached Part 10 yet. A `Reel` row with zero
`ReelHighlight` children is exactly what a max_clips=0 selection means.

**What this does NOT do, and why.** `Reel.file_path` stays NULL and
`Reel.status` stays whatever `status` the caller passes (default
`ReelStatus.PENDING`) — this module writes the reel's *composition*
(which highlights, in what order), not the actual rendered video file.
Same split as Part 7f leaving `Highlight.clip_file_path` NULL for Part 8
to fill in later: FFmpeg assembly is a later Part 10 stage's job, which
reads this row back (by id, or by match_id) and UPDATEs `file_path` +
flips `status` to `READY`/`FAILED` once a real file exists, rather than
this stage guessing at a path before any rendering has happened. Picking
`music_track` (PRD Module 4: swappable without regenerating the reel) is
likewise a caller decision this module just records, not a default it
invents.

**Idempotency.** Same "delete only what this stage owns, then reinsert"
shape as analyze_persistence_stage.py and player_statistics_persistence_
stage.py. Unlike those two, there's no partial-ownership slicing to get
right here — a match's *entire* `Reel` row (and, through it, every
`ReelHighlight` child) belongs to this stage, since PRD Module 4 is one
reel per match/video. Every `ReelHighlight` row for the match's existing
`Reel`(s) is deleted first, then the `Reel` row(s) themselves — children
before parent, because the `reel_highlights.reel_id` foreign key has no
ON DELETE CASCADE at the DB level (see the initial-schema migration) —
before the fresh set is written. Same reasoning as those two modules'
own idempotency sections: a retried caller (e.g. a re-run of whatever
later Part builds `timeline_entries`) never leaves stale `ReelHighlight`
rows from a previous attempt's now-superseded ordering sitting alongside
the fresh ones.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Sequence

from sqlalchemy import delete, select

from app.core.database import get_sync_db
from app.models.enums import ReelStatus
from app.models.highlight import Highlight
from app.models.reel import Reel
from app.models.reel_highlight import ReelHighlight

if TYPE_CHECKING:
    from ml.pipeline.reel_ordering import ReelTimelineEntry

logger = logging.getLogger(__name__)


class ReelPersistenceError(Exception):
    """Raised when `timeline_entries` can't be trusted for this match."""


def persist_reel(
    match_id: uuid.UUID,
    timeline_entries: Sequence["ReelTimelineEntry"],
    *,
    status: ReelStatus = ReelStatus.PENDING,
    music_track: str | None = None,
) -> uuid.UUID:
    """
    Writes one `Reel` row for `match_id` and one `ReelHighlight` row per
    entry in `timeline_entries` (`ReelHighlight.position` copied straight
    from `ReelTimelineEntry.position` — see module docstring on why
    that's a direct 1:1, not a re-derivation).

    Every entry's `highlight_id` is checked against this match's real
    `Highlight` rows before anything is written; any entry whose
    `highlight_id` isn't a `Highlight` of `match_id` (wrong match, a
    fabricated/non-UUID id, or a `Highlight` that's since been deleted
    and reinserted with a new id by a retried `analyze`) raises
    `ReelPersistenceError` and writes nothing — see module docstring. An
    empty `timeline_entries` is not an error; it writes a real, empty
    `Reel` row (see module docstring on why that honestly represents
    `max_clips=0`, rather than the stage silently not running).

    Returns the persisted `Reel.id`.
    """
    with get_sync_db() as db:
        match_highlight_ids = set(
            db.execute(select(Highlight.id).where(Highlight.match_id == match_id)).scalars()
        )

        entry_highlight_ids: list[uuid.UUID] = []
        for entry in timeline_entries:
            highlight_id = entry.highlight_id
            if not isinstance(highlight_id, uuid.UUID) or highlight_id not in match_highlight_ids:
                raise ReelPersistenceError(
                    f"match_id={match_id}: timeline entry at position {entry.position} has "
                    f"highlight_id={highlight_id!r}, which is not a Highlight of this match -- "
                    f"refusing to write a ReelHighlight row for a clip that wasn't cut for this match"
                )
            entry_highlight_ids.append(highlight_id)

        # Idempotency: clear only what this stage owns for this match --
        # every ReelHighlight of this match's existing Reel(s), then the
        # Reel row(s) themselves (children first: no ON DELETE CASCADE at
        # the DB level -- see module docstring) -- before reinserting.
        existing_reel_ids = list(db.execute(select(Reel.id).where(Reel.match_id == match_id)).scalars())
        if existing_reel_ids:
            db.execute(delete(ReelHighlight).where(ReelHighlight.reel_id.in_(existing_reel_ids)))
            db.execute(delete(Reel).where(Reel.match_id == match_id))

        reel = Reel(match_id=match_id, status=status, file_path=None, music_track=music_track)
        db.add(reel)
        db.flush()  # populates reel.id (Python-side default) for the ReelHighlight rows below

        for entry, highlight_id in zip(timeline_entries, entry_highlight_ids):
            db.add(ReelHighlight(reel_id=reel.id, highlight_id=highlight_id, position=entry.position))

        db.commit()
        reel_id = reel.id

    logger.info(
        "[reel] match_id=%s: persisted Reel id=%s with %d ReelHighlight row(s)",
        match_id, reel_id, len(timeline_entries),
    )
    return reel_id
