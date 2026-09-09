"""
Player-level Statistic persistence — Part 9e.

Part 9a (ml/pipeline/stats_aggregation.py) computes five StatTypes —
DISTANCE_COVERED, MOVEMENT_SPEED_AVG, REACTION_TIME_AVG,
SMASH_SUCCESS_RATE, NET_SUCCESS_RATE, the STATUS_COMPUTABLE_PENDING_
PLAYER_IDENTITY entries in that module's own STAT_TYPE_STATUS — as plain
StatValue dataclasses keyed by ByteTrack `track_id`, not `Player.id`. Part
9b (ml/pipeline/player_identity.py) gets as far as grouping each track_id
onto a stable, video-scoped SIDE_A/SIDE_B label, and its own module
docstring is explicit about where that mapping stops: two teammates share
one court side for the whole match, and this codebase has no
appearance-based signal (no jersey OCR, no face re-identification) to
tell them apart. Wiring a side to a real `MatchPlayer.team_number` — let
alone picking which of that team's two `Player.id`s a given track_id is —
is therefore genuinely unresolved by anything upstream of this module;
9b's docstring points at a human-in-the-loop confirmation step as the
most plausible source, not a pipeline stage.

This module does NOT try to close that gap itself. Doing so — even a
"best guess" side-order assumption — would be exactly the kind of
fabricated row analyze_persistence_stage.py (Part 7f) and
stats_aggregation.py (Part 9a) both already refuse elsewhere: see the
former's docstring on why WINNERS isn't derived from
OUTCOME_IN_BOUNDS_END, and the latter's STAT_TYPE_STATUS entries for
`winners`/`serve_percentage`/`momentum_possession`. Instead,
`persist_player_statistics` takes a `track_id_to_player_id` mapping as a
required argument, already resolved by whatever caller has the
information (or the human confirmation) to build it honestly. This
module's job is narrow and purely mechanical: given a match, 9a's
track_id-keyed StatValues, and that mapping, write one `Statistic` row
per (match, player, stat_type) — Part 2's schema — for exactly the
StatValues the mapping can vouch for, and skip (not guess at) the rest.

This is the first part of Part 9 that touches Postgres. 9a and 9b are
both pure-logic modules with no Postgres/Celery/`app` import (see their
own module docstrings); the match-level slice `Statistic` rows Part 7f
already writes (TOTAL_POINTS, RALLY_LENGTH_AVG, LONGEST_RALLY, ERRORS)
predates Part 9 entirely — analyze_persistence_stage.py's own docstring
calls it "a first, deliberately small slice" pending Part 9's work, not
Part 9 output itself.

**Sanity check, not identity resolution.** Before writing anything, this
module verifies every `Player.id` in `track_id_to_player_id` is actually
a `MatchPlayer` of the given match — i.e. it will refuse to write a
`Statistic` row with a `player_id` that didn't play in this match, the
same "don't write a row the data doesn't support" posture every
STATUS_NOT_COMPUTABLE verdict in 9a already takes. It does NOT verify
that the mapping picked the *correct* one of a team's two players — 9b's
own docstring is explicit no signal in this codebase can check that,
so there is nothing here to check it against either. That trust boundary
is deliberately the caller's, not this module's.

**Idempotency.** Same "delete only what this stage owns, then reinsert"
shape as analyze_persistence_stage.py's own retry handling: on each call,
every `Statistic` row for this match with a non-NULL `player_id` and a
`stat_type` in `PLAYER_LEVEL_STAT_TYPES` is deleted before the fresh set
is written, so re-running this stage (a retried Celery task, a
re-resolved identity mapping) never double-inserts and never touches
Part 7f's match-level rows (`player_id IS NULL`) or a future stage's
rows for any other StatType.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING, Mapping, Sequence

from sqlalchemy import delete, select

from app.core.database import get_sync_db
from app.models.enums import StatType
from app.models.match_player import MatchPlayer
from app.models.statistic import Statistic

if TYPE_CHECKING:
    from ml.pipeline.stats_aggregation import StatValue

logger = logging.getLogger(__name__)

# Exactly ml.pipeline.stats_aggregation.STAT_TYPE_STATUS's
# STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY subset -- the only stat_keys
# player_level_stat_values() ever emits, and therefore the only StatTypes
# this stage ever owns rows for. Keyed off the real enum's own .value
# strings (not a second hand-copied literal per stat, the way
# analyze_persistence_stage.py's MATCH_LEVEL_STAT_TYPES already keys off
# StatValue.stat_key 1:1 against StatType.value) so this can't silently
# drift from 9a's own inventory.
PLAYER_LEVEL_STAT_TYPES: tuple[StatType, ...] = (
    StatType.DISTANCE_COVERED,
    StatType.MOVEMENT_SPEED_AVG,
    StatType.REACTION_TIME_AVG,
    StatType.SMASH_SUCCESS_RATE,
    StatType.NET_SUCCESS_RATE,
)

_STAT_KEY_TO_TYPE: dict[str, StatType] = {stat_type.value: stat_type for stat_type in PLAYER_LEVEL_STAT_TYPES}


class PlayerStatisticsPersistenceError(Exception):
    """Raised when `track_id_to_player_id` can't be trusted for this match."""


def persist_player_statistics(
    match_id: uuid.UUID,
    stat_values: Sequence["StatValue"],
    track_id_to_player_id: Mapping[int, uuid.UUID],
) -> int:
    """
    Writes one `Statistic` row per (match, player, stat_type) for every
    `stat_values` entry whose `track_id` is present in
    `track_id_to_player_id` and whose `stat_key` is one of
    `PLAYER_LEVEL_STAT_TYPES` (see module docstring for why both filters
    matter: a `stat_values` list may also carry `stat_key`s or
    `track_id=None` this stage doesn't own, and a `track_id` this stage
    has no honest player mapping for is skipped rather than guessed).

    `track_id_to_player_id` is caller-supplied and NOT derived here — see
    module docstring for why 9b's own signals can't build it (the
    which-of-a-team's-two-players-is-this-track question has no answer
    in this codebase). Every value in it is checked against this match's
    real `MatchPlayer` rows before anything is written; a `player_id`
    that isn't a `MatchPlayer` of `match_id` raises
    `PlayerStatisticsPersistenceError` rather than silently writing a
    `Statistic` row for someone who didn't play in this match. An empty
    mapping is not an error -- it just means zero rows get written this
    call, the same "no signal, no row" posture 9a's own
    STATUS_NOT_COMPUTABLE verdicts already take, here because the
    identity side of the pipeline hasn't resolved anything yet rather
    than because the stat itself is uncomputable.

    Returns the number of `Statistic` rows written.
    """
    with get_sync_db() as db:
        if track_id_to_player_id:
            match_player_ids = set(
                db.execute(
                    select(MatchPlayer.player_id).where(MatchPlayer.match_id == match_id)
                ).scalars()
            )
            unknown_player_ids = set(track_id_to_player_id.values()) - match_player_ids
            if unknown_player_ids:
                raise PlayerStatisticsPersistenceError(
                    f"match_id={match_id}: track_id_to_player_id maps to player_id(s) "
                    f"{sorted(str(pid) for pid in unknown_player_ids)} that are not a MatchPlayer of this "
                    f"match -- refusing to write a Statistic row whose player_id doesn't reflect who "
                    f"actually played in this match"
                )

        # Idempotency: clear only what this stage owns for this match
        # before reinserting (see module docstring).
        db.execute(
            delete(Statistic).where(
                Statistic.match_id == match_id,
                Statistic.player_id.is_not(None),
                Statistic.stat_type.in_(PLAYER_LEVEL_STAT_TYPES),
            )
        )

        written = 0
        skipped_unmapped = 0
        skipped_not_owned = 0
        for stat_value in stat_values:
            stat_type = _STAT_KEY_TO_TYPE.get(stat_value.stat_key)
            if stat_type is None:
                # Not one of the five pending-player-identity stat_keys this
                # stage owns (e.g. a match-level StatValue with
                # track_id=None, already Part 7f's job) -- never this
                # stage's row to write.
                skipped_not_owned += 1
                continue

            player_id = track_id_to_player_id.get(stat_value.track_id) if stat_value.track_id is not None else None
            if player_id is None:
                skipped_unmapped += 1
                continue

            db.add(
                Statistic(
                    match_id=match_id,
                    player_id=player_id,
                    stat_type=stat_type,
                    value=stat_value.value,
                )
            )
            written += 1

        db.commit()

    if skipped_unmapped:
        logger.info(
            "[stats] match_id=%s: skipped %d player-level statistic(s) whose track_id had no "
            "Player.id mapping yet",
            match_id, skipped_unmapped,
        )
    if skipped_not_owned:
        logger.debug(
            "[stats] match_id=%s: skipped %d statistic(s) not owned by this stage (not one of %s)",
            match_id, skipped_not_owned, [t.value for t in PLAYER_LEVEL_STAT_TYPES],
        )

    logger.info(
        "[stats] match_id=%s: persisted %d player-level statistic(s) across %d mapped track_id(s)",
        match_id, written, len(track_id_to_player_id),
    )
    return written
