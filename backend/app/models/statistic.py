"""
Statistic — PRD Section 11 & Module 5.

Design choice (flagged before building): one row per (match, player,
stat_type) rather than one wide table with a fixed column per stat.
Reasons:
  1. The PRD lists ~12 stat types now and explicitly plans to add more
     post-MVP (heatmaps, reaction time, momentum) — a wide table means an
     ALTER TABLE + migration every time. This design means a new StatType
     enum value, no schema change.
  2. player_id is nullable: team-level stats (e.g. total team distance)
     and player-level stats (e.g. one player's serve %) share the same
     table instead of needing two.
  3. `value` is a single float, which covers every stat in the PRD list
     today. If we later need multi-dimensional stats (e.g. full heatmap
     grids), those get their own table — forcing them into `value` would
     be the wrong tradeoff.
"""

import uuid

from sqlalchemy import ForeignKey, Float, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase
from app.models.enums import StatType


class Statistic(Base, TimestampedBase):
    __tablename__ = "statistics"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"))
    # NULL player_id = team- or match-level stat
    player_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True)

    stat_type: Mapped[StatType] = mapped_column(SAEnum(StatType, name="stat_type"), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    match: Mapped["Match"] = relationship(back_populates="statistics")
    player: Mapped["Player | None"] = relationship(back_populates="statistics")

    def __repr__(self) -> str:
        who = self.player_id or "team"
        return f"<Statistic {self.stat_type}={self.value} for {who}>"
