"""
Match — PRD Section 11: "Metadata for each match (date, venue,
participants, format)."

This is the hub entity. Nearly everything else (Video, Highlight, Reel,
Statistic, Prediction, Report) belongs to exactly one Match. Deleting a
match should cascade to all of these — see cascade="all, delete-orphan"
below.
"""

import uuid
from datetime import datetime

from sqlalchemy import String, ForeignKey, DateTime
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.associationproxy import association_proxy, AssociationProxy
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase


class Match(Base, TimestampedBase):
    __tablename__ = "matches"

    played_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    venue: Mapped[str | None] = mapped_column(String(255), nullable=True)
    # e.g. "singles", "doubles" — PRD mentions "format" generically;
    # doubles is the default since padel is a 2v2 sport.
    format: Mapped[str] = mapped_column(String(50), default="doubles", nullable=False)

    uploaded_by: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("users.id"))

    uploaded_by_user: Mapped["User"] = relationship(back_populates="matches")

    match_players: Mapped[list["MatchPlayer"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    # Read-through convenience: match.players gives Player objects directly.
    # To ADD a player with a team assignment, create a MatchPlayer explicitly —
    # see the sanity check script for the pattern.
    players: AssociationProxy[list["Player"]] = association_proxy("match_players", "player")

    # one-to-one in practice (single camera per match, per PRD assumption 18.1)
    video: Mapped["Video"] = relationship(back_populates="match", uselist=False, cascade="all, delete-orphan")

    highlights: Mapped[list["Highlight"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    reels: Mapped[list["Reel"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    statistics: Mapped[list["Statistic"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    predictions: Mapped[list["Prediction"]] = relationship(back_populates="match", cascade="all, delete-orphan")
    reports: Mapped[list["Report"]] = relationship(back_populates="match", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Match {self.id} played_at={self.played_at}>"
