"""
Reel — PRD Section 11 & Module 4: generated reels, their composition, and
export status.

A reel is composed of an ordered sequence of highlights via the
ReelHighlight association object (position column), same pattern as
MatchPlayer/team_number.
"""

import uuid

from sqlalchemy import String, ForeignKey, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.associationproxy import association_proxy, AssociationProxy
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase
from app.models.enums import ReelStatus


class Reel(Base, TimestampedBase):
    __tablename__ = "reels"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"))

    status: Mapped[ReelStatus] = mapped_column(
        SAEnum(ReelStatus, name="reel_status"), default=ReelStatus.PENDING, nullable=False
    )
    file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    # Which background music track is applied — swappable without regenerating
    # the reel per PRD Module 4 acceptance criteria.
    music_track: Mapped[str | None] = mapped_column(String(255), nullable=True)

    match: Mapped["Match"] = relationship(back_populates="reels")

    reel_highlights: Mapped[list["ReelHighlight"]] = relationship(
        back_populates="reel", cascade="all, delete-orphan", order_by="ReelHighlight.position"
    )
    # Read-through convenience, already ordered by position via the relationship above.
    highlights: AssociationProxy[list["Highlight"]] = association_proxy("reel_highlights", "highlight")

    def __repr__(self) -> str:
        return f"<Reel {self.id} status={self.status}>"
