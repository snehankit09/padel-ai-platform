"""
Player — PRD Section 11: "Player profiles linked across matches for
longitudinal tracking."

A Player appears in many Matches via the MatchPlayer association object
(see match_player.py) rather than a plain many-to-many, because we need
team_number stored on each link.
"""

import uuid

from sqlalchemy import String, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.associationproxy import association_proxy, AssociationProxy
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase


class Player(Base, TimestampedBase):
    __tablename__ = "players"

    full_name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Optional link to a login account — not every tracked player has one
    # (e.g. an opponent captured on video who never signs up).
    user_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id"), nullable=True
    )

    match_players: Mapped[list["MatchPlayer"]] = relationship(back_populates="player")
    # Read-through convenience: player.matches gives you Match objects
    # directly, proxying through the match_players association rows.
    matches: AssociationProxy[list["Match"]] = association_proxy("match_players", "match")

    statistics: Mapped[list["Statistic"]] = relationship(back_populates="player")

    def __repr__(self) -> str:
        return f"<Player {self.full_name}>"
