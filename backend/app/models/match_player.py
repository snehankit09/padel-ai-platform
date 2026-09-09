"""
MatchPlayer — association *object* linking Match and Player.

Why not a plain Table + secondary=? Because we need to store team_number
on the join itself, and a bare association table can only add/remove
links — it has no way to set extra columns when you do
`match.players.append(player)`. Mapping the join table as its own class
lets us write:
    session.add(MatchPlayer(match=match, player=player_a, team_number=1))
which is explicit about which team the player was on for *this* match
(their partner can differ match to match).
"""

import uuid

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class MatchPlayer(Base):
    __tablename__ = "match_players"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"), primary_key=True)
    player_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"), primary_key=True)
    team_number: Mapped[int] = mapped_column(Integer, nullable=False)

    match: Mapped["Match"] = relationship(back_populates="match_players")
    player: Mapped["Player"] = relationship(back_populates="match_players")

    def __repr__(self) -> str:
        return f"<MatchPlayer match={self.match_id} player={self.player_id} team={self.team_number}>"
