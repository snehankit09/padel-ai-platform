"""
ReelHighlight — association object linking Reel and Highlight, carrying
the `position` column (order within the reel). Same reasoning as
MatchPlayer: a plain secondary= table can't set extra columns on append.
"""

import uuid

from sqlalchemy import ForeignKey, Integer
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class ReelHighlight(Base):
    __tablename__ = "reel_highlights"

    reel_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("reels.id"), primary_key=True)
    highlight_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("highlights.id"), primary_key=True)
    position: Mapped[int] = mapped_column(Integer, nullable=False)

    reel: Mapped["Reel"] = relationship(back_populates="reel_highlights")
    highlight: Mapped["Highlight"] = relationship()

    def __repr__(self) -> str:
        return f"<ReelHighlight reel={self.reel_id} highlight={self.highlight_id} pos={self.position}>"
