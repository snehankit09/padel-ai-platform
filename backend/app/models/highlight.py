"""
Highlight — PRD Section 11 & Module 3: individual detected highlight clips
with type, timestamp, and score.
"""

import uuid

from sqlalchemy import String, ForeignKey, Float, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase
from app.models.enums import HighlightType


class Highlight(Base, TimestampedBase):
    __tablename__ = "highlights"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"))

    event_type: Mapped[HighlightType] = mapped_column(SAEnum(HighlightType, name="highlight_type"), nullable=False)

    # In/out points within the source video, including configured pre/post-roll padding
    start_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    end_time_seconds: Mapped[float] = mapped_column(Float, nullable=False)

    # Computed importance score used for ranking (PRD Module 3 acceptance criteria)
    importance_score: Mapped[float] = mapped_column(Float, nullable=False)

    # Path to the trimmed clip file, once Module 3's clip generation step runs
    clip_file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    match: Mapped["Match"] = relationship(back_populates="highlights")

    def __repr__(self) -> str:
        return f"<Highlight {self.event_type} score={self.importance_score:.2f}>"
