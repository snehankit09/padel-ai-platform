"""
Report — PRD Section 11 & Module 7: generated downloadable report
artifacts (PDF summarizing statistics and predictions for a match).
"""

import uuid

from sqlalchemy import ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase


class Report(Base, TimestampedBase):
    __tablename__ = "reports"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"))
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)

    match: Mapped["Match"] = relationship(back_populates="reports")

    def __repr__(self) -> str:
        return f"<Report {self.id} for match={self.match_id}>"
