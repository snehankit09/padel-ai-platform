"""
Prediction — PRD Section 11 & Module 6: model outputs (win probability,
momentum, fatigue, rankings) per match.

Two things the PRD explicitly requires that shape this schema:
  - "Predictions are versioned so that model updates do not silently
    change historical results without disclosure" -> model_version column.
  - "Player rankings ... accompanied by the metrics that produced them,
    for explainability" -> explanation column (JSON) to hold the
    contributing metrics/SHAP values, not just the bare prediction.
  - Win probability is a *time series*, not a single value -> timestamp
    column, so multiple rows per match/type represent points over time.
"""

import uuid
from datetime import datetime

from sqlalchemy import ForeignKey, Float, String, DateTime, Enum as SAEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase
from app.models.enums import PredictionType


class Prediction(Base, TimestampedBase):
    __tablename__ = "predictions"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"))
    # Nullable: match-wide predictions (win probability) have no single
    # player; player-ranking predictions (strongest player) do.
    player_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("players.id"), nullable=True)

    prediction_type: Mapped[PredictionType] = mapped_column(SAEnum(PredictionType, name="prediction_type"), nullable=False)
    value: Mapped[float] = mapped_column(Float, nullable=False)

    # Point in match time this prediction applies to (for time-series types
    # like win_probability/momentum). NULL for single-shot, whole-match predictions.
    match_time_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    # Which contributing metrics produced this value — explainability requirement
    explanation: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    model_version: Mapped[str] = mapped_column(String(50), nullable=False)
    predicted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    match: Mapped["Match"] = relationship(back_populates="predictions")
    player: Mapped["Player | None"] = relationship()

    def __repr__(self) -> str:
        return f"<Prediction {self.prediction_type}={self.value} v{self.model_version}>"
