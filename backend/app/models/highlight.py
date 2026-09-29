"""
Highlight — PRD Section 11 & Module 3: individual detected highlight clips
with type, timestamp, and score.
"""

import uuid

from sqlalchemy import String, ForeignKey, Float, Integer, Enum as SAEnum
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

    # Path to this clip's single-frame poster image — Highlights Improvement
    # Roadmap Tier 1a. Same lifecycle as clip_file_path: NULL until Part 8b
    # (app/services/clip_extraction_stage.py) successfully extracts one, and
    # a clip whose extraction failed leaves both this and clip_file_path
    # unset rather than one succeeding without the other, since both come
    # from the same try/except block in that stage.
    thumbnail_file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    # Raw ByteTrack track_id of this event's one natural "owner" shot --
    # Highlights Improvement Roadmap Tier 3a (ml/pipeline/highlight_tagging.py's
    # HighlightEvent.player_track_id)/3b (this column). Set only for
    # POWERFUL_SMASH and SPECTACULAR_SAVE (see that module's own docstring
    # for why LONG_RALLY/FAST_EXCHANGE never get one). Deliberately NOT a
    # ForeignKey to app.models.player.Player: a ByteTrack track_id isn't a
    # stable Player identity anywhere in this codebase yet -- the same gap
    # app/services/player_statistics_stage.py's own module docstring
    # already documents for its own track_id-keyed stats (no track_id ->
    # Player.id mapping exists for an automated pipeline run). Resolving
    # this raw id to a human-readable label is player_label, below.
    player_track_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # player_track_id resolved to a human-readable label -- "Player A"
    # .."Player D", or a "Near Team"/"Far Team" fallback -- via Tier
    # 2's ml.pipeline.team_side_positions.assign_player_label. Highlights
    # Improvement Roadmap Tier 3c (app/services/team_side_label_resolution_stage.py):
    # set in a small UPDATE step at the very end of `analyze`, once BOTH
    # this row (Part 7f) and team_side_labels.json (Tier 2d) exist --
    # see that stage's own module docstring for why label resolution
    # can't happen any earlier in the same run. Stays NULL when
    # player_track_id is itself NULL (nothing to resolve), and also when
    # a non-NULL player_track_id has no entry in that run's track_labels
    # lookup (Tier 1d skipped that track_id entirely -- see
    # ml.pipeline.team_side_positions.merge_track_fragments' own
    # docstring for when that happens); either way this stays honestly
    # unset rather than guessing a label.
    player_label: Mapped[str | None] = mapped_column(String(64), nullable=True)

    match: Mapped["Match"] = relationship(back_populates="highlights")

    def __repr__(self) -> str:
        return f"<Highlight {self.event_type} score={self.importance_score:.2f}>"
