"""
Video — PRD Section 11: "Uploaded source video files and processing
status." This is what the /videos/upload and /videos/{id}/status
endpoints (Part 3) will read and write.
"""

import uuid

from sqlalchemy import String, ForeignKey, Integer, Float, Enum as SAEnum
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.mixins import TimestampedBase
from app.models.enums import VideoStatus


class Video(Base, TimestampedBase):
    __tablename__ = "videos"

    match_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("matches.id"), unique=True)

    # Path within the configured storage backend (local path or S3 key) —
    # never a public URL. The service layer resolves this to something
    # servable (signed URL, static mount, etc).
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)

    status: Mapped[VideoStatus] = mapped_column(
        SAEnum(VideoStatus, name="video_status"), default=VideoStatus.PENDING, nullable=False
    )
    # Which pipeline task is currently running (e.g. "detect", "track") while
    # status == PROCESSING (Part 4b/4c). Plain string, not an enum: it names
    # a Celery task, not a domain concept, and the set of stage names may
    # still change as Parts 5-9 flesh each one out. Null once status is
    # QUEUED, DONE, or FAILED.
    current_stage: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Set if status == FAILED, so users get a specific error, not a black box
    # (PRD Module 1 acceptance criteria: "clear, specific error message")
    error_message: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolution_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    resolution_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    file_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)

    match: Mapped["Match"] = relationship(back_populates="video")

    def __repr__(self) -> str:
        return f"<Video {self.id} status={self.status}>"
