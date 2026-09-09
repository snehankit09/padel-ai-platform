"""
Video upload / status schemas.

Kept separate from the SQLAlchemy models (app/models/video.py) on purpose:
the API contract and the DB shape are allowed to drift over time (e.g. we
may want to expose a computed field or hide an internal one), and mixing
them means every model change becomes an API change by accident.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enums import VideoStatus


class VideoUploadResponse(BaseModel):
    """Returned immediately after a successful upload (before any processing runs)."""

    video_id: uuid.UUID
    match_id: uuid.UUID
    status: VideoStatus
    original_filename: str


class VideoStatusResponse(BaseModel):
    """Returned by GET /videos/{id}/status — the shape a client polls."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    match_id: uuid.UUID
    status: VideoStatus
    current_stage: str | None
    error_message: str | None
    original_filename: str
    duration_seconds: float | None
    resolution_width: int | None
    resolution_height: int | None
    fps: float | None
    file_size_bytes: int | None
    created_at: datetime
    updated_at: datetime
