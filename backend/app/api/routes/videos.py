"""
Video upload & status routes.

Design recap (from the build log): this route validates, stores, and
writes a `Video` row with status=pending. It does NOT process the video —
that's the async job pipeline (Part 4). Keeping upload fast and
processing separate is why Celery/Redis are in the stack at all.

Flow for POST /videos/upload:
  1. Stream the upload to a temp file (never trust it's valid yet).
  2. Validate it with ffprobe (app.services.video_validation). Invalid ->
     422 with a specific reason, nothing touches the DB or permanent
     storage.
  3. Valid -> create the Match this video belongs to (uploaded_by is the
     placeholder user until real auth exists), move the file into
     permanent storage, write the Video row with status=pending and the
     metadata ffprobe already gave us.
"""

from __future__ import annotations

import os
import tempfile
import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.database import get_db
from app.models.enums import VideoStatus
from app.models.match import Match
from app.models.video import Video
from app.schemas.video import VideoStatusResponse, VideoUploadResponse
from app.services.placeholder_user import get_or_create_placeholder_user
from app.services.storage import (
    StorageError,
    StorageService,
    get_storage_service,
    make_video_destination_key,
)
from app.services.video_validation import VideoValidationError, validate_video
from app.workers.tasks import process_video

router = APIRouter()


@router.post("/upload", response_model=VideoUploadResponse, status_code=201)
async def upload_video(
    file: UploadFile = File(...),
    played_at: datetime | None = Form(None),
    venue: str | None = Form(None),
    format: str = Form("doubles"),
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings),
    storage: StorageService = Depends(get_storage_service),
) -> VideoUploadResponse:
    original_filename = file.filename or "upload"
    extension = os.path.splitext(original_filename)[1].lstrip(".").lower()
    if extension not in settings.allowed_video_formats_list:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '.{extension}'. "
                f"Allowed formats: {', '.join(settings.allowed_video_formats_list)}."
            ),
        )

    # Stream to a temp file rather than reading the whole upload into memory —
    # PRD allows videos up to max_upload_size_mb (default 2GB), which we do
    # not want sitting in process memory at once.
    temp_fd, temp_path = tempfile.mkstemp(suffix=f".{extension}")
    file_size_bytes = 0
    try:
        with os.fdopen(temp_fd, "wb") as temp_file:
            while chunk := await file.read(1024 * 1024):
                temp_file.write(chunk)
                file_size_bytes += len(chunk)

        try:
            probe_result = validate_video(temp_path, file_size_bytes, settings)
        except VideoValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        placeholder_user = await get_or_create_placeholder_user(db)

        match = Match(
            played_at=played_at or datetime.now(timezone.utc),
            venue=venue,
            format=format,
            uploaded_by=placeholder_user.id,
        )
        db.add(match)
        await db.flush()  # populate match.id for the storage key and Video FK

        destination_key = make_video_destination_key(match.id, original_filename)
        try:
            stored_path = storage.save(temp_path, destination_key)
        except StorageError as exc:
            raise HTTPException(status_code=500, detail="Could not save the uploaded file.") from exc

        video = Video(
            match_id=match.id,
            file_path=stored_path,
            original_filename=original_filename,
            status=VideoStatus.PENDING,
            duration_seconds=probe_result.duration_seconds,
            resolution_width=probe_result.width,
            resolution_height=probe_result.height,
            fps=probe_result.fps,
            file_size_bytes=probe_result.file_size_bytes,
        )
        db.add(video)
        await db.commit()
        await db.refresh(video)
        video.status = VideoStatus.QUEUED
        await db.commit()
        process_video.delay(str(video.id))

        return VideoUploadResponse(
            video_id=video.id,
            match_id=match.id,
            status=video.status,
            original_filename=video.original_filename,
        )
    finally:
        # storage.save() moves the temp file on success, so this is a no-op
        # in the happy path; it only matters when validation/storage failed
        # and the temp file was never moved.
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("/{video_id}/status", response_model=VideoStatusResponse)
async def get_video_status(video_id: uuid.UUID, db: AsyncSession = Depends(get_db)) -> Video:
    result = await db.execute(select(Video).where(Video.id == video_id))
    video = result.scalar_one_or_none()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found.")
    return video
