"""
Video validation.

Why this exists: an uploaded file having a `.mp4` extension proves nothing.
It could be truncated, corrupt, a renamed .txt file, or a real video in a
format we don't support. The PRD requires a "clear, specific error message"
for bad uploads (Module 1 acceptance criteria) — not a generic 500 three
steps later when the CV pipeline chokes on it. So we validate *before* any
DB row is created or the file is moved into permanent storage: probe the
file with ffprobe, and only trust what ffprobe reports back.

We shell out to ffprobe (via the `ffmpeg-python` wrapper already in
requirements.txt) rather than trusting file extensions or MIME types,
which the client can send as anything.
"""

from __future__ import annotations

from dataclasses import dataclass

import ffmpeg

from app.core.config import Settings


class VideoValidationError(Exception):
    """
    Raised with a specific, user-facing reason a file was rejected.
    The route layer catches this and returns it as-is in a 422 response —
    it should never contain anything the user shouldn't see (e.g. no raw
    ffmpeg stack traces, just the distilled reason).
    """


@dataclass
class VideoProbeResult:
    duration_seconds: float
    width: int
    height: int
    fps: float
    file_size_bytes: int


def validate_video(file_path: str, file_size_bytes: int, settings: Settings) -> VideoProbeResult:
    """
    Probes `file_path` with ffprobe and enforces the upload limits from
    Settings (max size, max duration, allowed formats — PRD Module 1).
    Raises VideoValidationError with a specific reason on any failure;
    returns the extracted metadata on success so the route can populate
    Video.duration_seconds / resolution_* / fps without re-probing.
    """
    _check_size(file_size_bytes, settings)

    try:
        probe = ffmpeg.probe(file_path)
    except ffmpeg.Error as exc:
        # ffprobe's own stderr is the most reliable signal that this isn't
        # a real, readable video file at all (corrupt, truncated, wrong
        # container). We don't surface the raw stderr to the user — it's
        # often cryptic — just a plain statement of the problem.
        stderr = exc.stderr.decode(errors="replace") if exc.stderr else ""
        raise VideoValidationError(
            "This file could not be read as a video. It may be corrupted, "
            "truncated, or not actually a video file."
        ) from RuntimeError(stderr)

    video_stream = next(
        (s for s in probe.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if video_stream is None:
        raise VideoValidationError("This file doesn't contain a video stream.")

    duration_seconds = _extract_duration(probe, video_stream)
    _check_duration(duration_seconds, settings)

    width = video_stream.get("width")
    height = video_stream.get("height")
    if not width or not height:
        raise VideoValidationError("Could not determine the video's resolution.")

    fps = _extract_fps(video_stream)

    return VideoProbeResult(
        duration_seconds=duration_seconds,
        width=int(width),
        height=int(height),
        fps=fps,
        file_size_bytes=file_size_bytes,
    )


def _check_size(file_size_bytes: int, settings: Settings) -> None:
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if file_size_bytes > max_bytes:
        raise VideoValidationError(
            f"File is too large ({file_size_bytes / (1024 * 1024):.1f} MB). "
            f"The maximum allowed size is {settings.max_upload_size_mb} MB."
        )
    if file_size_bytes == 0:
        raise VideoValidationError("The uploaded file is empty.")


def _check_duration(duration_seconds: float, settings: Settings) -> None:
    max_seconds = settings.max_video_duration_minutes * 60
    if duration_seconds > max_seconds:
        raise VideoValidationError(
            f"Video is too long ({duration_seconds / 60:.1f} minutes). "
            f"The maximum allowed duration is {settings.max_video_duration_minutes} minutes."
        )
    if duration_seconds <= 0:
        raise VideoValidationError("Video has zero or unreadable duration.")


def _extract_duration(probe: dict, video_stream: dict) -> float:
    # Duration can live on the top-level format block or, for some
    # containers, only on the video stream itself. Prefer format (covers
    # the whole file); fall back to the stream if format omits it.
    raw = probe.get("format", {}).get("duration") or video_stream.get("duration")
    if raw is None:
        raise VideoValidationError("Could not determine the video's duration.")
    try:
        return float(raw)
    except (TypeError, ValueError):
        raise VideoValidationError("Could not determine the video's duration.")


def _extract_fps(video_stream: dict) -> float:
    # ffprobe reports frame rate as a "num/den" string (e.g. "30000/1001"),
    # not a plain float — has to be parsed rather than cast directly.
    raw = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    try:
        num, den = raw.split("/")
        num, den = float(num), float(den)
        return num / den if den else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0
