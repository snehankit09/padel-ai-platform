"""
Frame extraction — pulls frames out of a source video at a fixed sampling
rate and writes them to disk as an ordered sequence of JPEGs.

Why sample instead of extracting every frame (Part 5a): a 90-minute match
at a native 25-30fps is well over 100k frames. Running detection (Part 5b+)
against every one of them wastes GPU time — consecutive frames a fraction
of a second apart look nearly identical to a detector. Sampling at a
fixed, lower rate (see settings.frame_sample_rate_fps in the backend)
keeps the detect stage's input size proportional to match *duration*, not
to the source video's native frame rate.

We shell out to the `ffmpeg` binary directly with `subprocess`, rather than
the `ffmpeg-python` wrapper already used for probing in
app/services/video_validation.py. That module only needs `ffmpeg.probe()`
(read-only, JSON out); this one needs to run a real filter+encode command
with an exact, testable argument list — `build_ffmpeg_frame_command` is
kept separate from `extract_frames` for exactly that reason: it can be
asserted on directly, with no filesystem or subprocess involved.

This module has no Celery/DB/app dependency on purpose. It's a plain,
independently testable function — the backend glues it to a specific
Video row and storage location (see
backend/app/services/frame_extraction_stage.py). Anything in ml/ that
needs frame extraction later (Part 5b+'s detector, a one-off debugging
script, ...) can import this directly without pulling in the backend.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

FRAME_FILENAME_PATTERN = "frame_%06d.jpg"
FRAME_GLOB_PATTERN = "frame_*.jpg"

# ffmpeg's -qscale:v is inverted (lower = better quality). 2 is
# near-lossless for JPEG without ballooning storage the way 1 would — this
# is prototype-stage detection input, not an archival copy of the match.
_JPEG_QSCALE = "2"


class FrameExtractionError(Exception):
    """Raised when ffmpeg fails to run, or runs but produces zero frames."""


@dataclass
class FrameExtractionResult:
    frames_dir: str
    frame_paths: list[str]
    frame_count: int
    sample_fps: float


def build_ffmpeg_frame_command(video_path: str, output_dir: str, sample_fps: float) -> list[str]:
    """
    Pure command-builder — no subprocess, no filesystem access — so the
    exact ffmpeg invocation can be asserted on in tests.

    Uses ffmpeg's `fps` filter rather than a manual "grab every Nth frame"
    loop: the filter resamples against the video's actual (possibly
    variable) frame rate, so `sample_fps` means what it says regardless of
    whether the source is 24, 25, 29.97, or 60fps.
    """
    if sample_fps <= 0:
        raise ValueError(f"sample_fps must be > 0, got {sample_fps!r}")
    pattern = str(Path(output_dir) / FRAME_FILENAME_PATTERN)
    return [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"fps={sample_fps}",
        "-start_number", "0",
        "-qscale:v", _JPEG_QSCALE,
        pattern,
        "-loglevel", "error",
    ]


def extract_frames(video_path: str, output_dir: str, sample_fps: float) -> FrameExtractionResult:
    """
    Extracts frames from `video_path` at `sample_fps` frames per second of
    video content, writing them as frame_000000.jpg, frame_000001.jpg, ...
    into `output_dir`.

    Idempotent: clears any existing frame_*.jpg files in `output_dir`
    first, so a retried stage (the backend's per-stage retry, Part 4d)
    never ends up with stale frames from a previous, failed attempt mixed
    in with the new ones.

    Raises FrameExtractionError if ffmpeg isn't available, fails to run,
    or produces zero frames. Zero frames is treated as a failure rather
    than a silent success — otherwise the detect stage would later run
    against an empty directory and quietly "succeed" at detecting nothing,
    which is a much worse failure mode than raising here, immediately,
    with a specific reason.
    """
    out_dir = Path(output_dir)
    _reset_output_dir(out_dir)

    command = build_ffmpeg_frame_command(video_path, str(out_dir), sample_fps)
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise FrameExtractionError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise FrameExtractionError(
            f"ffmpeg failed to extract frames from {video_path!r}: {stderr or 'unknown error'}"
        ) from exc

    frame_paths = sorted(str(p) for p in out_dir.glob(FRAME_GLOB_PATTERN))
    if not frame_paths:
        raise FrameExtractionError(
            f"ffmpeg produced no frames from {video_path!r} — the source file may be "
            "unreadable, silent/blank, or shorter than one sample interval."
        )

    return FrameExtractionResult(
        frames_dir=str(out_dir),
        frame_paths=frame_paths,
        frame_count=len(frame_paths),
        sample_fps=sample_fps,
    )


def list_frame_paths(frames_dir: str) -> list[str]:
    """
    Returns every frame file a previous extract_frames() call wrote into
    `frames_dir`, sorted in extraction order — for a later stage (Part 5d's
    `detect`) that already knows where frames live, via the frames_dir key
    extract_frames' caller recorded onto the pipeline payload, and just
    needs the file list back without re-running ffmpeg.
    """
    return sorted(str(p) for p in Path(frames_dir).glob(FRAME_GLOB_PATTERN))


def _reset_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        for existing in out_dir.glob(FRAME_GLOB_PATTERN):
            existing.unlink()
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
