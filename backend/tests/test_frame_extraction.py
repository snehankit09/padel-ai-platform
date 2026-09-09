"""
Tests for ml/common/frame_extraction.py — Part 5a.

Unlike the pipeline chain tests, this needs no DB, no Celery, no
Postgres/Redis — just ffmpeg and a filesystem, since the module itself has
no app dependency (see its docstring). That also means it's the one
pipeline-adjacent test file in this session that could actually be run
directly (`python3 -c ...` against real ffmpeg-generated files, not
mocked) rather than only written against the missing fastapi/sqlalchemy
stack — see the module docstring pattern in test_video_upload.py for the
general caveat that applies to the rest of this test suite.

Run with: pytest backend/tests/test_frame_extraction.py -v
"""

import subprocess
from pathlib import Path

import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.common.frame_extraction import (  # noqa: E402
    FrameExtractionError,
    build_ffmpeg_frame_command,
    extract_frames,
)


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory) -> Path:
    """A real, ffmpeg-generated 3-second/25fps video, reused across this module's tests."""
    video_dir = tmp_path_factory.mktemp("frame_extraction_fixtures")
    video_path = video_dir / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=320x240:rate=25",
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            str(video_path),
            "-loglevel", "error",
        ],
        check=True,
    )
    return video_path


# --- build_ffmpeg_frame_command: pure, no filesystem/subprocess ------------


def test_command_uses_fps_filter_not_frame_skip():
    """
    Must resample by content-rate (`fps=N` filter), not "grab every Nth
    frame" — the latter would silently mean something different on a 24fps
    vs a 60fps source for the same `sample_fps`.
    """
    cmd = build_ffmpeg_frame_command("in.mp4", "out_dir", sample_fps=5.0)
    assert cmd[0] == "ffmpeg"
    assert "-vf" in cmd
    assert cmd[cmd.index("-vf") + 1] == "fps=5.0"


def test_command_starts_numbering_at_zero_and_targets_output_dir():
    cmd = build_ffmpeg_frame_command("in.mp4", "out_dir", sample_fps=5.0)
    assert "-start_number" in cmd
    assert cmd[cmd.index("-start_number") + 1] == "0"
    assert cmd[-2] == str(Path("out_dir") / "frame_%06d.jpg")


def test_command_rejects_non_positive_sample_fps():
    with pytest.raises(ValueError, match="sample_fps must be > 0"):
        build_ffmpeg_frame_command("in.mp4", "out_dir", sample_fps=0)
    with pytest.raises(ValueError):
        build_ffmpeg_frame_command("in.mp4", "out_dir", sample_fps=-2.0)


# --- extract_frames: real ffmpeg, real filesystem ---------------------------


def test_extracts_expected_frame_count_for_duration_and_sample_rate(sample_video_path, tmp_path):
    # 3s source @ sample_fps=5 -> 15 frames, regardless of the source's own 25fps.
    result = extract_frames(str(sample_video_path), str(tmp_path / "frames"), sample_fps=5.0)

    assert result.frame_count == 15
    assert len(result.frame_paths) == 15
    assert result.sample_fps == 5.0
    assert result.frames_dir == str(tmp_path / "frames")


def test_frame_paths_are_sorted_and_zero_indexed(sample_video_path, tmp_path):
    result = extract_frames(str(sample_video_path), str(tmp_path / "frames"), sample_fps=5.0)

    assert result.frame_paths == sorted(result.frame_paths)
    assert Path(result.frame_paths[0]).name == "frame_000000.jpg"
    assert all(Path(p).exists() for p in result.frame_paths)


def test_lower_sample_rate_yields_fewer_frames(sample_video_path, tmp_path):
    fast = extract_frames(str(sample_video_path), str(tmp_path / "fast"), sample_fps=10.0)
    slow = extract_frames(str(sample_video_path), str(tmp_path / "slow"), sample_fps=2.0)

    assert fast.frame_count > slow.frame_count
    assert fast.frame_count == 30
    assert slow.frame_count == 6


def test_retrying_into_same_dir_clears_stale_frames_first(sample_video_path, tmp_path):
    """
    A stage that retries (Part 4d) must not accumulate frames across
    attempts — an interrupted first attempt's leftovers mixed with a
    second, successful attempt's frames would corrupt the sequence
    `detect` (Part 5b+) later reads.
    """
    out_dir = tmp_path / "frames"
    first = extract_frames(str(sample_video_path), str(out_dir), sample_fps=5.0)
    # Simulate a stale leftover from a previous, differently-configured attempt.
    stray = out_dir / "frame_999999.jpg"
    stray.write_bytes(b"not a real frame")

    second = extract_frames(str(sample_video_path), str(out_dir), sample_fps=5.0)

    assert second.frame_count == first.frame_count
    assert not stray.exists()
    assert "frame_999999.jpg" not in {Path(p).name for p in second.frame_paths}


def test_raises_specific_error_for_missing_source_file(tmp_path):
    with pytest.raises(FrameExtractionError, match="does_not_exist.mp4"):
        extract_frames(str(tmp_path / "does_not_exist.mp4"), str(tmp_path / "frames"), sample_fps=5.0)


def test_raises_specific_error_for_unreadable_source_file(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00" * 2000)

    with pytest.raises(FrameExtractionError):
        extract_frames(str(corrupt), str(tmp_path / "frames"), sample_fps=5.0)
