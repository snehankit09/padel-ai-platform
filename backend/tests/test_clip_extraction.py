"""
Tests for ml/common/clip_extraction.py — Part 8b.

Same "no DB, no Celery — just ffmpeg and a filesystem" approach as
test_frame_extraction.py, since the module itself has no app dependency
(see its docstring).

Run with: pytest backend/tests/test_clip_extraction.py -v
"""

import subprocess
from pathlib import Path

import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.common.clip_extraction import (  # noqa: E402
    ClipExtractionError,
    _escape_drawtext_text,
    build_ffmpeg_clip_command,
    extract_clip,
)


@pytest.fixture(scope="module")
def sample_video_path(tmp_path_factory) -> Path:
    """A real, ffmpeg-generated 10-second/25fps video with a tone, reused across this module's tests."""
    video_dir = tmp_path_factory.mktemp("clip_extraction_fixtures")
    video_path = video_dir / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=10:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=10",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video_path),
            "-loglevel", "error",
        ],
        check=True,
    )
    return video_path


def _ffprobe_duration_s(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    return float(result.stdout.strip())


# --- build_ffmpeg_clip_command: pure, no filesystem/subprocess -------------


def test_command_seeks_before_input_not_after():
    """
    `-ss` must come before `-i` (input seeking) — see module docstring for
    why this is both faster and (combined with a re-encode) frame-accurate,
    unlike output seeking or a keyframe-snapped `-c copy`.
    """
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=10.0, end_time_s=15.0)
    ss_index = cmd.index("-ss")
    i_index = cmd.index("-i")
    assert ss_index < i_index
    assert cmd[ss_index + 1] == "10.000"


def test_command_uses_duration_not_absolute_end_timestamp():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=10.0, end_time_s=15.5)
    assert "-t" in cmd
    assert cmd[cmd.index("-t") + 1] == "5.500"
    assert "-to" not in cmd


def test_command_re_encodes_rather_than_stream_copies():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0)
    assert "copy" not in cmd
    assert "-c:v" in cmd
    assert "-c:a" in cmd


def test_command_targets_output_path():
    cmd = build_ffmpeg_clip_command("in.mp4", "out/clip_000.mp4", start_time_s=0.0, end_time_s=1.0)
    assert "out/clip_000.mp4" in cmd


def test_command_rejects_non_positive_duration():
    with pytest.raises(ValueError, match="end_time_s"):
        build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=10.0, end_time_s=10.0)
    with pytest.raises(ValueError):
        build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=10.0, end_time_s=5.0)


# --- build_ffmpeg_clip_command: encoding/quality settings (Part 8c) --------


def test_command_caps_resolution_via_downscale_only_filter():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0)
    assert "-vf" in cmd
    vf = cmd[cmd.index("-vf") + 1]
    # min(ih, 1080) never upscales — see _scale_filter's own docstring —
    # and the second chained scale guarantees even output dimensions.
    assert "min(ih,1080)" in vf
    assert "trunc(iw/2)*2:trunc(ih/2)*2" in vf


def test_command_forces_yuv420p_pixel_format():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0)
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"


def test_command_normalizes_audio_regardless_of_source():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0)
    assert cmd[cmd.index("-ar") + 1] == "48000"
    assert cmd[cmd.index("-ac") + 1] == "2"
    assert cmd[cmd.index("-b:a") + 1] == "128k"


def test_command_enables_faststart_for_progressive_playback():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0)
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"


# --- build_ffmpeg_clip_command: highlight-type label overlay (Part 8d) -----


def test_command_omits_drawtext_when_no_label_given():
    cmd = build_ffmpeg_clip_command("in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0)
    assert "drawtext" not in cmd[cmd.index("-vf") + 1]


def test_command_includes_drawtext_with_the_given_label():
    cmd = build_ffmpeg_clip_command(
        "in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0, label_text="LONG RALLY"
    )
    vf = cmd[cmd.index("-vf") + 1]
    assert "drawtext=" in vf
    assert "text='LONG RALLY'" in vf
    # The label must come after the resolution-cap filters in the chain —
    # see _build_video_filter's docstring for why the order matters
    # (label pixel size should be constant regardless of source resolution).
    assert vf.index("scale=") < vf.index("drawtext=")


def test_command_references_a_font_file_not_a_fontconfig_family_lookup():
    # A `font=` (family-name/fontconfig) lookup would silently depend on
    # some matching font existing at runtime; this asserts the safer
    # `fontfile=<path>` form is used instead — see the module docstring.
    cmd = build_ffmpeg_clip_command(
        "in.mp4", "out.mp4", start_time_s=0.0, end_time_s=1.0, label_text="LONG RALLY"
    )
    vf = cmd[cmd.index("-vf") + 1]
    assert "fontfile=" in vf
    assert ".ttf" in vf


def test_escape_drawtext_text_escapes_filtergraph_special_characters():
    # Every character FFmpeg's filtergraph parser gives special meaning to
    # inside an option value must come back backslash-escaped.
    assert _escape_drawtext_text("a:b") == "a\\:b"
    assert _escape_drawtext_text("a,b") == "a\\,b"
    assert _escape_drawtext_text("a'b") == "a\\'b"
    assert _escape_drawtext_text("a\\b") == "a\\\\b"


def test_escape_drawtext_text_does_not_double_escape_existing_backslashes():
    # Escaping "\" has to happen first, or escaping the other characters
    # afterward would double-escape a backslash that escaping "\" already
    # introduced.
    assert _escape_drawtext_text("a\\:b") == "a\\\\\\:b"


# --- extract_clip: real ffmpeg, real filesystem -----------------------------


def test_extracts_a_clip_of_the_requested_duration(sample_video_path, tmp_path):
    output_path = tmp_path / "clips" / "clip_000.mp4"
    result = extract_clip(str(sample_video_path), str(output_path), start_time_s=2.0, end_time_s=6.0)

    assert result.duration_s == 4.0
    assert result.start_time_s == 2.0
    assert result.end_time_s == 6.0
    assert output_path.exists()
    assert result.file_size_bytes > 0
    assert result.file_size_bytes == output_path.stat().st_size

    # The cut file's own probed duration should be close to what was asked
    # for — some encoder rounding is expected, but not seconds' worth.
    assert _ffprobe_duration_s(output_path) == pytest.approx(4.0, abs=0.5)


def test_creates_parent_directory_if_missing(sample_video_path, tmp_path):
    output_path = tmp_path / "does" / "not" / "exist" / "clip.mp4"
    result = extract_clip(str(sample_video_path), str(output_path), start_time_s=0.0, end_time_s=2.0)
    assert Path(result.output_path).exists()


def test_raises_specific_error_for_missing_source_file(tmp_path):
    with pytest.raises(ClipExtractionError, match="does_not_exist.mp4"):
        extract_clip(
            str(tmp_path / "does_not_exist.mp4"), str(tmp_path / "clip.mp4"),
            start_time_s=0.0, end_time_s=1.0,
        )


def test_raises_specific_error_for_unreadable_source_file(tmp_path):
    corrupt = tmp_path / "corrupt.mp4"
    corrupt.write_bytes(b"\x00" * 2000)

    with pytest.raises(ClipExtractionError):
        extract_clip(str(corrupt), str(tmp_path / "clip.mp4"), start_time_s=0.0, end_time_s=1.0)


def _ffprobe_video_stream(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,pix_fmt",
            "-of", "csv=p=0",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    width, height, pix_fmt = result.stdout.strip().split(",")
    return {"width": int(width), "height": int(height), "pix_fmt": pix_fmt}


def _ffprobe_audio_stream(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a:0",
            "-show_entries", "stream=sample_rate,channels",
            "-of", "csv=p=0",
            str(path),
        ],
        check=True, capture_output=True, text=True,
    )
    sample_rate, channels = result.stdout.strip().split(",")
    return {"sample_rate": int(sample_rate), "channels": int(channels)}


@pytest.fixture(scope="module")
def oversized_video_path(tmp_path_factory) -> Path:
    """
    A real 1440p source, taller than the 1080p output cap — reused across
    this module's Part 8c (encoding/quality) tests to confirm the cap is
    actually enforced end-to-end, not just present in the built command.
    """
    video_dir = tmp_path_factory.mktemp("clip_extraction_oversized_fixture")
    video_path = video_dir / "oversized.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", "testsrc=duration=3:size=2560x1440:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(video_path),
            "-loglevel", "error",
        ],
        check=True,
    )
    return video_path


def test_extracted_clip_is_downscaled_to_the_output_cap(oversized_video_path, tmp_path):
    output_path = tmp_path / "clip.mp4"
    extract_clip(str(oversized_video_path), str(output_path), start_time_s=0.0, end_time_s=2.0)

    video = _ffprobe_video_stream(output_path)
    assert video["height"] == 1080
    # Width is derived to preserve the source's aspect ratio (16:9 here),
    # not hardcoded — 2560x1440 -> 1920x1080 is exactly 16:9 at the cap.
    assert video["width"] == 1920
    assert video["pix_fmt"] == "yuv420p"


def test_extracted_clip_from_a_source_already_under_the_cap_is_not_upscaled(sample_video_path, tmp_path):
    # sample_video_path is 320x240 — well under the 1080p cap.
    output_path = tmp_path / "clip.mp4"
    extract_clip(str(sample_video_path), str(output_path), start_time_s=0.0, end_time_s=2.0)

    video = _ffprobe_video_stream(output_path)
    assert video["width"] == 320
    assert video["height"] == 240


def test_extracted_clip_audio_is_normalized(oversized_video_path, tmp_path):
    output_path = tmp_path / "clip.mp4"
    extract_clip(str(oversized_video_path), str(output_path), start_time_s=0.0, end_time_s=2.0)

    audio = _ffprobe_audio_stream(output_path)
    assert audio["sample_rate"] == 48000
    assert audio["channels"] == 2


def test_extracted_clip_with_a_label_still_produces_a_valid_playable_file(sample_video_path, tmp_path):
    # Doesn't (and can't cheaply) assert on rendered pixel content — just
    # that adding the drawtext filter doesn't break the encode, on a
    # source small enough that the label overlay is a meaningful fraction
    # of the frame.
    output_path = tmp_path / "clip.mp4"
    result = extract_clip(
        str(sample_video_path), str(output_path), start_time_s=1.0, end_time_s=3.0,
        label_text="POWERFUL_SMASH",
    )
    assert output_path.exists()
    assert result.file_size_bytes > 0
    video = _ffprobe_video_stream(output_path)
    assert video["width"] > 0 and video["height"] > 0


def test_raises_specific_error_when_requested_window_is_past_the_video(sample_video_path, tmp_path):
    # sample_video_path is 10s long; requesting a window starting well past
    # that produces no frames at all, which must raise rather than silently
    # writing an empty/zero-byte file.
    with pytest.raises(ClipExtractionError):
        extract_clip(
            str(sample_video_path), str(tmp_path / "clip.mp4"),
            start_time_s=100.0, end_time_s=105.0,
        )
