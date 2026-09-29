"""
Reel assembly — Part 10c, the FFmpeg-touching sibling of Part 8b
(ml/common/clip_extraction.py). Parts 10a/10b already answer *which*
already-cut clips make the reel and *how* they're timed against each
other (ClipCandidate selection, ReelTimelineEntry pacing) — still just
numbers, no combined video file anywhere. This module is the last step:
turn an ordered list of real clip files plus their timeline into one
real, playable reel video.

**What this module deliberately does and doesn't do.** It concatenates
clips with a hard cut, filling any reserved `gap_before_s` (Part 10b)
with a plain black segment of matching duration — no crossfades, no
music, no title cards, no score overlays. This is the same "Part 8d
[overlay/annotation] is a separate, later decision, not attempted here"
posture ml.common.clip_extraction.py's own module docstring already
takes toward player names and score text: nothing else in this codebase
does audio mixing, transition rendering, or title-card generation, so
building any of that into this module would be inventing a capability
rather than using one that exists. A black-segment hard cut is the
honest floor every one of those richer treatments could later replace
piece by piece, without this module's own job (concatenate real files
into one real file) changing shape.

**Why every input clip AND every generated gap segment is re-encoded to
the exact same codec/resolution/pixel-format/audio settings as
ml.common.clip_extraction's own Part 8c normalization** (kept as a
second, independent copy of those constants here, not an import — see
ml.pipeline.point_outcome's own BallPoint for the precedent on why a
small, cheap-to-duplicate shape stays independent per module rather than
creating a cross-module dependency for its own sake). FFmpeg's concat
DEMUXER can stream-copy (`-c copy`, no re-encode, fast) when every input
already shares identical codec parameters — which Part 8's clips already
do, since every one of them came out of the exact same
build_ffmpeg_clip_command. The one thing that ISN'T already guaranteed to
match is a freshly-generated black gap segment, so this module builds
those to the identical spec on purpose, which is what makes the whole
concat safe to stream-copy rather than needing its own re-encode pass on
top of the two re-encodes (Part 8b's clip cut, Part 8c's normalization)
that already happened per clip.

**Why the concat DEMUXER (`-f concat`), not the concat FILTER
(`-filter_complex concat`).** The filter graph re-encodes everything it
touches, which would mean re-encoding the whole reel a third time on top
of Part 8's own two passes — real, avoidable cost for output that's
already uniform going in. The demuxer's stream-copy path only works
because of the normalization guarantee above; if that guarantee ever
stops holding (e.g. Part 8's clips start varying in resolution per
video), the concat filter's re-encode-everything behavior would become
the correct choice instead, at the cost this module's `-c copy` currently
avoids.

Same "pure, independently testable, no-Celery/DB" shape as
ml.common.clip_extraction.py, for the same reason: it knows nothing about
Reel, ReelHighlight, or Highlight rows — just a list of (real) file paths,
gap durations, and where to write the result. Whatever Part 10's own glue
stage becomes is what turns persisted ClipCandidates and a computed
timeline into the plain arguments this module actually needs.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Sequence

# Kept byte-for-byte identical to ml.common.clip_extraction's own Part 8c
# constants — see module docstring for why this is a second copy, not an
# import, and why keeping them identical (not just "close enough") is what
# makes -c copy concatenation safe in the first place.
_VIDEO_CODEC = "libx264"
_AUDIO_CODEC = "aac"
_VIDEO_CRF = "20"
_VIDEO_PRESET = "veryfast"
_PIXEL_FORMAT = "yuv420p"
_MAX_OUTPUT_HEIGHT_PX = 1080
_AUDIO_SAMPLE_RATE_HZ = "48000"
_AUDIO_CHANNELS = "2"
_AUDIO_BITRATE = "128k"
_MOVFLAGS = "+faststart"

_MIN_GAP_S_TO_RENDER = 0.05  # gaps shorter than this are imperceptible; skip generating a segment for them at all

# Title-card text styling (Part 10d) — same font file as
# ml.common.clip_extraction's own burned-in label (Part 8d), for a
# consistent look between what's drawn on individual clips and what's
# drawn on the reel's own intro/outro. Kept as an independent copy, not
# an import — see module docstring's precedent on why.
_LABEL_FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_TITLE_FONT_SIZE = "48"
_SUBTITLE_FONT_SIZE = "28"
_TITLE_FONT_COLOR = "white"
_TITLE_LINE_SPACING_PX = 56
_DEFAULT_TITLE_CARD_DURATION_S = 2.5


class ReelAssemblyError(Exception):
    """Raised when ffmpeg isn't available, fails to run, or produces no playable reel file."""


@dataclass
class ReelAssemblyResult:
    output_path: str
    clip_count: int
    total_duration_s: float
    file_size_bytes: int


def _probe_clip_frame_size(clip_path: str) -> tuple[int, int]:
    """
    Real width/height of an already-normalized Part 8 clip, via ffprobe —
    same "don't trust anything but ffprobe" posture
    app/services/video_validation.py and ml.common.clip_extraction's own
    _has_video_stream check already use. Needed to build a black gap
    segment at the exact same frame size (Part 8c already caps every
    clip's height at _MAX_OUTPUT_HEIGHT_PX and scales width to match the
    source's own aspect ratio, so two different source videos' clips can
    legitimately have different widths even after normalization — this
    reads the real value per-reel rather than assuming one).
    """
    result = subprocess.run(
        [
            "ffprobe", "-v", "quiet", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "json", clip_path,
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise ReelAssemblyError(f"ffprobe could not read frame size from {clip_path!r}: {result.stderr.strip()}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise ReelAssemblyError(f"{clip_path!r} has no video stream to read a frame size from")
    return int(streams[0]["width"]), int(streams[0]["height"])


def build_ffmpeg_gap_command(output_path: str, duration_s: float, width: int, height: int) -> list[str]:
    """
    Pure command-builder for one black gap segment, same split as
    build_ffmpeg_clip_command in ml.common.clip_extraction.py. Generates
    `duration_s` seconds of silent black video, encoded to the exact same
    spec as every real clip (see module docstring) so it concatenates
    with them via stream copy with no further re-encode.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
    return [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration_s:.3f}",
        "-f", "lavfi", "-i", f"anullsrc=r={_AUDIO_SAMPLE_RATE_HZ}:cl=stereo",
        "-t", f"{duration_s:.3f}",
        "-c:v", _VIDEO_CODEC, "-preset", _VIDEO_PRESET, "-crf", _VIDEO_CRF, "-pix_fmt", _PIXEL_FORMAT,
        "-c:a", _AUDIO_CODEC, "-ar", _AUDIO_SAMPLE_RATE_HZ, "-ac", _AUDIO_CHANNELS, "-b:a", _AUDIO_BITRATE,
        "-movflags", _MOVFLAGS,
        output_path,
        "-loglevel", "error",
    ]


def _escape_drawtext_text(text: str) -> str:
    """
    Independent copy of ml.common.clip_extraction._escape_drawtext_text —
    see that module's docstring for the exact escaping rules (four
    filtergraph-special characters, `\\` escaped first). Kept as a
    second copy rather than an import for the same per-module
    independence reason as this module's Part 8c constants above.
    """
    for char in ("\\", ":", "'", ","):
        text = text.replace(char, "\\" + char)
    return text


def _title_card_drawtext_filters(lines: Sequence[str]) -> str:
    """
    Centers `lines` as a vertically-stacked block around the frame's own
    mid-height — extends ml.common.clip_extraction._drawtext_filter's
    single-line, fixed-corner approach (right for a per-clip label) to
    the multi-line, centered layout a title card actually needs. The
    first line renders larger (`_TITLE_FONT_SIZE`) than the rest
    (`_SUBTITLE_FONT_SIZE`) — a title card is a heading plus supporting
    detail, not a list of equally-weighted lines.
    """
    if not lines:
        raise ValueError("_title_card_drawtext_filters needs at least one line")

    block_height_px = _TITLE_LINE_SPACING_PX * (len(lines) - 1)
    filters = []
    for index, line in enumerate(lines):
        escaped = _escape_drawtext_text(line)
        font_size = _TITLE_FONT_SIZE if index == 0 else _SUBTITLE_FONT_SIZE
        y_expr = f"(h-{block_height_px})/2+{index * _TITLE_LINE_SPACING_PX}-th/2"
        filters.append(
            f"drawtext=fontfile={_LABEL_FONT_FILE}:text='{escaped}'"
            f":fontsize={font_size}:fontcolor={_TITLE_FONT_COLOR}"
            f":x=(w-text_w)/2:y={y_expr}"
        )
    return ",".join(filters)


def build_ffmpeg_title_card_command(
    output_path: str,
    lines: Sequence[str],
    *,
    duration_s: float = _DEFAULT_TITLE_CARD_DURATION_S,
    width: int,
    height: int,
) -> list[str]:
    """
    Pure command-builder for one title/outro card (Part 10d) — a plain
    black background with `lines` of centered text burned in, encoded to
    the exact same spec as every real clip and gap segment (see module
    docstring) so it concatenates via stream copy right alongside them,
    no different from any other entry in assemble_reel's own
    `ordered_clip_paths`.

    Deliberately just text on black, not a branded template, animated
    logo, or score graphic — see module docstring's "what this module
    deliberately does and doesn't do" section: nothing in this codebase
    renders a real title-card design or has brand assets to draw from,
    so building either would be inventing a capability rather than using
    one that exists. This is the honest floor a real design could later
    replace, same relationship the black gap segment already has to a
    future crossfade.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s!r}")
    if not lines:
        raise ValueError("build_ffmpeg_title_card_command needs at least one line of text")
    return [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=black:s={width}x{height}:d={duration_s:.3f}",
        "-f", "lavfi", "-i", f"anullsrc=r={_AUDIO_SAMPLE_RATE_HZ}:cl=stereo",
        "-t", f"{duration_s:.3f}",
        "-vf", _title_card_drawtext_filters(lines),
        "-c:v", _VIDEO_CODEC, "-preset", _VIDEO_PRESET, "-crf", _VIDEO_CRF, "-pix_fmt", _PIXEL_FORMAT,
        "-c:a", _AUDIO_CODEC, "-ar", _AUDIO_SAMPLE_RATE_HZ, "-ac", _AUDIO_CHANNELS, "-b:a", _AUDIO_BITRATE,
        "-movflags", _MOVFLAGS,
        output_path,
        "-loglevel", "error",
    ]


def generate_title_card(
    output_path: str,
    lines: Sequence[str],
    *,
    duration_s: float = _DEFAULT_TITLE_CARD_DURATION_S,
    width: int,
    height: int,
) -> str:
    """
    Renders one title/outro card to a real file at `output_path` and
    returns that same path — the file-producing counterpart to
    build_ffmpeg_title_card_command, same split as extract_clip vs.
    build_ffmpeg_clip_command in ml.common.clip_extraction.py.

    A caller builds a reel WITH an intro/outro by generating one (or
    two) of these, then handing their paths to assemble_reel as the
    first/last entries of `ordered_clip_paths` — this function doesn't
    call assemble_reel itself, keeping "what's a title card" and "how do
    clips get concatenated" as separate, composable pieces, same "thin,
    composable pieces" reasoning ml.pipeline.reel_ordering.py's own
    module docstring already gives for keeping ordering and timeline
    math in separate functions.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    command = build_ffmpeg_title_card_command(output_path, lines, duration_s=duration_s, width=width, height=height)
    _run_ffmpeg(command, context="title card")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ReelAssemblyError(f"ffmpeg produced no output file at {output_path!r}")
    return output_path


def build_ffmpeg_concat_command(concat_list_path: str, output_path: str) -> list[str]:
    """
    Pure command-builder for the final concat step. `-safe 0` allows the
    concat list to reference absolute paths (every clip/gap segment this
    module deals with already is one) — without it, ffmpeg's concat
    demuxer refuses any path containing a `/`, which would reject every
    real path this pipeline ever produces.
    """
    return [
        "ffmpeg", "-y",
        "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy",  # safe only because every input already shares identical encoding — see module docstring
        "-movflags", _MOVFLAGS,
        output_path,
        "-loglevel", "error",
    ]


def assemble_reel(
    ordered_clip_paths: Sequence[str],
    gap_before_s: Sequence[float],
    output_path: str,
) -> ReelAssemblyResult:
    """
    Concatenates `ordered_clip_paths` (already-cut, already-ordered Part 8
    clip files, real paths on disk) into one reel file at `output_path`,
    inserting a black segment of `gap_before_s[i]` seconds before clip `i`
    (0.0 for the first clip — see ml.pipeline.reel_ordering.build_reel_timeline,
    whose ReelTimelineEntry.gap_before_s values are exactly what a caller
    passes here, in the same order as the ClipCandidates it built the
    timeline from).

    Raises ReelAssemblyError (not a bare ValueError/IndexError) for
    anything this module itself catches — a length mismatch between the
    two input sequences, ffmpeg missing, ffmpeg failing, or a final file
    that doesn't actually have a playable video stream — same "one
    exception type for anything the caller needs to catch" posture
    ClipExtractionError already establishes.
    """
    if len(ordered_clip_paths) != len(gap_before_s):
        raise ReelAssemblyError(
            f"ordered_clip_paths ({len(ordered_clip_paths)}) and gap_before_s "
            f"({len(gap_before_s)}) must be the same length — one gap value per clip"
        )
    if not ordered_clip_paths:
        raise ReelAssemblyError("assemble_reel needs at least one clip — got an empty ordered_clip_paths")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    width, height = _probe_clip_frame_size(ordered_clip_paths[0])

    with tempfile.TemporaryDirectory(prefix="padel_reel_assembly_") as tmp_dir:
        concat_entries: list[str] = []
        for index, (clip_path, gap_s) in enumerate(zip(ordered_clip_paths, gap_before_s)):
            if gap_s > _MIN_GAP_S_TO_RENDER:
                gap_path = os.path.join(tmp_dir, f"gap_{index:03d}.mp4")
                _run_ffmpeg(build_ffmpeg_gap_command(gap_path, gap_s, width, height), context=f"gap before clip {index}")
                concat_entries.append(gap_path)
            concat_entries.append(clip_path)

        concat_list_path = os.path.join(tmp_dir, "concat_list.txt")
        with open(concat_list_path, "w") as f:
            for entry_path in concat_entries:
                # Resolved to an absolute path before writing: ffmpeg's
                # concat demuxer treats a relative path in the list file as
                # relative to the list file's OWN location (this temp
                # directory), not the caller's current working directory —
                # a caller-supplied relative clip path would silently
                # resolve to the wrong file (usually "not found") without
                # this. Real Part 8 clip paths from storage are already
                # absolute, but this makes the function correct for any
                # caller, not just that one.
                absolute_path = os.path.abspath(entry_path)
                # ffmpeg's concat demuxer format: each line is `file '<path>'`,
                # with any single quote in the path itself escaped per its own
                # documented convention — real Part 8 clip paths never
                # contain one, but this is cheap correctness to not skip.
                escaped = absolute_path.replace("'", r"'\''")
                f.write(f"file '{escaped}'\n")

        _run_ffmpeg(build_ffmpeg_concat_command(concat_list_path, output_path), context="final concat")

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise ReelAssemblyError(f"ffmpeg produced no output file at {output_path!r}")

    total_duration_s = _probe_duration_s(output_path)
    return ReelAssemblyResult(
        output_path=output_path,
        clip_count=len(ordered_clip_paths),
        total_duration_s=total_duration_s,
        file_size_bytes=os.path.getsize(output_path),
    )


def _run_ffmpeg(command: list[str], *, context: str) -> None:
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ReelAssemblyError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise ReelAssemblyError(f"ffmpeg failed during {context}: {stderr or 'unknown error'}") from exc


def _probe_duration_s(path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise ReelAssemblyError(f"ffprobe could not read a duration from {path!r}: {result.stderr.strip()}")
    return float(result.stdout.strip())
