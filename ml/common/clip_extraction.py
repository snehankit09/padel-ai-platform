"""
Clip extraction — Part 8b, the second and final piece of Part 8 (highlight
clip generation). Given a source video path and an (in, out) pair of
timestamps, cuts a real, standalone clip file via FFmpeg.

ml/pipeline/clip_boundaries.py (Part 8a) already answers "what should this
clip's in/out points be" — padded, clamped ClipBoundary objects, still just
numbers on a JSON payload, no video file anywhere. This module is the next
and last step: turn one (start_time_s, end_time_s) pair into bytes on disk
that a browser can actually play. It knows nothing about ClipBoundary,
HighlightEvent, rally_index, or any other pipeline-domain concept — just a
source path and two floats — the same "pure, independently testable,
no-Celery/DB" shape as ml/common/frame_extraction.py (Part 5a), and for the
same reason: app/services/clip_extraction_stage.py is what knows how to
turn a padded ClipBoundary and a Video row into the source_path/output_path/
start_time_s/end_time_s this module actually needs.

**Why re-encode instead of `-c copy`.** Stream-copying (no re-encode) is
much faster and lossless, but it can only cut on a keyframe — the actual
clip that comes out is silently snapped to the nearest keyframe *before*
the requested start, which for a source video with keyframes seconds apart
(typical for long single-take match footage, not edited-for-cuts content)
would routinely start a "17.0s" clip several seconds early. That's exactly
the kind of surprise Part 8a's pre/post-roll padding math wouldn't know to
account for. Re-encoding costs CPU time this module doesn't try to hide,
but it's what makes start_time_s/end_time_s mean what they say.

**Why `-ss` before `-i`, not after.** Since FFmpeg 2.1, input seeking
(`-ss` before `-i`) combined with a re-encode (not `-c copy`) is frame-
accurate — FFmpeg decodes and discards frames up to the seek point rather
than only seeking to the nearest keyframe, the same accuracy output seeking
(`-ss` after `-i`) gives. Input seeking is also faster: FFmpeg can skip
straight to (approximately) the right spot in the demuxer before it starts
decoding, rather than decoding the entire file from the start and only
discarding output frames after the fact — meaningful here since source
videos can run up to settings.max_video_duration_minutes (120 minutes by
default) and a single video may have several clips extracted from it.

**Why cap resolution, force a fixed pixel format, and normalize audio
(Part 8c).** allowed_video_formats (mp4/mov/avi) puts no ceiling on
upload resolution — a 4K or even higher source is a real, expected input,
not an edge case — and one match's clips get watched back-to-back in a
reel alongside another match's, possibly recorded on a completely
different camera. Without a cap, "highlight clip size" would just be
"whatever the uploader's camera happened to shoot", which is both a
storage-cost problem across potentially dozens of clips per match and an
inconsistent-viewing-experience problem (some clips crisp and huge, others
small). `_MAX_OUTPUT_HEIGHT_PX` fixes an upper bound instead: clips scale
*down* to it, never up — see `_scale_filter` — so a source already at or
below the cap (a phone recording at 720p, say) is left at its native
resolution rather than padded or upscaled into fake sharpness.
`_PIXEL_FORMAT` ("yuv420p") is a compatibility floor, not a quality
choice: some source cameras/containers produce 4:2:2 or 4:4:4 chroma
subsampling, which several common players (Safari/iOS chief among them)
either refuse to play or render incorrectly, so every clip is normalized
to the one subsampling scheme guaranteed to play everywhere, regardless of
what the source used. The audio settings (`_AUDIO_SAMPLE_RATE_HZ`,
`_AUDIO_CHANNELS`, `_AUDIO_BITRATE`) exist for the same reason applied to
audio: a source might be mono, 5.1 surround, or a high sample rate a
browser's audio pipeline doesn't need for a highlight clip, so every clip
is normalized to one small, stereo, web-standard target instead of
inheriting whatever the source happened to have.

**Why `-movflags +faststart`.** By default an MP4's index (`moov` atom)
is written at the *end* of the file, which forces a browser to download
the entire clip before it can start playing — fine for a small file, but
these clips are meant to be scrubbed through in a reel UI, not downloaded
first. `+faststart` makes FFmpeg do a second, cheap pass at the end of the
encode to move that index to the front, so playback (and, in supporting
players, byte-range seeking within the clip) can start as soon as enough
of the file has streamed in rather than waiting on the whole thing.

**Part 8d — the highlight-type label overlay, and what it deliberately
doesn't cover.** The PRD (Module 3) isn't included in this codebase as a
document to check against, and nothing else in the codebase — models,
enums, prior part docstrings — mentions overlays at all. What the
existing data model rules out on its own: `Highlight` (app/models/
highlight.py) has no player/track identity linkage of any kind, only
match_id, and no stage anywhere in this codebase tracks live game/set/
match score state (the same missing signal `ml/pipeline/highlight_tagging.py`
already cites as the reason MATCH_POINT/BREAK_POINT aren't tagged at all
yet). So a player-name or score overlay would have nothing real to draw
from today — building either now would mean inventing placeholder data
just to have something to burn in, throwaway work that'd need to be redone
once that identity/score data actually exists (real Part 9+ territory).

A highlight-*type* label (e.g. "LONG RALLY", "POWERFUL_SMASH") has no such
gap: `Highlight.event_type` (Part 7e) is already populated on every row
this stage processes, so `label_text` is accepted as a plain, pre-
formatted string — humanizing a HighlightType enum value into display
text is app/services/clip_extraction_stage.py's job (a domain decision),
not this module's; see this module's own opening paragraph on why it
stays domain-agnostic about ClipBoundary/HighlightEvent concepts.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass

# Re-encode, not `-c copy` — see module docstring for why frame-accurate
# cuts require it. libx264/aac is the same "widely-playable, not an
# archival master" tradeoff ml/common/frame_extraction.py makes for its own
# JPEG output (_JPEG_QSCALE): this is a highlight reel clip meant to be
# watched, not a lossless intermediate.
_VIDEO_CODEC = "libx264"
_AUDIO_CODEC = "aac"
# Constant-quality encode (lower = better/larger, per libx264 convention —
# same inverted-scale idea as frame_extraction's _JPEG_QSCALE, different
# codec's own knob). 20 is visually close to source for web playback
# without ballooning storage the way a near-lossless setting would across
# potentially dozens of clips per match.
_VIDEO_CRF = "20"
# Speed/compression tradeoff, not quality — "veryfast" keeps per-clip
# extraction time low across a match's worth of highlights rather than
# optimizing for the smallest possible file at a given CRF.
_VIDEO_PRESET = "veryfast"
# Chroma subsampling every clip is normalized to, regardless of what the
# source used — see the module docstring's "Part 8c" section for why this
# is a compatibility floor (Safari/iOS playback), not a quality knob.
_PIXEL_FORMAT = "yuv420p"
# Downscale-only cap on output height, in pixels — a clip already at or
# below this stays at its native resolution (see _scale_filter); one
# above it is scaled down to it. 1080 is "visually full quality for a
# highlight reel watched on a phone or laptop, not the multi-GB-per-clip
# cost of preserving a 4K source verbatim".
_MAX_OUTPUT_HEIGHT_PX = 1080
# Audio is normalized the same way video resolution is — one small,
# widely-playable target regardless of what the source had (mono, 5.1,
# a high sample rate a highlight clip has no use for).
_AUDIO_SAMPLE_RATE_HZ = "48000"
_AUDIO_CHANNELS = "2"
_AUDIO_BITRATE = "128k"
# Relocates the MP4 index to the front of the file post-encode so a clip
# can start playing/seeking as it streams in rather than only after a full
# download — see the module docstring's "Part 8c" section.
_MOVFLAGS = "+faststart"

# Slow-motion (Highlights Improvement Roadmap Tier 1b): the minimum
# length either normal-speed side segment needs to be worth splitting out
# at all. Below this, a "slowed window near the very edge of the clip"
# isn't meaningfully different from "the whole clip is slow, with a tiny
# fast sliver at one end" — see build_ffmpeg_slowmo_clip_command.
_SLOWMO_MIN_SEGMENT_S = 0.2

# --- Highlight-type label overlay (Part 8d) ---------------------------------
# A fixed font *file* path, not a fontconfig family name lookup
# (`font='DejaVu Sans Bold'`) — the worker container is a slim base image
# that has no fonts installed by default (see backend/Dockerfile, which
# this feature adds `fonts-dejavu-core` to specifically to guarantee this
# exact path exists). A family-name lookup would depend on fontconfig
# finding *some* matching font at runtime, which is exactly the kind of
# "works on my machine, breaks in the container" gap an explicit path
# avoids entirely.
_LABEL_FONT_FILE = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
_LABEL_FONT_SIZE = "28"
_LABEL_FONT_COLOR = "white"
# A semi-transparent box behind the text, not just the text on its own —
# a highlight clip's background is real match footage (court, players,
# sky), so white-on-white or white-on-light-grass would otherwise make the
# label unreadable for exactly the clips where it matters.
_LABEL_BOX_COLOR = "black@0.5"
_LABEL_BOX_BORDER_PX = "12"
_LABEL_MARGIN_PX = "24"


def _escape_drawtext_text(text: str) -> str:
    """
    Escapes a plain string for safe use as drawtext's `text` value.

    FFmpeg's filtergraph syntax gives four characters special meaning
    inside a filter's option value: `:` separates a filter's own
    key=value options, `,` separates filters within a chain (see
    `_scale_filter`'s docstring for the same issue), `'` starts/ends a
    quoted span, and `\\` is the escape character itself — each has to be
    backslash-escaped, `\\` first so an already-escaped character in the
    input isn't double-escaped by escaping the others afterward. Every
    caller today only ever passes a HighlightType value formatted by
    app/services/clip_extraction_stage.py (e.g. "LONG RALLY" — no
    punctuation at all), so none of this fires in practice yet, but this
    module still doesn't assume that will always be true — see the
    module docstring on why it stays domain-agnostic about what text it's
    asked to draw.
    """
    for char in ("\\", ":", "'", ","):
        text = text.replace(char, "\\" + char)
    return text


def _drawtext_filter(label_text: str) -> str:
    """
    Builds the `drawtext` filter that burns `label_text` into a clip's
    bottom-left corner — see the module docstring's "Part 8d" section for
    why this exists and what it deliberately doesn't cover yet (player
    names, score).

    Position uses drawtext's own `text_h` (`th`) expression variable
    rather than a hardcoded y-offset, so the label sits a fixed margin
    above the bottom edge regardless of the chosen font size — an
    explicit dependency on `_LABEL_FONT_SIZE` here would silently go
    stale the next time that constant changes.
    """
    escaped = _escape_drawtext_text(label_text)
    return (
        f"drawtext=fontfile={_LABEL_FONT_FILE}:text='{escaped}'"
        f":fontsize={_LABEL_FONT_SIZE}:fontcolor={_LABEL_FONT_COLOR}"
        f":box=1:boxcolor={_LABEL_BOX_COLOR}:boxborderw={_LABEL_BOX_BORDER_PX}"
        f":x={_LABEL_MARGIN_PX}:y=h-th-{_LABEL_MARGIN_PX}"
    )


def _build_video_filter(max_output_height_px: int, label_text: str | None) -> str:
    """
    Combines the always-on resolution cap (`_scale_filter`, Part 8c) with
    the optional label overlay (`_drawtext_filter`, Part 8d) into the
    single `-vf` chain FFmpeg expects, in that order: scaling before
    drawtext keeps the label's font size and margins meaning the same
    fixed pixel amount on the final output frame, regardless of the
    source's own resolution — drawtext-then-scale would instead shrink or
    grow the label along with the video.
    """
    filters = [_scale_filter(max_output_height_px)]
    if label_text is not None:
        filters.append(_drawtext_filter(label_text))
    return ",".join(filters)


def _scale_filter(max_output_height_px: int) -> str:
    """
    Builds the `-vf` filter chain that caps output height at
    `max_output_height_px` (never upscales — see module docstring) while
    guaranteeing both output dimensions are even, which libx264 requires
    and a downscaled source's dimensions aren't guaranteed to already be.

    Two chained `scale` filters, not one, because they do two genuinely
    different jobs:
      1. `scale=-2:'min(ih,{cap})'` picks the target height (the source's
         own height if it's already <= the cap, else the cap) and derives
         a width from it that preserves aspect ratio and is already even
         (`-2`, vs. `-1` which doesn't guarantee evenness).
      2. `scale=trunc(iw/2)*2:trunc(ih/2)*2` is a no-op for the common
         case where step 1's height was already even (the cap itself, or
         an even source height under it), and only does real work for the
         one case step 1 can't fix on its own: an *odd* source height at
         or under the cap, which step 1 passes through unchanged via
         `min()` — this step floors it to the nearest even value.
    The `min(...)` expression is wrapped in single quotes because FFmpeg's
    filtergraph parser treats a bare comma as a filter separator; quoting
    the whole expression is how a comma *inside* one filter's option
    survives being parsed as the start of the next filter.
    """
    return (
        f"scale=-2:'min(ih,{max_output_height_px})',"
        "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    )


class ClipExtractionError(Exception):
    """Raised when ffmpeg isn't available, fails to run, or produces no playable output file."""


@dataclass
class ClipExtractionResult:
    output_path: str
    start_time_s: float
    end_time_s: float
    duration_s: float
    file_size_bytes: int


def build_ffmpeg_clip_command(
    source_path: str,
    output_path: str,
    start_time_s: float,
    end_time_s: float,
    label_text: str | None = None,
) -> list[str]:
    """
    Pure command-builder — no subprocess, no filesystem access — so the
    exact ffmpeg invocation can be asserted on in tests, same split as
    build_ffmpeg_frame_command in ml/common/frame_extraction.py.

    Uses `-t <duration>` (a length) rather than a second `-ss`/`-to`
    absolute timestamp for the out point: with `-ss` already placed before
    `-i` for the in point (see module docstring), FFmpeg's own output
    timestamps restart from ~0 at the seek point, so a second absolute
    timestamp would have to be re-derived relative to that seek rather than
    the source file's original timeline — a duration sidesteps the
    conversion entirely and says exactly what it means regardless of where
    the seek landed.

    Video/audio normalization (resolution cap, pixel format, audio
    sample rate/channels/bitrate, faststart) is the "Part 8c" work — see
    the module docstring's own section for why each exists. It's applied
    unconditionally here rather than exposed as a parameter: unlike
    start_time_s/end_time_s (per-clip, driven by a specific ClipBoundary),
    these are "what does *any* clip this module produces look like"
    decisions, the same one answer for every clip regardless of which
    highlight it came from.

    `label_text`, unlike those, genuinely does vary per clip — a
    LONG_RALLY clip and a POWERFUL_SMASH clip shouldn't carry the same
    burned-in caption — so it's the one piece of "Part 8d" left as a
    parameter rather than a module constant. `None` (the default) skips
    the drawtext filter entirely rather than drawing an empty label; see
    the module docstring's "Part 8d" section for why player names and
    score aren't accepted here at all yet.
    """
    if end_time_s <= start_time_s:
        raise ValueError(
            f"end_time_s ({end_time_s!r}) must be greater than start_time_s ({start_time_s!r})"
        )
    duration_s = end_time_s - start_time_s
    return [
        "ffmpeg", "-y",
        "-ss", f"{start_time_s:.3f}",
        "-i", source_path,
        "-t", f"{duration_s:.3f}",
        "-vf", _build_video_filter(_MAX_OUTPUT_HEIGHT_PX, label_text),
        "-c:v", _VIDEO_CODEC,
        "-preset", _VIDEO_PRESET,
        "-crf", _VIDEO_CRF,
        "-pix_fmt", _PIXEL_FORMAT,
        "-c:a", _AUDIO_CODEC,
        "-ar", _AUDIO_SAMPLE_RATE_HZ,
        "-ac", _AUDIO_CHANNELS,
        "-b:a", _AUDIO_BITRATE,
        "-movflags", _MOVFLAGS,
        "-avoid_negative_ts", "make_zero",
        output_path,
        "-loglevel", "error",
    ]


def extract_clip(
    source_path: str,
    output_path: str,
    start_time_s: float,
    end_time_s: float,
    label_text: str | None = None,
) -> ClipExtractionResult:
    """
    Cuts `source_path` down to [start_time_s, end_time_s) and writes the
    result to `output_path`, re-encoding for frame-accurate boundaries (see
    module docstring). Creates output_path's parent directory if it
    doesn't exist yet, same as extract_frames does for its own output_dir.

    `label_text`, if given, is burned into the clip's bottom-left corner
    (Part 8d) — see build_ffmpeg_clip_command's docstring for why this is
    a parameter rather than one of Part 8c's fixed module constants.

    Raises ClipExtractionError if ffmpeg isn't available, fails to run, or
    produces a missing/empty output file. An empty output is treated as a
    failure rather than a silent success — same reasoning as
    extract_frames' zero-frames check: a Highlight row later UPDATEd with
    clip_file_path pointing at a zero-byte file is a much worse, harder to
    diagnose failure mode than raising here, immediately, with a specific
    reason.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    command = build_ffmpeg_clip_command(
        source_path, output_path, start_time_s, end_time_s, label_text=label_text
    )
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ClipExtractionError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise ClipExtractionError(
            f"ffmpeg failed to extract a clip from {source_path!r} "
            f"[{start_time_s:.3f}s - {end_time_s:.3f}s]: {stderr or 'unknown error'}"
        ) from exc

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0 or not _has_video_stream(output_path):
        # A start_time_s past the source's actual duration doesn't make
        # ffmpeg itself exit non-zero (input seeking past EOF just yields
        # zero encoded frames, "successfully") — it makes ffmpeg write a
        # small, valid-looking container with no streams in it at all.
        # os.path.getsize alone wouldn't catch that (that empty container
        # is a couple hundred bytes, not zero), so this checks for an
        # actual video stream via ffprobe, the same "don't trust the file
        # existing, trust what ffprobe reports back" approach
        # app/services/video_validation.py already uses for uploads.
        raise ClipExtractionError(
            f"ffmpeg produced no playable output for {source_path!r} "
            f"[{start_time_s:.3f}s - {end_time_s:.3f}s] — the requested window is likely "
            "outside the source video's actual duration"
        )

    return ClipExtractionResult(
        output_path=output_path,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        duration_s=end_time_s - start_time_s,
        file_size_bytes=os.path.getsize(output_path),
    )


def build_ffmpeg_thumbnail_command(source_path: str, output_path: str, timestamp_s: float) -> list[str]:
    """
    Pure command-builder for a single-frame poster image — Highlights
    Improvement Roadmap Tier 1a — same "no subprocess, no filesystem
    access" split as build_ffmpeg_clip_command above, for the same
    testability reason.

    Reuses `_scale_filter` (Part 8c) unmodified so a thumbnail is capped
    to the exact same resolution ceiling as the clip it belongs to,
    rather than introducing a second, independent sizing decision this
    module would then have to keep in sync by hand.

    `-ss` before `-i` for the same frame-accurate-seek reason
    build_ffmpeg_clip_command uses it (see module docstring); `-frames:v 1`
    stops the encode after exactly one frame instead of decoding further
    than necessary. `-q:v 2` is JPEG's own quality knob (2-31, lower is
    better) — 2 is "visually near-lossless single frame", cheap to afford
    since this produces one image per clip, not a duration's worth of
    frames the way frame_extraction.py's _JPEG_QSCALE has to budget for.
    """
    return [
        "ffmpeg", "-y",
        "-ss", f"{timestamp_s:.3f}",
        "-i", source_path,
        "-frames:v", "1",
        "-vf", _scale_filter(_MAX_OUTPUT_HEIGHT_PX),
        "-q:v", "2",
        output_path,
        "-loglevel", "error",
    ]


def extract_thumbnail(source_path: str, output_path: str, timestamp_s: float) -> None:
    """
    Writes a single JPEG frame from `source_path` at `timestamp_s` to
    `output_path` — Highlights Improvement Roadmap Tier 1a's poster image
    for one highlight clip. Creates output_path's parent directory if it
    doesn't exist yet, same as extract_clip does for its own output_path.

    Raises ClipExtractionError under the same conditions extract_clip
    does (ffmpeg missing/failed, or no playable frame produced) — reusing
    that one exception type rather than introducing a second, since
    callers already have to handle ClipExtractionError from extract_clip
    in the same code path (see clip_extraction_stage.py) and a second,
    thumbnail-specific exception type would only mean catching two things
    where one already covers it.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    command = build_ffmpeg_thumbnail_command(source_path, output_path, timestamp_s)
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ClipExtractionError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise ClipExtractionError(
            f"ffmpeg failed to extract a thumbnail from {source_path!r} "
            f"at {timestamp_s:.3f}s: {stderr or 'unknown error'}"
        ) from exc

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        # Same "a past-EOF seek doesn't make ffmpeg exit non-zero" trap
        # extract_clip's own check guards against — here it'd mean
        # timestamp_s landed past the source's actual duration.
        raise ClipExtractionError(
            f"ffmpeg produced no thumbnail for {source_path!r} at {timestamp_s:.3f}s — "
            "the requested timestamp is likely outside the source video's actual duration"
        )


def _build_atempo_chain(factor: float) -> str:
    """
    ffmpeg's `atempo` filter only accepts a single speed multiplier in
    [0.5, 2.0] per instance — anything outside that range (our own
    slow-motion use is well under 0.5, since `highlight_slowmo_factor`
    defaults to 0.45) has to be reached by chaining multiple `atempo`
    instances whose multipliers compound to the target factor, the
    standard workaround for this ffmpeg limitation.
    """
    if factor <= 0:
        raise ValueError(f"atempo factor must be positive, got {factor!r}")
    filters: list[str] = []
    remaining = factor
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    filters.append(f"atempo={remaining:.6f}")
    return ",".join(filters)


class SlowMotionWindowError(ValueError):
    """
    Raised when a requested slow-motion window doesn't leave enough real
    clip on both sides to be worth the added filter-graph complexity —
    e.g. a very short clip, or a decisive-frame timestamp too close to
    one edge. Callers (clip_extraction_stage.py) should catch this and
    fall back to a plain extract_clip() call rather than treating it as a
    hard failure — same "a clip is still better than no clip" posture
    Part 8f already established for extraction failures generally.
    """


def build_ffmpeg_slowmo_clip_command(
    source_path: str,
    output_path: str,
    start_time_s: float,
    end_time_s: float,
    slowmo_center_s: float,
    slowmo_window_s: float,
    slowmo_factor: float,
    label_text: str | None = None,
) -> list[str]:
    """
    Highlights Improvement Roadmap Tier 1b — builds a three-segment
    filter-graph command: normal speed, then a `slowmo_window_s`-wide
    window centered on `slowmo_center_s` (an ABSOLUTE video timestamp,
    same convention as start_time_s/end_time_s — converted to
    clip-relative internally) played back at `slowmo_factor` of real-time,
    then normal speed again.

    **Why three segments via trim+setpts+concat, not a single setpts
    expression over the whole clip.** `setpts` applies one constant
    multiplier to a stream's presentation timestamps — there's no
    "slow down only between t1 and t2" mode built into the filter itself.
    Three `trim`'d sub-streams, each re-based to start at PTS=0
    (`setpts=PTS-STARTPTS`) so they splice back together without gaps,
    then `concat`, is the standard ffmpeg pattern for a variable-speed
    single output. Audio gets the matching treatment: `atrim` + `atempo`
    (via `_build_atempo_chain`, since our target factor is outside
    atempo's own single-instance range) + `concat` with paired video/audio
    streams so playback stays in sync.

    **This does not itself decide whether slow motion will look good.**
    See settings.enable_highlight_slowmo's own comment in core/config.py
    for the real risk (choppy playback on typical source frame rates) —
    this function just builds the command faithfully; judging the output
    is on whoever turns the feature on.

    Raises SlowMotionWindowError if the computed window doesn't leave a
    real segment on at least one side — see that exception's docstring.
    """
    if end_time_s <= start_time_s:
        raise ValueError(f"end_time_s ({end_time_s!r}) must be greater than start_time_s ({start_time_s!r})")
    if slowmo_factor <= 0 or slowmo_factor >= 1:
        raise ValueError(f"slowmo_factor must be in (0, 1), got {slowmo_factor!r}")

    duration_s = end_time_s - start_time_s
    center_rel_s = slowmo_center_s - start_time_s
    half_window_s = slowmo_window_s / 2

    t1 = max(0.0, center_rel_s - half_window_s)
    t2 = min(duration_s, center_rel_s + half_window_s)

    if t2 - t1 < 0.05 or (t1 < _SLOWMO_MIN_SEGMENT_S and duration_s - t2 < _SLOWMO_MIN_SEGMENT_S):
        raise SlowMotionWindowError(
            f"slow-motion window [{t1:.3f}s, {t2:.3f}s] leaves no meaningful "
            f"normal-speed segment within a {duration_s:.3f}s clip"
        )

    video_labels: list[str] = []
    audio_labels: list[str] = []
    filter_parts: list[str] = []
    segments = [(0.0, t1, 1.0), (t1, t2, slowmo_factor), (t2, duration_s, 1.0)]

    for index, (seg_start, seg_end, speed) in enumerate(segments):
        if seg_end - seg_start < 0.01:
            continue  # degenerate edge segment (window touches a clip boundary) — skip it entirely
        v_label = f"sv{index}"
        a_label = f"sa{index}"
        pts_multiplier = 1.0 / speed
        filter_parts.append(
            f"[0:v]trim={seg_start:.3f}:{seg_end:.3f},setpts=(PTS-STARTPTS)*{pts_multiplier:.6f}[{v_label}]"
        )
        atempo = "" if speed == 1.0 else f",{_build_atempo_chain(speed)}"
        filter_parts.append(f"[0:a]atrim={seg_start:.3f}:{seg_end:.3f},asetpts=PTS-STARTPTS{atempo}[{a_label}]")
        video_labels.append(f"[{v_label}]")
        audio_labels.append(f"[{a_label}]")

    concat_inputs = "".join(f"{v}{a}" for v, a in zip(video_labels, audio_labels))
    filter_parts.append(f"{concat_inputs}concat=n={len(video_labels)}:v=1:a=1[vcat][acat]")
    filter_parts.append(f"[vcat]{_build_video_filter(_MAX_OUTPUT_HEIGHT_PX, label_text)}[vout]")

    return [
        "ffmpeg", "-y",
        "-i", source_path,
        "-filter_complex", ";".join(filter_parts),
        "-map", "[vout]",
        "-map", "[acat]",
        "-c:v", _VIDEO_CODEC,
        "-preset", _VIDEO_PRESET,
        "-crf", _VIDEO_CRF,
        "-pix_fmt", _PIXEL_FORMAT,
        "-c:a", _AUDIO_CODEC,
        "-ar", _AUDIO_SAMPLE_RATE_HZ,
        "-ac", _AUDIO_CHANNELS,
        "-b:a", _AUDIO_BITRATE,
        "-movflags", _MOVFLAGS,
        output_path,
        "-loglevel", "error",
    ]


def extract_clip_with_slowmo(
    source_path: str,
    output_path: str,
    start_time_s: float,
    end_time_s: float,
    slowmo_center_s: float,
    slowmo_window_s: float,
    slowmo_factor: float,
    label_text: str | None = None,
) -> ClipExtractionResult:
    """
    Same contract as extract_clip (frame-accurate window, re-encoded,
    same normalization, raises ClipExtractionError on failure), but
    produces a clip with a slowed window around `slowmo_center_s` — see
    build_ffmpeg_slowmo_clip_command's docstring for the technique and its
    real quality tradeoff.

    **Not input-seeked the way extract_clip is.** extract_clip places
    `-ss` before `-i` for a fast, frame-accurate seek directly to
    start_time_s — safe there because that whole window plays at one
    constant speed. Here, the filter graph itself needs frame-accurate
    `trim` points at THREE different offsets (0, t1, t2) within the
    requested window, which requires decoding from the start of that
    window forward; input-seeking first would shift what "0" means
    inside the filter graph's own trim math in a way that's easy to get
    subtly wrong. The practical cost: this decodes the full
    [start_time_s, end_time_s] span itself (a few seconds, per Part 8a's
    padding) rather than skipping to it — not the multi-minute source
    file scan input-seeking avoids elsewhere in this module.

    Raises SlowMotionWindowError (a ValueError) if the window doesn't fit
    this clip — see build_ffmpeg_slowmo_clip_command. Callers should treat
    that the same as a soft failure and fall back to plain extract_clip(),
    not as this clip failing to extract at all.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    command = build_ffmpeg_slowmo_clip_command(
        source_path, output_path, start_time_s, end_time_s,
        slowmo_center_s=slowmo_center_s, slowmo_window_s=slowmo_window_s,
        slowmo_factor=slowmo_factor, label_text=label_text,
    )
    try:
        subprocess.run(command, check=True, capture_output=True)
    except FileNotFoundError as exc:
        raise ClipExtractionError("ffmpeg is not installed or not on PATH") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace").strip() if exc.stderr else ""
        raise ClipExtractionError(
            f"ffmpeg failed to extract a slow-motion clip from {source_path!r} "
            f"[{start_time_s:.3f}s - {end_time_s:.3f}s]: {stderr or 'unknown error'}"
        ) from exc

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0 or not _has_video_stream(output_path):
        raise ClipExtractionError(
            f"ffmpeg produced no playable slow-motion output for {source_path!r} "
            f"[{start_time_s:.3f}s - {end_time_s:.3f}s]"
        )

    return ClipExtractionResult(
        output_path=output_path,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        duration_s=end_time_s - start_time_s,
        file_size_bytes=os.path.getsize(output_path),
    )


def _has_video_stream(path: str) -> bool:
    """
    Cheap post-encode sanity check via ffprobe: does `path` actually contain
    a video stream? Used only to tell a real (if tiny) clip apart from the
    empty-but-valid container ffmpeg writes when the requested window
    landed entirely past the source's own duration — see extract_clip.
    Any ffprobe failure (missing binary, unreadable file) is treated as
    "no video stream" rather than propagated, since either way the caller's
    own conclusion (this isn't a usable clip) is the same.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                path,
            ],
            capture_output=True,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0 and b"video" in result.stdout
