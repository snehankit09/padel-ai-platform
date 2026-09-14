"""
Part 8g — end-to-end verification of Part 8 (highlight clip generation)
against a real Part 7 output: computes real clip boundaries (8a), cuts
real, encoded, labeled clip files with FFmpeg (8b/8c/8d), and prepares a
human-review checklist so a person can actually watch a handful of them
and confirm "in/out points feel right, no abrupt cuts, audio/video sync
holds up" (the task this script exists to satisfy) — before trusting
clip generation at scale.

Same relationship to test_clip_boundaries.py / test_clip_extraction.py as
verify_analyze_pipeline.py (7g) has to test_rally_detection.py and friends:
those tests cover each module in isolation against synthetic fixtures.
This runs the real pieces together, in the exact order the `analyze`
stage actually calls them (see app/workers/tasks.py's _analyze):

    compute_clip_boundaries (8a, ml/pipeline/clip_boundaries.py)
      -> extract_clip per boundary (8b/8c/8d, ml/common/clip_extraction.py)

Deliberately calls ml/pipeline/clip_boundaries.py and
ml/common/clip_extraction.py directly, not the app/services/
clip_boundary_stage.py / clip_extraction_stage.py glue — those wrappers
require a real Postgres `Video` row and Highlight rows already inserted
by Part 7f, which is Part 8a/8b's own job to read back (already covered
by test_pipeline_stages.py's eager-mode chain test). This script has no
Postgres dependency, same reasoning as every other verify_*_pipeline.py
script before it: it exists purely to answer "does the output look and
sound right", which a DB-backed integration test can't tell you either.

**"A real Part 7 output" — what this script expects as input.** Part 7g
(verify_analyze_pipeline.py) is what actually produces one: run against a
real match video, it writes rallies.json/serves.json/shots.json/
outcomes.json/highlights.json plus a verification_report.json recording
which video they came from. This script's normal input IS that 7g run
directory (`--run-dir`) — it does not re-run detection/tracking/rally
analysis itself, on purpose: 7g already exists to answer "is Part 7's
output trustworthy", and re-deriving highlights.json here would both
duplicate that work and let a stale/synthetic Part 7 output quietly feed
Part 8 without anyone noticing. `--video`/`--highlights-json` are
provided only to override individual pieces of a `--run-dir` (e.g. the
video file moved) or to point at a highlights.json produced some other
way.

**Why this can't just be another automated pass/fail check.** This
script's own probes (`probe_media`) can tell you a clip file exists, has
a video and (if the source did) an audio stream, and is close to its
requested duration — real, useful checks that a corrupt or truncated
clip fails loudly instead of silently. What they cannot tell you is
whether cutting into a rally 3 seconds before contact reads as "a beat of
anticipation" or "an abrupt, context-free jump cut", or whether the
audio drifts out of sync with the video over the course of a longer clip
— both are perceptual judgments nothing in this codebase has ground
truth for. Hence this script's real output isn't a verdict, it's a
`clips/` directory of real, playable files plus a markdown checklist for
a person to actually open and watch a handful of.

Run with (from backend/):
    python -m scripts.verify_clip_extraction_pipeline --run-dir <a 7g output dir> [--sample-clips N] [--out-dir DIR]
    python -m scripts.verify_clip_extraction_pipeline --video match.mp4 --highlights-json highlights.json

With neither `--run-dir` nor `--video`/`--highlights-json` given, this
falls back to running Part 7g itself (as a subprocess, against a
synthetic sample video — see verify_detection_pipeline.py's
generate_synthetic_sample_video) so the script is runnable with no other
setup. Same as every other verify_*_pipeline.py script's synthetic
fallback, this is explicitly NOT what 8g's task is asking for: run this
against a real Part 7g run directory before trusting anything about
clip quality.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.ml_path import ensure_ml_importable  # noqa: E402

ensure_ml_importable()

# How many clips actually get cut and handed to a human by default, on top
# of any auto-flagged ones (see select_clips_to_render) — a 90-minute match
# could tag dozens of highlights, and nobody's watching all of them in one
# sitting; that's the whole point of 8g's task: check "a handful", not
# every clip.
DEFAULT_SAMPLE_CLIPS = 6

# A rendered clip's actual duration is allowed to differ from its requested
# (end_time_s - start_time_s) by this much before probe_media's technical
# check flags it — FFmpeg's own frame-accurate re-encode (see
# ml/common/clip_extraction.py's module docstring) lands within a frame or
# two of the request, so anything past this is worth a second look, not
# encoder rounding.
DURATION_TOLERANCE_S = 0.5

# A boundary that ended up with less than this fraction of its intended
# pre-/post-roll (almost always because it got clamped against the video's
# own [0, duration] edge — see ml/pipeline/clip_boundaries.py) is flagged:
# the resulting clip is the one most likely to start or end mid-action.
PRE_POST_ROLL_FLAG_RATIO = 0.5


def probe_media(path: str) -> dict:
    """
    ffprobe summary of a media file: whether it has a video/audio stream,
    its duration, and its video resolution. Same "trust ffprobe, not the
    file existing" posture as ml/common/clip_extraction.py's own
    _has_video_stream / app/services/video_validation.py — used here on
    both the source video (once, to know whether audio-sync is even
    something to check) and every rendered clip (to catch a broken output
    file before asking a person to watch it).
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,width,height",
                "-show_entries", "format=duration",
                "-of", "json",
                path,
            ],
            capture_output=True,
        )
    except FileNotFoundError:
        return {"ok": False, "has_video": False, "has_audio": False, "duration_s": None, "width": None, "height": None}

    if result.returncode != 0:
        return {"ok": False, "has_video": False, "has_audio": False, "duration_s": None, "width": None, "height": None}

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    duration_raw = data.get("format", {}).get("duration")
    return {
        "ok": True,
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "duration_s": float(duration_raw) if duration_raw is not None else None,
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
    }


def flag_boundary(
    boundary: dict,
    *,
    padding_by_type: dict[str, tuple[float, float]],
    default_pre_roll_s: float,
    default_post_roll_s: float,
) -> list[str]:
    """
    Heuristic reasons a *boundary* (before it's even cut) is worth a
    human's limited attention first — same "auto-flag, don't auto-judge"
    posture as verify_analyze_pipeline.py's flag_rally, applied to the one
    class of error that's specific to clip generation: padding that got
    silently clamped away by the video's own edges (see
    ml/pipeline/clip_boundaries.py's _expand_to_min_duration), which is
    exactly what produces a clip that starts or ends mid-action.

    Compares against this boundary's own HighlightType's expected padding
    (per-type since Part 8a's "1c" follow-up), not one uniform pre/post-roll
    across every type — a smash's shorter expected pre-roll shouldn't get
    flagged against a long rally's longer one, or vice versa.
    """
    from ml.pipeline.clip_boundaries import get_padding_for_type

    pre_roll_s, post_roll_s = get_padding_for_type(
        boundary["highlight_type"],
        padding_by_type=padding_by_type,
        default_pre_roll_s=default_pre_roll_s,
        default_post_roll_s=default_post_roll_s,
    )
    min_duration_s = pre_roll_s + post_roll_s
    reasons = []
    actual_pre_roll = boundary["event_start_time_s"] - boundary["start_time_s"]
    actual_post_roll = boundary["end_time_s"] - boundary["event_end_time_s"]

    if pre_roll_s > 0 and actual_pre_roll < pre_roll_s * PRE_POST_ROLL_FLAG_RATIO:
        reasons.append(
            f"only {actual_pre_roll:.2f}s of lead-up before the highlight moment "
            f"(wanted {pre_roll_s:.1f}s) — check the clip doesn't start mid-action"
        )
    if post_roll_s > 0 and actual_post_roll < post_roll_s * PRE_POST_ROLL_FLAG_RATIO:
        reasons.append(
            f"only {actual_post_roll:.2f}s of follow-through after the highlight moment "
            f"(wanted {post_roll_s:.1f}s) — check the clip doesn't cut off before the point finishes"
        )
    duration = boundary["end_time_s"] - boundary["start_time_s"]
    if duration <= min_duration_s + 0.05:
        reasons.append(
            f"clip pinned at the {min_duration_s:.1f}s minimum duration — likely clamped hard "
            "against the start or end of the video"
        )
    return reasons


def technical_check(expected_duration_s: float, probe: dict, *, source_has_audio: bool) -> list[str]:
    """
    Automated sanity checks on an already-rendered clip file — catches a
    broken/corrupt/silently-truncated output before it wastes a person's
    time. Deliberately narrow: these can confirm a clip is technically
    playable and roughly the right length, never that it *feels* right —
    see module docstring on why the actual verdict is still a human's.
    """
    issues = []
    if not probe["ok"]:
        issues.append("ffprobe could not read this file at all")
        return issues
    if not probe["has_video"]:
        issues.append("no video stream in the extracted clip")
    if source_has_audio and not probe["has_audio"]:
        issues.append("source video has audio but this clip has no audio stream")
    if probe["duration_s"] is not None and abs(probe["duration_s"] - expected_duration_s) > DURATION_TOLERANCE_S:
        issues.append(
            f"extracted duration {probe['duration_s']:.2f}s differs from the requested "
            f"{expected_duration_s:.2f}s by more than {DURATION_TOLERANCE_S}s"
        )
    if probe["height"] and probe["height"] > 1080:
        issues.append(f"output height {probe['height']}px exceeds the 1080p cap (Part 8c)")
    return issues


def select_clips_to_render(boundaries: list[dict], flags_by_index: dict[int, list[str]], sample_clips: int) -> list[int]:
    """
    Which boundary indices actually get cut into real clip files. Round-
    robins across distinct highlight_type values (in chronological order
    within each type) rather than just taking the first N overall, so a
    match with 20 LONG_RALLY events and 2 POWERFUL_SMASH events doesn't
    hand a reviewer six long-rally clips and call it a representative
    sample — every unlabeled overlay/label-text combination (Part 8d)
    should get at least one look. Every flagged boundary (see
    flag_boundary) is always included on top of that, same "auto-flag,
    don't auto-judge, but never let a flagged one go unseen" posture as
    verify_analyze_pipeline.py's select_rallies_to_render.
    """
    ordered = sorted(range(len(boundaries)), key=lambda i: boundaries[i]["event_start_time_s"])
    by_type: dict[str, list[int]] = {}
    for i in ordered:
        by_type.setdefault(boundaries[i]["highlight_type"], []).append(i)

    diverse: list[int] = []
    type_keys = sorted(by_type.keys())
    while len(diverse) < sample_clips and any(by_type[t] for t in type_keys):
        for t in type_keys:
            if by_type[t]:
                diverse.append(by_type[t].pop(0))
            if len(diverse) >= sample_clips:
                break

    flagged = [i for i, reasons in flags_by_index.items() if reasons]
    return sorted(set(diverse) | set(flagged))


def resolve_inputs(args, work_dir: Path) -> tuple[Path, Path]:
    """
    Figures out (video_path, highlights_path) from whichever combination
    of --run-dir/--video/--highlights-json was given — see module
    docstring for the three supported modes and why --run-dir (a real
    Part 7g output) is the normal one.
    """
    if args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = run_dir / "verification_report.json"
        if not report_path.exists():
            print(f"[8g] ERROR: {report_path} not found — --run-dir should point at a "
                  "verify_analyze_pipeline.py (7g) output directory.")
            sys.exit(1)
        report = json.loads(report_path.read_text())
        video_path = Path(args.video) if args.video else Path(report["video"])
        highlights_path = Path(args.highlights_json) if args.highlights_json else (run_dir / "highlights.json")
        print(f"[8g] Using Part 7 output from {run_dir} (video: {video_path})")
        return video_path, highlights_path

    if args.video and args.highlights_json:
        print(f"[8g] Using explicit video={args.video} highlights_json={args.highlights_json}")
        return Path(args.video), Path(args.highlights_json)

    if args.video or args.highlights_json:
        print("[8g] ERROR: --video and --highlights-json must be given together "
              "(or use --run-dir instead).")
        sys.exit(1)

    print("[8g] No --run-dir/--video/--highlights-json given — generating a synthetic sample "
          "and running Part 7g against it (plumbing check only).")
    print("[8g] NOTE: this is NOT what 8g's task is asking for. Re-run with --run-dir pointing "
          "at a real verify_analyze_pipeline.py output before trusting anything below.")
    part7_dir = work_dir / "part7"
    subprocess.run(
        [sys.executable, "-m", "scripts.verify_analyze_pipeline", "--out-dir", str(part7_dir)],
        check=True, cwd=str(BACKEND_ROOT),
    )
    report = json.loads((part7_dir / "verification_report.json").read_text())
    return Path(report["video"]), part7_dir / "highlights.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=None,
                         help="A verify_analyze_pipeline.py (7g) output directory — the normal input; see module docstring.")
    parser.add_argument("--video", default=None, help="Override/supply the source video path.")
    parser.add_argument("--highlights-json", default=None, help="Override/supply the highlights.json path.")
    parser.add_argument("--out-dir", default=None, help="Where to write clips/results. Defaults to a temp dir.")
    parser.add_argument("--sample-clips", type=int, default=DEFAULT_SAMPLE_CLIPS,
                         help=f"How many clips (diverse across highlight_type) to actually cut and hand to a "
                              f"human, on top of any auto-flagged ones (default {DEFAULT_SAMPLE_CLIPS}).")
    parser.add_argument("--no-label", action="store_true",
                         help="Skip the Part 8d highlight-type label overlay (debugging only — production always burns it in).")
    args = parser.parse_args()

    settings = get_settings()
    work_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="padel_8g_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = work_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_path, highlights_path = resolve_inputs(args, work_dir)
    if not video_path.exists():
        print(f"[8g] ERROR: video not found at {video_path}")
        sys.exit(1)
    if not highlights_path.exists():
        print(f"[8g] ERROR: highlights.json not found at {highlights_path}")
        sys.exit(1)

    print(f"\n=== Step 1/4: probing source video ({video_path}) ===")
    source_probe = probe_media(str(video_path))
    if not source_probe["ok"] or not source_probe["has_video"] or source_probe["duration_s"] is None:
        print(f"[8g] ERROR: ffprobe could not read a usable video stream/duration from {video_path}")
        sys.exit(1)
    video_duration_s = source_probe["duration_s"]
    source_has_audio = source_probe["has_audio"]
    print(f"[8g] duration={video_duration_s:.2f}s  has_audio={source_has_audio}  "
          f"resolution={source_probe['width']}x{source_probe['height']}")
    if not source_has_audio:
        print("[8g] NOTE: source video has no audio stream at all — the 'audio/video sync' half "
              "of this task's checklist won't apply to any clip from this video.")

    print(f"\n=== Step 2/4: clip boundary calculation (Part 8a, real settings) ===")
    ensure_ml_importable()
    from ml.pipeline.clip_boundaries import compute_clip_boundaries, summarize_clip_boundaries, to_serializable
    from ml.pipeline.highlight_tagging import (
        HIGHLIGHT_TYPE_LONG_RALLY,
        HIGHLIGHT_TYPE_POWERFUL_SMASH,
        HighlightEvent,
    )

    highlights_data = json.loads(highlights_path.read_text())
    events = [HighlightEvent(**e) for e in highlights_data.get("highlights", [])]
    if not events:
        print("[8g] No highlight events in this Part 7 output — nothing for Part 8 to cut. Exiting.")
        sys.exit(0)

    # Per-HighlightType padding (Part 8a's "1c" follow-up) — see
    # ml/pipeline/clip_boundaries.py's CLIP_PADDING_BY_HIGHLIGHT_TYPE and
    # app/services/clip_boundary_stage.py, which builds this same mapping
    # from these same settings for the real `analyze` stage.
    padding_by_type = {
        HIGHLIGHT_TYPE_POWERFUL_SMASH: (
            settings.clip_powerful_smash_pre_roll_s,
            settings.clip_powerful_smash_post_roll_s,
        ),
        HIGHLIGHT_TYPE_LONG_RALLY: (
            settings.clip_long_rally_pre_roll_s,
            settings.clip_long_rally_post_roll_s,
        ),
    }

    clip_boundaries = compute_clip_boundaries(
        events,
        video_duration_s=video_duration_s,
        padding_by_type=padding_by_type,
        default_pre_roll_s=settings.clip_pre_roll_s,
        default_post_roll_s=settings.clip_post_roll_s,
    )
    boundaries = to_serializable(clip_boundaries)
    summary = summarize_clip_boundaries(clip_boundaries)
    (work_dir / "clip_boundaries.json").write_text(json.dumps(
        {
            "default_pre_roll_s": settings.clip_pre_roll_s,
            "default_post_roll_s": settings.clip_post_roll_s,
            "clip_padding_by_type": {
                highlight_type: {"pre_roll_s": pre, "post_roll_s": post}
                for highlight_type, (pre, post) in padding_by_type.items()
            },
            "video_duration_s": video_duration_s,
            **summary,
            "clip_boundaries": boundaries,
        },
        indent=2,
    ))
    print(f"[8g] {summary['clip_count']} clip boundary(s) computed "
          f"(avg {summary['avg_clip_duration_s']:.1f}s) -> {work_dir / 'clip_boundaries.json'}")

    flags_by_index = {
        i: flag_boundary(
            b,
            padding_by_type=padding_by_type,
            default_pre_roll_s=settings.clip_pre_roll_s,
            default_post_roll_s=settings.clip_post_roll_s,
        )
        for i, b in enumerate(boundaries)
    }
    render_set = select_clips_to_render(boundaries, flags_by_index, args.sample_clips)
    print(f"[8g] rendering {len(render_set)}/{len(boundaries)} clip(s): {render_set}")

    print(f"\n=== Step 3/4: cutting {len(render_set)} real clip(s) with FFmpeg (Part 8b/8c/8d) ===")
    from ml.common.clip_extraction import ClipExtractionError, extract_clip

    entries = []
    for i in render_set:
        boundary = boundaries[i]
        expected_duration = boundary["end_time_s"] - boundary["start_time_s"]
        label_text = None if args.no_label else boundary["highlight_type"].replace("_", " ").upper()
        clip_path = clips_dir / f"clip_{i:03d}_{boundary['highlight_type']}.mp4"

        try:
            result = extract_clip(
                source_path=str(video_path),
                output_path=str(clip_path),
                start_time_s=boundary["start_time_s"],
                end_time_s=boundary["end_time_s"],
                label_text=label_text,
            )
        except ClipExtractionError as exc:
            print(f"[8g]   clip {i}: EXTRACTION FAILED: {exc}")
            entries.append({
                "index": i, "boundary": boundary, "flags": flags_by_index[i],
                "clip_path": None, "extraction_error": str(exc), "technical_issues": [],
            })
            continue

        clip_probe = probe_media(str(clip_path))
        issues = technical_check(expected_duration, clip_probe, source_has_audio=source_has_audio)
        status = "OK" if not issues else "ISSUES: " + "; ".join(issues)
        print(f"[8g]   clip {i} ({boundary['highlight_type']}, "
              f"{result.duration_s:.1f}s, {result.file_size_bytes / 1024:.0f}KB): {status}")

        entries.append({
            "index": i,
            "boundary": boundary,
            "flags": flags_by_index[i],
            "clip_path": str(clip_path.relative_to(work_dir)),
            "extraction_error": None,
            "technical_issues": issues,
            "probe": clip_probe,
        })

    print("\n=== Step 4/4: writing human-review checklist ===")
    report = {
        "video": str(video_path),
        "video_duration_s": video_duration_s,
        "source_has_audio": source_has_audio,
        "default_pre_roll_s": settings.clip_pre_roll_s,
        "default_post_roll_s": settings.clip_post_roll_s,
        "clip_padding_by_type": {
            highlight_type: {"pre_roll_s": pre, "post_roll_s": post}
            for highlight_type, (pre, post) in padding_by_type.items()
        },
        "boundary_count": len(boundaries),
        "clips_rendered": len(render_set),
        "clips_with_extraction_errors": sum(1 for e in entries if e["extraction_error"]),
        "clips_with_technical_issues": sum(1 for e in entries if e["technical_issues"]),
        "entries": entries,
    }
    report_path = work_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    md_path = work_dir / "human_review_checklist.md"
    md_path.write_text(_render_markdown_checklist(report))

    print("\n=== Summary ===")
    print(f"video:                        {video_path}")
    print(f"clip boundaries computed:     {len(boundaries)}")
    print(f"clips rendered:               {len(render_set)}")
    print(f"clips with extraction errors: {report['clips_with_extraction_errors']}")
    print(f"clips with technical issues:  {report['clips_with_technical_issues']}")
    print(f"\nreview checklist:             {md_path}")
    print(f"clip files:                   {clips_dir}")
    print(f"full report:                  {report_path}")
    print(
        "\n[8g] Next step is manual, not automated: open human_review_checklist.md, actually play "
        "each linked clip, and check off the three things per clip — in/out points feel right, no "
        "abrupt cut at either end, audio/video sync holds up. A clip with no technical issues above "
        "has only been confirmed *playable*, not confirmed to feel right — see module docstring."
    )


def _render_markdown_checklist(report: dict) -> str:
    lines = [
        "# Part 8g manual review checklist",
        "",
        f"Video: `{report['video']}` ({report['video_duration_s']:.1f}s, "
        f"audio: {'yes' if report['source_has_audio'] else 'no'})",
        f"Default padding (any type without its own override): "
        f"pre-roll {report['default_pre_roll_s']:.1f}s / post-roll {report['default_post_roll_s']:.1f}s",
        "Per-type padding: " + ", ".join(
            f"{highlight_type} (pre {p['pre_roll_s']:.1f}s / post {p['post_roll_s']:.1f}s)"
            for highlight_type, p in report["clip_padding_by_type"].items()
        ),
        f"Clip boundaries: {report['boundary_count']} | Clips rendered: {report['clips_rendered']} | "
        f"Extraction errors: {report['clips_with_extraction_errors']} | "
        f"Technical issues: {report['clips_with_technical_issues']}",
        "",
        "For each clip below: actually open and play the linked file. A flagged clip (⚠) is one "
        "this script's own heuristics singled out as worth checking first (heavily-padded, "
        "clamped against the video's own edge, or failing an automated technical check) — not a "
        "confirmed error. Passing the automated technical check only means the file is playable "
        "and roughly the right length; it says nothing about whether the cut feels right.",
        "",
    ]
    for entry in report["entries"]:
        b = entry["boundary"]
        flag_marker = " ⚠" if entry["flags"] or entry["technical_issues"] or entry["extraction_error"] else ""
        lines.append(f"## Clip {entry['index']} — {b['highlight_type']}{flag_marker}")
        lines.append(
            f"- Source window: {b['start_time_s']:.2f}s – {b['end_time_s']:.2f}s "
            f"(highlight moment: {b['event_start_time_s']:.2f}s – {b['event_end_time_s']:.2f}s, "
            f"reason: {b['reason']})"
        )
        for f in entry["flags"]:
            lines.append(f"- ⚠ {f}")
        if entry["extraction_error"]:
            lines.append(f"- ⚠ **Extraction failed**: {entry['extraction_error']}")
            lines.append("")
            continue
        for issue in entry["technical_issues"]:
            lines.append(f"- ⚠ Automated check: {issue}")
        lines.append(f"- File: [{entry['clip_path']}]({entry['clip_path']})")
        lines.append("- [ ] In/out points feel right (not too tight, not too loose)")
        lines.append("- [ ] No abrupt cut at the start")
        lines.append("- [ ] No abrupt cut at the end")
        if report["source_has_audio"]:
            lines.append("- [ ] Audio/video sync holds up throughout")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
