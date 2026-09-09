"""
Part 10f — end-to-end verification of Part 10 (reel generation) against a
real Part 8 output: runs real clip selection (10a) and ordering/pacing
(10b) over real, already-cut clip files, concatenates them with real
FFmpeg (10c) into one real, playable `reel.mp4`, and prepares a
human-review checklist so a person can actually open and watch the
*assembled reel* end to end — not just individual clips — and confirm
"the right moments are in it, in a sensible order, with clean transitions
and no A/V sync drift across the cuts" (the task this script exists to
satisfy) before trusting reel generation at scale.

Same relationship to test_reel_selection.py / test_reel_ordering.py /
test_reel_generation_stage.py as verify_clip_extraction_pipeline.py (8g)
has to test_clip_boundaries.py / test_clip_extraction.py: those tests
cover each piece in isolation against synthetic fixtures or a handful of
throwaway clips. This runs the real pieces together, in the exact order
the real glue stage (app/services/reel_generation_stage.py, Part 10f)
calls them:

    select_clips_for_reel (10a, ml/pipeline/reel_selection.py)
      -> order_clips_for_reel + build_reel_timeline (10b, ml/pipeline/reel_ordering.py)
      -> assemble_reel (10c, ml/common/reel_assembly.py)

Deliberately calls ml/pipeline/reel_selection.py, ml/pipeline/reel_ordering.py,
and ml/common/reel_assembly.py directly, not app/services/reel_generation_stage.py
itself: that module requires a real Postgres `Video`/`Highlight` row set and a
Celery payload (already covered by test_reel_generation_stage.py's real-ffmpeg,
throwaway-SQLite tests). This script has no Postgres dependency, same reasoning
as every other verify_*_pipeline.py script before it: it exists purely to
answer "does the assembled reel look and sound right", which a DB-backed
integration test can't tell you either.

**"A real Part 8 output" — what this script expects as input.** Part 8g
(verify_clip_extraction_pipeline.py) is what actually produces one: run
against a real match video, it cuts real, encoded, labeled clip files and
writes a verification_report.json recording each clip's boundary data
(highlight_type, padded start/end, importance_score) and where its file
landed. This script's normal input IS that 8g run directory (`--run-dir`)
— it does not re-run detection/tracking/analysis/clip-cutting itself, on
purpose: 8g already exists to answer "are these clips trustworthy", and
re-deriving them here would both duplicate that work and let stale or
synthetic clips quietly feed Part 10 without anyone noticing.

**Why this uses 8g's (sampled) clip set rather than cutting every
highlight itself.** 8g deliberately only cuts a representative sample of
clips by default (see its own module docstring — "check a handful, not
every clip"), not the full highlight set a real 90-minute match would
produce. Real clip selection (10a) only ever ranks clips that already
have a cut file (`ClipCandidate.clip_file_path` is a required `str`, not
optional — same "cuttable clips only" framing app/services/
reel_generation_stage.py's own module docstring already gives), so this
script's candidate pool is exactly whatever 8g actually rendered. That's
narrower than a full match's real corpus, but it's real, honestly-cut
footage assembled by the real 10a/10b/10c pipeline — not synthetic
placeholder clips — which is what answers this task's actual question
("does a real reel from real clips look and sound right"). Pass a larger
`--sample-clips` to the 8g run this script consumes if you want a bigger
candidate pool.

**Why this can't just be another automated pass/fail check.** This
script's own probe (`probe_media`, reused from 8g via the same reasoning)
can confirm the assembled reel file is playable, has video and (if any
source clip did) audio, and runs roughly as long as the timeline math
predicts. What it cannot tell you is whether cutting from one rally
straight into the next reads as coherent pacing or a jarring non-sequitur,
whether the black transition gaps feel like a deliberate beat or a stall,
or whether audio drifts out of sync over the course of the concatenated
file — all perceptual judgments nothing in this codebase has ground truth
for. Hence this script's real output isn't a verdict, it's one real
`reel.mp4` plus a markdown checklist for a person to actually open and
watch, start to finish.

Run with (from backend/):
    python -m scripts.verify_reel_generation_pipeline --run-dir <an 8g output dir> [--out-dir DIR]
    python -m scripts.verify_reel_generation_pipeline --run-dir <dir> --max-clips 5 --target-duration 45 --strategy importance

With no `--run-dir` given, this falls back to running Part 8g itself (as a
subprocess, which in turn falls back to Part 7g against a synthetic sample
video if it isn't given one either — see verify_clip_extraction_pipeline.py)
so the script is runnable with no other setup. Same as every other
verify_*_pipeline.py script's synthetic fallback, this is explicitly NOT
what 10f's task is asking for: run this against a real 8g output before
trusting anything about how an assembled reel actually looks and sounds.
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

from ml.pipeline.reel_selection import ClipCandidate  # noqa: E402

# Reused, not re-implemented -- see module docstring.
from scripts.verify_clip_extraction_pipeline import probe_media  # noqa: E402


def resolve_run_dir(args, work_dir: Path) -> Path:
    """
    Figures out which 8g output directory to consume -- see module
    docstring for why an explicit --run-dir is the normal, expected input
    and the no-argument fallback explicitly is not.
    """
    if args.run_dir:
        run_dir = Path(args.run_dir)
        report_path = run_dir / "verification_report.json"
        if not report_path.exists():
            print(f"[10f] ERROR: {report_path} not found -- --run-dir should point at a "
                  "verify_clip_extraction_pipeline.py (8g) output directory.")
            sys.exit(1)
        print(f"[10f] Using Part 8 output from {run_dir}")
        return run_dir

    print("[10f] No --run-dir given -- generating a synthetic sample and running Part 8g "
          "against it (plumbing check only).")
    print("[10f] NOTE: this is NOT what 10f's task is asking for. Re-run with --run-dir "
          "pointing at a real verify_clip_extraction_pipeline.py output before trusting "
          "anything below about how a real reel looks and sounds.")
    part8_dir = work_dir / "part8"
    subprocess.run(
        [sys.executable, "-m", "scripts.verify_clip_extraction_pipeline", "--out-dir", str(part8_dir)],
        check=True, cwd=str(BACKEND_ROOT),
    )
    return part8_dir


def load_candidates(run_dir: Path) -> tuple[list[ClipCandidate], list[dict]]:
    """
    Turns 8g's verification_report.json entries into real ClipCandidates
    (10a) -- only entries with a real, successfully-extracted clip file
    are candidates at all, same "cuttable clips only" filter
    app/services/reel_generation_stage.py applies against real `Highlight`
    rows (see that module's docstring). `highlight_id` is each entry's own
    8g `index` (a plain int, not a UUID) -- exactly the "no database
    dependency" use ClipCandidate.highlight_id is typed `object` for (see
    ml/pipeline/reel_selection.py's own docstring).

    Returns (candidates, skipped) where `skipped` records every entry left
    out and why, for the report this script itself writes.
    """
    report = json.loads((run_dir / "verification_report.json").read_text())
    candidates: list[ClipCandidate] = []
    skipped: list[dict] = []

    for entry in report["entries"]:
        boundary = entry["boundary"]
        if entry.get("clip_path") is None:
            skipped.append({"index": entry["index"], "reason": entry.get("extraction_error") or "no clip file"})
            continue
        clip_path = run_dir / entry["clip_path"]
        if not clip_path.exists():
            skipped.append({"index": entry["index"], "reason": f"clip file missing on disk: {clip_path}"})
            continue

        candidates.append(
            ClipCandidate(
                highlight_id=entry["index"],
                event_type=boundary["highlight_type"],
                start_time_s=boundary["start_time_s"],
                end_time_s=boundary["end_time_s"],
                importance_score=boundary["importance_score"],
                clip_file_path=str(clip_path),
            )
        )

    return candidates, skipped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", default=None,
                         help="A verify_clip_extraction_pipeline.py (8g) output directory -- the normal input; see module docstring.")
    parser.add_argument("--out-dir", default=None, help="Where to write the reel + report. Defaults to a temp dir.")
    parser.add_argument("--max-clips", type=int, default=None,
                         help="Override settings.reel_max_clips (10a hard count cap).")
    parser.add_argument("--target-duration", type=float, default=None,
                         help="Override settings.reel_target_duration_s (10a soft duration budget, seconds).")
    parser.add_argument("--strategy", choices=["chronological", "importance"], default=None,
                         help="Override settings.reel_ordering_strategy (10b).")
    parser.add_argument("--transition-gap", type=float, default=None,
                         help="Override settings.reel_transition_gap_s (10b/10c black-segment gap, seconds).")
    args = parser.parse_args()

    settings = get_settings()
    max_clips = args.max_clips if args.max_clips is not None else settings.reel_max_clips
    target_duration_s = args.target_duration if args.target_duration is not None else settings.reel_target_duration_s
    strategy = args.strategy or settings.reel_ordering_strategy
    transition_gap_s = args.transition_gap if args.transition_gap is not None else settings.reel_transition_gap_s

    work_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="padel_10f_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    run_dir = resolve_run_dir(args, work_dir)

    if not (run_dir / "verification_report.json").exists():
        # Same "no highlight events -> nothing to cut, exit 0" outcome 8g's own
        # module docstring describes -- most likely from the synthetic fallback
        # chain (a short static clip rarely has anything eventful in it). A
        # real, valid outcome, not this script's own bug -- see module
        # docstring on why the synthetic path is a plumbing check only.
        print(f"\n[10f] {run_dir} has no verification_report.json -- Part 8g found nothing to cut "
              "for this input (most likely the synthetic fallback chain, which rarely produces any "
              "highlight events at all). Nothing for Part 10 to select from. Exiting.")
        print("[10f] Re-run with --run-dir pointing at a real verify_clip_extraction_pipeline.py "
              "output (itself run against a real match video) to actually exercise reel assembly.")
        sys.exit(0)

    print(f"\n=== Step 1/4: loading real, already-cut clips from {run_dir} ===")
    candidates, skipped_no_file = load_candidates(run_dir)
    print(f"[10f] {len(candidates)} cuttable clip(s) available "
          f"({len(skipped_no_file)} entries had no usable clip file)")
    if not candidates:
        print("[10f] No cuttable clips found in this run-dir -- nothing for Part 10 to select from. Exiting.")
        sys.exit(0)

    print(f"\n=== Step 2/4: clip selection (10a) + ordering/pacing (10b) ===")
    print(f"[10f] settings: max_clips={max_clips} target_duration_s={target_duration_s} "
          f"strategy={strategy!r} transition_gap_s={transition_gap_s}")
    from ml.pipeline.reel_ordering import build_reel_timeline, order_clips_for_reel, summarize_timeline
    from ml.pipeline.reel_selection import select_clips_for_reel, summarize_selection

    selected = select_clips_for_reel(candidates, max_clips=max_clips, target_duration_s=target_duration_s)
    excluded_ids = {c.highlight_id for c in candidates} - {c.highlight_id for c in selected}
    selection_summary = summarize_selection(selected)
    print(f"[10f] selected {len(selected)}/{len(candidates)} clip(s) "
          f"(total {selection_summary['total_duration_s']:.1f}s, "
          f"avg importance {selection_summary['avg_importance_score']:.2f})")

    if not selected:
        print("[10f] Selection produced an empty reel (max_clips=0, or no candidates at all) -- "
              "a real, valid outcome, but there's nothing to assemble. Exiting.")
        sys.exit(0)

    ordered = order_clips_for_reel(selected, strategy=strategy)
    timeline = build_reel_timeline(ordered, transition_gap_s=transition_gap_s)
    timeline_summary = summarize_timeline(timeline)
    print(f"[10f] ordered ({strategy}); timeline: {timeline_summary['clip_count']} clip(s), "
          f"{timeline_summary['total_clip_duration_s']:.1f}s of clips + "
          f"{timeline_summary['total_gap_duration_s']:.1f}s of gaps = "
          f"{timeline_summary['total_reel_duration_s']:.1f}s total")

    print(f"\n=== Step 3/4: assembling one real reel with FFmpeg (10c) ===")
    from ml.common.reel_assembly import ReelAssemblyError, assemble_reel

    reel_path = work_dir / "reel.mp4"
    ordered_clip_paths = [c.clip_file_path for c in ordered]
    gap_before_s = [e.gap_before_s for e in timeline]

    try:
        result = assemble_reel(ordered_clip_paths, gap_before_s, str(reel_path))
    except ReelAssemblyError as exc:
        print(f"[10f] ERROR: reel assembly failed: {exc}")
        sys.exit(1)

    reel_probe = probe_media(str(reel_path))
    print(f"[10f] assembled: {result.clip_count} clip(s), {result.total_duration_s:.1f}s, "
          f"{result.file_size_bytes / 1024:.0f}KB, has_audio={reel_probe['has_audio']}")

    print("\n=== Step 4/4: writing human-review checklist ===")
    entries = []
    for entry, timeline_entry in zip(ordered, timeline):
        entries.append({
            "position": timeline_entry.position,
            "highlight_id": entry.highlight_id,
            "event_type": entry.event_type,
            "importance_score": entry.importance_score,
            "source_duration_s": entry.duration_s,
            "gap_before_s": timeline_entry.gap_before_s,
            "reel_start_s": timeline_entry.reel_start_s,
            "reel_end_s": timeline_entry.reel_end_s,
            "source_clip": str(Path(entry.clip_file_path).relative_to(run_dir))
                if str(entry.clip_file_path).startswith(str(run_dir)) else entry.clip_file_path,
        })

    report = {
        "run_dir": str(run_dir),
        "settings": {
            "reel_max_clips": max_clips,
            "reel_target_duration_s": target_duration_s,
            "reel_ordering_strategy": strategy,
            "reel_transition_gap_s": transition_gap_s,
        },
        "candidates_available": len(candidates),
        "candidates_skipped": skipped_no_file,
        "clips_selected": len(selected),
        "clips_excluded_by_caps": sorted(excluded_ids, key=str),
        "reel_path": str(reel_path),
        "reel_duration_s": result.total_duration_s,
        "reel_has_audio": reel_probe["has_audio"],
        "reel_has_video": reel_probe["has_video"],
        "entries": entries,
    }
    report_path = work_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    md_path = work_dir / "human_review_checklist.md"
    md_path.write_text(_render_markdown_checklist(report))

    print("\n=== Summary ===")
    print(f"clips available:       {len(candidates)}")
    print(f"clips selected:        {len(selected)}")
    print(f"reel duration:         {result.total_duration_s:.1f}s")
    print(f"reel file:             {reel_path}")
    print(f"review checklist:      {md_path}")
    print(f"full report:           {report_path}")
    print(
        "\n[10f] Next step is manual, not automated: open human_review_checklist.md, then actually "
        "play reel.mp4 start to finish, and check off the items below. A reel with no technical "
        "issues above has only been confirmed *playable*, not confirmed to feel right -- see "
        "module docstring."
    )


def _render_markdown_checklist(report: dict) -> str:
    lines = [
        "# Part 10f manual review checklist",
        "",
        f"Source: `{report['run_dir']}`",
        f"Settings: max_clips={report['settings']['reel_max_clips']} "
        f"target_duration_s={report['settings']['reel_target_duration_s']} "
        f"strategy={report['settings']['reel_ordering_strategy']!r} "
        f"transition_gap_s={report['settings']['reel_transition_gap_s']}",
        f"Candidates available: {report['candidates_available']} | "
        f"Selected: {report['clips_selected']} | "
        f"Reel duration: {report['reel_duration_s']:.1f}s | "
        f"Audio: {'yes' if report['reel_has_audio'] else 'no'}",
        "",
        f"## Watch this: [{Path(report['reel_path']).name}]({Path(report['reel_path']).name})",
        "",
        "- [ ] Reel plays from start to finish with no playback errors",
        "- [ ] Clips appear in a sensible order (context precedes payoff for a chronological "
        "reel; best-first if strategy=importance)",
        "- [ ] Transitions between clips (the black gap, if `transition_gap_s > 0`) feel like a "
        "deliberate beat, not a stall or a jarring instant cut",
        "- [ ] Audio stays in sync with video for the entire reel, including after each cut",
        "- [ ] No clip appears cut off mid-action at its start or end within the reel",
        "",
        "## Composition (what's in it, in play order)",
        "",
    ]
    for e in report["entries"]:
        lines.append(
            f"{e['position']}. **{e['event_type']}** (importance {e['importance_score']:.2f}, "
            f"{e['source_duration_s']:.1f}s source, gap before {e['gap_before_s']:.1f}s) -- "
            f"reel time {e['reel_start_s']:.1f}s\u2013{e['reel_end_s']:.1f}s -- `{e['source_clip']}`"
        )
    if report["clips_excluded_by_caps"]:
        lines.append("")
        lines.append(
            f"Excluded by `reel_max_clips`/`reel_target_duration_s` caps (available but not "
            f"selected): {report['clips_excluded_by_caps']}"
        )
    if report["candidates_skipped"]:
        lines.append("")
        lines.append("Skipped entirely (no usable clip file in the source run-dir):")
        for s in report["candidates_skipped"]:
            lines.append(f"- index {s['index']}: {s['reason']}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
