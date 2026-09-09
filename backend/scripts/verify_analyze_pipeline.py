"""
Part 7g — end-to-end verification of the Parts 5-7 pipeline (detection ->
tracking -> rally/serve/shot/outcome/highlight detection) against a real
match video, with a human-review artifact set: annotated frame snapshots
at every rally boundary and every classified shot, plus a markdown
checklist, so a person can manually confirm "correct start/end, plausible
shot labels, correct point outcome" for a handful of rallies before
trusting the pipeline at scale (the task this script exists to satisfy).

Same relationship to test_rally_detection.py / test_shot_classification.py
/ test_point_outcome.py / test_highlight_tagging.py as
verify_tracking_pipeline.py (Part 6e) has to test_byte_tracker.py: those
tests each cover one module in isolation against synthetic fixtures. This
runs the REAL pieces together, in the exact order analyze_stage actually
calls them (see app/workers/tasks.py's _analyze):

    extract_frames (5a)
      -> detect_court on a handful of candidate frames (5c, optional —
         non-fatal if it fails, same as 5f/6e)
      -> detect_players_and_ball_many across every frame (5d)
      -> player-only + ball-only ByteTracker, real settings.*_track_*
         tuning, ball gaps interpolated (6b/6c)
      -> detect_rally_segments (7a)
      -> detect_serves (7b)
      -> detect_shots (7c)
      -> detect_point_outcomes (7d)
      -> detect_highlights (7e)

Deliberately calls ml/pipeline/*.py directly, not the app/services/
*_stage.py glue — those wrappers require a real Postgres `Video` row and
Celery payload, which is Part 7f's job (already covered by
test_pipeline_stages.py's eager-mode chain test). This script has no
Postgres dependency, same reasoning as verify_tracking_pipeline.py's own
module docstring: it exists purely to answer "does this look right on
real footage", which a DB-backed integration test can't tell you either.

**Why this can't just be another automated pass/fail check.** 7a-7e each
already log a warning when their own summary numbers look suspicious (0
rallies, most serves unidentified, most shots unknown, most outcomes
undetermined) — see each stage module's own docstring. Those are useful
but blunt: a video can sail through every one of those checks and still
have, say, rally boundaries that are each a couple hundred milliseconds
too early, or a `net` outcome that was actually a clean winner. Nothing
in this codebase has ground-truth labels for a real match video, so the
only way to catch that class of error is a person looking at the actual
moment being described — hence this script's real output isn't a
pass/fail, it's frame snapshots for a person to look at.

Run with (from backend/):
    python -m scripts.verify_analyze_pipeline [video.mp4] [--sample-rallies N] [--out-dir DIR]

With no video argument, generates a short synthetic sample clip (same one
verify_detection_pipeline.py builds) so the script is runnable with no
other setup — useful to prove the pipeline plumbing is sound, but this is
explicitly NOT what 7g's task is asking for: run this against a real
match video before trusting anything at scale.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.ml_path import ensure_ml_importable  # noqa: E402

ensure_ml_importable()

import cv2  # noqa: E402

from ml.common.frame_extraction import extract_frames  # noqa: E402
from ml.detection.court_detector import CourtDetectionError, detect_court  # noqa: E402
from ml.detection.player_ball_detection import detect_players_and_ball_many  # noqa: E402
from ml.detection.yolo_detector import COCO_PERSON_CLASS_ID, COCO_SPORTS_BALL_CLASS_ID  # noqa: E402
from ml.tracking.ball_interpolation import interpolate_ball_gaps  # noqa: E402
from ml.tracking.byte_tracker import ByteTracker  # noqa: E402
from ml.tracking.byte_tracker import to_serializable as tracks_to_serializable  # noqa: E402

# Reused, not re-implemented — see module docstring.
from scripts.verify_detection_pipeline import (  # noqa: E402
    COURT_CALIBRATION_MAX_ATTEMPTS,
    generate_synthetic_sample_video,
    load_real_or_standin_model,
)
from scripts.verify_tracking_pipeline import build_frame_detections  # noqa: E402

FRAME_SAMPLE_FPS = 5.0

# How many rallies get full image snapshots by default. Every rally still
# gets its numbers in verification_report.json/.md — this only caps how
# many get PNGs, since a 90-minute match could have 100+ rallies and
# nobody's manually reviewing all of them in one sitting (that's the
# whole point of 7g's task: check "a handful", not the entire match).
DEFAULT_SAMPLE_RALLIES = 6

# A rally auto-flagged by one of these heuristics is always added to the
# render set even past --sample-rallies, on top of whichever rallies were
# picked first — these are the ones most worth a human's limited
# attention, not a verdict on their own (see module docstring: no ground
# truth exists here to actually confirm any of them are wrong).
FLAG_SHORT_DURATION_S = 1.0
FLAG_LONG_DURATION_S = 40.0
FLAG_UNKNOWN_SHOT_RATE = 0.5

TEXT_COLOR = (255, 255, 255)
TEXT_BG = (0, 0, 0)
BALL_MARKER_COLOR = (0, 0, 255)
BOUNDARY_MARKER_COLOR = (0, 220, 0)


def build_ball_and_player_tracks(frame_paths: list[str], detection_results: list, settings):
    """
    Steps 2-3 of 6e's main(), factored out so this script can call it once
    and then feed the same in-memory tracks into both 6c's on-disk JSON
    shape (for 7a-7d's frames_from_*_tracks_json readers, which all
    expect that exact {"frames": [...]} shape written by
    ml.tracking.byte_tracker.to_serializable — see rally_detection_stage.py
    / shot_classification_stage.py's docstrings) and this script's own
    annotation drawing (which wants the live FrameTracks objects, not a
    JSON round-trip).
    """
    player_frames_in = build_frame_detections(frame_paths, detection_results, COCO_PERSON_CLASS_ID)
    ball_frames_in = build_frame_detections(frame_paths, detection_results, COCO_SPORTS_BALL_CLASS_ID)

    player_tracker = ByteTracker(
        track_thresh=settings.player_track_thresh,
        match_thresh_low=settings.player_track_match_thresh_low,
        iou_threshold=settings.player_track_iou_threshold,
        max_age=settings.player_track_max_age,
        min_hits=settings.player_track_min_hits,
    )
    player_tracks = player_tracker.update_many(player_frames_in)

    ball_tracker = ByteTracker(
        track_thresh=settings.ball_track_thresh,
        match_thresh_low=settings.ball_track_match_thresh_low,
        iou_threshold=settings.ball_track_iou_threshold,
        max_age=settings.ball_track_max_age,
        min_hits=settings.ball_track_min_hits,
    )
    ball_tracks_raw = ball_tracker.update_many(ball_frames_in)
    ball_tracks = interpolate_ball_gaps(
        ball_tracks_raw, max_gap_frames=settings.ball_track_max_interpolation_gap_frames
    )
    return player_tracks, ball_tracks


def calibrate_court(frame_paths: list[str]):
    """Same best-effort, non-fatal court calibration as verify_detection_pipeline.py's Step 2/3."""
    step = max(1, len(frame_paths) // COURT_CALIBRATION_MAX_ATTEMPTS)
    candidates = frame_paths[::step][:COURT_CALIBRATION_MAX_ATTEMPTS]
    for candidate in candidates:
        try:
            calibration = detect_court(candidate)
            print(f"[7g] Court calibrated from {candidate}")
            return calibration
        except CourtDetectionError as exc:
            print(f"[7g]   {candidate}: {exc}")
    print("[7g] WARNING: could not calibrate the court from any candidate frame (non-fatal, "
          "same as 5f/6e) — shot volley/groundstroke split and point outcomes will report "
          "unknown/undetermined for this run.")
    return None


def flag_rally(rally: dict, shots_for_rally: list[dict], outcome: dict | None) -> list[str]:
    """Heuristic reasons a rally is worth a human's limited attention first — see constants above."""
    reasons = []
    if rally["duration_s"] < FLAG_SHORT_DURATION_S:
        reasons.append(f"very short ({rally['duration_s']:.2f}s) — check this is a real rally, not noise")
    if rally["duration_s"] > FLAG_LONG_DURATION_S:
        reasons.append(f"very long ({rally['duration_s']:.1f}s) — check it isn't two rallies stitched together")
    if not shots_for_rally:
        reasons.append("zero shots detected in this rally — check the start/end boundary against the video")
    else:
        unknown = sum(1 for s in shots_for_rally if s["shot_type"] == "unknown")
        if unknown / len(shots_for_rally) > FLAG_UNKNOWN_SHOT_RATE:
            reasons.append(f"{unknown}/{len(shots_for_rally)} shots classified unknown")
    if outcome is not None and outcome["outcome"] == "undetermined":
        reasons.append(f"outcome undetermined ({outcome['reason']})")
    return reasons


def _draw_label(image, lines: list[str], origin=(10, 10)) -> None:
    """Small multi-line text label with a solid background, top-left anchored at `origin`."""
    x, y = origin
    line_h = 20
    for i, line in enumerate(lines):
        ly = y + i * line_h
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(image, (x - 4, ly - 2), (x + tw + 4, ly + th + 6), TEXT_BG, thickness=-1)
        cv2.putText(image, line, (x, ly + th), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)


def snapshot_boundary_frame(frame_path: str, out_path: Path, lines: list[str]) -> None:
    """Rally start/end snapshot: the frame itself plus a text label, no marker (no single point to mark)."""
    image = cv2.imread(frame_path)
    if image is None:
        return
    _draw_label(image, lines)
    cv2.imwrite(str(out_path), image)


def snapshot_shot_frame(frame_path: str, out_path: Path, lines: list[str], ball_center) -> None:
    """Shot contact snapshot: the frame, a text label, and a marker circle at the ball's contact position."""
    image = cv2.imread(frame_path)
    if image is None:
        return
    if ball_center is not None:
        cx, cy = int(ball_center[0]), int(ball_center[1])
        cv2.circle(image, (cx, cy), 10, BALL_MARKER_COLOR, thickness=2)
        cv2.line(image, (cx - 14, cy), (cx + 14, cy), BALL_MARKER_COLOR, 1)
        cv2.line(image, (cx, cy - 14), (cx, cy + 14), BALL_MARKER_COLOR, 1)
    _draw_label(image, lines)
    cv2.imwrite(str(out_path), image)


def ball_center_at_frame(ball_tracks, frame_index: int):
    """First ball track's bbox center at a given sampled-frame index, or None if the ball had no track that frame."""
    if frame_index is None or frame_index >= len(ball_tracks):
        return None
    tracks = ball_tracks[frame_index].tracks
    return tracks[0].bbox.center if tracks else None


def select_rallies_to_render(rallies: list[dict], flags_by_rally: dict[int, list[str]], sample_rallies: int) -> list[int]:
    """First `sample_rallies` rally_index values, plus every flagged rally, deduped and sorted."""
    first_n = [r["rally_index"] for r in rallies[:sample_rallies]]
    flagged = [idx for idx, reasons in flags_by_rally.items() if reasons]
    return sorted(set(first_n) | set(flagged))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", nargs="?", default=None,
                         help="Path to a real match video. Omit to generate a synthetic one (plumbing check only).")
    parser.add_argument("--out-dir", default=None, help="Where to write frames/results. Defaults to a temp dir.")
    parser.add_argument("--sample-rallies", type=int, default=DEFAULT_SAMPLE_RALLIES,
                         help=f"How many rallies (in chronological order) to render snapshots for, on top of any "
                              f"auto-flagged ones (default {DEFAULT_SAMPLE_RALLIES}).")
    args = parser.parse_args()

    settings = get_settings()
    work_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="padel_7g_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"
    review_dir = work_dir / "review"
    review_dir.mkdir(parents=True, exist_ok=True)

    if args.video:
        video_path = Path(args.video)
        print(f"[7g] Using real video: {video_path}")
    else:
        video_path = work_dir / "synthetic_sample.mp4"
        generate_synthetic_sample_video(video_path)
        print(f"[7g] No video given — generated a synthetic sample at {video_path}")
        print("[7g] NOTE: this is a plumbing check only. Re-run against a real match video "
              "before trusting anything about the numbers below.")

    print(f"\n=== Step 1/6: frame extraction (real ffmpeg, sample_fps={FRAME_SAMPLE_FPS}) ===")
    extraction_result = extract_frames(str(video_path), str(frames_dir), sample_fps=FRAME_SAMPLE_FPS)
    frame_paths = extraction_result.frame_paths
    print(f"[7g] Extracted {extraction_result.frame_count} frames -> {extraction_result.frames_dir}")
    assert extraction_result.frame_count > 0, "extract_frames should never return zero frames without raising"

    print(f"\n=== Step 2/6: court calibration (real OpenCV, up to {COURT_CALIBRATION_MAX_ATTEMPTS} candidate frames) ===")
    calibration = calibrate_court(frame_paths)
    if calibration is not None:
        to_court_meters = calibration.pixel_to_court
        court_length_m = calibration.court_length_m
        court_width_m = calibration.court_width_m
    else:
        to_court_meters = None
        court_length_m = settings.court_length_m
        court_width_m = settings.court_width_m

    print(f"\n=== Step 3/6: player & ball detection across {len(frame_paths)} frames ===")
    model, model_kind = load_real_or_standin_model()
    print(f"[7g] model kind: {model_kind}")
    detection_results = detect_players_and_ball_many(model, frame_paths)

    print("\n=== Step 4/6: player & ball tracking (Parts 6a-6c, real settings) ===")
    player_tracks, ball_tracks = build_ball_and_player_tracks(frame_paths, detection_results, settings)
    player_tracks_data = {"frames": tracks_to_serializable(player_tracks)}
    ball_tracks_data = {"frames": tracks_to_serializable(ball_tracks)}
    (work_dir / "player_tracks.json").write_text(json.dumps(player_tracks_data, indent=2))
    (work_dir / "ball_tracks.json").write_text(json.dumps(ball_tracks_data, indent=2))

    print("\n=== Step 5/6: rally/serve/shot/outcome/highlight detection (Parts 7a-7e, real logic) ===")
    from ml.pipeline.highlight_tagging import detect_highlights
    from ml.pipeline.highlight_tagging import to_serializable as highlights_to_serializable
    from ml.pipeline.point_outcome import ball_points_from_tracks_json as outcome_ball_points_from_tracks_json
    from ml.pipeline.point_outcome import detect_point_outcomes
    from ml.pipeline.point_outcome import to_serializable as outcomes_to_serializable
    from ml.pipeline.rally_detection import detect_rally_segments, frames_from_ball_tracks_json
    from ml.pipeline.rally_detection import to_serializable as rallies_to_serializable
    from ml.pipeline.serve_detection import detect_serves
    from ml.pipeline.serve_detection import frames_from_tracks_json as serve_frames_from_tracks_json
    from ml.pipeline.serve_detection import to_serializable as serves_to_serializable
    from ml.pipeline.shot_classification import ball_points_from_tracks_json as shot_ball_points_from_tracks_json
    from ml.pipeline.shot_classification import detect_shots
    from ml.pipeline.shot_classification import player_frames_from_tracks_json
    from ml.pipeline.shot_classification import to_serializable as shots_to_serializable

    rally_signals = frames_from_ball_tracks_json(ball_tracks_data)
    rally_segments = detect_rally_segments(
        rally_signals,
        sample_fps=FRAME_SAMPLE_FPS,
        activity_gap_tolerance_frames=settings.rally_activity_gap_tolerance_frames,
        min_rally_duration_frames=settings.rally_min_duration_frames,
    )
    rallies = rallies_to_serializable(rally_segments)
    print(f"[7g] {len(rallies)} rally segment(s) detected")

    serve_ball_frames = serve_frames_from_tracks_json(ball_tracks_data)
    serve_player_frames = serve_frames_from_tracks_json(player_tracks_data)
    serve_events = detect_serves(
        rally_segments, serve_ball_frames, serve_player_frames,
        window_frames=settings.serve_detection_window_frames,
        max_distance_m=settings.serve_max_ball_player_distance_m,
        max_distance_px=settings.serve_max_ball_player_distance_px,
        to_court_meters=to_court_meters,
    )
    serves = serves_to_serializable(serve_events)

    shot_ball_points = shot_ball_points_from_tracks_json(ball_tracks_data)
    shot_player_frames = player_frames_from_tracks_json(player_tracks_data)
    shot_objs = detect_shots(
        rally_segments, shot_ball_points, shot_player_frames,
        max_distance_m=settings.shot_max_contact_player_distance_m,
        max_distance_px=settings.shot_max_contact_player_distance_px,
        smash_height_ratio=settings.shot_smash_height_ratio,
        lob_min_airborne_frames=settings.shot_lob_min_airborne_frames,
        net_proximity_m=settings.shot_net_proximity_m,
        court_length_m=court_length_m,
        to_court_meters=to_court_meters,
    )
    shots = shots_to_serializable(shot_objs)
    print(f"[7g] {len(shots)} shot(s) classified")

    outcome_ball_points = outcome_ball_points_from_tracks_json(ball_tracks_data)
    outcome_objs = detect_point_outcomes(
        rally_segments, outcome_ball_points,
        out_of_bounds_margin_m=settings.point_outcome_out_of_bounds_margin_m,
        net_zone_m=settings.point_outcome_net_zone_m,
        net_deceleration_ratio=settings.point_outcome_net_deceleration_ratio,
        court_width_m=court_width_m,
        court_length_m=court_length_m,
        to_court_meters=to_court_meters,
    )
    outcomes = outcomes_to_serializable(outcome_objs)

    highlight_events = detect_highlights(
        rally_segments, shot_objs,
        sample_fps=FRAME_SAMPLE_FPS,
        long_rally_min_duration_s=settings.highlight_long_rally_min_duration_s,
        long_rally_score_saturation_s=settings.highlight_long_rally_score_saturation_s,
        fast_exchange_max_interval_s=settings.highlight_fast_exchange_max_interval_s,
        fast_exchange_min_shot_count=settings.highlight_fast_exchange_min_shot_count,
        fast_exchange_score_saturation_count=settings.highlight_fast_exchange_score_saturation_count,
        smash_height_ratio=settings.shot_smash_height_ratio,
        powerful_smash_score_ceiling_ratio=settings.highlight_powerful_smash_score_ceiling_ratio,
        spectacular_save_max_response_s=settings.highlight_spectacular_save_max_response_s,
    )
    highlights = highlights_to_serializable(highlight_events)

    for name, data in [
        ("rallies.json", {"rallies": rallies}),
        ("serves.json", {"serves": serves}),
        ("shots.json", {"shots": shots}),
        ("outcomes.json", {"outcomes": outcomes}),
        ("highlights.json", {"highlights": highlights}),
    ]:
        (work_dir / name).write_text(json.dumps(data, indent=2))
    print(f"[7g] wrote rallies/serves/shots/outcomes/highlights JSON -> {work_dir}")

    print("\n=== Step 6/6: building human-review snapshots + checklist ===")
    shots_by_rally: dict[int, list[dict]] = {}
    for s in shots:
        shots_by_rally.setdefault(s["rally_index"], []).append(s)
    outcome_by_rally = {o["rally_index"]: o for o in outcomes}
    serve_by_rally = {sv["rally_index"]: sv for sv in serves}

    flags_by_rally = {
        r["rally_index"]: flag_rally(r, shots_by_rally.get(r["rally_index"], []), outcome_by_rally.get(r["rally_index"]))
        for r in rallies
    }
    render_set = select_rallies_to_render(rallies, flags_by_rally, args.sample_rallies)
    print(f"[7g] rendering snapshots for {len(render_set)}/{len(rallies)} rally(s): {render_set}")

    checklist_entries = []
    for rally in rallies:
        idx = rally["rally_index"]
        if idx not in render_set:
            continue
        rally_dir = review_dir / f"rally_{idx:03d}"
        rally_dir.mkdir(exist_ok=True)
        rally_shots = sorted(shots_by_rally.get(idx, []), key=lambda s: s["frame_index"])
        outcome = outcome_by_rally.get(idx)
        serve = serve_by_rally.get(idx)

        start_png = rally_dir / "start.png"
        snapshot_boundary_frame(
            frame_paths[rally["start_frame"]], start_png,
            [f"RALLY {idx} START", f"frame {rally['start_frame']}  t={rally['start_time_s']:.2f}s"],
        )
        end_png = rally_dir / "end.png"
        snapshot_boundary_frame(
            frame_paths[rally["end_frame"]], end_png,
            [f"RALLY {idx} END", f"frame {rally['end_frame']}  t={rally['end_time_s']:.2f}s",
             f"outcome: {outcome['outcome'] if outcome else 'n/a'}"],
        )

        shot_pngs = []
        for j, shot in enumerate(rally_shots):
            shot_png = rally_dir / f"shot_{j + 1:02d}_{shot['shot_type']}.png"
            label = [
                f"rally {idx} shot {j + 1}: {shot['shot_type']}",
                f"frame {shot['frame_index']}  player_track={shot['player_track_id']}",
            ]
            if shot["contact_height_ratio"] is not None:
                label.append(f"contact_height_ratio={shot['contact_height_ratio']:.2f}")
            if shot["airborne_frames_after"] is not None:
                label.append(f"airborne_frames_after={shot['airborne_frames_after']}")
            if shot["distance_from_net_m"] is not None:
                label.append(f"dist_from_net_m={shot['distance_from_net_m']:.2f}")
            snapshot_shot_frame(
                frame_paths[shot["frame_index"]], shot_png, label,
                ball_center_at_frame(ball_tracks, shot["frame_index"]),
            )
            shot_pngs.append(str(shot_png.relative_to(work_dir)))

        checklist_entries.append({
            "rally_index": idx,
            "start_frame": rally["start_frame"],
            "end_frame": rally["end_frame"],
            "start_time_s": rally["start_time_s"],
            "end_time_s": rally["end_time_s"],
            "duration_s": rally["duration_s"],
            "serve": serve,
            "shots": rally_shots,
            "outcome": outcome,
            "flags": flags_by_rally[idx],
            "start_png": str(start_png.relative_to(work_dir)),
            "end_png": str(end_png.relative_to(work_dir)),
            "shot_pngs": shot_pngs,
        })

    report = {
        "video": str(video_path),
        "model_kind": model_kind,
        "court_calibrated": calibration is not None,
        "frame_count": len(frame_paths),
        "rally_count": len(rallies),
        "shot_count": len(shots),
        "outcome_counts": {
            o: sum(1 for x in outcomes if x["outcome"] == o)
            for o in {"out_of_bounds", "net", "in_bounds_end", "undetermined"}
        },
        "rallies_rendered": render_set,
        "flags_by_rally": {str(k): v for k, v in flags_by_rally.items() if v},
        "review_entries": checklist_entries,
    }
    report_path = work_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2))

    md_path = work_dir / "human_review_checklist.md"
    md_path.write_text(_render_markdown_checklist(report))

    print("\n=== Summary ===")
    print(f"frames extracted:        {len(frame_paths)}")
    print(f"model kind:              {model_kind}")
    print(f"court calibrated:        {'yes' if calibration is not None else 'no'}")
    print(f"rallies detected:        {len(rallies)}")
    print(f"shots classified:        {len(shots)}")
    print(f"outcome counts:          {report['outcome_counts']}")
    print(f"rallies flagged:         {len(report['flags_by_rally'])} {list(report['flags_by_rally'].keys())}")
    print(f"rallies rendered:        {render_set}")
    print(f"\nreview checklist:        {md_path}")
    print(f"full report:             {report_path}")
    print(f"work dir:                {work_dir}")
    print(
        "\n[7g] Next step is manual, not automated: open human_review_checklist.md, look at each "
        "rally's start.png/end.png/shot_*.png against the actual video around that timestamp, and "
        "check the three things off per rally — start/end correct, shot labels plausible, point "
        "outcome correct. This script has no ground truth, so nothing above is a pass/fail verdict "
        "on its own — see module docstring."
    )


def _render_markdown_checklist(report: dict) -> str:
    lines = [
        "# Part 7g manual review checklist",
        "",
        f"Video: `{report['video']}`",
        f"Model: {report['model_kind']} | Court calibrated: {report['court_calibrated']}",
        f"Rallies detected: {report['rally_count']} | Shots classified: {report['shot_count']} | "
        f"Outcome counts: {report['outcome_counts']}",
        "",
        "For each rally below: open the linked PNGs next to the real video at the given timestamps "
        "and check off each item. A flagged rally (⚠) is one this script's own heuristics singled "
        "out as worth checking first — not a confirmed error.",
        "",
    ]
    for entry in report["review_entries"]:
        flag_marker = " ⚠" if entry["flags"] else ""
        lines.append(f"## Rally {entry['rally_index']}{flag_marker}")
        if entry["flags"]:
            for f in entry["flags"]:
                lines.append(f"- ⚠ {f}")
        lines.append(
            f"- Window: frame {entry['start_frame']}–{entry['end_frame']} "
            f"({entry['start_time_s']:.2f}s–{entry['end_time_s']:.2f}s, {entry['duration_s']:.2f}s)"
        )
        serve = entry["serve"]
        if serve:
            lines.append(
                f"- Serve: {'player ' + str(serve['server_track_id']) if serve['identified'] else 'not identified'}"
            )
        lines.append(f"- [ ] Start/end correct — [{entry['start_png']}]({entry['start_png']}) / "
                      f"[{entry['end_png']}]({entry['end_png']})")
        if entry["shots"]:
            lines.append(f"- [ ] Shot labels plausible ({len(entry['shots'])} shot(s)):")
            for shot, png in zip(entry["shots"], entry["shot_pngs"]):
                lines.append(f"  - [ ] frame {shot['frame_index']}: **{shot['shot_type']}** — [{png}]({png})")
        else:
            lines.append("- [ ] Shot labels plausible — **0 shots detected in this rally**, check against the video")
        outcome = entry["outcome"]
        if outcome:
            lines.append(f"- [ ] Point outcome correct — **{outcome['outcome']}** ({outcome['reason']})")
        else:
            lines.append("- [ ] Point outcome correct — no outcome recorded")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
