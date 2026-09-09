"""
Part 6e — end-to-end verification of the Parts 6a-6d tracking pipeline
against Part 5's real detection output, with a track-continuity check.

Same relationship to test_byte_tracker.py / test_ball_interpolation.py /
test_player_tracking_stage.py / test_ball_tracking_stage.py as
verify_detection_pipeline.py (Part 5f) has to test_yolo_detector.py /
test_court_detector.py: those tests each cover one module in isolation,
mostly against synthetic fakes built to exercise one specific behavior.
This runs the REAL pieces together in the order track_stage actually
calls them:

    extract_frames (5a, real ffmpeg)
      -> detect_players_and_ball_many across every frame (5d, real Part 5
         detection output — this is "Part 5's real detection output" the
         Part 6e task description asks for, not a canned detections.json)
      -> a player-only ByteTracker, tuned with the real
         settings.player_track_* values (6b)
      -> a ball-only ByteTracker, same real settings.ball_track_* values,
         through ball_interpolation.interpolate_ball_gaps (6c)
      -> continuity checks + a trajectory overlay drawn on one real frame

Deliberately doesn't touch Postgres/Celery/StorageService — those are
app/services/{player,ball}_tracking_stage.py's job (Part 6d), already
covered by test_pipeline_stages.py's eager-mode chain test. This script
exists purely to answer the two questions those tests can't, because they
never look at *sequences* of real tracked positions: does the same player
keep the same ID across a rally, and does the ball's track avoid jumping
around erratically. Uses the same real-model-or-OpenCV-stand-in fallback
as verify_detection_pipeline.py (imported from it directly, not
duplicated) for the same reason: this environment has no network access
to install ultralytics / download weights, and a stand-in built from real
pixel contours is more honest about what's actually being verified than
canned fixed boxes would be. Once a real model is available this script
picks it up automatically with zero changes, same as 5f.

"Continuity" here means two distinct things, checked separately (see
player_continuity_report / ball_jump_report below for the exact logic):

  1. Player ID stability: walking each track_id's real (non-predicted)
     appearances in frame order and flagging any *consecutive-sampled-
     frame* jump bigger than a fraction of the frame diagonal — a jump
     that large for something an IoU-gated matcher accepted as "the same
     track" suggests two different players got stitched into one ID
     (Kalman positions can technically still satisfy the IoU gate right
     after a track's rebirth; see byte_tracker.py's known-limitation
     docstring).
  2. Ball trajectory smoothness: same idea but across the *whole*
     observed+interpolated sequence rather than restricted to one
     track_id, because the ball's own known failure mode (per
     byte_tracker.py's docstring) is usually a *new* track_id being
     minted after a fast movement, not the same ID jumping — a per-ID
     check like the player one would miss exactly that case.

Neither check hard-fails (no assert, no non-zero exit) — like 5f's ball
detection rate / player track count warnings, a flagged jump can be a
genuinely fast lunge or a hard volley rather than a tracker bug, and this
script has no ground truth to tell the two apart on real footage. It
prints what it found, writes it to tracking_report.json, and leaves the
judgment call to whoever's looking at the annotated trajectory image.

Run with (from backend/):  python -m scripts.verify_tracking_pipeline [video.mp4]
"""

from __future__ import annotations

import argparse
import json
import math
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
from ml.detection.player_ball_detection import (  # noqa: E402
    detect_players_and_ball_many,
    to_serializable as detections_to_serializable,
)
from ml.detection.yolo_detector import (  # noqa: E402
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    FrameDetections,
)
from ml.tracking.ball_interpolation import (  # noqa: E402
    interpolate_ball_gaps,
    summarize_ball_track_coverage,
)
from ml.tracking.byte_tracker import ByteTracker, FrameTracks  # noqa: E402
from ml.tracking.byte_tracker import to_serializable as tracks_to_serializable  # noqa: E402

# Reused, not re-implemented — see module docstring.
from scripts.verify_detection_pipeline import (  # noqa: E402
    generate_synthetic_sample_video,
    load_real_or_standin_model,
)

FRAME_SAMPLE_FPS = 5.0

# --- continuity thresholds ---------------------------------------------------
# Fraction of the frame diagonal a track's center is allowed to move
# between two *consecutive sampled* frames of the *same* track_id before
# it's flagged. Not physically derived — a generous sanity margin, tuned
# to catch "the tracker clearly stitched two different objects together"
# rather than "a player moved unusually fast this frame". Kept as two
# separate constants (not one shared value) since a player's box is much
# larger relative to the frame than the ball's, so what counts as a
# suspicious jump legitimately differs between them.
PLAYER_JUMP_FRACTION_OF_DIAGONAL = 0.25
BALL_JUMP_FRACTION_OF_DIAGONAL = 0.35


def _center(bbox) -> tuple[float, float]:
    return bbox.center


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def build_frame_detections(
    frame_paths: list[str], detection_results: list, class_id: int
) -> list[FrameDetections]:
    """
    Reshapes Part 5's FramePlayerBallDetections (players/balls already
    split, per detection_stage.py's convention) into the plain
    FrameDetections ByteTracker expects — same filtering
    player_tracking_stage.py / ball_tracking_stage.py do when reading
    detections.json back off disk, just skipping the JSON round-trip
    since this script has the real objects in memory already.
    """
    frames = []
    for frame_path, det in zip(frame_paths, detection_results):
        source = det.players if class_id == COCO_PERSON_CLASS_ID else det.balls
        frames.append(FrameDetections(frame_path=frame_path, detections=list(source)))
    return frames


def player_continuity_report(frame_tracks: list[FrameTracks], *, frame_diagonal: float) -> dict:
    """"Same player keeps the same ID across a rally" — see module docstring point 1."""
    appearances: dict[int, list[tuple[int, tuple[float, float]]]] = {}
    for frame_idx, ft in enumerate(frame_tracks):
        for t in ft.tracks:
            if not t.is_predicted:
                appearances.setdefault(t.track_id, []).append((frame_idx, _center(t.bbox)))

    jump_threshold = PLAYER_JUMP_FRACTION_OF_DIAGONAL * frame_diagonal
    suspicious_jumps = []
    track_lengths = {}
    for track_id, points in appearances.items():
        points.sort(key=lambda p: p[0])
        track_lengths[track_id] = len(points)
        for (idx_a, pos_a), (idx_b, pos_b) in zip(points, points[1:]):
            if idx_b - idx_a != 1:
                continue  # a real sampling gap, not a jump between adjacent frames
            d = _dist(pos_a, pos_b)
            if d > jump_threshold:
                suspicious_jumps.append(
                    {"track_id": track_id, "from_frame": idx_a, "to_frame": idx_b, "distance_px": d}
                )

    return {
        "unique_track_ids": sorted(appearances.keys()),
        "unique_track_count": len(appearances),
        "track_lengths_frames": track_lengths,
        "longest_track_frames": max(track_lengths.values(), default=0),
        "suspicious_same_id_jumps": suspicious_jumps,
    }


def ball_jump_report(frame_tracks: list[FrameTracks], *, frame_diagonal: float) -> dict:
    """"Ball track doesn't jump erratically" — see module docstring point 2.

    Deliberately NOT restricted to one track_id (unlike
    player_continuity_report): the ball's own well-documented failure mode
    is a fast movement causing ByteTracker to give up and mint a *new*
    track_id rather than stretch the old one, so a per-ID check would
    silently miss exactly the case this is meant to catch.
    """
    positions: list[tuple[int, tuple[float, float]] | None] = []
    for frame_idx, ft in enumerate(frame_tracks):
        # A dedicated ball-only tracker should report at most one live
        # track per frame; more than one is itself worth surfacing.
        positions.append((frame_idx, _center(ft.tracks[0].bbox)) if ft.tracks else None)

    jump_threshold = BALL_JUMP_FRACTION_OF_DIAGONAL * frame_diagonal
    jumps = []
    distances = []
    prev = None
    for entry in positions:
        if entry is None:
            continue
        idx, pos = entry
        if prev is not None and idx - prev[0] == 1:
            d = _dist(prev[1], pos)
            distances.append(d)
            if d > jump_threshold:
                jumps.append({"from_frame": prev[0], "to_frame": idx, "distance_px": d})
        prev = entry

    return {
        "frames_with_ball": sum(1 for e in positions if e is not None),
        "erratic_jumps": jumps,
        "max_consecutive_displacement_px": max(distances, default=0.0),
        "mean_consecutive_displacement_px": (sum(distances) / len(distances)) if distances else 0.0,
    }


def draw_trajectories(
    frame_path: str,
    player_frame_tracks: list[FrameTracks],
    ball_frame_tracks: list[FrameTracks],
    out_path: Path,
) -> None:
    """
    Visual sanity check, same role as verify_detection_pipeline.py's
    draw_annotations: traces each player track_id's path in its own color
    (a color that switches partway across a real rally is the visual
    signature of an ID switch) and the ball's full observed+interpolated
    path in red, all on one representative frame.
    """
    image = cv2.imread(frame_path)
    palette = [(255, 100, 0), (0, 200, 0), (0, 165, 255), (200, 0, 200), (255, 0, 255), (0, 255, 255)]

    per_player: dict[int, list[tuple[float, float]]] = {}
    for ft in player_frame_tracks:
        for t in ft.tracks:
            per_player.setdefault(t.track_id, []).append(_center(t.bbox))
    for i, (track_id, pts) in enumerate(per_player.items()):
        color = palette[i % len(palette)]
        pts_int = [(int(x), int(y)) for x, y in pts]
        for a, b in zip(pts_int, pts_int[1:]):
            cv2.line(image, a, b, color, 2)
        if pts_int:
            cv2.putText(image, f"player {track_id}", pts_int[-1],
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    ball_pts = [_center(ft.tracks[0].bbox) for ft in ball_frame_tracks if ft.tracks]
    ball_pts_int = [(int(x), int(y)) for x, y in ball_pts]
    for a, b in zip(ball_pts_int, ball_pts_int[1:]):
        cv2.line(image, a, b, (0, 0, 255), 1)
    for p in ball_pts_int:
        cv2.circle(image, p, 3, (0, 0, 255), thickness=-1)

    cv2.imwrite(str(out_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default=None,
                         help="Path to a real sample video. Omit to generate a synthetic one.")
    parser.add_argument("--out-dir", default=None, help="Where to write frames/results. Defaults to a temp dir.")
    args = parser.parse_args()

    settings = get_settings()
    work_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="padel_6e_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"

    if args.video:
        video_path = Path(args.video)
        print(f"[6e] Using real video: {video_path}")
    else:
        video_path = work_dir / "synthetic_sample.mp4"
        generate_synthetic_sample_video(video_path)
        print(f"[6e] No video given — generated a synthetic sample at {video_path}")

    print(f"\n=== Step 1/4: frame extraction (real ffmpeg, sample_fps={FRAME_SAMPLE_FPS}) ===")
    extraction_result = extract_frames(str(video_path), str(frames_dir), sample_fps=FRAME_SAMPLE_FPS)
    frame_paths = extraction_result.frame_paths
    print(f"[6e] Extracted {extraction_result.frame_count} frames -> {extraction_result.frames_dir}")
    assert extraction_result.frame_count > 0, "extract_frames should never return zero frames without raising"

    print(f"\n=== Step 2/4: player & ball detection (Part 5, real logic) across {len(frame_paths)} frames ===")
    model, model_kind = load_real_or_standin_model()
    print(f"[6e] model kind: {model_kind}")
    detection_results = detect_players_and_ball_many(model, frame_paths)

    detections_json_path = work_dir / "detections.json"
    detections_json_path.write_text(json.dumps(detections_to_serializable(detection_results), indent=2))
    print(f"[6e] wrote per-frame detections -> {detections_json_path}")

    print("\n=== Step 3/4: player & ball tracking (Parts 6a-6c, real logic, real settings.*_track_* tuning) ===")
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
    coverage = summarize_ball_track_coverage(ball_tracks)

    player_tracks_json = work_dir / "player_tracks.json"
    player_tracks_json.write_text(json.dumps(tracks_to_serializable(player_tracks), indent=2))
    ball_tracks_json = work_dir / "ball_tracks.json"
    ball_tracks_json.write_text(json.dumps(tracks_to_serializable(ball_tracks), indent=2))
    print(f"[6e] wrote tracks -> {player_tracks_json}, {ball_tracks_json}")

    first_frame = cv2.imread(frame_paths[0])
    frame_h, frame_w = first_frame.shape[:2]
    frame_diagonal = math.hypot(frame_w, frame_h)

    print("\n=== Step 4/4: track continuity checks ===")
    player_report = player_continuity_report(player_tracks, frame_diagonal=frame_diagonal)
    ball_report = ball_jump_report(ball_tracks, frame_diagonal=frame_diagonal)

    trajectory_path = work_dir / "annotated_trajectories.png"
    mid_frame_path = frame_paths[len(frame_paths) // 2]
    draw_trajectories(mid_frame_path, player_tracks, ball_tracks, trajectory_path)
    print(f"[6e] wrote trajectory overlay -> {trajectory_path}")

    report = {
        "model_kind": model_kind,
        "frame_count": len(frame_paths),
        "player_continuity": player_report,
        "ball_jumps": ball_report,
        "ball_coverage": coverage,
    }
    report_path = work_dir / "tracking_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"[6e] wrote full report -> {report_path}")

    print("\n=== Summary ===")
    print(f"frames:                          {len(frame_paths)}")
    print(f"unique player track IDs:         {player_report['unique_track_count']} {player_report['unique_track_ids']}")
    print(f"longest player track (frames):   {player_report['longest_track_frames']}")
    print(f"suspicious same-ID player jumps: {len(player_report['suspicious_same_id_jumps'])}")
    for j in player_report["suspicious_same_id_jumps"]:
        print(f"    track {j['track_id']}: frame {j['from_frame']}->{j['to_frame']}, {j['distance_px']:.1f}px")
    print(f"ball coverage (observed+interp): {coverage['ball_coverage_rate'] * 100:.1f}%")
    print(f"ball max frame-to-frame jump:    {ball_report['max_consecutive_displacement_px']:.1f}px")
    print(f"ball erratic jumps flagged:      {len(ball_report['erratic_jumps'])}")
    for j in ball_report["erratic_jumps"]:
        print(f"    frame {j['from_frame']}->{j['to_frame']}: {j['distance_px']:.1f}px")

    if player_report["suspicious_same_id_jumps"] or ball_report["erratic_jumps"]:
        print(
            "\n[6e] WARNING: continuity check flagged suspicious jump(s) — this is a signal to go "
            "look at annotated_trajectories.png and tracking_report.json, not a guaranteed bug (a "
            "genuinely fast lunge or volley can trip these thresholds too, especially on the "
            "synthetic sample or the OpenCV stand-in model — see module docstring)."
        )
    else:
        print("\n[6e] Continuity checks passed: no same-ID player jumps or erratic ball jumps flagged.")

    print(f"\nwork dir: {work_dir}")


if __name__ == "__main__":
    main()
