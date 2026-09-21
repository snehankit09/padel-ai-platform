"""
Highlights Improvement Roadmap, Tier 3a — team-side attribution
investigation, NOT a build script.

The roadmap's own prompt for this item was explicit: "investigate
feasibility ... before any implementation." This script is that
investigation, made runnable against a REAL match video's already-computed
pipeline output rather than staying a theoretical question — if you've
already run a video through the full pipeline (detect -> track -> analyze),
the exact data this script needs is already sitting in local storage.

**What this answers, and what it can't.** Court calibration
(ml/detection/court_detector.py, Part 5c) already produces a real
pixel-to-court-meters homography, trusted enough that shot_classification,
point_outcome, and serve_detection all already use it per-shot. Player
tracking (Part 6b) does NOT currently apply that same conversion
continuously across a rally — this script is the first thing that does,
purely to measure one question: for each tracked player, in each rally,
what fraction of frames do they spend on each half of the court, and how
often does their side flip? A track that cleanly stays on one side almost
the whole rally is a good team-side attribution candidate; a track that
flips sides constantly (net poaching, ID swaps between the ByteTracker's
own track IDs, or genuinely aggressive doubles positioning) is not — no
further build should happen on this feature until real numbers like these
say which case is typical, not just which case seems plausible on paper.

**Real-world court axis convention** (see court_detector.py's
compute_homography docstring): x runs across the court's WIDTH (0 to
10m), y runs along its LENGTH (0 to 20m) — so the net divides the court
along the Y axis, at y = court_length_m / 2, NOT along X. Getting this
backwards would silently produce a script that "works" but measures the
wrong line entirely — worth stating explicitly here rather than assuming
whoever reads this remembers court_detector.py's own convention.

A player's "position" per frame is taken as their bounding box's
bottom-center (foot position) in pixel space, converted via
pixel_to_court_point — the standard "where is this person standing on the
ground" convention, not the bbox center (which drifts upward for a tall
detected box, e.g. mid-jump for a smash).

Run with (from backend/):
    python -m scripts.investigate_team_side_attribution <video_id>

Reads directly from local storage (courts/<video_id>/calibration.json,
tracks/<video_id>/player_tracks.json, rallies/<video_id>/rallies.json) —
no DB, no Celery, no app.core.config Settings() needed, since all three
inputs are already plain files on disk once a video has gone through the
full pipeline. Only works against STORAGE_BACKEND=local's on-disk layout
(make_court_calibration_destination_path / make_player_tracks_destination_path
/ make_rally_segments_destination_path in app/services/storage.py) — pass
--storage-path if LOCAL_STORAGE_PATH isn't the default ./storage.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

FRAME_PATH_RE = re.compile(r"frame_(\d+)\.jpg$")


def _frame_index_from_path(frame_path: str) -> int:
    """Recovers the numeric frame index from a frame_%06d.jpg-style path (see ml/common/frame_extraction.py)."""
    match = FRAME_PATH_RE.search(frame_path)
    if not match:
        raise ValueError(f"Could not parse a frame index out of {frame_path!r}")
    return int(match.group(1))


def _pixel_to_court_point(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """
    Inlined rather than imported from ml.detection.court_detector, on
    purpose: this script is meant to be runnable standalone against
    already-computed JSON, without needing ensure_ml_importable() or the
    rest of that module's own dependency surface for one three-line
    perspective-transform call.
    """
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    x, y = dst[0, 0]
    return (float(x), float(y))


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} doesn't exist — has this video actually completed the full "
            "pipeline (detect through analyze), not just partway?"
        )
    with open(path) as f:
        return json.load(f)


def investigate(storage_path: Path, video_id: str) -> None:
    calibration_data = _load_json(storage_path / "courts" / video_id / "calibration.json")
    tracks_data = _load_json(storage_path / "tracks" / video_id / "player_tracks.json")
    rallies_data = _load_json(storage_path / "rallies" / video_id / "rallies.json")

    homography = np.array(calibration_data["homography"], dtype=np.float64)
    court_length_m = calibration_data.get("court_length_m", 20.0)
    net_y = court_length_m / 2.0

    # frame_index -> {track_id -> (foot_x_px, foot_y_px)}, built once so
    # each rally's lookup below is a dict access, not a re-scan of every
    # frame in the whole video per rally.
    positions_by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for frame_entry in tracks_data["frames"]:
        frame_index = _frame_index_from_path(frame_entry["frame_path"])
        per_track: dict[int, tuple[float, float]] = {}
        for t in frame_entry["tracks"]:
            bbox = t["bbox"]
            foot_x = (bbox["x1"] + bbox["x2"]) / 2.0
            foot_y = bbox["y2"]  # bottom edge = feet, not bbox center
            per_track[t["track_id"]] = (foot_x, foot_y)
        positions_by_frame[frame_index] = per_track

    rallies = rallies_data["rallies"]
    if not rallies:
        print(f"No rallies found for video_id={video_id} — nothing to analyze.")
        return

    print(f"video_id={video_id}")
    print(f"court_length_m={court_length_m}  net_y={net_y:.2f}m  (net divides the Y axis, not X)")
    print(f"{len(rallies)} rally segment(s) to check\n")

    header = f"{'rally':>5}  {'track':>5}  {'frames':>6}  {'%near (y<net)':>13}  {'%far (y>net)':>12}  {'side flips':>10}  verdict"
    print(header)
    print("-" * len(header))

    total_clean = 0
    total_tracks_seen = 0

    for rally in rallies:
        rally_index = rally["rally_index"]
        start_frame = rally["start_frame"]
        end_frame = rally["end_frame"]

        # track_id -> ordered list of court_y values across this rally's frame range
        court_y_by_track: dict[int, list[float]] = defaultdict(list)
        for frame_index in range(start_frame, end_frame + 1):
            for track_id, (px, py) in positions_by_frame.get(frame_index, {}).items():
                _, court_y = _pixel_to_court_point(homography, (px, py))
                court_y_by_track[track_id].append(court_y)

        for track_id, y_values in sorted(court_y_by_track.items()):
            if len(y_values) < 3:
                continue  # too few observed frames in this rally to say anything meaningful

            total_tracks_seen += 1
            sides = ["near" if y < net_y else "far" for y in y_values]
            near_pct = 100.0 * sides.count("near") / len(sides)
            far_pct = 100.0 * sides.count("far") / len(sides)
            flips = sum(1 for a, b in zip(sides, sides[1:]) if a != b)

            dominant_pct = max(near_pct, far_pct)
            if dominant_pct >= 90.0:
                verdict = "CLEAN"
                total_clean += 1
            elif dominant_pct >= 75.0:
                verdict = "mostly one side"
            else:
                verdict = "MIXED — flips a lot"

            print(
                f"{rally_index:>5}  {track_id:>5}  {len(y_values):>6}  "
                f"{near_pct:>12.1f}%  {far_pct:>11.1f}%  {flips:>10}  {verdict}"
            )

    print()
    if total_tracks_seen == 0:
        print("No track had enough observed frames in any rally to evaluate. Nothing to conclude.")
        return

    clean_pct = 100.0 * total_clean / total_tracks_seen
    print(f"{total_clean}/{total_tracks_seen} track-rally pairs ({clean_pct:.0f}%) stayed on one side >=90% of the time.")
    print()
    if clean_pct >= 85.0:
        print(
            "READ: team-side attribution from position alone looks like a reasonable bet on this "
            "video. Worth checking against at least one more real match (ideally one with more "
            "net play/poaching) before committing to building it for real — one video is one data "
            "point, not a pattern."
        )
    elif clean_pct >= 50.0:
        print(
            "READ: mixed result. Team-side attribution would be right often enough to maybe be "
            "useful, but wrong often enough that shipping it as a confident-looking label on every "
            "highlight clip risks being actively misleading rather than just occasionally silent. "
            "Consider only surfacing it when a track's own dominant-side percentage clears some "
            "threshold (e.g. 90%), and saying nothing for the rest, rather than guessing either way."
        )
    else:
        print(
            "READ: players don't reliably stay on one side in this video. Position-only team-side "
            "attribution is probably not reliable enough to build as-is — this doubles court's "
            "actual play style (heavy poaching/switching) may just not fit the assumption this "
            "feature was hoping to lean on."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video_id", help="UUID of an already-fully-processed video")
    parser.add_argument(
        "--storage-path", default="../storage",
        help="Path to the local storage root (default: ../storage, i.e. LOCAL_STORAGE_PATH's default when run from backend/)",
    )
    args = parser.parse_args()

    try:
        investigate(Path(args.storage_path), args.video_id)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
