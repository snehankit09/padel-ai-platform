"""
Part 5f — end-to-end verification of the Parts 5a-5e detection pipeline
against a real sample video, with a visual sanity check.

Unlike test_yolo_detector.py / test_court_detector.py / etc. (which each
test one module in isolation, mostly against synthetic fakes), this runs
the REAL pieces together in the order detect_stage actually calls them:

    extract_frames (5a, real ffmpeg)
      -> detect_court on a handful of candidate frames (5c, real OpenCV)
      -> detect_players_and_ball_many across every frame (5d)
      -> draws the results back onto one real frame and saves it as a PNG

Deliberately doesn't touch Postgres/Celery/StorageService — those are
app/services/*_stage.py's job (Part 5e), already covered by
test_pipeline_stages.py's eager-mode tests. This script exists purely to
answer the question those tests can't: "does this look right on an actual
frame, not just structurally correct in a database row."

YOLO detection needs a real trained model to be meaningful against real
footage, and this environment has no network access to install
ultralytics or download weights (see the printed banner at the top of a
run for which path was taken). Rather than fake fixed canned boxes (which
would prove nothing about whether the pipeline's drawing/serialization
code is even looking at the right coordinates), the fallback stand-in
model finds real bounding boxes in the real frame via OpenCV colour/
contour detection — so what gets drawn is genuinely derived from the
frame's pixels, only the "is this a person or a ball" classification step
is a stand-in for a trained model rather than the model itself. Once
ultralytics + a real checkpoint are available, this script uses them
automatically (see load_model's own try/except) with zero changes needed
— that's the whole point of yolo_detector.py's model-agnostic
`.predict()` contract.

Run with (from backend/):  python -m scripts.verify_detection_pipeline [video.mp4]

With no video argument, generates a short synthetic sample clip (a static
trapezoid boundary standing in for a court, plus three moving circles
standing in for two players and a ball) so the script is runnable with no
other setup — useful to prove the pipeline plumbing is sound, but NOT a
substitute for running this against real match footage, which is the
first thing to do once any is available.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_ROOT))

from app.core.ml_path import ensure_ml_importable  # noqa: E402

ensure_ml_importable()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from ml.common.frame_extraction import extract_frames  # noqa: E402
from ml.detection.court_detector import CourtDetectionError, detect_court  # noqa: E402
from ml.detection.player_ball_detection import (  # noqa: E402
    detect_players_and_ball_many,
    summarize_ball_detection_rate,
    to_serializable,
)
from ml.detection.yolo_detector import (  # noqa: E402
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    ModelLoadError,
    load_model,
)

COURT_CALIBRATION_MAX_ATTEMPTS = 5
FRAME_SAMPLE_FPS = 5.0


# --- synthetic sample video (only used when no real video is supplied) -----


def generate_synthetic_sample_video(path: Path, *, seconds: float = 3.0, fps: int = 25) -> None:
    """
    A static trapezoid boundary (a stand-in "court" for detect_court to
    calibrate against — same shape used in test_court_detector.py's own
    fixtures) plus two mid-size circles drifting horizontally (players)
    and one small fast circle on a diagonal (ball). Real video file, real
    pixels — extract_frames and detect_court both run against it for
    real; only the video's *content* is fabricated, not the processing.
    """
    width, height = 800, 600
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    court_pts = np.array([[150, 500], [650, 500], [700, 100], [100, 100]], dtype=np.int32)

    total_frames = int(seconds * fps)
    for i in range(total_frames):
        t = i / total_frames
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.polylines(frame, [court_pts], isClosed=True, color=(255, 255, 255), thickness=4)

        # Two "players" (larger, slower, blue/green) drifting across the court.
        p1_x = int(200 + 300 * t)
        cv2.circle(frame, (p1_x, 420), 22, (255, 100, 0), thickness=-1)  # BGR: blue-ish
        p2_x = int(600 - 300 * t)
        cv2.circle(frame, (p2_x, 250), 22, (0, 200, 0), thickness=-1)  # green

        # One "ball" (small, fast, red), on a diagonal that crosses the whole frame twice.
        ball_phase = (t * 2) % 1.0
        ball_x = int(120 + 560 * ball_phase)
        ball_y = int(150 + 300 * abs(0.5 - ball_phase) * 2)
        cv2.circle(frame, (ball_x, ball_y), 7, (0, 0, 255), thickness=-1)  # red

        writer.write(frame)
    writer.release()


# --- OpenCV-based stand-in for a trained YOLO model -------------------------


class _ColorBlobBoxes:
    def __init__(self, xyxy: list, conf: list, cls: list):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self) -> int:
        return len(self.cls)


class _ColorBlobResult:
    def __init__(self, xyxy: list, conf: list, cls: list, names: dict):
        self.boxes = _ColorBlobBoxes(xyxy, conf, cls)
        self.names = names


class ColorBlobStandInModel:
    """
    Stands in for a real trained YOLO checkpoint when ultralytics/weights
    aren't available (see module docstring for why this finds real boxes
    via OpenCV rather than returning fixed canned coordinates). Matches
    the same `.predict(frame, conf=..., classes=..., verbose=...)` -> list
    of Results-shaped objects contract detect_frame expects from a real
    ultralytics model — nothing downstream needs to know which kind it's
    talking to.

    Only meaningful against the synthetic sample video's colour-coded
    circles (blue/green -> "person", red -> "sports ball"); this is a
    pipeline-plumbing check, not a substitute for real inference on real
    footage.
    """

    NAMES = {COCO_PERSON_CLASS_ID: "person", COCO_SPORTS_BALL_CLASS_ID: "sports ball"}
    # (BGR lower, BGR upper, class_id) — matches the colours drawn in
    # generate_synthetic_sample_video above.
    _COLOR_RANGES = [
        ((240, 80, 0), (255, 120, 20), COCO_PERSON_CLASS_ID),      # player 1 (blue-ish)
        ((0, 180, 0), (20, 220, 20), COCO_PERSON_CLASS_ID),        # player 2 (green)
        ((0, 0, 240), (20, 20, 255), COCO_SPORTS_BALL_CLASS_ID),   # ball (red)
    ]

    def predict(self, frame, **kwargs):
        image = cv2.imread(frame) if isinstance(frame, str) else frame
        xyxy, conf, cls = [], [], []
        for lower, upper, class_id in self._COLOR_RANGES:
            mask = cv2.inRange(image, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                if cv2.contourArea(contour) < 10:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                xyxy.append([float(x), float(y), float(x + w), float(y + h)])
                conf.append(0.9)  # stand-in confidence — there's no real model score to report
                cls.append(class_id)
        return [_ColorBlobResult(xyxy, conf, cls, self.NAMES)]


def load_real_or_standin_model():
    """Tries a real ultralytics model first; falls back to the OpenCV stand-in, loudly either way."""
    try:
        model = load_model(DEFAULT_MODEL_NAME, DEFAULT_DEVICE)
        print(f"[5f] Loaded a REAL ultralytics model ({DEFAULT_MODEL_NAME}, device={DEFAULT_DEVICE}).")
        return model, "real"
    except ModelLoadError as exc:
        print(
            f"[5f] Could not load a real YOLO model ({exc}). "
            "Falling back to the OpenCV color-blob stand-in — see module docstring. "
            "This still exercises the real pipeline plumbing, NOT real detection quality."
        )
        return ColorBlobStandInModel(), "stand-in"


# --- visualization -----------------------------------------------------------


def draw_annotations(frame_path: str, calibration, frame_detections, out_path: Path) -> None:
    image = cv2.imread(frame_path)
    if calibration is not None:
        pts = calibration.corners.as_array().astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
    for player in frame_detections.players:
        b = player.bbox
        cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (255, 100, 0), 2)
        cv2.putText(image, f"player {player.confidence:.2f}", (int(b.x1), int(b.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 0), 1)
    for ball in frame_detections.balls:
        b = ball.bbox
        cv2.rectangle(image, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)), (0, 0, 255), 2)
        cv2.putText(image, f"ball {ball.confidence:.2f}", (int(b.x1), int(b.y1) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(str(out_path), image)


# --- main ---------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", nargs="?", default=None, help="Path to a real sample video. Omit to generate a synthetic one.")
    parser.add_argument("--out-dir", default=None, help="Where to write frames/results. Defaults to a temp dir.")
    args = parser.parse_args()

    work_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="padel_5f_"))
    work_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = work_dir / "frames"

    if args.video:
        video_path = Path(args.video)
        print(f"[5f] Using real video: {video_path}")
    else:
        video_path = work_dir / "synthetic_sample.mp4"
        generate_synthetic_sample_video(video_path)
        print(f"[5f] No video given — generated a synthetic sample at {video_path}")

    print(f"\n=== Step 1/3: frame extraction (real ffmpeg, sample_fps={FRAME_SAMPLE_FPS}) ===")
    extraction_result = extract_frames(str(video_path), str(frames_dir), sample_fps=FRAME_SAMPLE_FPS)
    print(f"[5f] Extracted {extraction_result.frame_count} frames -> {extraction_result.frames_dir}")
    assert extraction_result.frame_count > 0, "extract_frames should never return zero frames without raising"

    print(f"\n=== Step 2/3: court calibration (real OpenCV, up to {COURT_CALIBRATION_MAX_ATTEMPTS} candidate frames) ===")
    frame_paths = extraction_result.frame_paths
    step = max(1, len(frame_paths) // COURT_CALIBRATION_MAX_ATTEMPTS)
    candidates = frame_paths[::step][:COURT_CALIBRATION_MAX_ATTEMPTS]
    calibration = None
    for candidate in candidates:
        try:
            calibration = detect_court(candidate)
            print(f"[5f] Court calibrated from {candidate}")
            print(f"     corners: {calibration.corners}")
            break
        except CourtDetectionError as exc:
            print(f"[5f]   {candidate}: {exc}")
    if calibration is None:
        print("[5f] WARNING: could not calibrate the court from any candidate frame (non-fatal, see Part 5e).")

    print(f"\n=== Step 3/3: player & ball detection across all {len(frame_paths)} frames ===")
    model, model_kind = load_real_or_standin_model()
    results = detect_players_and_ball_many(model, frame_paths)
    ball_rate = summarize_ball_detection_rate(results)
    total_player_detections = sum(r.player_count for r in results)
    print(f"[5f] model kind: {model_kind}")
    print(f"[5f] total player detections across all frames: {total_player_detections}")
    print(f"[5f] ball detected in {ball_rate * 100:.1f}% of frames")

    detections_json_path = work_dir / "detections.json"
    detections_json_path.write_text(json.dumps(to_serializable(results), indent=2))
    print(f"[5f] wrote per-frame detections -> {detections_json_path}")

    # Visual sanity check: annotate a frame from partway through the clip
    # (not frame 0) so it's more likely to have a representative mix of
    # detections rather than a startup transient.
    mid_index = len(frame_paths) // 2
    annotated_path = work_dir / "annotated_sample_frame.png"
    draw_annotations(frame_paths[mid_index], calibration, results[mid_index], annotated_path)
    print(f"[5f] wrote annotated frame -> {annotated_path}")

    print("\n=== Summary ===")
    print(f"frames extracted:        {extraction_result.frame_count}")
    print(f"court calibrated:        {'yes' if calibration is not None else 'no'}")
    print(f"detection model:         {model_kind}")
    print(f"ball detection rate:     {ball_rate * 100:.1f}%")
    print(f"total player detections: {total_player_detections}")
    print(f"annotated frame:         {annotated_path}")
    print(f"work dir:                {work_dir}")


if __name__ == "__main__":
    main()
