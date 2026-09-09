"""
Court, player, and ball detection (Part 5b+).

Part 5b (yolo_detector.py) is the reusable core for generic object
detection: load a pretrained YOLO model, run it against a single frame,
get back a plain list of Detection dataclasses — used for players and the
ball. Part 5c (court_detector.py) handles the court itself with a
different technique (classical line detection + homography, not a
learned detector — see that module's docstring for why) and hands back a
CourtCalibration that converts pixel coordinates to real-world court
meters.
"""

from ml.detection.court_detector import (
    DEFAULT_APPROX_EPSILON_RATIO,
    DEFAULT_CANNY_THRESHOLDS,
    DEFAULT_COURT_LENGTH_M,
    DEFAULT_COURT_WIDTH_M,
    DEFAULT_HOUGH_MAX_LINE_GAP,
    DEFAULT_HOUGH_MIN_LINE_LENGTH,
    DEFAULT_HOUGH_THRESHOLD,
    CourtCalibration,
    CourtCorners,
    CourtDetectionError,
    LineSegment,
    compute_homography,
    detect_court,
    detect_line_segments,
    find_court_quadrilateral,
    order_corners,
    pixel_to_court_point,
)
from ml.detection.yolo_detector import (
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    DEFAULT_CLASS_IDS,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_DEVICE,
    DEFAULT_MODEL_NAME,
    BoundingBox,
    Detection,
    DetectionError,
    FrameDetections,
    ModelLoadError,
    YOLODetector,
    detect_frame,
    detect_frames,
    get_cached_model,
    load_model,
    parse_yolo_result,
)

__all__ = [
    # yolo_detector (Part 5b)
    "COCO_PERSON_CLASS_ID",
    "COCO_SPORTS_BALL_CLASS_ID",
    "DEFAULT_CLASS_IDS",
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_DEVICE",
    "DEFAULT_MODEL_NAME",
    "BoundingBox",
    "Detection",
    "DetectionError",
    "FrameDetections",
    "ModelLoadError",
    "YOLODetector",
    "detect_frame",
    "detect_frames",
    "get_cached_model",
    "load_model",
    "parse_yolo_result",
    # court_detector (Part 5c)
    "DEFAULT_APPROX_EPSILON_RATIO",
    "DEFAULT_CANNY_THRESHOLDS",
    "DEFAULT_COURT_LENGTH_M",
    "DEFAULT_COURT_WIDTH_M",
    "DEFAULT_HOUGH_MAX_LINE_GAP",
    "DEFAULT_HOUGH_MIN_LINE_LENGTH",
    "DEFAULT_HOUGH_THRESHOLD",
    "CourtCalibration",
    "CourtCorners",
    "CourtDetectionError",
    "LineSegment",
    "compute_homography",
    "detect_court",
    "detect_line_segments",
    "find_court_quadrilateral",
    "order_corners",
    "pixel_to_court_point",
]

