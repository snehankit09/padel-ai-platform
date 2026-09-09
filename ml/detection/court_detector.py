"""
Court detection & calibration — Part 5c.

Deliberately a different technique from ml/detection/yolo_detector.py
(Part 5b), for the reason called out in that module's docstring: COCO
(what the pretrained YOLO checkpoint knows) has no "court" class at all,
and a court isn't really an object to draw one box around anyway — it's a
flat rectangle seen at an angle, and what every later stage actually needs
from it is a mapping from pixel coordinates to real-world court
coordinates (PRD 5.2: "Court detection and calibration"; PRD 5.5's
heatmaps and movement stats explicitly require "court-relative
coordinates, not raw pixel coordinates").

So this module finds the court's four outer boundary corners with
classical line detection (Canny edges + a Hough transform, both plain
OpenCV — no model weights, no network, no training data required) instead
of a learned detector, then solves a perspective transform (homography)
from those four pixel corners to the four corners of a real, flat
20m x 10m rectangle. Padel courts are internationally standardized to
that one size (FIP regulations) — unlike tennis there's no
singles/doubles width to choose between — so that rectangle is a safe
default rather than a per-venue guess.

Same "frame in -> result out" shape and input convention as
yolo_detector.detect_frame (a path or an already-decoded array), and the
same "fail loud, don't silently produce nothing useful" stance as
extract_frames (Part 5a) and detect_frame (Part 5b): a wrong-but-returned
calibration would quietly corrupt every court-relative coordinate
computed from it downstream, which is worse than raising here with a
specific reason.

Layered the same way as Part 5b, for the same testability reason: the
pure geometry (find_court_quadrilateral, order_corners,
compute_homography, pixel_to_court_point) needs no image, no OpenCV video
I/O, and no real match footage to unit test — synthetic point sets and
synthetic drawn images are enough, and deterministic, since nothing here
is a trained model with weights that vary by environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

# Padel courts are a fixed, internationally standardized size (FIP
# regulations), not a per-venue variable — so these are safe defaults,
# not a guess. Length is the longer (baseline-to-baseline) dimension.
DEFAULT_COURT_LENGTH_M = 20.0
DEFAULT_COURT_WIDTH_M = 10.0

DEFAULT_CANNY_THRESHOLDS: tuple[int, int] = (50, 150)
DEFAULT_HOUGH_THRESHOLD = 80
DEFAULT_HOUGH_MIN_LINE_LENGTH = 60
DEFAULT_HOUGH_MAX_LINE_GAP = 20
# Fraction of the convex hull's perimeter used as cv2.approxPolyDP's
# epsilon — how aggressively nearby hull vertices get merged before
# checking whether what's left is a clean quadrilateral.
DEFAULT_APPROX_EPSILON_RATIO = 0.02


class CourtDetectionError(Exception):
    """Raised when a frame's court boundary can't be reliably found or calibrated."""


@dataclass(frozen=True)
class LineSegment:
    """One straight segment found by the Hough transform, in source-frame pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)


@dataclass(frozen=True)
class CourtCorners:
    """
    The court's four outer corners in source-frame pixel coordinates, in a
    fixed order (top_left, top_right, bottom_right, bottom_left) —
    everything downstream (compute_homography, and anything that reads
    these fields directly) depends on that order being consistent, which
    is exactly what order_corners exists to guarantee regardless of what
    order detection happened to produce them in.
    """

    top_left: tuple[float, float]
    top_right: tuple[float, float]
    bottom_right: tuple[float, float]
    bottom_left: tuple[float, float]

    def as_array(self) -> np.ndarray:
        """The four corners as a (4, 2) float32 array, in the same fixed order — what cv2's API wants."""
        return np.array(
            [self.top_left, self.top_right, self.bottom_right, self.bottom_left],
            dtype=np.float32,
        )


@dataclass(frozen=True, eq=False)
class CourtCalibration:
    """
    One court's detected corners plus the homography that maps pixel
    coordinates in that frame to real-world court meters.

    eq=False deliberately: the default dataclass equality would compare
    `homography` (a numpy array) with `==`, which raises "truth value of
    an array is ambiguous" the moment two instances are compared — instead
    of a false negative that's easy to miss, this makes that a hard error
    at the language level. Callers that need to compare two calibrations
    should compare specific fields (e.g. `.corners`) instead.
    """

    corners: CourtCorners
    homography: np.ndarray
    court_length_m: float = DEFAULT_COURT_LENGTH_M
    court_width_m: float = DEFAULT_COURT_WIDTH_M

    def pixel_to_court(self, point: tuple[float, float]) -> tuple[float, float]:
        """Maps one pixel coordinate in the calibrated frame to real-world court meters."""
        return pixel_to_court_point(self.homography, point)

    def pixels_to_court(self, points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [self.pixel_to_court(p) for p in points]


def load_frame(frame) -> np.ndarray:
    """
    Accepts a path (str/Path) or an already-decoded ndarray and returns a
    decoded ndarray either way — same input convention as
    ml.detection.yolo_detector.detect_frame, so callers can pass the same
    kind of value (e.g. one of frame_extraction's frame_paths) to either
    module without converting it first.
    """
    if isinstance(frame, (str, Path)):
        image = cv2.imread(str(frame))
        if image is None:
            raise CourtDetectionError(f"could not read image at {frame!r}")
        return image
    return frame


def detect_line_segments(
    frame: np.ndarray,
    *,
    canny_thresholds: tuple[int, int] = DEFAULT_CANNY_THRESHOLDS,
    hough_threshold: int = DEFAULT_HOUGH_THRESHOLD,
    min_line_length: int = DEFAULT_HOUGH_MIN_LINE_LENGTH,
    max_line_gap: int = DEFAULT_HOUGH_MAX_LINE_GAP,
) -> list[LineSegment]:
    """
    Canny edge detection followed by a probabilistic Hough transform — the
    raw material court boundary (and other) lines are made of. `frame`
    must already be a decoded ndarray (grayscale or BGR); use load_frame
    first if you're starting from a path.

    Returns every straight segment found, boundary and non-boundary alike
    (a net line, a service line, or an unrelated straight edge in the
    background will all show up here too) — separating the true court
    boundary out from that mixed bag is find_court_quadrilateral's job,
    not this function's, so this stays a thin, easily-testable wrapper
    around two stock OpenCV calls.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, canny_thresholds[0], canny_thresholds[1])
    raw = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if raw is None:
        return []
    return [LineSegment(float(x1), float(y1), float(x2), float(y2)) for x1, y1, x2, y2 in raw[:, 0, :]]


def find_court_quadrilateral(
    lines: Sequence[LineSegment],
    *,
    approx_epsilon_ratio: float = DEFAULT_APPROX_EPSILON_RATIO,
) -> list[tuple[float, float]] | None:
    """
    Reduces a bag of line-segment endpoints down to the 4 corners of the
    court's *outer* boundary.

    Approach: every segment's endpoints necessarily lie on or near the
    true court lines, so the convex hull of all of them approximates the
    outer boundary even with inner lines (service line, net) or noise
    mixed into `lines` — those lie inside the hull, not on it, so they
    don't affect the result. cv2.approxPolyDP then simplifies that hull
    down to as few vertices as the epsilon allows.

    Returns None — rather than guessing — if the simplified hull doesn't
    reduce to exactly 4 points. That means either too few/noisy lines were
    found, or the scene doesn't have a clean rectangular boundary in it;
    either way a 5- or 3-point "quadrilateral" isn't a court corner set
    worth trusting for a homography.
    """
    if len(lines) < 4:
        return None

    points = np.array(
        [pt for seg in lines for pt in ((seg.x1, seg.y1), (seg.x2, seg.y2))],
        dtype=np.float32,
    )
    hull = cv2.convexHull(points)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, approx_epsilon_ratio * perimeter, True)

    if len(approx) != 4:
        return None

    return [(float(p[0][0]), float(p[0][1])) for p in approx]


def order_corners(points: Sequence[tuple[float, float]]) -> CourtCorners:
    """
    Takes 4 unordered (x, y) points and returns them as a CourtCorners in a
    fixed order (top-left, top-right, bottom-right, bottom-left) —
    cv2.convexHull/approxPolyDP hand back points in hull-traversal order,
    which depends on where in the frame the court happens to sit, not on
    which corner is which.

    Standard sum/difference trick, works for any convex quadrilateral
    (not just axis-aligned ones, which is what a camera-angled court
    photo actually produces): top-left has the smallest (x + y),
    bottom-right the largest; top-right has the smallest (y - x),
    bottom-left the largest.
    """
    if len(points) != 4:
        raise ValueError(f"order_corners requires exactly 4 points, got {len(points)}")

    pts = np.array(points, dtype=np.float64)
    sums = pts.sum(axis=1)
    diffs = pts[:, 1] - pts[:, 0]  # y - x

    top_left = pts[np.argmin(sums)]
    bottom_right = pts[np.argmax(sums)]
    top_right = pts[np.argmin(diffs)]
    bottom_left = pts[np.argmax(diffs)]

    return CourtCorners(
        top_left=(float(top_left[0]), float(top_left[1])),
        top_right=(float(top_right[0]), float(top_right[1])),
        bottom_right=(float(bottom_right[0]), float(bottom_right[1])),
        bottom_left=(float(bottom_left[0]), float(bottom_left[1])),
    )


def compute_homography(
    corners: CourtCorners,
    *,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
) -> np.ndarray:
    """
    Builds the pixel-space -> real-world-meters perspective transform for
    one calibrated court.

    Real-world axes, as a fixed convention every caller of
    pixel_to_court_point can rely on: x runs across the court's width (0
    -> court_width_m), y runs along its length (0 -> court_length_m),
    origin at the top-left corner per CourtCorners' own order.

    Uses cv2.getPerspectiveTransform (an exact solve for exactly 4 point
    correspondences), not cv2.findHomography (a least-squares fit meant
    for >4, noisy correspondences) — the 4 ordered corners are the entire
    input here, so there's nothing to average over.
    """
    src = corners.as_array()
    dst = np.array(
        [
            (0.0, 0.0),
            (court_width_m, 0.0),
            (court_width_m, court_length_m),
            (0.0, court_length_m),
        ],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def pixel_to_court_point(homography: np.ndarray, point: tuple[float, float]) -> tuple[float, float]:
    """Applies a homography to map one pixel coordinate to real-world court meters."""
    src = np.array([[[point[0], point[1]]]], dtype=np.float32)
    dst = cv2.perspectiveTransform(src, homography)
    x, y = dst[0, 0]
    return (float(x), float(y))


def detect_court(
    frame,
    *,
    court_length_m: float = DEFAULT_COURT_LENGTH_M,
    court_width_m: float = DEFAULT_COURT_WIDTH_M,
    canny_thresholds: tuple[int, int] = DEFAULT_CANNY_THRESHOLDS,
    hough_threshold: int = DEFAULT_HOUGH_THRESHOLD,
    min_line_length: int = DEFAULT_HOUGH_MIN_LINE_LENGTH,
    max_line_gap: int = DEFAULT_HOUGH_MAX_LINE_GAP,
    approx_epsilon_ratio: float = DEFAULT_APPROX_EPSILON_RATIO,
) -> CourtCalibration:
    """
    The reusable core: frame in -> CourtCalibration out. Runs the full
    pipeline — line detection, boundary-quadrilateral extraction, corner
    ordering, homography — and hands back an object any later stage can
    call `.pixel_to_court(...)` on to convert a tracked player/ball pixel
    position (Part 6+) into real-world court coordinates.

    `frame` is a path (str/Path) or an already-decoded ndarray, same
    convention as yolo_detector.detect_frame.

    Raises CourtDetectionError if no confident 4-corner boundary can be
    found at any stage of the pipeline, with a reason specific to which
    stage failed — never returns a wrong-but-plausible-looking
    calibration. A silently bad one would poison every court-relative
    coordinate computed from it downstream (PRD 5.2, 5.5) with no signal
    that anything was wrong.
    """
    image = load_frame(frame)
    lines = detect_line_segments(
        image,
        canny_thresholds=canny_thresholds,
        hough_threshold=hough_threshold,
        min_line_length=min_line_length,
        max_line_gap=max_line_gap,
    )
    if len(lines) < 4:
        raise CourtDetectionError(
            f"found only {len(lines)} line segment(s) — not enough to determine a court boundary"
        )

    quad = find_court_quadrilateral(lines, approx_epsilon_ratio=approx_epsilon_ratio)
    if quad is None:
        raise CourtDetectionError(
            "detected lines did not reduce to a clean 4-corner boundary — "
            "scene may lack a clear court outline, or lighting/occlusion "
            "made the edges too noisy (see PRD Section 13 risks: Lighting "
            "changes, Occlusion)"
        )

    corners = order_corners(quad)
    homography = compute_homography(corners, court_length_m=court_length_m, court_width_m=court_width_m)

    return CourtCalibration(
        corners=corners,
        homography=homography,
        court_length_m=court_length_m,
        court_width_m=court_width_m,
    )
