"""
Tests for ml/detection/court_detector.py — Part 5c.

Unlike test_yolo_detector.py's real-model tests (Part 5b), nothing here
needs network access or downloaded weights — this module is plain OpenCV
line detection + geometry, so every test runs against synthetic points or
synthetic drawn images and is fully deterministic. No skips.

Run with: pytest backend/tests/test_court_detector.py -v
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
import cv2  # noqa: E402

from ml.detection.court_detector import (  # noqa: E402
    CourtCorners,
    CourtDetectionError,
    LineSegment,
    compute_homography,
    detect_court,
    detect_line_segments,
    find_court_quadrilateral,
    load_frame,
    order_corners,
    pixel_to_court_point,
)

# A camera-angled rectangle: not axis-aligned, the way a real court photo
# from an elevated/tripod angle wouldn't be either. Reused by several tests
# below so they agree on what "the court" looks like.
TRUE_CORNERS = {
    "top_left": (150.0, 80.0),
    "top_right": (490.0, 60.0),
    "bottom_right": (600.0, 420.0),
    "bottom_left": (60.0, 440.0),
}


def _draw_synthetic_court(size=(480, 640), with_inner_line=True) -> np.ndarray:
    """A white quadrilateral outline on a black background, standing in for a court's boundary lines."""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    pts = np.array(
        [TRUE_CORNERS["top_left"], TRUE_CORNERS["top_right"], TRUE_CORNERS["bottom_right"], TRUE_CORNERS["bottom_left"]],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=4)
    if with_inner_line:
        # A net-like inner line — must not confuse boundary detection into a 5- or 6-point shape.
        cv2.line(img, (325, 70), (330, 430), color=(255, 255, 255), thickness=3)
    return img


# --- LineSegment ---------------------------------------------------------


def test_line_segment_length():
    seg = LineSegment(0.0, 0.0, 3.0, 4.0)
    assert seg.length == 5.0


# --- order_corners: pure geometry, no image involved -----------------------


def test_order_corners_identifies_each_corner_regardless_of_input_order():
    points = list(TRUE_CORNERS.values())
    random.Random(7).shuffle(points)

    corners = order_corners(points)

    assert corners.top_left == TRUE_CORNERS["top_left"]
    assert corners.top_right == TRUE_CORNERS["top_right"]
    assert corners.bottom_right == TRUE_CORNERS["bottom_right"]
    assert corners.bottom_left == TRUE_CORNERS["bottom_left"]


def test_order_corners_works_for_axis_aligned_rectangle_too():
    points = [(0.0, 0.0), (100.0, 0.0), (100.0, 50.0), (0.0, 50.0)]
    random.Random(3).shuffle(points)

    corners = order_corners(points)

    assert corners.top_left == (0.0, 0.0)
    assert corners.top_right == (100.0, 0.0)
    assert corners.bottom_right == (100.0, 50.0)
    assert corners.bottom_left == (0.0, 50.0)


def test_order_corners_rejects_wrong_point_count():
    with pytest.raises(ValueError, match="exactly 4 points"):
        order_corners([(0, 0), (1, 1), (2, 2)])
    with pytest.raises(ValueError):
        order_corners([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])


# --- compute_homography / pixel_to_court_point: pure geometry --------------


def test_homography_maps_corners_to_exact_real_world_rectangle():
    corners = order_corners(list(TRUE_CORNERS.values()))
    homography = compute_homography(corners, court_length_m=20.0, court_width_m=10.0)

    tl = pixel_to_court_point(homography, corners.top_left)
    tr = pixel_to_court_point(homography, corners.top_right)
    br = pixel_to_court_point(homography, corners.bottom_right)
    bl = pixel_to_court_point(homography, corners.bottom_left)

    # An exact 4-point solve: the 4 defining correspondences map perfectly, no tolerance needed.
    assert tl == pytest.approx((0.0, 0.0), abs=1e-4)
    assert tr == pytest.approx((10.0, 0.0), abs=1e-4)
    assert br == pytest.approx((10.0, 20.0), abs=1e-4)
    assert bl == pytest.approx((0.0, 20.0), abs=1e-4)


def test_homography_respects_custom_court_dimensions():
    corners = order_corners([(0.0, 0.0), (200.0, 0.0), (200.0, 200.0), (0.0, 200.0)])
    homography = compute_homography(corners, court_length_m=1.0, court_width_m=1.0)

    center = pixel_to_court_point(homography, (100.0, 100.0))

    assert center == pytest.approx((0.5, 0.5), abs=1e-4)


def test_pixel_to_court_maps_a_general_point_between_corners():
    corners = order_corners([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)])
    homography = compute_homography(corners, court_length_m=20.0, court_width_m=10.0)

    # exact center pixel of an axis-aligned square maps to exact court center
    assert pixel_to_court_point(homography, (5.0, 5.0)) == pytest.approx((5.0, 10.0), abs=1e-3)


# --- find_court_quadrilateral: pure geometry on synthetic points -----------


def test_find_court_quadrilateral_from_rectangle_segments():
    corners = list(TRUE_CORNERS.values())
    # Build segments directly tracing the 4 boundary edges (no image/Hough involved).
    ordered = [corners[0], corners[1], corners[2], corners[3]]
    lines = [
        LineSegment(*ordered[i], *ordered[(i + 1) % 4])
        for i in range(4)
    ]

    quad = find_court_quadrilateral(lines)

    assert quad is not None
    assert len(quad) == 4
    result_set = {tuple(round(v) for v in p) for p in quad}
    expected_set = {tuple(round(v) for v in p) for p in corners}
    assert result_set == expected_set


def test_find_court_quadrilateral_ignores_an_inner_line():
    corners = list(TRUE_CORNERS.values())
    lines = [LineSegment(*corners[i], *corners[(i + 1) % 4]) for i in range(4)]
    # An inner "net" segment, well inside the hull — must not create a 5th hull vertex.
    lines.append(LineSegment(325, 70, 330, 430))

    quad = find_court_quadrilateral(lines)

    assert quad is not None
    assert len(quad) == 4


def test_find_court_quadrilateral_returns_none_for_too_few_lines():
    assert find_court_quadrilateral([LineSegment(0, 0, 1, 1)]) is None
    assert find_court_quadrilateral([]) is None


def test_find_court_quadrilateral_returns_none_for_non_quadrilateral_shape():
    # Points roughly on a circle approximate a many-sided polygon, not a quadrilateral.
    import math

    lines = []
    n = 12
    radius = 100
    pts = [
        (100 + radius * math.cos(2 * math.pi * i / n), 100 + radius * math.sin(2 * math.pi * i / n))
        for i in range(n)
    ]
    for i in range(n):
        lines.append(LineSegment(*pts[i], *pts[(i + 1) % n]))

    quad = find_court_quadrilateral(lines, approx_epsilon_ratio=0.01)

    assert quad is None


# --- detect_line_segments / load_frame: real OpenCV, synthetic image -------


def test_detect_line_segments_finds_the_drawn_boundary():
    img = _draw_synthetic_court(with_inner_line=False)

    lines = detect_line_segments(img)

    assert len(lines) >= 4


def test_detect_line_segments_returns_empty_list_for_blank_image():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)

    assert detect_line_segments(blank) == []


def test_detect_line_segments_accepts_grayscale_input():
    img = _draw_synthetic_court(with_inner_line=False)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lines = detect_line_segments(gray)

    assert len(lines) >= 4


def test_load_frame_passes_through_an_already_decoded_array():
    img = _draw_synthetic_court()
    assert load_frame(img) is img


def test_load_frame_reads_a_real_image_from_disk(tmp_path):
    img = _draw_synthetic_court()
    path = tmp_path / "court.jpg"
    cv2.imwrite(str(path), img)

    loaded = load_frame(str(path))

    assert loaded.shape == img.shape


def test_load_frame_raises_specific_error_for_unreadable_path(tmp_path):
    missing = tmp_path / "does_not_exist.jpg"
    with pytest.raises(CourtDetectionError, match="does_not_exist.jpg"):
        load_frame(str(missing))


# --- detect_court: full pipeline, synthetic image end-to-end ---------------


def test_detect_court_recovers_approximate_true_corners():
    img = _draw_synthetic_court()

    calibration = detect_court(img)

    for expected, actual in zip(TRUE_CORNERS.values(), _as_ordered_list(calibration.corners)):
        assert actual == pytest.approx(expected, abs=6.0)  # a few px of tolerance for line thickness/Hough noise


def test_detect_court_pixel_to_court_maps_corners_near_real_world_rectangle():
    img = _draw_synthetic_court()

    calibration = detect_court(img)

    assert calibration.pixel_to_court(calibration.corners.top_left) == pytest.approx((0.0, 0.0), abs=0.5)
    assert calibration.pixel_to_court(calibration.corners.bottom_right) == pytest.approx(
        (calibration.court_width_m, calibration.court_length_m), abs=0.5
    )


def test_detect_court_uses_default_padel_dimensions():
    img = _draw_synthetic_court()

    calibration = detect_court(img)

    assert calibration.court_length_m == 20.0
    assert calibration.court_width_m == 10.0


def test_detect_court_respects_custom_dimensions():
    img = _draw_synthetic_court()

    calibration = detect_court(img, court_length_m=20.0, court_width_m=10.0)
    br = calibration.pixel_to_court(calibration.corners.bottom_right)

    assert br == pytest.approx((10.0, 20.0), abs=0.5)


def test_detect_court_accepts_a_path(tmp_path):
    img = _draw_synthetic_court()
    path = tmp_path / "court.jpg"
    cv2.imwrite(str(path), img)

    calibration = detect_court(str(path))

    assert calibration.corners is not None


def test_detect_court_raises_for_blank_frame():
    blank = np.zeros((200, 200, 3), dtype=np.uint8)
    with pytest.raises(CourtDetectionError, match="not enough"):
        detect_court(blank)


def test_detect_court_raises_for_non_court_shaped_scene():
    """A scene with a clean 5-sided outline (not a quadrilateral) should not fabricate a 4-corner calibration."""
    import math

    img = np.zeros((300, 300, 3), dtype=np.uint8)
    n = 5
    radius = 120
    pts = np.array(
        [
            (150 + radius * math.cos(2 * math.pi * i / n - math.pi / 2), 150 + radius * math.sin(2 * math.pi * i / n - math.pi / 2))
            for i in range(n)
        ],
        dtype=np.int32,
    )
    cv2.polylines(img, [pts], isClosed=True, color=(255, 255, 255), thickness=3)

    with pytest.raises(CourtDetectionError):
        detect_court(img)


def test_detect_court_pixels_to_court_batch_matches_individual_calls():
    img = _draw_synthetic_court()
    calibration = detect_court(img)
    points = [calibration.corners.top_left, calibration.corners.bottom_right]

    batch = calibration.pixels_to_court(points)

    assert batch == [calibration.pixel_to_court(p) for p in points]


def _as_ordered_list(corners: CourtCorners) -> list[tuple[float, float]]:
    return [corners.top_left, corners.top_right, corners.bottom_right, corners.bottom_left]
