"""
Tests for ml/tracking/ball_interpolation.py — Part 6c.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_byte_tracker.py: builds FrameTracks/TrackedDetection objects
directly and feeds them straight into interpolate_ball_gaps.

Run with: pytest backend/tests/test_ball_interpolation.py -v
"""

from __future__ import annotations

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.detection.yolo_detector import BoundingBox  # noqa: E402
from ml.tracking.ball_interpolation import (  # noqa: E402
    interpolate_ball_gaps,
    summarize_ball_track_coverage,
)
from ml.tracking.byte_tracker import FrameTracks, TrackedDetection  # noqa: E402

BALL = 32


def _tracked(track_id, x, *, conf=0.5) -> TrackedDetection:
    return TrackedDetection(
        track_id=track_id,
        class_id=BALL,
        class_name="sports ball",
        confidence=conf,
        bbox=BoundingBox(x, 100, x + 8, 108),
        age=1,
        hits=1,
        time_since_update=0,
    )


def _frame(frame_path, tracks) -> FrameTracks:
    return FrameTracks(frame_path=frame_path, tracks=tracks)


def test_short_gap_gets_interpolated():
    # frame 0: real detection. frames 1-2: missed. frame 3: real detection.
    frames = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", []),
        _frame("f3.jpg", [_tracked(1, 30)]),
    ]
    result = interpolate_ball_gaps(frames, max_gap_frames=4)

    assert len(result[1].tracks) == 1, "a 2-frame gap should be bridged"
    assert len(result[2].tracks) == 1
    # Straight line from x=0 to x=30 over 3 steps: frame1 -> x=10, frame2 -> x=20.
    assert result[1].tracks[0].bbox.x1 == 10.0
    assert result[2].tracks[0].bbox.x1 == 20.0
    assert result[1].tracks[0].track_id == 1
    assert result[1].tracks[0].is_predicted, "interpolated frames must be flagged, not treated as real detections"
    assert not result[0].tracks[0].is_predicted, "the real anchor detections must stay untouched"
    assert not result[3].tracks[0].is_predicted


def test_gap_longer_than_cap_is_left_alone():
    frames = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", []),
        _frame("f3.jpg", []),
        _frame("f4.jpg", []),
        _frame("f5.jpg", [_tracked(1, 100)]),
    ]
    # gap is 4 frames; cap is 2 -> must not be bridged (e.g. ball left frame on a lob)
    result = interpolate_ball_gaps(frames, max_gap_frames=2)

    for fr in result[1:5]:
        assert fr.tracks == [], "a gap longer than the cap must be left empty, not fabricated"


def test_gap_exactly_at_cap_is_bridged_gap_one_more_is_not():
    frames_at_cap = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", []),
        _frame("f3.jpg", [_tracked(1, 30)]),
    ]
    result_at_cap = interpolate_ball_gaps(frames_at_cap, max_gap_frames=2)
    assert len(result_at_cap[1].tracks) == 1
    assert len(result_at_cap[2].tracks) == 1

    frames_over_cap = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", []),
        _frame("f3.jpg", []),
        _frame("f4.jpg", [_tracked(1, 30)]),
    ]
    result_over_cap = interpolate_ball_gaps(frames_over_cap, max_gap_frames=2)
    assert result_over_cap[1].tracks == []
    assert result_over_cap[2].tracks == []
    assert result_over_cap[3].tracks == []


def test_different_track_ids_are_never_bridged_across_each_other():
    """A gap where ByteTracker itself started a new track_id must never be filled — see module docstring."""
    frames = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", [_tracked(2, 30)]),  # different identity — track 1 was lost, this is a new track
    ]
    result = interpolate_ball_gaps(frames, max_gap_frames=4)
    assert result[1].tracks == [], "no bridging across two different track_ids"


def test_no_gap_at_all_is_a_no_op():
    frames = [_frame(f"f{i}.jpg", [_tracked(1, i * 10)]) for i in range(4)]
    result = interpolate_ball_gaps(frames, max_gap_frames=4)
    for original, produced in zip(frames, result):
        assert produced.tracks == original.tracks


def test_input_list_is_not_mutated():
    frames = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", [_tracked(1, 20)]),
    ]
    interpolate_ball_gaps(frames, max_gap_frames=4)
    assert frames[1].tracks == [], "interpolate_ball_gaps must return a new list, not mutate its input"


def test_summarize_ball_track_coverage_splits_observed_from_interpolated():
    frames = [
        _frame("f0.jpg", [_tracked(1, 0)]),
        _frame("f1.jpg", []),
        _frame("f2.jpg", [_tracked(1, 20)]),
        _frame("f3.jpg", []),  # gap too long relative to cap=0 below -> stays empty
    ]
    result = interpolate_ball_gaps(frames, max_gap_frames=1)
    summary = summarize_ball_track_coverage(result)

    assert summary["frame_count"] == 4
    assert summary["observed_ball_detections"] == 2
    assert summary["interpolated_ball_detections"] == 1
    assert summary["frames_with_ball"] == 3
    assert summary["ball_coverage_rate"] == 3 / 4


def test_summarize_ball_track_coverage_handles_empty_input():
    assert summarize_ball_track_coverage([]) == {
        "frame_count": 0,
        "frames_with_ball": 0,
        "observed_ball_detections": 0,
        "interpolated_ball_detections": 0,
        "ball_coverage_rate": 0.0,
    }
