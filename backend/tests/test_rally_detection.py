"""
Tests for ml/pipeline/rally_detection.py — Part 7a.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_ball_interpolation.py: builds BallFrameSignal objects directly
(or plain ball_tracks.json-shaped dicts, for frames_from_ball_tracks_json)
and feeds them straight into detect_rally_segments.

Run with: pytest backend/tests/test_rally_detection.py -v
"""

from __future__ import annotations

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.rally_detection import (  # noqa: E402
    BallFrameSignal,
    RallyDetectionError,
    detect_rally_segments,
    frames_from_ball_tracks_json,
    summarize_rally_segments,
    to_serializable,
)


def _signal(idx: int, present: bool, *, observed: bool | None = None, conf: float | None = None) -> BallFrameSignal:
    if observed is None:
        observed = present
    if conf is None and present:
        conf = 0.5
    return BallFrameSignal(frame_index=idx, frame_path=f"f{idx}.jpg", ball_present=present, observed=observed, confidence=conf)


def _signals(pattern: str) -> list[BallFrameSignal]:
    """Builds a list of signals from a compact 'X' (present) / '.' (absent) string, e.g. 'XXX..XXX'."""
    return [_signal(i, ch == "X") for i, ch in enumerate(pattern)]


# --- core boundary detection -------------------------------------------------


def test_continuous_activity_is_one_rally():
    signals = _signals("XXXXXXXX")
    result = detect_rally_segments(signals, sample_fps=5.0)

    assert len(result) == 1
    rally = result[0]
    assert rally.rally_index == 1
    assert rally.start_frame == 0
    assert rally.end_frame == 7
    assert rally.frame_count == 8


def test_short_gap_within_tolerance_does_not_split_the_rally():
    # 3-frame gap, tolerance 4 -> bridged into one rally.
    signals = _signals("XXXXX...XXXXX")
    result = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=4, min_rally_duration_frames=1)

    assert len(result) == 1
    assert result[0].start_frame == 0
    assert result[0].end_frame == 12


def test_long_gap_beyond_tolerance_splits_into_two_rallies():
    # Same 3-frame gap, tolerance 1 -> NOT bridged, two separate rallies.
    signals = _signals("XXXXX...XXXXX")
    result = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=1, min_rally_duration_frames=1)

    assert len(result) == 2
    assert result[0].start_frame == 0 and result[0].end_frame == 4
    assert result[1].start_frame == 8 and result[1].end_frame == 12
    assert result[0].rally_index == 1
    assert result[1].rally_index == 2


def test_gap_touching_the_start_is_never_bridged():
    # No activity before frame 3 -> the leading dead time must never
    # become part of a rally, no matter how generous the tolerance.
    signals = _signals("...XXXXX")
    result = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=100, min_rally_duration_frames=1)

    assert len(result) == 1
    assert result[0].start_frame == 3
    assert result[0].end_frame == 7


def test_gap_touching_the_end_is_never_bridged():
    signals = _signals("XXXXX...")
    result = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=100, min_rally_duration_frames=1)

    assert len(result) == 1
    assert result[0].start_frame == 0
    assert result[0].end_frame == 4


def test_short_blip_below_min_duration_is_dropped_as_noise():
    signals = _signals("...X....XXXXXXXX")  # a 1-frame blip, then a real rally
    result = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=0, min_rally_duration_frames=5)

    assert len(result) == 1, "the 1-frame blip must not be reported as its own rally"
    assert result[0].start_frame == 8


def test_no_activity_at_all_produces_no_rallies():
    signals = _signals("............")
    result = detect_rally_segments(signals, sample_fps=5.0)
    assert result == []


def test_empty_input_produces_no_rallies():
    assert detect_rally_segments([], sample_fps=5.0) == []


def test_invalid_sample_fps_raises():
    signals = _signals("XXX")
    for bad_fps in (0.0, -1.0):
        try:
            detect_rally_segments(signals, sample_fps=bad_fps)
            assert False, "expected RallyDetectionError"
        except RallyDetectionError:
            pass


# --- timestamp conversion -----------------------------------------------


def test_timestamps_derived_from_sample_fps():
    signals = _signals("XXXXX")  # frames 0-4
    result = detect_rally_segments(signals, sample_fps=5.0, min_rally_duration_frames=1)

    rally = result[0]
    assert rally.start_time_s == 0.0
    assert rally.end_time_s == 1.0  # (4 + 1) / 5.0
    assert rally.duration_s == 1.0


def test_timestamps_at_a_different_sample_rate():
    signals = _signals("XXXXXXXXXX")  # 10 frames
    result = detect_rally_segments(signals, sample_fps=2.0, min_rally_duration_frames=1)

    rally = result[0]
    assert rally.start_time_s == 0.0
    assert rally.end_time_s == 5.0  # 10 frames / 2fps
    assert rally.duration_s == 5.0


# --- summary stats within a segment --------------------------------------


def test_ball_active_and_observed_counts_within_a_bridged_rally():
    signals = [
        _signal(0, True, observed=True, conf=0.9),
        _signal(1, False),
        _signal(2, True, observed=False, conf=0.3),  # interpolated
        _signal(3, True, observed=True, conf=0.7),
    ]
    result = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=2, min_rally_duration_frames=1)

    assert len(result) == 1
    rally = result[0]
    assert rally.frame_count == 4
    assert rally.ball_active_frame_count == 3, "frame 1 was bridged, not actually ball_present"
    assert rally.ball_observed_frame_count == 2
    assert rally.mean_ball_confidence == (0.9 + 0.3 + 0.7) / 3


# --- frames_from_ball_tracks_json ----------------------------------------


def test_frames_from_ball_tracks_json_marks_presence_and_observed_flags():
    data = {
        "frames": [
            {"frame_path": "f0.jpg", "tracks": [{"confidence": 0.8, "time_since_update": 0}]},
            {"frame_path": "f1.jpg", "tracks": []},
            {"frame_path": "f2.jpg", "tracks": [{"confidence": 0.4, "time_since_update": 2}]},
        ]
    }
    signals = frames_from_ball_tracks_json(data)

    assert len(signals) == 3
    assert signals[0].ball_present and signals[0].observed and signals[0].confidence == 0.8
    assert not signals[1].ball_present and not signals[1].observed and signals[1].confidence is None
    assert signals[2].ball_present and not signals[2].observed, "interpolated (time_since_update>0) is present but not observed"


def test_frames_from_ball_tracks_json_handles_missing_frames_key():
    assert frames_from_ball_tracks_json({}) == []


def test_frames_from_ball_tracks_json_frame_index_matches_list_position():
    data = {"frames": [{"frame_path": "a.jpg", "tracks": []}, {"frame_path": "b.jpg", "tracks": []}]}
    signals = frames_from_ball_tracks_json(data)
    assert [s.frame_index for s in signals] == [0, 1]


# --- serialization / summary ---------------------------------------------


def test_to_serializable_shape():
    signals = _signals("XXXXX")
    segments = detect_rally_segments(signals, sample_fps=5.0, min_rally_duration_frames=1)
    serialized = to_serializable(segments)

    assert serialized == [
        {
            "rally_index": 1,
            "start_frame": 0,
            "end_frame": 4,
            "start_time_s": 0.0,
            "end_time_s": 1.0,
            "duration_s": 1.0,
            "frame_count": 5,
            "ball_active_frame_count": 5,
            "ball_observed_frame_count": 5,
            "mean_ball_confidence": 0.5,
        }
    ]


def test_summarize_rally_segments_aggregates_across_rallies():
    signals = _signals("XXXXX...XXXXXXXXXX")  # 5-frame rally, 3-frame dead time, 10-frame rally
    segments = detect_rally_segments(signals, sample_fps=5.0, activity_gap_tolerance_frames=1, min_rally_duration_frames=1)
    summary = summarize_rally_segments(segments, total_frame_count=len(signals))

    assert summary["rally_count"] == 2
    assert summary["total_rally_frame_count"] == 15
    assert summary["longest_rally_duration_s"] == 2.0
    assert summary["shortest_rally_duration_s"] == 1.0
    assert summary["avg_rally_duration_s"] == 1.5
    assert summary["rally_frame_coverage_rate"] == 15 / len(signals)


def test_summarize_rally_segments_handles_no_rallies():
    assert summarize_rally_segments([], total_frame_count=10) == {
        "rally_count": 0,
        "total_rally_frame_count": 0,
        "total_rally_duration_s": 0.0,
        "avg_rally_duration_s": 0.0,
        "longest_rally_duration_s": 0.0,
        "shortest_rally_duration_s": 0.0,
        "rally_frame_coverage_rate": 0.0,
    }


def test_summarize_rally_segments_handles_zero_total_frames():
    assert summarize_rally_segments([], total_frame_count=0)["rally_frame_coverage_rate"] == 0.0
