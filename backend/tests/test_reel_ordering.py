"""
Tests for ml/pipeline/reel_ordering.py — Part 10b.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_reel_selection.py: builds ClipCandidate objects directly and
feeds them into order_clips_for_reel / build_reel_timeline.

Run with: pytest backend/tests/test_reel_ordering.py -v
"""

from __future__ import annotations

import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.reel_selection import ClipCandidate  # noqa: E402
from ml.pipeline.reel_ordering import (  # noqa: E402
    build_reel_timeline,
    order_clips_for_reel,
    summarize_timeline,
    to_serializable,
)


def _clip(
    highlight_id: str,
    *,
    start: float,
    end: float,
    importance_score: float,
    event_type: str = "long_rally",
) -> ClipCandidate:
    return ClipCandidate(
        highlight_id=highlight_id,
        event_type=event_type,
        start_time_s=start,
        end_time_s=end,
        importance_score=importance_score,
        clip_file_path=f"clips/{highlight_id}.mp4",
    )


# --- ordering ----------------------------------------------------------


def test_chronological_is_default_and_sorts_by_start_time():
    clips = [
        _clip("b", start=20, end=30, importance_score=0.9),
        _clip("a", start=0, end=10, importance_score=0.2),
        _clip("c", start=40, end=50, importance_score=0.5),
    ]
    ordered = order_clips_for_reel(clips)
    assert [c.highlight_id for c in ordered] == ["a", "b", "c"]


def test_chronological_ties_broken_by_original_order():
    clips = [
        _clip("first", start=10, end=15, importance_score=0.1),
        _clip("second", start=10, end=15, importance_score=0.9),
    ]
    ordered = order_clips_for_reel(clips, strategy="chronological")
    assert [c.highlight_id for c in ordered] == ["first", "second"]


def test_importance_strategy_sorts_best_first():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.2),
        _clip("b", start=20, end=30, importance_score=0.9),
        _clip("c", start=40, end=50, importance_score=0.5),
    ]
    ordered = order_clips_for_reel(clips, strategy="importance")
    assert [c.highlight_id for c in ordered] == ["b", "c", "a"]


def test_importance_ties_broken_by_original_order():
    clips = [
        _clip("first", start=0, end=10, importance_score=0.7),
        _clip("second", start=20, end=30, importance_score=0.7),
    ]
    ordered = order_clips_for_reel(clips, strategy="importance")
    assert [c.highlight_id for c in ordered] == ["first", "second"]


def test_unknown_strategy_raises():
    clips = [_clip("a", start=0, end=10, importance_score=0.5)]
    with pytest.raises(ValueError):
        order_clips_for_reel(clips, strategy="random")  # type: ignore[arg-type]


def test_ordering_does_not_select_or_drop_clips():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.2),
        _clip("b", start=20, end=30, importance_score=0.9),
    ]
    ordered = order_clips_for_reel(clips)
    assert {c.highlight_id for c in ordered} == {"a", "b"}


# --- timeline / pacing ---------------------------------------------------


def test_timeline_no_gap_by_default():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.5),
        _clip("b", start=20, end=25, importance_score=0.5),
    ]
    entries = build_reel_timeline(clips)
    assert entries[0].gap_before_s == 0.0
    assert entries[0].reel_start_s == 0.0
    assert entries[0].reel_end_s == 10.0
    assert entries[1].gap_before_s == 0.0
    assert entries[1].reel_start_s == 10.0
    assert entries[1].reel_end_s == 15.0


def test_timeline_reserves_gap_between_clips_only():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.5),  # 10s clip
        _clip("b", start=20, end=25, importance_score=0.5),  # 5s clip
        _clip("c", start=40, end=48, importance_score=0.5),  # 8s clip
    ]
    entries = build_reel_timeline(clips, transition_gap_s=2.0)

    # No gap reserved before the first clip.
    assert entries[0].gap_before_s == 0.0
    assert entries[0].reel_start_s == 0.0
    assert entries[0].reel_end_s == 10.0

    # 2s gap reserved before each subsequent clip.
    assert entries[1].gap_before_s == 2.0
    assert entries[1].reel_start_s == 12.0
    assert entries[1].reel_end_s == 17.0

    assert entries[2].gap_before_s == 2.0
    assert entries[2].reel_start_s == 19.0
    assert entries[2].reel_end_s == 27.0


def test_timeline_negative_gap_clamped_to_zero():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.5),
        _clip("b", start=20, end=25, importance_score=0.5),
    ]
    entries = build_reel_timeline(clips, transition_gap_s=-3.0)
    assert entries[1].gap_before_s == 0.0
    assert entries[1].reel_start_s == 10.0


def test_timeline_positions_are_sequential_from_zero():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.5),
        _clip("b", start=20, end=25, importance_score=0.5),
        _clip("c", start=40, end=48, importance_score=0.5),
    ]
    entries = build_reel_timeline(clips)
    assert [e.position for e in entries] == [0, 1, 2]


def test_timeline_empty_input_returns_empty():
    assert build_reel_timeline([]) == []


def test_timeline_preserves_source_in_out_points():
    clips = [_clip("a", start=12.5, end=20.0, importance_score=0.5)]
    entries = build_reel_timeline(clips)
    assert entries[0].source_start_s == 12.5
    assert entries[0].source_end_s == 20.0
    assert entries[0].clip_duration_s == 7.5


def test_to_serializable_round_trip_fields():
    clips = [_clip("a", start=0, end=10, importance_score=0.5)]
    entries = build_reel_timeline(clips, transition_gap_s=1.0)
    serialized = to_serializable(entries)
    assert serialized[0]["highlight_id"] == "a"
    assert serialized[0]["position"] == 0
    assert serialized[0]["reel_start_s"] == 0.0
    assert serialized[0]["reel_end_s"] == 10.0


def test_summarize_timeline_totals():
    clips = [
        _clip("a", start=0, end=10, importance_score=0.5),
        _clip("b", start=20, end=25, importance_score=0.5),
    ]
    entries = build_reel_timeline(clips, transition_gap_s=2.0)
    summary = summarize_timeline(entries)
    assert summary["clip_count"] == 2
    assert summary["total_clip_duration_s"] == 15.0
    assert summary["total_gap_duration_s"] == 2.0
    assert summary["total_reel_duration_s"] == 17.0


def test_summarize_timeline_empty():
    summary = summarize_timeline([])
    assert summary == {
        "clip_count": 0,
        "total_clip_duration_s": 0.0,
        "total_gap_duration_s": 0.0,
        "total_reel_duration_s": 0.0,
    }


# --- end-to-end: selection (10a) -> ordering/pacing (10b) ----------------


def test_selection_then_ordering_pipeline():
    from ml.pipeline.reel_selection import select_top_n

    clips = [
        _clip("a", start=0, end=10, importance_score=0.9),
        _clip("b", start=20, end=30, importance_score=0.2),
        _clip("c", start=40, end=48, importance_score=0.8),
        _clip("d", start=60, end=65, importance_score=0.1),
    ]
    selected = select_top_n(clips, 3)  # drops "d", stays chronological
    ordered = order_clips_for_reel(selected, strategy="chronological")
    entries = build_reel_timeline(ordered, transition_gap_s=1.0)

    assert [e.highlight_id for e in entries] == ["a", "b", "c"]
    assert entries[-1].reel_end_s == pytest.approx(10.0 + 1.0 + 10.0 + 1.0 + 8.0)
