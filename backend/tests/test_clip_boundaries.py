"""
Tests for ml/pipeline/clip_boundaries.py — Part 8a.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_highlight_tagging.py: builds HighlightEvent objects directly and
feeds them straight into compute_clip_boundary(ies).

Run with: pytest backend/tests/test_clip_boundaries.py -v
"""

from __future__ import annotations

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.highlight_tagging import HighlightEvent  # noqa: E402
from ml.pipeline.clip_boundaries import (  # noqa: E402
    compute_clip_boundaries,
    compute_clip_boundary,
    summarize_clip_boundaries,
    to_serializable,
)

PRE = 3.0
POST = 2.0
MIN_DURATION = 5.0


def _event(
    start_time_s: float,
    end_time_s: float,
    *,
    rally_index: int = 1,
    highlight_type: str = "long_rally",
    importance_score: float = 0.5,
    source_frame_index: int | None = None,
) -> HighlightEvent:
    return HighlightEvent(
        rally_index=rally_index,
        highlight_type=highlight_type,
        start_time_s=start_time_s,
        end_time_s=end_time_s,
        importance_score=importance_score,
        reason="test",
        source_frame_index=source_frame_index,
    )


# --- plain padding, no clamping ------------------------------------------


def test_pads_before_and_after_when_there_is_room():
    event = _event(20.0, 30.0)
    boundary = compute_clip_boundary(
        event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.start_time_s == 17.0
    assert boundary.end_time_s == 32.0


def test_tight_event_boundaries_are_preserved_alongside_padded_ones():
    event = _event(20.0, 30.0)
    boundary = compute_clip_boundary(
        event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.event_start_time_s == 20.0
    assert boundary.event_end_time_s == 30.0


def test_carries_through_identifying_fields_from_the_event():
    event = _event(20.0, 30.0, rally_index=7, highlight_type="powerful_smash", importance_score=0.9, source_frame_index=42)
    boundary = compute_clip_boundary(event, video_duration_s=100.0)
    assert boundary.rally_index == 7
    assert boundary.highlight_type == "powerful_smash"
    assert boundary.importance_score == 0.9
    assert boundary.source_frame_index == 42


# --- clamping against the video boundary ---------------------------------


def test_clamps_start_to_zero_near_beginning_of_video():
    event = _event(1.0, 5.0)
    boundary = compute_clip_boundary(
        event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.start_time_s == 0.0


def test_clamps_end_to_video_duration_near_the_end():
    event = _event(95.0, 99.0)
    boundary = compute_clip_boundary(
        event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.end_time_s == 100.0


# --- minimum duration top-up ----------------------------------------------


def test_instantaneous_event_still_gets_full_pre_and_post_roll():
    # A zero-duration event (e.g. a smash with no airborne frames after
    # contact) with room on both sides should just get plain padding —
    # the min_duration_s floor shouldn't need to do anything extra here.
    event = _event(50.0, 50.0)
    boundary = compute_clip_boundary(
        event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.start_time_s == 47.0
    assert boundary.end_time_s == 52.0
    assert (boundary.end_time_s - boundary.start_time_s) == MIN_DURATION


def test_min_duration_pulls_extra_room_from_the_open_side_when_clamped():
    # Event starts essentially at t=0, so there's no room to pad before
    # it at all; the shortfall should come entirely from extra post-roll
    # instead of leaving the clip shorter than min_duration_s.
    event = _event(0.0, 0.5)
    boundary = compute_clip_boundary(
        event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.start_time_s == 0.0
    assert (boundary.end_time_s - boundary.start_time_s) >= MIN_DURATION


def test_video_shorter_than_min_duration_returns_the_full_video_span_without_raising():
    event = _event(1.0, 2.0)
    boundary = compute_clip_boundary(
        event, video_duration_s=4.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION
    )
    assert boundary.start_time_s == 0.0
    assert boundary.end_time_s == 4.0


# --- batch behavior --------------------------------------------------------


def test_compute_clip_boundaries_preserves_order_and_count():
    events = [_event(10.0, 12.0, rally_index=1), _event(40.0, 41.0, rally_index=2), _event(80.0, 85.0, rally_index=3)]
    boundaries = compute_clip_boundaries(events, video_duration_s=100.0)
    assert [b.rally_index for b in boundaries] == [1, 2, 3]


def test_empty_events_list_produces_empty_boundaries():
    assert compute_clip_boundaries([], video_duration_s=100.0) == []


# --- serialization / summary -----------------------------------------------


def test_to_serializable_round_trips_expected_keys():
    boundary = compute_clip_boundary(_event(20.0, 30.0), video_duration_s=100.0)
    [serialized] = to_serializable([boundary])
    assert serialized["start_time_s"] == boundary.start_time_s
    assert serialized["end_time_s"] == boundary.end_time_s
    assert serialized["event_start_time_s"] == boundary.event_start_time_s
    assert serialized["highlight_type"] == boundary.highlight_type


def test_summarize_clip_boundaries_on_empty_list():
    summary = summarize_clip_boundaries([])
    assert summary == {"clip_count": 0, "total_clip_duration_s": 0.0, "avg_clip_duration_s": 0.0}


# --- per-HighlightType padding (Part 8a's "1c" follow-up) -----------------


def test_powerful_smash_gets_its_own_default_padding_when_no_override_given():
    # No pre_roll_s/post_roll_s passed -> resolved from
    # CLIP_PADDING_BY_HIGHLIGHT_TYPE's powerful_smash entry (2.0, 3.0),
    # not the module's shared (3.0, 2.0) default.
    event = _event(20.0, 30.0, highlight_type="powerful_smash")
    boundary = compute_clip_boundary(event, video_duration_s=100.0)
    assert boundary.start_time_s == 18.0
    assert boundary.end_time_s == 33.0


def test_long_rally_gets_its_own_default_padding_when_no_override_given():
    # long_rally's entry is (4.0, 2.0) -> longer lead-in than the shared default.
    event = _event(20.0, 30.0, highlight_type="long_rally")
    boundary = compute_clip_boundary(event, video_duration_s=100.0)
    assert boundary.start_time_s == 16.0
    assert boundary.end_time_s == 32.0


def test_unmapped_highlight_type_falls_back_to_shared_default():
    # fast_exchange has no entry in CLIP_PADDING_BY_HIGHLIGHT_TYPE -> falls
    # back to (DEFAULT_CLIP_PRE_ROLL_S, DEFAULT_CLIP_POST_ROLL_S) = (3.0, 2.0).
    event = _event(20.0, 30.0, highlight_type="fast_exchange")
    boundary = compute_clip_boundary(event, video_duration_s=100.0)
    assert boundary.start_time_s == 17.0
    assert boundary.end_time_s == 32.0


def test_explicit_pre_roll_and_post_roll_still_override_the_per_type_default():
    # Passing pre_roll_s/post_roll_s explicitly still applies uniformly,
    # same as this function's old always-uniform behavior, even for a
    # highlight_type that has its own entry in the per-type table.
    event = _event(20.0, 30.0, highlight_type="powerful_smash")
    boundary = compute_clip_boundary(event, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST)
    assert boundary.start_time_s == 17.0
    assert boundary.end_time_s == 32.0


def test_min_duration_derives_from_the_resolved_per_type_padding():
    # An instantaneous powerful_smash with room on both sides should land
    # exactly at its own type's pre+post sum (2.0 + 3.0 = 5.0), not the
    # module's shared default sum.
    event = _event(50.0, 50.0, highlight_type="powerful_smash")
    boundary = compute_clip_boundary(event, video_duration_s=100.0)
    assert boundary.start_time_s == 48.0
    assert boundary.end_time_s == 53.0


def test_compute_clip_boundaries_resolves_each_event_by_its_own_type():
    # A batch mixing types gets each event its own type-appropriate
    # padding, not one padding applied across the whole batch.
    events = [
        _event(20.0, 30.0, rally_index=1, highlight_type="long_rally"),
        _event(60.0, 60.0, rally_index=2, highlight_type="powerful_smash"),
    ]
    boundaries = {b.rally_index: b for b in compute_clip_boundaries(events, video_duration_s=100.0)}
    assert boundaries[1].start_time_s == 16.0  # long_rally: 4.0s pre-roll
    assert boundaries[2].start_time_s == 58.0  # powerful_smash: 2.0s pre-roll


def test_summarize_clip_boundaries_computes_averages():
    events = [_event(0.0, 5.0), _event(20.0, 20.0)]
    boundaries = compute_clip_boundaries(events, video_duration_s=100.0, pre_roll_s=PRE, post_roll_s=POST, min_duration_s=MIN_DURATION)
    summary = summarize_clip_boundaries(boundaries)
    assert summary["clip_count"] == 2
    expected_total = sum(b.end_time_s - b.start_time_s for b in boundaries)
    assert summary["total_clip_duration_s"] == expected_total
    assert summary["avg_clip_duration_s"] == expected_total / 2
