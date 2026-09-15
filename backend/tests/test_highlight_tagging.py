"""
Tests for ml/pipeline/highlight_tagging.py — Part 7e.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_rally_detection.py / test_shot_classification.py: builds
RallySegment/Shot objects directly and feeds them straight into
detect_highlights and the individual tag_* functions.

Run with: pytest backend/tests/test_highlight_tagging.py -v
"""

from __future__ import annotations

import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.rally_detection import RallySegment  # noqa: E402
from ml.pipeline.shot_classification import (  # noqa: E402
    SHOT_TYPE_GROUNDSTROKE,
    SHOT_TYPE_SMASH,
    Shot,
)
from ml.pipeline.highlight_tagging import (  # noqa: E402
    HIGHLIGHT_TYPE_FAST_EXCHANGE,
    HIGHLIGHT_TYPE_LONG_RALLY,
    HIGHLIGHT_TYPE_POWERFUL_SMASH,
    HIGHLIGHT_TYPE_SCORE_WEIGHT,
    HIGHLIGHT_TYPE_SPECTACULAR_SAVE,
    detect_highlights,
    summarize_highlights,
    tag_fast_exchanges,
    tag_long_rally,
    tag_powerful_smash,
    tag_spectacular_saves,
    to_serializable,
)

SAMPLE_FPS = 5.0


def _rally(rally_index: int, duration_s: float, *, start_frame: int = 0) -> RallySegment:
    frame_count = int(duration_s * SAMPLE_FPS)
    end_frame = start_frame + frame_count - 1
    return RallySegment(
        rally_index=rally_index, start_frame=start_frame, end_frame=end_frame,
        start_time_s=start_frame / SAMPLE_FPS, end_time_s=(end_frame + 1) / SAMPLE_FPS,
        duration_s=duration_s, frame_count=frame_count,
        ball_active_frame_count=frame_count, ball_observed_frame_count=frame_count,
        mean_ball_confidence=0.6,
    )


def _shot(
    frame_index: int,
    *,
    rally_index: int = 1,
    shot_type: str = SHOT_TYPE_GROUNDSTROKE,
    player_track_id: int | None = 1,
    contact_height_ratio: float | None = None,
    airborne_frames_after: int | None = 2,
) -> Shot:
    return Shot(
        rally_index=rally_index, frame_index=frame_index, player_track_id=player_track_id,
        shot_type=shot_type, contact_height_ratio=contact_height_ratio,
        airborne_frames_after=airborne_frames_after, distance_from_net_m=None,
        used_court_calibration=False,
    )


# --- long rally ---------------------------------------------------------


def test_rally_below_threshold_is_not_tagged():
    rally = _rally(1, duration_s=10.0)
    assert tag_long_rally(rally, min_duration_s=15.0) is None


def test_rally_at_or_above_threshold_is_tagged():
    rally = _rally(1, duration_s=20.0)
    event = tag_long_rally(rally, min_duration_s=15.0, score_saturation_s=35.0)

    assert event is not None
    assert event.highlight_type == HIGHLIGHT_TYPE_LONG_RALLY
    assert event.start_time_s == rally.start_time_s
    assert event.end_time_s == rally.end_time_s
    assert 0.0 < event.importance_score < 1.0


def test_long_rally_score_saturates_at_ceiling():
    rally = _rally(1, duration_s=50.0)
    event = tag_long_rally(rally, min_duration_s=15.0, score_saturation_s=35.0)
    assert event.importance_score == 1.0


def test_long_rally_score_is_near_zero_right_at_the_floor():
    rally = _rally(1, duration_s=15.0)
    event = tag_long_rally(rally, min_duration_s=15.0, score_saturation_s=35.0)
    assert event.importance_score == 0.0


# --- fast exchange -------------------------------------------------------


def test_short_run_below_min_count_is_not_tagged():
    rally = _rally(1, duration_s=5.0)
    shots = [_shot(i) for i in (0, 1, 2)]  # 3 shots, min is 4
    events = tag_fast_exchanges(rally, shots, sample_fps=SAMPLE_FPS, max_interval_s=1.0, min_shot_count=4)
    assert events == []


def test_run_of_close_shots_is_tagged_fast_exchange():
    rally = _rally(1, duration_s=5.0)
    # frames 0,2,4,6,8 at 5fps -> 0.4s apart, well under 1.0s max_interval_s
    shots = [_shot(i) for i in (0, 2, 4, 6, 8)]
    events = tag_fast_exchanges(rally, shots, sample_fps=SAMPLE_FPS, max_interval_s=1.0, min_shot_count=4)

    assert len(events) == 1
    event = events[0]
    assert event.highlight_type == HIGHLIGHT_TYPE_FAST_EXCHANGE
    assert event.start_time_s == 0.0
    assert event.end_time_s == 8 / SAMPLE_FPS


def test_a_slow_gap_splits_one_run_into_two():
    rally = _rally(1, duration_s=10.0)
    # 0,2,4,6 close together, then a big gap (frame 30), then 32,34,36,38 close together
    shots = [_shot(i) for i in (0, 2, 4, 6, 30, 32, 34, 36, 38)]
    events = tag_fast_exchanges(rally, shots, sample_fps=SAMPLE_FPS, max_interval_s=1.0, min_shot_count=4)

    assert len(events) == 2
    assert events[0].start_time_s == 0.0
    assert events[0].end_time_s == 6 / SAMPLE_FPS
    assert events[1].start_time_s == 30 / SAMPLE_FPS
    assert events[1].end_time_s == 38 / SAMPLE_FPS


def test_fast_exchange_score_saturates_with_more_shots():
    rally = _rally(1, duration_s=10.0)
    short_run = [_shot(i) for i in (0, 2, 4, 6)]  # exactly min_shot_count
    long_run = [_shot(i) for i in range(0, 32, 2)]  # 16 shots, well past saturation

    short_events = tag_fast_exchanges(
        rally, short_run, sample_fps=SAMPLE_FPS, max_interval_s=1.0, min_shot_count=4, score_saturation_count=8
    )
    long_events = tag_fast_exchanges(
        rally, long_run, sample_fps=SAMPLE_FPS, max_interval_s=1.0, min_shot_count=4, score_saturation_count=8
    )

    assert short_events[0].importance_score == 0.0
    assert long_events[0].importance_score == 1.0


# --- powerful smash --------------------------------------------------------


def test_non_smash_shot_is_not_tagged():
    shot = _shot(10, shot_type=SHOT_TYPE_GROUNDSTROKE)
    assert tag_powerful_smash(shot, rally_index=1, sample_fps=SAMPLE_FPS, smash_height_ratio=1.05) is None


def test_smash_at_threshold_scores_at_the_floor():
    shot = _shot(10, shot_type=SHOT_TYPE_SMASH, contact_height_ratio=1.05, airborne_frames_after=3)
    event = tag_powerful_smash(
        shot, rally_index=1, sample_fps=SAMPLE_FPS, smash_height_ratio=1.05, score_ceiling_ratio=1.6
    )

    assert event is not None
    assert event.highlight_type == HIGHLIGHT_TYPE_POWERFUL_SMASH
    # Tier 2: the floor is 0.0 now, the same convention every other tag_*
    # function's own "just barely qualifies" case uses — see
    # test_long_rally_score_is_near_zero_right_at_the_floor and
    # test_fast_exchange_score_saturates_with_more_shots (its short_events
    # assertion) below for the equivalent case on the other two types.
    assert event.importance_score == 0.0
    assert event.source_frame_index == 10
    assert event.start_time_s == 10 / SAMPLE_FPS
    assert event.end_time_s == 13 / SAMPLE_FPS


def test_smash_at_or_above_ceiling_ratio_saturates_to_one():
    shot = _shot(10, shot_type=SHOT_TYPE_SMASH, contact_height_ratio=1.8)
    event = tag_powerful_smash(
        shot, rally_index=1, sample_fps=SAMPLE_FPS, smash_height_ratio=1.05, score_ceiling_ratio=1.6
    )
    assert event.importance_score == 1.0


def test_smash_with_no_height_ratio_falls_back_to_a_neutral_score():
    # Tier 2: 0.5 here is a deliberately different thing than "the floor"
    # (the floor is 0.0 now) — it's a neutral middle value for a case
    # where contact_height_ratio is unexpectedly missing, unrelated to
    # where the pre-Tier-2 floor used to sit at the same number.
    shot = _shot(10, shot_type=SHOT_TYPE_SMASH, contact_height_ratio=None)
    event = tag_powerful_smash(shot, rally_index=1, sample_fps=SAMPLE_FPS, smash_height_ratio=1.05)
    assert event.importance_score == 0.5


# --- spectacular save --------------------------------------------------------


def test_quick_return_by_a_different_player_is_a_save():
    rally = _rally(1, duration_s=5.0)
    smash = _shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=1)
    save = _shot(12, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2)  # 0.4s later
    events = tag_spectacular_saves(rally, [smash, save], sample_fps=SAMPLE_FPS, max_response_s=0.8)

    assert len(events) == 1
    event = events[0]
    assert event.highlight_type == HIGHLIGHT_TYPE_SPECTACULAR_SAVE
    assert event.source_frame_index == 12
    # 0.4s response out of a 0.8s max -> exactly halfway between the
    # floor and the ceiling under Tier 2's linear scale. pytest.approx
    # since (max_response_s - response_s) / max_response_s doesn't land
    # on an exact float 0.5 here (0.8 - 0.4 != 0.4 in binary floating
    # point) -- a precision artifact of the division, not a logic bug.
    assert event.importance_score == pytest.approx(0.5)


def test_instant_save_saturates_to_one():
    rally = _rally(1, duration_s=5.0)
    smash = _shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=1)
    save = _shot(10, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2)  # same frame -> 0.0s response
    events = tag_spectacular_saves(rally, [smash, save], sample_fps=SAMPLE_FPS, max_response_s=0.8)
    assert events[0].importance_score == 1.0


def test_save_at_max_response_scores_at_the_floor():
    # Tier 2: response_s == max_response_s is the slowest response that
    # still counts as a save at all -- same "0.0 at the qualifying
    # threshold" convention every other tag_* function now uses. 0.8s at
    # SAMPLE_FPS=5.0 is exactly 4 frames later. pytest.approx: computing
    # response_s via frame_index/sample_fps subtraction doesn't land on
    # an exact float 0.8 (binary floating point can't represent 0.8
    # exactly), so the resulting score is a hair above 0.0, not exactly
    # 0.0 -- a precision artifact, not a boundary-exclusion bug (confirmed
    # separately: response_s still compares as <= max_response_s, so the
    # event isn't dropped at this boundary).
    rally = _rally(1, duration_s=5.0)
    smash = _shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=1)
    save = _shot(14, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2)  # 0.8s later
    events = tag_spectacular_saves(rally, [smash, save], sample_fps=SAMPLE_FPS, max_response_s=0.8)
    assert len(events) == 1
    assert events[0].importance_score == pytest.approx(0.0, abs=1e-9)


def test_slow_response_is_not_a_save():
    rally = _rally(1, duration_s=5.0)
    smash = _shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=1)
    late_return = _shot(20, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2)  # 2.0s later
    events = tag_spectacular_saves(rally, [smash, late_return], sample_fps=SAMPLE_FPS, max_response_s=0.8)
    assert events == []


def test_same_player_returning_their_own_smash_is_not_a_save():
    rally = _rally(1, duration_s=5.0)
    smash = _shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=1)
    same_player_shot = _shot(12, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=1)
    events = tag_spectacular_saves(rally, [smash, same_player_shot], sample_fps=SAMPLE_FPS, max_response_s=0.8)
    assert events == []


def test_unresolved_player_is_not_a_save():
    rally = _rally(1, duration_s=5.0)
    smash = _shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=None)
    save = _shot(12, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2)
    events = tag_spectacular_saves(rally, [smash, save], sample_fps=SAMPLE_FPS, max_response_s=0.8)
    assert events == []


def test_non_smash_shot_before_a_return_is_not_a_save():
    rally = _rally(1, duration_s=5.0)
    groundstroke = _shot(10, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=1)
    return_shot = _shot(12, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2)
    events = tag_spectacular_saves(rally, [groundstroke, return_shot], sample_fps=SAMPLE_FPS, max_response_s=0.8)
    assert events == []


# --- cross-type score calibration (Highlights Improvement Roadmap Tier 2) ---


def test_every_type_scores_zero_at_its_own_barely_qualifying_threshold():
    """
    The whole point of Tier 2, as one assertion: construct the least
    impressive still-qualifying event of each type and confirm every one
    of them scores exactly 0.0, not four different numbers. Before Tier
    2, a barely-qualifying smash scored 0.5 and a barely-qualifying save
    scored 0.6 here -- this test would have failed against the old
    formulas, which is exactly the bug this tier fixed.
    """
    long_rally_event = tag_long_rally(_rally(1, duration_s=15.0), min_duration_s=15.0, score_saturation_s=35.0)
    fast_exchange_events = tag_fast_exchanges(
        _rally(2, duration_s=5.0), [_shot(i) for i in (0, 2, 4, 6)],
        sample_fps=SAMPLE_FPS, max_interval_s=1.0, min_shot_count=4, score_saturation_count=8,
    )
    smash_event = tag_powerful_smash(
        _shot(10, shot_type=SHOT_TYPE_SMASH, contact_height_ratio=1.05),
        rally_index=3, sample_fps=SAMPLE_FPS, smash_height_ratio=1.05, score_ceiling_ratio=1.6,
    )
    save_events = tag_spectacular_saves(
        _rally(4, duration_s=5.0),
        [_shot(10, shot_type=SHOT_TYPE_SMASH, player_track_id=1), _shot(14, player_track_id=2)],
        sample_fps=SAMPLE_FPS, max_response_s=0.8,
    )

    assert long_rally_event.importance_score == 0.0
    assert fast_exchange_events[0].importance_score == 0.0
    assert smash_event.importance_score == 0.0
    # Same float-precision note as test_save_at_max_response_scores_at_the_floor.
    assert save_events[0].importance_score == pytest.approx(0.0, abs=1e-9)


def test_score_weights_default_to_a_no_op():
    """
    HIGHLIGHT_TYPE_SCORE_WEIGHT is the extension point for real,
    footage-informed cross-type weighting -- but not yet. This just
    guards against someone bumping a weight in a drive-by edit without
    the real footage review that's supposed to justify it (see the
    constant's own comment): every weight must stay 1.0 until that
    review actually happens.
    """
    assert all(weight == 1.0 for weight in HIGHLIGHT_TYPE_SCORE_WEIGHT.values())




def test_detect_highlights_combines_every_rule_across_rallies():
    long_rally = _rally(1, duration_s=20.0, start_frame=0)
    short_rally = _rally(2, duration_s=3.0, start_frame=200)

    shots = [
        # A powerful smash immediately saved by a different player, inside the long rally.
        _shot(10, rally_index=1, shot_type=SHOT_TYPE_SMASH, player_track_id=1, contact_height_ratio=1.3),
        _shot(12, rally_index=1, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2),
        # A fast exchange run inside the long rally.
        _shot(50, rally_index=1, player_track_id=1),
        _shot(52, rally_index=1, player_track_id=2),
        _shot(54, rally_index=1, player_track_id=1),
        _shot(56, rally_index=1, player_track_id=2),
        # A single ordinary shot in the short rally -- nothing should fire here.
        _shot(201, rally_index=2, player_track_id=1),
    ]

    events = detect_highlights(
        [long_rally, short_rally], shots,
        sample_fps=SAMPLE_FPS,
        long_rally_min_duration_s=15.0,
        fast_exchange_max_interval_s=1.0,
        fast_exchange_min_shot_count=4,
        smash_height_ratio=1.05,
    )

    types = {e.highlight_type for e in events}
    assert HIGHLIGHT_TYPE_LONG_RALLY in types
    assert HIGHLIGHT_TYPE_POWERFUL_SMASH in types
    assert HIGHLIGHT_TYPE_SPECTACULAR_SAVE in types
    assert HIGHLIGHT_TYPE_FAST_EXCHANGE in types
    assert all(e.rally_index in (1, 2) for e in events)
    # Nothing in the short rally clears any threshold.
    assert all(e.rally_index != 2 for e in events)


def test_detect_highlights_with_no_rallies_or_shots_is_empty():
    assert detect_highlights([], [], sample_fps=SAMPLE_FPS, smash_height_ratio=1.05) == []


def test_shots_missing_from_a_rally_are_simply_skipped():
    rally = _rally(1, duration_s=3.0)  # too short for long_rally
    events = detect_highlights([rally], [], sample_fps=SAMPLE_FPS, smash_height_ratio=1.05)
    assert events == []


# --- serialization / summary -------------------------------------------------


def test_to_serializable_round_trips_every_field():
    rally = _rally(1, duration_s=20.0)
    event = tag_long_rally(rally, min_duration_s=15.0)
    data = to_serializable([event])[0]

    assert data["rally_index"] == event.rally_index
    assert data["highlight_type"] == HIGHLIGHT_TYPE_LONG_RALLY
    assert data["start_time_s"] == event.start_time_s
    assert data["end_time_s"] == event.end_time_s
    assert data["importance_score"] == event.importance_score
    assert data["reason"] == event.reason
    assert data["source_frame_index"] is None


def test_summarize_highlights_counts_by_type_and_averages_score():
    rally = _rally(1, duration_s=20.0)
    event_a = tag_long_rally(rally, min_duration_s=15.0, score_saturation_s=35.0)
    event_b = tag_long_rally(_rally(2, duration_s=35.0), min_duration_s=15.0, score_saturation_s=35.0)

    summary = summarize_highlights([event_a, event_b])

    assert summary["highlight_count"] == 2
    assert summary["highlight_counts_by_type"][HIGHLIGHT_TYPE_LONG_RALLY] == 2
    assert summary["highlight_counts_by_type"][HIGHLIGHT_TYPE_FAST_EXCHANGE] == 0
    assert summary["avg_importance_score"] == (event_a.importance_score + event_b.importance_score) / 2


def test_summarize_highlights_with_no_events():
    summary = summarize_highlights([])
    assert summary["highlight_count"] == 0
    assert summary["avg_importance_score"] == 0.0
