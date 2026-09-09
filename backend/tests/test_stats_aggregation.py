"""
Tests for ml/pipeline/stats_aggregation.py — Part 9a.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_rally_detection.py / test_highlight_tagging.py: builds
RallySegment/Shot/PointOutcome/TrackPoint objects directly and feeds them
straight into the compute_* functions.

Run with: pytest backend/tests/test_stats_aggregation.py -v
"""

from __future__ import annotations

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.point_outcome import (  # noqa: E402
    OUTCOME_IN_BOUNDS_END,
    OUTCOME_NET,
    OUTCOME_OUT_OF_BOUNDS,
    OUTCOME_UNDETERMINED,
    PointOutcome,
)
from ml.pipeline.rally_detection import RallySegment  # noqa: E402
from ml.pipeline.serve_detection import ServeEvent, TrackPoint  # noqa: E402
from ml.pipeline.shot_classification import (  # noqa: E402
    SHOT_TYPE_GROUNDSTROKE,
    SHOT_TYPE_SMASH,
    SHOT_TYPE_VOLLEY,
    Shot,
)
from ml.pipeline.stats_aggregation import (  # noqa: E402
    STAT_ERRORS,
    STAT_LONGEST_RALLY,
    STAT_RALLY_LENGTH_AVG,
    STAT_TOTAL_POINTS,
    STAT_TYPE_STATUS,
    STATUS_COMPUTABLE_MATCH_LEVEL,
    STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY,
    STATUS_NOT_COMPUTABLE,
    compute_distance_and_speed,
    compute_match_level_stats,
    compute_rally_ending_shot_success_rate,
    compute_reaction_times,
    player_level_stat_values,
    summarize_match_diagnostics,
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
) -> Shot:
    return Shot(
        rally_index=rally_index, frame_index=frame_index, player_track_id=player_track_id,
        shot_type=shot_type, contact_height_ratio=None, airborne_frames_after=2,
        distance_from_net_m=None, used_court_calibration=False,
    )


def _outcome(rally_index: int, outcome: str, *, used_court_calibration: bool = True) -> PointOutcome:
    return PointOutcome(
        rally_index=rally_index, outcome=outcome, reason="test",
        last_ball_frame_index=None, last_ball_x_m=None, last_ball_y_m=None,
        speed_ratio=None, used_court_calibration=used_court_calibration,
    )


# --- STAT_TYPE_STATUS inventory ------------------------------------------


def test_stat_type_status_covers_every_real_stat_type_exactly_once():
    from app.models.enums import StatType

    assert set(STAT_TYPE_STATUS.keys()) == {s.value for s in StatType}


def test_stat_type_status_only_uses_known_statuses():
    valid = {STATUS_COMPUTABLE_MATCH_LEVEL, STATUS_COMPUTABLE_PENDING_PLAYER_IDENTITY, STATUS_NOT_COMPUTABLE}
    for entry in STAT_TYPE_STATUS.values():
        assert entry["status"] in valid
        assert entry["reason"]


def test_winners_and_momentum_and_serve_percentage_are_not_computable():
    assert STAT_TYPE_STATUS["winners"]["status"] == STATUS_NOT_COMPUTABLE
    assert STAT_TYPE_STATUS["momentum_possession"]["status"] == STATUS_NOT_COMPUTABLE
    assert STAT_TYPE_STATUS["serve_percentage"]["status"] == STATUS_NOT_COMPUTABLE


# --- match-level stats -----------------------------------------------------


def test_match_level_stats_basic_counts():
    rallies = [_rally(1, 10.0), _rally(2, 20.0), _rally(3, 5.0)]
    outcomes = [
        _outcome(1, OUTCOME_IN_BOUNDS_END),
        _outcome(2, OUTCOME_OUT_OF_BOUNDS),
        _outcome(3, OUTCOME_NET),
    ]

    values = compute_match_level_stats(rallies, outcomes)
    by_key = {v.stat_key: v for v in values}

    assert by_key[STAT_TOTAL_POINTS].value == 3
    assert by_key[STAT_RALLY_LENGTH_AVG].value == (10.0 + 20.0 + 5.0) / 3
    assert by_key[STAT_LONGEST_RALLY].value == 20.0
    # errors = out_of_bounds + net, NOT in_bounds_end (winner/UE ambiguous)
    assert by_key[STAT_ERRORS].value == 2
    assert by_key[STAT_ERRORS].sample_size == 3


def test_errors_ignores_undetermined_and_in_bounds_end():
    rallies = [_rally(1, 10.0)]
    outcomes = [_outcome(1, OUTCOME_UNDETERMINED, used_court_calibration=False)]

    values = compute_match_level_stats(rallies, outcomes)
    errors = next(v for v in values if v.stat_key == STAT_ERRORS)
    assert errors.value == 0
    assert errors.sample_size == 1


def test_no_rallies_yields_only_total_points_and_errors():
    values = compute_match_level_stats([], [])
    keys = {v.stat_key for v in values}
    assert keys == {STAT_TOTAL_POINTS, STAT_ERRORS}
    assert next(v for v in values if v.stat_key == STAT_TOTAL_POINTS).value == 0


# --- reaction time -----------------------------------------------------------


def test_reaction_time_credits_the_responding_player():
    rallies = [_rally(1, 4.0)]
    # player 1 hits at frame 0, player 2 responds at frame 2 (0.4s later)
    shots = [
        _shot(0, player_track_id=1),
        _shot(2, player_track_id=2),
    ]

    results = compute_reaction_times(rallies, shots, sample_fps=SAMPLE_FPS)

    assert 1 not in results  # player 1 never responded to anyone
    assert results[2].response_count == 1
    assert results[2].avg_reaction_time_s == 2 / SAMPLE_FPS


def test_reaction_time_ignores_same_player_consecutive_shots():
    rallies = [_rally(1, 4.0)]
    shots = [_shot(0, player_track_id=1), _shot(2, player_track_id=1)]

    results = compute_reaction_times(rallies, shots, sample_fps=SAMPLE_FPS)
    assert results == {}


def test_reaction_time_skips_unresolved_player_shots():
    rallies = [_rally(1, 4.0)]
    shots = [_shot(0, player_track_id=None), _shot(2, player_track_id=2)]

    results = compute_reaction_times(rallies, shots, sample_fps=SAMPLE_FPS)
    assert results == {}


# --- rally-ending shot success rate -------------------------------------------


def test_smash_success_rate_only_counts_rally_ending_smashes():
    rallies = [_rally(1, 4.0), _rally(2, 4.0)]
    shots = [
        # rally 1: smash is the last shot, rally ended out of bounds -> failure
        _shot(0, rally_index=1, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=1),
        _shot(2, rally_index=1, shot_type=SHOT_TYPE_SMASH, player_track_id=1),
        # rally 2: smash is NOT the last shot -> doesn't count at all
        _shot(0, rally_index=2, shot_type=SHOT_TYPE_SMASH, player_track_id=1),
        _shot(2, rally_index=2, shot_type=SHOT_TYPE_GROUNDSTROKE, player_track_id=2),
    ]
    outcomes = [
        _outcome(1, OUTCOME_OUT_OF_BOUNDS),
        _outcome(2, OUTCOME_IN_BOUNDS_END),
    ]

    results = compute_rally_ending_shot_success_rate(rallies, shots, outcomes, shot_type=SHOT_TYPE_SMASH)

    assert results[1].attempt_count == 1
    assert results[1].success_count == 0
    assert results[1].success_rate == 0.0


def test_shot_success_rate_counts_in_bounds_end_as_success():
    rallies = [_rally(1, 4.0)]
    shots = [_shot(0, rally_index=1, shot_type=SHOT_TYPE_VOLLEY, player_track_id=3)]
    outcomes = [_outcome(1, OUTCOME_IN_BOUNDS_END)]

    results = compute_rally_ending_shot_success_rate(rallies, shots, outcomes, shot_type=SHOT_TYPE_VOLLEY)

    assert results[3].attempt_count == 1
    assert results[3].success_count == 1


def test_shot_success_rate_skips_undetermined_outcomes():
    rallies = [_rally(1, 4.0)]
    shots = [_shot(0, rally_index=1, shot_type=SHOT_TYPE_SMASH, player_track_id=1)]
    outcomes = [_outcome(1, OUTCOME_UNDETERMINED, used_court_calibration=False)]

    results = compute_rally_ending_shot_success_rate(rallies, shots, outcomes, shot_type=SHOT_TYPE_SMASH)
    assert results == {}


# --- distance / speed ---------------------------------------------------------


def _identity_to_court_meters(point: tuple[float, float]) -> tuple[float, float]:
    # Trivial 1px == 1m mapping so displacement math is easy to check by hand.
    return point


def test_distance_and_speed_requires_calibration():
    player_frames = [[TrackPoint(track_id=1, x=0.0, y=0.0, is_predicted=False)]]
    results = compute_distance_and_speed(player_frames, sample_fps=SAMPLE_FPS, to_court_meters=None)
    assert results == {}


def test_distance_and_speed_sums_consecutive_displacement():
    # track 1 moves (0,0) -> (3,4) -> (3,4) over frames 0,1,2 at 5fps
    player_frames = [
        [TrackPoint(track_id=1, x=0.0, y=0.0, is_predicted=False)],
        [TrackPoint(track_id=1, x=3.0, y=4.0, is_predicted=False)],
        [TrackPoint(track_id=1, x=3.0, y=4.0, is_predicted=False)],
    ]

    results = compute_distance_and_speed(
        player_frames, sample_fps=SAMPLE_FPS, to_court_meters=_identity_to_court_meters
    )

    assert results[1].total_distance_m == 5.0  # 3-4-5 triangle
    assert results[1].elapsed_time_s == 2 / SAMPLE_FPS
    assert results[1].avg_speed_mps == 5.0 / (2 / SAMPLE_FPS)
    assert results[1].observed_frame_count == 3
    assert results[1].predicted_frame_count == 0


def test_distance_and_speed_tracks_are_independent():
    player_frames = [
        [TrackPoint(track_id=1, x=0.0, y=0.0, is_predicted=False), TrackPoint(track_id=2, x=10.0, y=0.0, is_predicted=False)],
        [TrackPoint(track_id=1, x=1.0, y=0.0, is_predicted=False)],
    ]
    results = compute_distance_and_speed(
        player_frames, sample_fps=SAMPLE_FPS, to_court_meters=_identity_to_court_meters
    )
    assert results[1].total_distance_m == 1.0
    # track 2 only appears once -- no displacement, no crash
    assert results[2].total_distance_m == 0.0
    assert results[2].elapsed_time_s == 0.0
    assert results[2].avg_speed_mps == 0.0


# --- player_level_stat_values wiring -------------------------------------------


def test_player_level_stat_values_omits_distance_without_calibration():
    rallies = [_rally(1, 4.0)]
    shots = [_shot(0, rally_index=1, player_track_id=1), _shot(2, rally_index=1, player_track_id=2)]
    outcomes = [_outcome(1, OUTCOME_IN_BOUNDS_END)]
    player_frames = [
        [TrackPoint(track_id=1, x=0.0, y=0.0, is_predicted=False)],
        [TrackPoint(track_id=1, x=1.0, y=0.0, is_predicted=False)],
    ]

    values = player_level_stat_values(
        rallies, shots, outcomes, player_frames, sample_fps=SAMPLE_FPS, to_court_meters=None
    )

    stat_keys = {v.stat_key for v in values}
    assert "distance_covered" not in stat_keys
    assert "movement_speed_avg" not in stat_keys
    assert "reaction_time_avg" in stat_keys  # unaffected by calibration


def test_player_level_stat_values_includes_distance_with_calibration():
    rallies = [_rally(1, 4.0)]
    shots = []
    outcomes = []
    player_frames = [
        [TrackPoint(track_id=1, x=0.0, y=0.0, is_predicted=False)],
        [TrackPoint(track_id=1, x=3.0, y=4.0, is_predicted=False)],
    ]

    values = player_level_stat_values(
        rallies, shots, outcomes, player_frames,
        sample_fps=SAMPLE_FPS, to_court_meters=_identity_to_court_meters,
    )

    distance = next(v for v in values if v.stat_key == "distance_covered")
    assert distance.track_id == 1
    assert distance.value == 5.0


# --- diagnostics + serialization -----------------------------------------------


def test_summarize_match_diagnostics_reuses_upstream_summaries():
    shots = [_shot(0, shot_type=SHOT_TYPE_SMASH), _shot(2, shot_type=SHOT_TYPE_GROUNDSTROKE)]
    serves = [
        ServeEvent(rally_index=1, frame_index=0, server_track_id=1, distance_m=1.0, distance_px=None,
                   used_court_calibration=True, identified=True),
    ]

    diagnostics = summarize_match_diagnostics(shots, serves)

    assert diagnostics["shot_counts_by_type"]["shot_count"] == 2
    assert diagnostics["serve_identification"]["identified_count"] == 1
    assert diagnostics["serve_counts_by_player"][1]["serve_count"] == 1


# --- serve counts by player ---------------------------------------------------


def test_serve_counts_by_player_splits_by_server_track_id():
    from ml.pipeline.serve_detection import summarize_serve_events_by_track_id

    serves = [
        ServeEvent(rally_index=1, frame_index=0, server_track_id=1, distance_m=1.0, distance_px=None,
                   used_court_calibration=True, identified=True),
        ServeEvent(rally_index=2, frame_index=5, server_track_id=2, distance_m=None, distance_px=40.0,
                   used_court_calibration=False, identified=True),
        ServeEvent(rally_index=3, frame_index=10, server_track_id=1, distance_m=0.5, distance_px=None,
                   used_court_calibration=True, identified=True),
    ]

    by_player = summarize_serve_events_by_track_id(serves)

    assert by_player[1] == {"serve_count": 2, "calibrated_count": 2, "pixel_count": 0}
    assert by_player[2] == {"serve_count": 1, "calibrated_count": 0, "pixel_count": 1}


def test_serve_counts_by_player_excludes_unidentified_serves():
    from ml.pipeline.serve_detection import summarize_serve_events_by_track_id

    serves = [
        ServeEvent(rally_index=1, frame_index=None, server_track_id=None, distance_m=None, distance_px=None,
                   used_court_calibration=False, identified=False),
    ]

    assert summarize_serve_events_by_track_id(serves) == {}


def test_serve_counts_by_player_empty_for_no_serves():
    from ml.pipeline.serve_detection import summarize_serve_events_by_track_id

    assert summarize_serve_events_by_track_id([]) == {}


def test_to_serializable_round_trips_fields():
    values = compute_match_level_stats([_rally(1, 10.0)], [_outcome(1, OUTCOME_NET)])
    serialized = to_serializable(values)
    assert all({"stat_key", "track_id", "value", "sample_size"} <= set(d.keys()) for d in serialized)
