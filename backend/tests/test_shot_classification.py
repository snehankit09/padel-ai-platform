"""
Tests for ml/pipeline/shot_classification.py — Part 7c.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_rally_detection.py / test_ball_interpolation.py: builds
BallPoint/PlayerBox/RallySegment objects directly and feeds them straight
into detect_shots/find_contact_points/classify_shot.

Run with: pytest backend/tests/test_shot_classification.py -v
"""

from __future__ import annotations

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.pipeline.rally_detection import RallySegment  # noqa: E402
from ml.pipeline.shot_classification import (  # noqa: E402
    SHOT_TYPE_GROUNDSTROKE,
    SHOT_TYPE_LOB,
    SHOT_TYPE_SMASH,
    SHOT_TYPE_UNKNOWN,
    SHOT_TYPE_VOLLEY,
    BallPoint,
    PlayerBox,
    ball_points_from_tracks_json,
    contact_height_ratio,
    detect_shots,
    find_contact_points,
    player_frames_from_tracks_json,
    summarize_shots,
    to_serializable,
)


def _rally(n_frames: int, rally_index: int = 1) -> RallySegment:
    return RallySegment(
        rally_index=rally_index, start_frame=0, end_frame=n_frames - 1,
        start_time_s=0.0, end_time_s=n_frames / 5.0, duration_s=n_frames / 5.0,
        frame_count=n_frames, ball_active_frame_count=n_frames,
        ball_observed_frame_count=n_frames, mean_ball_confidence=0.6,
    )


def _ball_points(ys, *, x=100.0) -> list[BallPoint]:
    return [BallPoint(frame_index=i, x=x, y=y, is_predicted=False) for i, y in enumerate(ys)]


def _players_every_frame(n_frames, *, x=100.0, y=150.0, top=100.0, bottom=190.0, track_id=1) -> list[list[PlayerBox]]:
    return [[PlayerBox(frame_index=i, track_id=track_id, x=x, y=y, top=top, bottom=bottom)] for i in range(n_frames)]


def _identity_court_meters(scale=20.0):
    return lambda pt: (pt[0] / scale, pt[1] / scale)


# --- contact_height_ratio ----------------------------------------------------


def test_contact_height_ratio_at_bbox_bottom_is_zero():
    player = PlayerBox(frame_index=0, track_id=1, x=0, y=0, top=0, bottom=100)
    assert contact_height_ratio(100, player) == 0.0


def test_contact_height_ratio_above_bbox_top_exceeds_one():
    player = PlayerBox(frame_index=0, track_id=1, x=0, y=0, top=0, bottom=100)
    assert contact_height_ratio(-10, player) == 1.1


def test_contact_height_ratio_handles_zero_height_box():
    player = PlayerBox(frame_index=0, track_id=1, x=0, y=0, top=50, bottom=50)
    assert contact_height_ratio(50, player) is None


# --- find_contact_points ------------------------------------------------------


def test_finds_a_single_direction_reversal():
    ys = [200, 180, 160, 140, 160, 180, 200]  # falls then rises -> local min at frame 3
    rally = _rally(len(ys))
    contacts = find_contact_points(rally, _ball_points(ys), _players_every_frame(len(ys)))
    assert len(contacts) == 1
    assert contacts[0].frame_index == 3


def test_monotonic_motion_has_no_contacts():
    ys = [200, 190, 180, 170, 160, 150]
    rally = _rally(len(ys))
    contacts = find_contact_points(rally, _ball_points(ys), _players_every_frame(len(ys)))
    assert contacts == []


def test_interpolated_points_are_ignored_for_reversal_detection():
    ys = [200, 180, 160, 140, 160, 180, 200]
    points = _ball_points(ys)
    # Mark the reversal frame itself as interpolated -- it must be skipped,
    # even though its neighbors are real, so no contact should be found there.
    points[3] = BallPoint(frame_index=3, x=100, y=points[3].y, is_predicted=True)
    rally = _rally(len(ys))
    contacts = find_contact_points(rally, points, _players_every_frame(len(ys)))
    assert contacts == []


def test_contact_with_no_nearby_player_is_still_returned_unresolved():
    ys = [200, 180, 160, 140, 160, 180, 200]
    rally = _rally(len(ys))
    far_players = _players_every_frame(len(ys), x=100000.0)  # nowhere near the ball
    contacts = find_contact_points(rally, _ball_points(ys), far_players)
    assert len(contacts) == 1
    assert contacts[0].player_track_id is None
    assert contacts[0].contact_height_ratio is None


def test_contact_outside_rally_window_is_excluded():
    ys = [200, 180, 160, 140, 160, 180, 200]
    rally = RallySegment(
        rally_index=1, start_frame=0, end_frame=1,  # window ends before the reversal at frame 3
        start_time_s=0.0, end_time_s=0.4, duration_s=0.4,
        frame_count=2, ball_active_frame_count=2, ball_observed_frame_count=2, mean_ball_confidence=0.6,
    )
    contacts = find_contact_points(rally, _ball_points(ys), _players_every_frame(len(ys)))
    assert contacts == []


# --- classify_shot / detect_shots: the four shot types + unknown -------------


def test_smash_is_classified_from_overhead_contact_height():
    ys = [200, 150, 100, 70, 110, 160, 220]  # sharp reversal well above player bbox top
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), top=80.0, bottom=190.0)
    shots = detect_shots([rally], _ball_points(ys), players)
    assert len(shots) == 1
    assert shots[0].shot_type == SHOT_TYPE_SMASH
    assert shots[0].contact_height_ratio >= 1.05


def test_lob_is_classified_from_long_airborne_duration():
    ys = [200, 180, 160, 150, 160, 180, 210, 250, 300, 350, 400, 420, 400, 350]
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), y=190.0, top=150.0, bottom=230.0)
    shots = detect_shots([rally], _ball_points(ys), players)
    assert shots[0].shot_type == SHOT_TYPE_LOB
    assert shots[0].airborne_frames_after >= 6


def test_volley_is_classified_near_the_net_with_calibration():
    ys = [190, 185, 180, 175, 180, 185, 190]  # stays near pixel y=180 -> near the net at scale 20 (court_length 20m)
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), y=180.0, top=150.0, bottom=210.0)
    shots = detect_shots(
        [rally], _ball_points(ys), players,
        to_court_meters=_identity_court_meters(scale=20.0), court_length_m=20.0,
    )
    assert shots[0].shot_type == SHOT_TYPE_VOLLEY
    assert shots[0].used_court_calibration is True
    assert shots[0].distance_from_net_m is not None
    assert shots[0].distance_from_net_m <= 3.0


def test_groundstroke_is_classified_near_the_baseline_with_calibration():
    ys = [40, 35, 30, 25, 30, 35, 40]  # near pixel y=30 -> near the baseline, far from the net at scale 20
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), y=30.0, top=5.0, bottom=60.0)
    shots = detect_shots(
        [rally], _ball_points(ys), players,
        to_court_meters=_identity_court_meters(scale=20.0), court_length_m=20.0,
    )
    assert shots[0].shot_type == SHOT_TYPE_GROUNDSTROKE
    assert shots[0].distance_from_net_m > 3.0


def test_no_calibration_and_no_smash_or_lob_signal_is_unknown():
    ys = [200, 180, 160, 140, 160, 180, 200]
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), top=80.0, bottom=190.0)  # contact height well below smash threshold
    shots = detect_shots([rally], _ball_points(ys), players)  # no to_court_meters given
    assert shots[0].shot_type == SHOT_TYPE_UNKNOWN
    assert shots[0].used_court_calibration is False


def test_rally_with_no_findable_contact_contributes_no_shots():
    ys = [200, 190, 180, 170]  # monotonic -- no reversal
    rally = _rally(len(ys))
    shots = detect_shots([rally], _ball_points(ys), _players_every_frame(len(ys)))
    assert shots == []


def test_multiple_rallies_each_contribute_their_own_shots():
    ys = [200, 180, 160, 140, 160, 180, 200]
    rally1 = _rally(len(ys), rally_index=1)
    rally2 = RallySegment(
        rally_index=2, start_frame=len(ys), end_frame=2 * len(ys) - 1,
        start_time_s=1.4, end_time_s=2.8, duration_s=1.4,
        frame_count=len(ys), ball_active_frame_count=len(ys), ball_observed_frame_count=len(ys), mean_ball_confidence=0.6,
    )
    all_ys = ys + ys
    players = _players_every_frame(len(all_ys), top=80.0, bottom=190.0)
    shots = detect_shots([rally1, rally2], _ball_points(all_ys), players)
    assert {s.rally_index for s in shots} == {1, 2}


# --- JSON round-trip helpers ---------------------------------------------------


def test_ball_points_from_tracks_json_skips_ambiguous_frames():
    data = {
        "frames": [
            {"tracks": []},  # no ball -- skipped
            {"tracks": [{"track_id": 1, "class_id": 32, "class_name": "sports ball", "confidence": 0.5,
                         "bbox": {"x1": 10, "y1": 20, "x2": 18, "y2": 28}, "age": 1, "hits": 1, "time_since_update": 0}]},
            {"tracks": [{"track_id": 1, "bbox": {"x1": 0, "y1": 0, "x2": 8, "y2": 8}},
                        {"track_id": 2, "bbox": {"x1": 0, "y1": 0, "x2": 8, "y2": 8}}]},  # ambiguous -- skipped
        ]
    }
    points = ball_points_from_tracks_json(data)
    assert len(points) == 1
    assert points[0].frame_index == 1
    assert points[0].x == 14.0 and points[0].y == 24.0
    assert points[0].is_predicted is False


def test_player_frames_from_tracks_json_preserves_bbox_extent():
    data = {"frames": [{"tracks": [{"track_id": 5, "bbox": {"x1": 0, "y1": 10, "x2": 20, "y2": 90}}]}]}
    frames = player_frames_from_tracks_json(data)
    assert len(frames) == 1
    assert frames[0][0].track_id == 5
    assert frames[0][0].top == 10
    assert frames[0][0].bottom == 90


def test_to_serializable_round_trips_shot_fields():
    ys = [200, 180, 160, 140, 160, 180, 200]
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), top=80.0, bottom=190.0)
    shots = detect_shots([rally], _ball_points(ys), players)
    serialized = to_serializable(shots)
    assert serialized[0]["shot_type"] == shots[0].shot_type
    assert serialized[0]["rally_index"] == shots[0].rally_index


def test_summarize_shots_counts_by_type_and_unknown_rate():
    ys = [200, 180, 160, 140, 160, 180, 200]
    rally = _rally(len(ys))
    players = _players_every_frame(len(ys), top=80.0, bottom=190.0)
    shots = detect_shots([rally], _ball_points(ys), players)  # -> unknown, no calibration
    summary = summarize_shots(shots)
    assert summary["shot_count"] == 1
    assert summary["shot_counts_by_type"][SHOT_TYPE_UNKNOWN] == 1
    assert summary["unknown_rate"] == 1.0


def test_summarize_shots_handles_empty_input():
    summary = summarize_shots([])
    assert summary["shot_count"] == 0
    assert summary["unknown_rate"] == 0.0
