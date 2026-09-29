"""
Tests for ml/pipeline/team_side_positions.py — team-side attribution
build, Tiers 1a-1b.

Pure logic throughout — no model, no video, no Celery/DB — same approach
as test_court_detector.py / test_stats_aggregation.py: Tier 1a's tests
build a real homography from a synthetic court and feed hand-built tracks
JSON straight into load_track_positions; Tier 1b's need nothing but bare
coordinate pairs, since classify_quadrant takes no homography or tracks
data at all.

Run with: pytest backend/tests/test_team_side_positions.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.detection.court_detector import compute_homography, order_corners  # noqa: E402
from ml.pipeline.team_side_positions import (  # noqa: E402
    DEFAULT_LABEL_CONFIDENCE_THRESHOLD,
    FAR_TEAM_LABEL,
    NEAR_TEAM_LABEL,
    QUADRANT_FAR_LEFT,
    QUADRANT_FAR_RIGHT,
    QUADRANT_LABELS,
    QUADRANT_NEAR_LEFT,
    QUADRANT_NEAR_RIGHT,
    QUADRANT_TEAM_LABELS,
    CourtPosition,
    DominantQuadrant,
    MergedPlayer,
    TrackPositionError,
    assign_player_label,
    build_track_label_lookup,
    classify_quadrant,
    dominant_quadrant,
    load_track_positions,
    merge_track_fragments,
    quadrant_to_label,
)

# A real, non-trivial (camera-angled) homography — reused across tests so
# they agree on what pixel positions map to which court-meter positions.
_CORNERS = order_corners(
    [
        (150.0, 80.0),
        (490.0, 60.0),
        (600.0, 420.0),
        (60.0, 440.0),
    ]
)
_COURT_LENGTH_M = 20.0
_COURT_WIDTH_M = 10.0
HOMOGRAPHY = compute_homography(_CORNERS, court_length_m=_COURT_LENGTH_M, court_width_m=_COURT_WIDTH_M)


def _bbox(x1: float, y1: float, x2: float, y2: float) -> dict:
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _frame(index: int, tracks: list[dict]) -> dict:
    return {"frame_path": f"/storage/frames/vid/frame_{index:06d}.jpg", "tracks": tracks}


def _track(track_id: int, bbox: dict, *, time_since_update: int = 0) -> dict:
    return {"track_id": track_id, "bbox": bbox, "time_since_update": time_since_update}


def test_returns_full_sequence_across_every_frame_the_track_appears_in():
    tracks_data = {
        "frames": [
            _frame(0, [_track(1, _bbox(300, 100, 340, 200))]),
            _frame(1, [_track(1, _bbox(305, 105, 345, 205))]),
            _frame(2, [_track(1, _bbox(310, 110, 350, 210))]),
        ]
    }

    positions = load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)

    assert [p.frame_index for p in positions] == [0, 1, 2]
    assert all(p.track_id == 1 for p in positions)


def test_skips_frames_where_the_track_id_does_not_appear():
    tracks_data = {
        "frames": [
            _frame(0, [_track(1, _bbox(300, 100, 340, 200))]),
            _frame(1, [_track(2, _bbox(100, 100, 140, 200))]),  # different track only
            _frame(2, [_track(1, _bbox(310, 110, 350, 210))]),
        ]
    }

    positions = load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)

    assert [p.frame_index for p in positions] == [0, 2]


def test_is_not_limited_to_a_single_rally_window():
    # Frames spanning well beyond what any one rally's start/end window
    # would cover -- the whole point of Tier 1a over Tier 3a's per-rally loop.
    tracks_data = {
        "frames": [
            _frame(i, [_track(1, _bbox(300 + i, 100, 340 + i, 200))]) for i in range(0, 500, 50)
        ]
    }

    positions = load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)

    assert [p.frame_index for p in positions] == list(range(0, 500, 50))


def test_uses_bbox_bottom_center_not_bbox_center_as_the_foot_position():
    # A very tall bbox: bottom-center and true center land in very
    # different places on the court -- this pins down which one wins.
    bbox = _bbox(x1=300, y1=50, x2=340, y2=400)
    tracks_data = {"frames": [_frame(0, [_track(1, bbox)])]}

    [position] = load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)

    from ml.detection.court_detector import pixel_to_court_point

    expected_x, expected_y = pixel_to_court_point(HOMOGRAPHY, (320.0, 400.0))  # bottom-center
    wrong_center_x, wrong_center_y = pixel_to_court_point(HOMOGRAPHY, (320.0, 225.0))  # bbox center

    assert position.court_x == pytest.approx(expected_x)
    assert position.court_y == pytest.approx(expected_y)
    assert position.court_y != pytest.approx(wrong_center_y)


def test_preserves_is_predicted_from_time_since_update():
    tracks_data = {
        "frames": [
            _frame(0, [_track(1, _bbox(300, 100, 340, 200), time_since_update=0)]),
            _frame(1, [_track(1, _bbox(305, 105, 345, 205), time_since_update=1)]),
        ]
    }

    positions = load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)

    assert positions[0].is_predicted is False
    assert positions[1].is_predicted is True


def test_returns_positions_sorted_by_frame_index_even_if_input_is_out_of_order():
    tracks_data = {
        "frames": [
            _frame(2, [_track(1, _bbox(310, 110, 350, 210))]),
            _frame(0, [_track(1, _bbox(300, 100, 340, 200))]),
            _frame(1, [_track(1, _bbox(305, 105, 345, 205))]),
        ]
    }

    positions = load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)

    assert [p.frame_index for p in positions] == [0, 1, 2]


def test_raises_when_track_id_never_appears():
    tracks_data = {"frames": [_frame(0, [_track(1, _bbox(300, 100, 340, 200))])]}

    with pytest.raises(TrackPositionError):
        load_track_positions(tracks_data, HOMOGRAPHY, track_id=999)


def test_raises_on_unparseable_frame_path():
    tracks_data = {
        "frames": [
            {"frame_path": "/storage/frames/vid/not_a_frame_file.jpg", "tracks": [_track(1, _bbox(300, 100, 340, 200))]}
        ]
    }

    with pytest.raises(TrackPositionError):
        load_track_positions(tracks_data, HOMOGRAPHY, track_id=1)


def test_court_position_is_a_frozen_dataclass_with_expected_fields():
    position = CourtPosition(frame_index=0, track_id=1, court_x=1.0, court_y=2.0, is_predicted=False)
    assert position.frame_index == 0
    assert position.track_id == 1
    assert position.court_x == 1.0
    assert position.court_y == 2.0
    assert position.is_predicted is False


# --- classify_quadrant: Tier 1b, pure geometry, no fixtures needed ---------


# Default court: 10m wide, 20m long -- midpoints at x=5, y=10.
def test_classifies_near_left():
    assert classify_quadrant(2.0, 3.0) == QUADRANT_NEAR_LEFT


def test_classifies_near_right():
    assert classify_quadrant(8.0, 3.0) == QUADRANT_NEAR_RIGHT


def test_classifies_far_left():
    assert classify_quadrant(2.0, 15.0) == QUADRANT_FAR_LEFT


def test_classifies_far_right():
    assert classify_quadrant(8.0, 15.0) == QUADRANT_FAR_RIGHT


def test_point_exactly_on_the_net_lands_far_not_near():
    # y == net_y (10.0): Tier 3a's own "near" if y < net_y else "far"
    # convention puts the exact midpoint on the "far" side.
    assert classify_quadrant(2.0, 10.0) == QUADRANT_FAR_LEFT
    assert classify_quadrant(8.0, 10.0) == QUADRANT_FAR_RIGHT


def test_point_exactly_on_the_width_midpoint_lands_right_not_left():
    # x == mid_x (5.0): same half-open convention on the new axis.
    assert classify_quadrant(5.0, 3.0) == QUADRANT_NEAR_RIGHT
    assert classify_quadrant(5.0, 15.0) == QUADRANT_FAR_RIGHT


def test_every_court_point_maps_to_exactly_one_quadrant():
    valid_quadrants = {QUADRANT_NEAR_LEFT, QUADRANT_NEAR_RIGHT, QUADRANT_FAR_LEFT, QUADRANT_FAR_RIGHT}
    for x in (-1.0, 0.0, 4.999, 5.0, 5.001, 10.0, 11.0):
        for y in (-1.0, 0.0, 9.999, 10.0, 10.001, 20.0, 21.0):
            assert classify_quadrant(x, y) in valid_quadrants


def test_respects_custom_court_dimensions():
    # A non-default court (e.g. a different court_width_m/court_length_m
    # from calibration.json) should shift both midpoints accordingly.
    assert classify_quadrant(3.0, 4.0, court_width_m=8.0, court_length_m=16.0) == QUADRANT_NEAR_LEFT
    assert classify_quadrant(5.0, 4.0, court_width_m=8.0, court_length_m=16.0) == QUADRANT_NEAR_RIGHT
    assert classify_quadrant(3.0, 9.0, court_width_m=8.0, court_length_m=16.0) == QUADRANT_FAR_LEFT


# --- dominant_quadrant: Tier 1c ---------------------------------------------


def _pos(frame_index: int, track_id: int, court_x: float, court_y: float, *, is_predicted: bool = False) -> CourtPosition:
    return CourtPosition(
        frame_index=frame_index, track_id=track_id, court_x=court_x, court_y=court_y, is_predicted=is_predicted
    )


def test_reports_the_quadrant_with_the_most_frames():
    positions = [
        _pos(0, 1, 2.0, 3.0),  # near_left
        _pos(1, 1, 2.0, 3.0),  # near_left
        _pos(2, 1, 2.0, 3.0),  # near_left
        _pos(3, 1, 8.0, 3.0),  # near_right
    ]

    result = dominant_quadrant(positions)

    assert result.track_id == 1
    assert result.quadrant == QUADRANT_NEAR_LEFT
    assert result.frames_observed == 4
    assert result.confidence == pytest.approx(0.75)


def test_confidence_is_1_when_every_frame_agrees():
    positions = [_pos(i, 1, 2.0, 3.0) for i in range(5)]

    result = dominant_quadrant(positions)

    assert result.quadrant == QUADRANT_NEAR_LEFT
    assert result.confidence == pytest.approx(1.0)


def test_excludes_predicted_frames_from_the_vote():
    positions = [
        _pos(0, 1, 2.0, 3.0),  # near_left, real
        _pos(1, 1, 8.0, 15.0, is_predicted=True),  # far_right, but predicted -- should not count
        _pos(2, 1, 2.0, 3.0),  # near_left, real
    ]

    result = dominant_quadrant(positions)

    assert result.quadrant == QUADRANT_NEAR_LEFT
    assert result.frames_observed == 2  # only the 2 real frames
    assert result.confidence == pytest.approx(1.0)


def test_ties_break_toward_earlier_quadrant_in_fixed_order():
    # near_left and far_right tied 1-1 -- near_left must win, deterministically.
    positions = [
        _pos(0, 1, 2.0, 3.0),  # near_left
        _pos(1, 1, 8.0, 15.0),  # far_right
    ]

    result = dominant_quadrant(positions)

    assert result.quadrant == QUADRANT_NEAR_LEFT
    assert result.confidence == pytest.approx(0.5)


def test_raises_on_empty_positions():
    with pytest.raises(TrackPositionError):
        dominant_quadrant([])


def test_raises_when_positions_mix_multiple_track_ids():
    positions = [_pos(0, 1, 2.0, 3.0), _pos(1, 2, 2.0, 3.0)]

    with pytest.raises(TrackPositionError):
        dominant_quadrant(positions)


def test_raises_when_every_position_is_predicted():
    positions = [_pos(0, 1, 2.0, 3.0, is_predicted=True), _pos(1, 1, 2.0, 3.0, is_predicted=True)]

    with pytest.raises(TrackPositionError):
        dominant_quadrant(positions)


def test_respects_custom_court_dimensions_when_classifying():
    positions = [_pos(0, 1, 3.0, 4.0), _pos(1, 1, 3.0, 4.0)]

    result = dominant_quadrant(positions, court_width_m=8.0, court_length_m=16.0)

    assert result.quadrant == QUADRANT_NEAR_LEFT


def test_dominant_quadrant_is_a_frozen_dataclass_with_expected_fields():
    result = DominantQuadrant(track_id=1, quadrant=QUADRANT_NEAR_LEFT, frames_observed=4, confidence=0.75)
    assert result.track_id == 1
    assert result.quadrant == QUADRANT_NEAR_LEFT
    assert result.frames_observed == 4
    assert result.confidence == 0.75


# --- merge_track_fragments: Tier 1d -----------------------------------------
#
# An identity homography (pixel coords == court-meter coords) so fragments
# can be placed by quadrant directly, without reasoning about the camera-
# angled HOMOGRAPHY above -- Tier 1d's own logic (grouping + interval
# partitioning) has nothing to do with the homography's specific shape, the
# same reasoning classify_quadrant's own tests use bare coordinate pairs.
IDENTITY_HOMOGRAPHY = np.eye(3, dtype=np.float64)

# Default court is 10m wide x 20m long (midpoints at x=5, y=10) -- these are
# one point comfortably inside each quadrant.
NEAR_LEFT_PT = (2.0, 3.0)
NEAR_RIGHT_PT = (8.0, 3.0)
FAR_LEFT_PT = (2.0, 15.0)
FAR_RIGHT_PT = (8.0, 15.0)


def _bbox_at(x: float, y: float) -> dict:
    # Zero-size box: bottom-center == (x, y) exactly, so foot position is
    # this point with no need to reason about box width/height.
    return _bbox(x, y, x, y)


def _fragment_frames(track_id: int, frame_indices: range, point: tuple[float, float]) -> list[dict]:
    return [_frame(i, [_track(track_id, _bbox_at(*point))]) for i in frame_indices]


def _tracks_data(*fragment_groups: list[dict]) -> dict:
    frames: list[dict] = []
    for group in fragment_groups:
        frames.extend(group)
    return {"frames": frames}


def test_merges_two_non_overlapping_fragments_in_the_same_quadrant():
    tracks_data = _tracks_data(
        _fragment_frames(1, range(0, 3), NEAR_LEFT_PT),  # frames 0-2
        _fragment_frames(2, range(5, 8), NEAR_LEFT_PT),  # frames 5-7, no overlap with track 1
    )

    [player] = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert player.quadrant == QUADRANT_NEAR_LEFT
    assert player.track_ids == (1, 2)
    assert player.start_frame == 0
    assert player.end_frame == 7
    assert player.frames_observed == 6


def test_does_not_merge_fragments_that_overlap_even_in_the_same_quadrant():
    tracks_data = _tracks_data(
        _fragment_frames(1, range(0, 6), NEAR_LEFT_PT),  # frames 0-5
        _fragment_frames(2, range(3, 9), NEAR_LEFT_PT),  # frames 3-8, overlaps 3-5 with track 1
    )

    players = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert len(players) == 2
    assert all(p.quadrant == QUADRANT_NEAR_LEFT for p in players)
    assert players[0].track_ids == (1,)
    assert players[1].track_ids == (2,)


def test_never_merges_fragments_from_different_quadrants_even_without_time_overlap():
    tracks_data = _tracks_data(
        _fragment_frames(1, range(0, 3), NEAR_LEFT_PT),
        _fragment_frames(2, range(5, 8), FAR_RIGHT_PT),
    )

    players = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert len(players) == 2
    quadrants = {p.quadrant for p in players}
    assert quadrants == {QUADRANT_NEAR_LEFT, QUADRANT_FAR_RIGHT}


def test_prefers_extending_the_latest_eligible_chain():
    # track 1 (0-3) and track 2 (2-6) overlap -> forced into separate
    # chains (ending at frame 3 and frame 6 respectively). track 3 (10-12)
    # is eligible to extend either chain; it should join track 2's chain
    # (the more recently active one), not track 1's.
    tracks_data = _tracks_data(
        _fragment_frames(1, range(0, 4), NEAR_LEFT_PT),  # frames 0-3
        _fragment_frames(2, range(2, 7), NEAR_LEFT_PT),  # frames 2-6, overlaps track 1
        _fragment_frames(3, range(10, 13), NEAR_LEFT_PT),  # frames 10-12, overlaps neither
    )

    players = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert len(players) == 2
    by_first_track = {p.track_ids[0]: p for p in players}
    assert by_first_track[1].track_ids == (1,)
    assert by_first_track[2].track_ids == (2, 3)


def test_skips_a_track_with_no_real_non_predicted_positions():
    all_predicted = [
        _frame(i, [_track(1, _bbox_at(*NEAR_LEFT_PT), time_since_update=1)]) for i in range(3)
    ]
    real_track = _fragment_frames(2, range(0, 3), NEAR_RIGHT_PT)
    tracks_data = _tracks_data(all_predicted, real_track)

    [player] = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert player.track_ids == (2,)


def test_returns_empty_list_for_tracks_data_with_no_frames():
    assert merge_track_fragments({"frames": []}, IDENTITY_HOMOGRAPHY) == []


def test_results_sorted_by_start_frame_across_quadrants():
    tracks_data = _tracks_data(
        _fragment_frames(1, range(10, 13), NEAR_LEFT_PT),
        _fragment_frames(2, range(0, 3), FAR_RIGHT_PT),
    )

    players = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert [p.track_ids for p in players] == [(2,), (1,)]


def test_confidence_is_the_minimum_across_merged_fragments():
    # track 1: perfectly confident (1.0). track 2: one stray frame lands
    # in a different quadrant late in its own timeline, but its DOMINANT
    # quadrant is still near_left (3 of 4 frames), confidence 0.75 -- see
    # test_reports_the_quadrant_with_the_most_frames for the same shape.
    track_1_frames = _fragment_frames(1, range(0, 3), NEAR_LEFT_PT)  # frames 0-2, conf 1.0
    track_2_frames = [
        _frame(10, [_track(2, _bbox_at(*NEAR_LEFT_PT))]),
        _frame(11, [_track(2, _bbox_at(*NEAR_LEFT_PT))]),
        _frame(12, [_track(2, _bbox_at(*NEAR_LEFT_PT))]),
        _frame(13, [_track(2, _bbox_at(*NEAR_RIGHT_PT))]),  # stray frame, still near_left-dominant
    ]
    tracks_data = _tracks_data(track_1_frames, track_2_frames)

    [player] = merge_track_fragments(tracks_data, IDENTITY_HOMOGRAPHY)

    assert player.track_ids == (1, 2)
    assert player.confidence == pytest.approx(0.75)


def test_merged_player_is_a_frozen_dataclass_with_expected_fields():
    player = MergedPlayer(
        quadrant=QUADRANT_NEAR_LEFT, track_ids=(1, 2), start_frame=0, end_frame=7, frames_observed=6, confidence=1.0
    )
    assert player.quadrant == QUADRANT_NEAR_LEFT
    assert player.track_ids == (1, 2)
    assert player.start_frame == 0
    assert player.end_frame == 7
    assert player.frames_observed == 6
    assert player.confidence == 1.0


# --- assign_player_label: Tier 2b ---------------------------------------

def _merged_player(quadrant: str, confidence: float) -> MergedPlayer:
    return MergedPlayer(
        quadrant=quadrant, track_ids=(1,), start_frame=0, end_frame=10, frames_observed=10, confidence=confidence
    )


@pytest.mark.parametrize(
    "quadrant, expected_label",
    [
        (QUADRANT_NEAR_LEFT, "Player A"),
        (QUADRANT_NEAR_RIGHT, "Player B"),
        (QUADRANT_FAR_LEFT, "Player C"),
        (QUADRANT_FAR_RIGHT, "Player D"),
    ],
)
def test_assign_player_label_returns_individual_label_when_confident(quadrant, expected_label):
    player = _merged_player(quadrant, confidence=0.75)
    assert assign_player_label(player) == expected_label


@pytest.mark.parametrize(
    "quadrant, expected_team_label",
    [
        (QUADRANT_NEAR_LEFT, NEAR_TEAM_LABEL),
        (QUADRANT_NEAR_RIGHT, NEAR_TEAM_LABEL),
        (QUADRANT_FAR_LEFT, FAR_TEAM_LABEL),
        (QUADRANT_FAR_RIGHT, FAR_TEAM_LABEL),
    ],
)
def test_assign_player_label_falls_back_to_team_label_when_not_confident(quadrant, expected_team_label):
    player = _merged_player(quadrant, confidence=0.3)
    assert assign_player_label(player) == expected_team_label


def test_assign_player_label_threshold_is_inclusive_at_exactly_the_boundary():
    player = _merged_player(QUADRANT_NEAR_LEFT, confidence=DEFAULT_LABEL_CONFIDENCE_THRESHOLD)
    assert assign_player_label(player) == "Player A"


def test_assign_player_label_falls_back_just_below_the_boundary():
    player = _merged_player(QUADRANT_NEAR_LEFT, confidence=DEFAULT_LABEL_CONFIDENCE_THRESHOLD - 0.0001)
    assert assign_player_label(player) == NEAR_TEAM_LABEL


def test_assign_player_label_respects_a_custom_threshold():
    player = _merged_player(QUADRANT_FAR_RIGHT, confidence=0.8)
    assert assign_player_label(player, confidence_threshold=0.5) == "Player D"
    assert assign_player_label(player, confidence_threshold=0.9) == FAR_TEAM_LABEL


def test_assign_player_label_rejects_unrecognized_quadrant():
    player = _merged_player("near_center", confidence=0.9)
    with pytest.raises(ValueError, match="unrecognized quadrant"):
        assign_player_label(player)


def test_quadrant_team_labels_covers_all_four_quadrants_with_two_distinct_teams():
    assert set(QUADRANT_TEAM_LABELS) == set(QUADRANT_LABELS)
    assert set(QUADRANT_TEAM_LABELS.values()) == {NEAR_TEAM_LABEL, FAR_TEAM_LABEL}


# --- build_track_label_lookup: Tier 2c -----------------------------------

def test_build_track_label_lookup_maps_every_track_id_to_its_players_label():
    players = [
        _merged_player(QUADRANT_NEAR_LEFT, confidence=0.9),  # single track_id: 1
        MergedPlayer(
            quadrant=QUADRANT_FAR_RIGHT,
            track_ids=(2, 3),
            start_frame=0,
            end_frame=20,
            frames_observed=15,
            confidence=0.9,
        ),
    ]

    lookup = build_track_label_lookup(players)

    assert lookup == {1: "Player A", 2: "Player D", 3: "Player D"}


def test_build_track_label_lookup_uses_team_fallback_for_low_confidence_players():
    players = [_merged_player(QUADRANT_NEAR_RIGHT, confidence=0.3)]

    lookup = build_track_label_lookup(players)

    assert lookup == {1: NEAR_TEAM_LABEL}


def test_build_track_label_lookup_respects_a_custom_threshold():
    players = [_merged_player(QUADRANT_FAR_LEFT, confidence=0.6)]

    assert build_track_label_lookup(players, confidence_threshold=0.5) == {1: "Player C"}
    assert build_track_label_lookup(players, confidence_threshold=0.7) == {1: FAR_TEAM_LABEL}


def test_build_track_label_lookup_returns_empty_dict_for_no_players():
    assert build_track_label_lookup([]) == {}


def test_build_track_label_lookup_omits_track_ids_never_passed_in():
    players = [_merged_player(QUADRANT_NEAR_LEFT, confidence=0.9)]  # only track_id 1

    lookup = build_track_label_lookup(players)

    assert 999 not in lookup
    with pytest.raises(KeyError):
        lookup[999]


def test_build_track_label_lookup_rejects_a_track_id_in_two_merged_players():
    players = [
        _merged_player(QUADRANT_NEAR_LEFT, confidence=0.9),  # track_id 1
        MergedPlayer(
            quadrant=QUADRANT_FAR_RIGHT,
            track_ids=(1,),  # collides with the player above
            start_frame=50,
            end_frame=60,
            frames_observed=10,
            confidence=0.9,
        ),
    ]

    with pytest.raises(ValueError, match="appears in more than one MergedPlayer"):
        build_track_label_lookup(players)


# --- quadrant_to_label: Tier 2a ----------------------------------------

@pytest.mark.parametrize(
    "quadrant, expected_label",
    [
        (QUADRANT_NEAR_LEFT, "Player A"),
        (QUADRANT_NEAR_RIGHT, "Player B"),
        (QUADRANT_FAR_LEFT, "Player C"),
        (QUADRANT_FAR_RIGHT, "Player D"),
    ],
)
def test_quadrant_to_label_fixed_convention(quadrant, expected_label):
    assert quadrant_to_label(quadrant) == expected_label


def test_quadrant_to_label_covers_all_four_quadrant_constants():
    assert set(QUADRANT_LABELS) == {
        QUADRANT_NEAR_LEFT,
        QUADRANT_NEAR_RIGHT,
        QUADRANT_FAR_LEFT,
        QUADRANT_FAR_RIGHT,
    }
    assert len(set(QUADRANT_LABELS.values())) == 4  # no two quadrants share a label


def test_quadrant_to_label_rejects_unrecognized_quadrant():
    with pytest.raises(ValueError, match="unrecognized quadrant"):
        quadrant_to_label("near_center")


def test_quadrant_to_label_rejects_empty_string():
    with pytest.raises(ValueError, match="unrecognized quadrant"):
        quadrant_to_label("")
