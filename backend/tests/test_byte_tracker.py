"""
Tests for ml/tracking/byte_tracker.py — Part 6a.

Pure logic throughout — no model, no video, no Celery/DB — so every test
here constructs plain Detection/BoundingBox objects directly (see
ml/detection/yolo_detector.py's own dataclasses) and feeds them straight
into a ByteTracker, the same "no heavy dependency needed to exercise
control flow" approach test_yolo_detector.py takes with its fake
model/Results objects.

Run with: pytest backend/tests/test_byte_tracker.py -v
"""

from __future__ import annotations

import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.detection.yolo_detector import BoundingBox, Detection, FrameDetections  # noqa: E402
from ml.tracking.byte_tracker import (  # noqa: E402
    ByteTracker,
    FrameTracks,
    TrackedDetection,
    iou,
    match_by_iou,
    to_serializable,
)

PERSON = 0
BALL = 32


def _det(x1, y1, x2, y2, *, conf=0.9, class_id=PERSON, name="person") -> Detection:
    return Detection(class_id=class_id, class_name=name, confidence=conf, bbox=BoundingBox(x1, y1, x2, y2))


def _box(x1, y1, x2, y2) -> BoundingBox:
    return BoundingBox(x1, y1, x2, y2)


# --- iou ----------------------------------------------------------------


def test_iou_of_identical_boxes_is_one():
    b = _box(0, 0, 10, 10)
    assert iou(b, b) == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    assert iou(_box(0, 0, 10, 10), _box(20, 20, 30, 30)) == 0.0


def test_iou_of_partially_overlapping_boxes():
    # (0,0)-(10,10) and (5,5)-(15,15): intersection 5x5=25, union 100+100-25=175
    assert iou(_box(0, 0, 10, 10), _box(5, 5, 15, 15)) == pytest.approx(25 / 175)


def test_iou_is_symmetric():
    a, b = _box(0, 0, 10, 10), _box(4, 4, 12, 12)
    assert iou(a, b) == pytest.approx(iou(b, a))


# --- match_by_iou ---------------------------------------------------------


def test_match_by_iou_pairs_overlapping_boxes():
    tracks = [_box(0, 0, 10, 10), _box(100, 100, 110, 110)]
    detections = [_box(101, 101, 111, 111), _box(1, 1, 11, 11)]  # deliberately shuffled
    matches, unmatched_tracks, unmatched_dets = match_by_iou(tracks, detections)

    assert set(matches) == {(0, 1), (1, 0)}
    assert unmatched_tracks == []
    assert unmatched_dets == []


def test_match_by_iou_below_threshold_is_unmatched():
    tracks = [_box(0, 0, 10, 10)]
    detections = [_box(9, 9, 20, 20)]  # tiny sliver of overlap
    matches, unmatched_tracks, unmatched_dets = match_by_iou(tracks, detections, iou_threshold=0.5)

    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_dets == [0]


def test_match_by_iou_handles_empty_inputs():
    assert match_by_iou([], []) == ([], [], [])
    assert match_by_iou([_box(0, 0, 1, 1)], []) == ([], [0], [])
    assert match_by_iou([], [_box(0, 0, 1, 1)]) == ([], [], [0])


# --- ByteTracker: basic identity persistence -------------------------------


def test_same_object_keeps_same_track_id_across_frames():
    tracker = ByteTracker()
    frame1 = [_det(0, 0, 20, 40, conf=0.9)]
    frame2 = [_det(2, 1, 22, 41, conf=0.9)]  # small, plausible motion

    tracks1 = tracker.update(frame1)
    tracks2 = tracker.update(frame2)

    assert len(tracks1) == 1
    assert len(tracks2) == 1
    assert tracks1[0].track_id == tracks2[0].track_id


def test_two_simultaneous_objects_get_distinct_ids():
    tracker = ByteTracker()
    frame = [_det(0, 0, 20, 40, conf=0.9), _det(200, 0, 220, 40, conf=0.9)]

    tracks = tracker.update(frame)

    assert len(tracks) == 2
    assert tracks[0].track_id != tracks[1].track_id


def test_new_object_appearing_mid_sequence_gets_a_new_id():
    tracker = ByteTracker()
    tracker.update([_det(0, 0, 20, 40, conf=0.9)])
    existing_id = tracker.update([_det(1, 1, 21, 41, conf=0.9)])[0].track_id

    # a second object shows up, far from the first
    tracks = tracker.update([_det(2, 1, 22, 41, conf=0.9), _det(300, 300, 320, 340, conf=0.9)])

    ids = {t.track_id for t in tracks}
    assert existing_id in ids
    assert len(ids) == 2


# --- ByteTracker: occlusion / brief disappearance ---------------------------


def test_track_survives_a_brief_gap_within_max_age():
    tracker = ByteTracker(max_age=3, min_hits=1)
    tracker.update([_det(0, 0, 20, 40, conf=0.9)])
    track_id = tracker.update([_det(1, 1, 21, 41, conf=0.9)])[0].track_id

    # object vanishes for 2 frames (e.g. occluded) — within max_age=3
    tracker.update([])
    tracker.update([])

    # reappears close to its predicted (Kalman-extrapolated) position
    tracks = tracker.update([_det(4, 3, 24, 43, conf=0.9)])

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id


def test_track_is_dropped_after_exceeding_max_age():
    tracker = ByteTracker(max_age=2, min_hits=1)
    tracker.update([_det(0, 0, 20, 40, conf=0.9)])
    tracker.update([_det(1, 1, 21, 41, conf=0.9)])
    assert tracker.active_track_count == 1

    # vanish for longer than max_age
    tracker.update([])
    tracker.update([])
    tracker.update([])

    assert tracker.active_track_count == 0


def test_low_confidence_detection_recovers_an_occluded_track():
    """
    The actual BYTE idea: a detection below track_thresh shouldn't start a
    new track on its own, but SHOULD be able to rescue a track that
    stage-2 (high-confidence) matching left unmatched this frame.
    """
    tracker = ByteTracker(track_thresh=0.5, match_thresh_low=0.1, max_age=3, min_hits=1)
    tracker.update([_det(0, 0, 20, 40, conf=0.9)])
    track_id = tracker.update([_det(1, 1, 21, 41, conf=0.9)])[0].track_id

    # this frame's detection is real but low-confidence (e.g. motion blur)
    low_conf_detection = _det(3, 2, 23, 42, conf=0.2)
    tracks = tracker.update([low_conf_detection])

    assert len(tracks) == 1
    assert tracks[0].track_id == track_id
    assert tracks[0].time_since_update == 0  # recovered, not just predicted


def test_low_confidence_detection_alone_never_starts_a_new_track():
    tracker = ByteTracker(track_thresh=0.5, match_thresh_low=0.1)
    tracks = tracker.update([_det(0, 0, 20, 40, conf=0.2)])
    assert tracks == []
    assert tracker.active_track_count == 0


def test_detection_below_match_thresh_low_is_ignored_entirely():
    tracker = ByteTracker(track_thresh=0.5, match_thresh_low=0.1)
    tracks = tracker.update([_det(0, 0, 20, 40, conf=0.05)])
    assert tracks == []
    assert tracker.active_track_count == 0


# --- ByteTracker: class handling -------------------------------------------


def test_tracks_never_match_across_classes():
    """A ball-class detection should never be assigned to a player track, even at perfect IoU overlap."""
    tracker = ByteTracker()
    tracker.update([_det(0, 0, 20, 40, conf=0.9, class_id=PERSON, name="person")])
    tracker.update([_det(0, 0, 20, 40, conf=0.9, class_id=PERSON, name="person")])

    tracks = tracker.update([_det(0, 0, 20, 40, conf=0.9, class_id=BALL, name="sports ball")])

    # two tracks now: the untouched (aged) person track's successor never
    # shows up this frame (unmatched, predicted only) and a brand new ball track
    class_ids = {t.class_id for t in tracks}
    assert BALL in class_ids
    assert PERSON not in class_ids  # person track went unmatched this frame, not reported (time_since_update > 0)


def test_min_hits_hides_a_track_until_confirmed():
    tracker = ByteTracker(min_hits=2)
    tracks_frame1 = tracker.update([_det(0, 0, 20, 40, conf=0.9)])
    assert tracks_frame1 == []  # only 1 hit so far, min_hits=2

    tracks_frame2 = tracker.update([_det(1, 1, 21, 41, conf=0.9)])
    assert len(tracks_frame2) == 1  # now confirmed


# --- FrameTracks / update_frame / update_many -------------------------------


def test_update_frame_preserves_frame_path():
    tracker = ByteTracker()
    fd = FrameDetections(frame_path="frames/frame_000001.jpg", detections=[_det(0, 0, 20, 40, conf=0.9)])

    result = tracker.update_frame(fd)

    assert isinstance(result, FrameTracks)
    assert result.frame_path == "frames/frame_000001.jpg"
    assert len(result) == 1


def test_frame_tracks_of_class_filters():
    tracker = ByteTracker()
    fd = FrameDetections(
        frame_path=None,
        detections=[
            _det(0, 0, 20, 40, conf=0.9, class_id=PERSON, name="person"),
            _det(50, 50, 55, 55, conf=0.9, class_id=BALL, name="sports ball"),
        ],
    )
    result = tracker.update_frame(fd)

    assert len(result.of_class(PERSON)) == 1
    assert len(result.of_class(BALL)) == 1


def test_update_many_maintains_identity_across_the_whole_sequence():
    tracker = ByteTracker()
    frames = [
        FrameDetections(frame_path=f"f{i}.jpg", detections=[_det(i, i, i + 20, i + 40, conf=0.9)])
        for i in range(5)
    ]

    results = tracker.update_many(frames)

    assert len(results) == 5
    ids = {t.track_id for r in results for t in r.tracks}
    assert ids == {results[0].tracks[0].track_id}  # exactly one identity, used throughout


# --- to_serializable ---------------------------------------------------------


def test_to_serializable_round_trips_expected_shape():
    tracker = ByteTracker()
    fd = FrameDetections(frame_path="f0.jpg", detections=[_det(0, 0, 20, 40, conf=0.9, class_id=PERSON, name="person")])
    result = tracker.update_frame(fd)

    serialized = to_serializable([result])

    assert serialized == [
        {
            "frame_path": "f0.jpg",
            "tracks": [
                {
                    "track_id": serialized[0]["tracks"][0]["track_id"],
                    "class_id": PERSON,
                    "class_name": "person",
                    "confidence": pytest.approx(0.9),
                    "bbox": {
                        "x1": pytest.approx(0.0, abs=1e-6),
                        "y1": pytest.approx(0.0, abs=1e-6),
                        "x2": pytest.approx(20.0, abs=1e-6),
                        "y2": pytest.approx(40.0, abs=1e-6),
                    },
                    "age": 1,
                    "hits": 1,
                    "time_since_update": 0,
                }
            ],
        }
    ]


# --- TrackedDetection.is_predicted ------------------------------------------


def test_is_predicted_reflects_time_since_update():
    tracker = ByteTracker(max_age=3, min_hits=1)
    tracker.update([_det(0, 0, 20, 40, conf=0.9)])
    track = tracker.update([_det(1, 1, 21, 41, conf=0.9)])[0]
    assert not track.is_predicted

    # For a track to show up as "predicted" in the returned list it would
    # need time_since_update == 0 per update()'s own filter — so
    # is_predicted is only ever observable false on returned tracks;
    # verified via internal state instead for the "gap" case.
    tracker.update([])  # object vanishes
    internal_track = tracker._tracks[0]
    assert internal_track.time_since_update == 1


# --- known limitation: fast, small objects vs. bbox-IoU gating -------------


def test_realistic_object_motion_keeps_identity_across_occlusion():
    """
    A small, fast object (the ball) whose per-frame displacement is modest
    relative to its own box size keeps its identity across a brief
    occlusion — same guarantee test_track_survives_a_brief_gap already
    covers, restated with a moving (not static) object to confirm the
    Kalman motion model, not just the IoU gate on a stationary box, is
    doing real work.
    """
    tracker = ByteTracker(max_age=3, min_hits=1)
    ball_id = None
    for i in range(4):
        x = 100 + i * 3  # modest displacement relative to an 8px box
        tracks = tracker.update([_det(x, 150, x + 8, 158, conf=0.6, class_id=BALL, name="sports ball")])
        ball_id = tracks[0].track_id

    tracker.update([])  # occluded
    tracker.update([])  # still occluded

    x = 100 + 6 * 3
    tracks = tracker.update([_det(x, 150, x + 8, 158, conf=0.6, class_id=BALL, name="sports ball")])
    assert len(tracks) == 1
    assert tracks[0].track_id == ball_id


def test_fast_small_object_loses_identity_when_displacement_exceeds_iou_gate():
    """
    Documents a real, inherent limitation rather than leaving it as a
    silent trap: bbox-IoU association (this tracker, and bbox-IoU-based
    trackers generally) has no way to associate a detection with its
    predecessor once per-frame displacement exceeds the object's own box
    size — a single prior observation gives the Kalman filter no velocity
    estimate to extrapolate from (that needs >= 2 observations), so the
    predicted box IS the previous box, and an 8px box that moved 30px has
    zero IoU with where it used to be.

    This is exactly the risk PRD Section 13 names for ball tracking
    specifically: a small, fast object sampled at a low
    frame_sample_rate_fps (Part 5a) can easily move farther between
    *sampled* frames than its own size. Fixing it for real footage is
    Part 6b's problem (see ml/tracking/byte_tracker.py's module
    docstring), not this module's — this test exists so that if a future
    change to match_by_iou or the Kalman model accidentally "fixes" this
    case, it's a deliberate, reviewed change, not an unnoticed one.
    """
    tracker = ByteTracker(max_age=3, min_hits=1)
    tracker.update([_det(100, 150, 108, 158, conf=0.6, class_id=BALL, name="sports ball")])
    first_id = tracker.update([_det(103, 150, 111, 158, conf=0.6, class_id=BALL, name="sports ball")])[0].track_id

    # a 30px jump for an 8px-wide box — far larger than the object itself
    tracks = tracker.update([_det(133, 150, 141, 158, conf=0.6, class_id=BALL, name="sports ball")])

    assert len(tracks) == 1
    assert tracks[0].track_id != first_id, (
        "expected identity loss here — see docstring; if this now passes, "
        "the association/motion model changed and this test should be revisited"
    )
