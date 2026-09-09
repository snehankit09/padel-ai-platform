"""
Tests for ml/detection/player_ball_detection.py — Part 5d.

Same fake-model approach as test_yolo_detector.py (Part 5b): a lightweight
stand-in for ultralytics' predict()/Results API, so these exercise this
module's own control flow — the per-class threshold split, the
single-inference-call-at-the-min-threshold optimization, serialization —
without ultralytics installed and without downloading any real weights.
Nothing here needs a real model; see test_yolo_detector.py's own
load_model/get_cached_model tests for the (skippable) real-weights path.

Run with: pytest backend/tests/test_player_ball_detection.py -v
"""

from __future__ import annotations

import json

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.detection.player_ball_detection import (  # noqa: E402
    DEFAULT_BALL_CONFIDENCE_THRESHOLD,
    DEFAULT_PLAYER_CONFIDENCE_THRESHOLD,
    FramePlayerBallDetections,
    detect_players_and_ball,
    detect_players_and_ball_many,
    split_player_ball_detections,
    summarize_ball_detection_rate,
    to_serializable,
)
from ml.detection.yolo_detector import (  # noqa: E402
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    BoundingBox,
    Detection,
    FrameDetections,
)

_PERSON = Detection(class_id=COCO_PERSON_CLASS_ID, class_name="person", confidence=0.9, bbox=BoundingBox(0, 0, 10, 10))
_BALL = Detection(class_id=COCO_SPORTS_BALL_CLASS_ID, class_name="sports ball", confidence=0.5, bbox=BoundingBox(1, 1, 2, 2))


# --- fakes: shaped like ultralytics' model API, nothing more ---------------


class _FakeBoxes:
    def __init__(self, xyxy, conf, cls):
        self.xyxy = xyxy
        self.conf = conf
        self.cls = cls

    def __len__(self):
        return len(self.cls)


class _FakeResult:
    def __init__(self, xyxy, conf, cls, names):
        self.boxes = _FakeBoxes(xyxy, conf, cls)
        self.names = names


_COCO_LIKE_NAMES = {COCO_PERSON_CLASS_ID: "person", COCO_SPORTS_BALL_CLASS_ID: "sports ball"}


class _FakeModel:
    """Records the kwargs it was called with, so tests can assert on the conf/classes passed to predict()."""

    def __init__(self, result):
        self._result = result
        self.calls: list[dict] = []

    def predict(self, frame, **kwargs):
        self.calls.append({"frame": frame, **kwargs})
        return [self._result]


# --- split_player_ball_detections: pure, no model/inference -----------------


def test_split_separates_by_class():
    frame_detections = FrameDetections(frame_path="f.jpg", detections=[_PERSON, _BALL])

    split = split_player_ball_detections(frame_detections)

    assert split.players == [_PERSON]
    assert split.balls == [_BALL]
    assert split.player_count == 1
    assert split.ball_count == 1
    assert split.has_ball is True


def test_split_applies_independent_per_class_thresholds():
    low_conf_ball = Detection(class_id=COCO_SPORTS_BALL_CLASS_ID, class_name="sports ball", confidence=0.15, bbox=BoundingBox(0, 0, 1, 1))
    low_conf_player = Detection(class_id=COCO_PERSON_CLASS_ID, class_name="person", confidence=0.15, bbox=BoundingBox(0, 0, 1, 1))
    frame_detections = FrameDetections(frame_path=None, detections=[low_conf_ball, low_conf_player])

    # 0.15 clears the (lower) default ball threshold but not the player one.
    split = split_player_ball_detections(
        frame_detections,
        player_confidence_threshold=0.25,
        ball_confidence_threshold=0.10,
    )

    assert split.balls == [low_conf_ball]
    assert split.players == []


def test_split_keeps_every_ball_candidate_not_just_the_best():
    ball_a = Detection(class_id=COCO_SPORTS_BALL_CLASS_ID, class_name="sports ball", confidence=0.9, bbox=BoundingBox(0, 0, 1, 1))
    ball_b = Detection(class_id=COCO_SPORTS_BALL_CLASS_ID, class_name="sports ball", confidence=0.11, bbox=BoundingBox(5, 5, 6, 6))
    frame_detections = FrameDetections(frame_path=None, detections=[ball_a, ball_b])

    split = split_player_ball_detections(frame_detections)

    assert split.ball_count == 2
    assert ball_a in split.balls and ball_b in split.balls


def test_frame_has_no_ball_when_none_clears_threshold():
    frame_detections = FrameDetections(frame_path=None, detections=[_PERSON])
    split = split_player_ball_detections(frame_detections)
    assert split.has_ball is False
    assert split.ball_count == 0


# --- detect_players_and_ball: single inference call, correct conf ----------


def test_runs_single_predict_call_at_the_lower_threshold():
    result = _FakeResult(xyxy=[[0, 0, 1, 1]], conf=[0.9], cls=[COCO_PERSON_CLASS_ID], names=_COCO_LIKE_NAMES)
    model = _FakeModel(result)

    detect_players_and_ball(
        model, "frame.jpg",
        player_confidence_threshold=0.25,
        ball_confidence_threshold=0.10,
    )

    assert len(model.calls) == 1, "must not call predict() twice per frame"
    assert model.calls[0]["conf"] == 0.10  # the min of the two thresholds


def test_default_thresholds_are_asymmetric_ball_lower_than_player():
    # The whole reason this module exists rather than one shared threshold.
    assert DEFAULT_BALL_CONFIDENCE_THRESHOLD < DEFAULT_PLAYER_CONFIDENCE_THRESHOLD


def test_end_to_end_split_from_fake_model_result():
    result = _FakeResult(
        xyxy=[[0, 0, 10, 10], [20, 20, 30, 30], [5, 5, 7, 7]],
        conf=[0.9, 0.8, 0.12],
        cls=[COCO_PERSON_CLASS_ID, COCO_PERSON_CLASS_ID, COCO_SPORTS_BALL_CLASS_ID],
        names=_COCO_LIKE_NAMES,
    )
    model = _FakeModel(result)

    detections = detect_players_and_ball(model, "frame.jpg")

    assert detections.player_count == 2
    assert detections.ball_count == 1
    assert detections.frame_path == "frame.jpg"


def test_ball_below_even_the_low_threshold_is_dropped():
    result = _FakeResult(
        xyxy=[[0, 0, 1, 1]], conf=[0.05], cls=[COCO_SPORTS_BALL_CLASS_ID], names=_COCO_LIKE_NAMES,
    )
    model = _FakeModel(result)

    # min-threshold inference conf is 0.05 or lower here, so ultralytics
    # itself wouldn't drop it — but the ball threshold post-filter should.
    detections = detect_players_and_ball(model, "frame.jpg", ball_confidence_threshold=0.10)

    assert detections.ball_count == 0


# --- detect_players_and_ball_many + summarize_ball_detection_rate ----------


def test_detect_many_preserves_frame_order():
    result_with_ball = _FakeResult(xyxy=[[0, 0, 1, 1]], conf=[0.9], cls=[COCO_SPORTS_BALL_CLASS_ID], names=_COCO_LIKE_NAMES)
    model = _FakeModel(result_with_ball)

    results = detect_players_and_ball_many(model, ["a.jpg", "b.jpg", "c.jpg"])

    assert [r.frame_path for r in results] == ["a.jpg", "b.jpg", "c.jpg"]


def test_summarize_ball_detection_rate_counts_frames_with_at_least_one_ball():
    with_ball = FramePlayerBallDetections(frame_path="a", players=[], balls=[_BALL])
    without_ball = FramePlayerBallDetections(frame_path="b", players=[_PERSON], balls=[])

    rate = summarize_ball_detection_rate([with_ball, without_ball, with_ball])

    assert rate == 2 / 3


def test_summarize_ball_detection_rate_empty_input_is_zero_not_error():
    assert summarize_ball_detection_rate([]) == 0.0


def test_summarize_ball_detection_rate_all_frames_have_ball():
    with_ball = FramePlayerBallDetections(frame_path="a", players=[], balls=[_BALL])
    assert summarize_ball_detection_rate([with_ball, with_ball]) == 1.0


# --- to_serializable: JSON-safe, round-trippable ----------------------------


def test_to_serializable_is_valid_json():
    frame = FramePlayerBallDetections(frame_path="f.jpg", players=[_PERSON], balls=[_BALL])

    data = to_serializable([frame])
    raw = json.dumps(data)
    round_tripped = json.loads(raw)

    assert round_tripped[0]["frame_path"] == "f.jpg"
    assert round_tripped[0]["players"][0]["class_name"] == "person"
    assert round_tripped[0]["players"][0]["confidence"] == 0.9
    assert round_tripped[0]["balls"][0]["bbox"] == {"x1": 1.0, "y1": 1.0, "x2": 2.0, "y2": 2.0}


def test_to_serializable_handles_empty_results_list():
    assert to_serializable([]) == []
