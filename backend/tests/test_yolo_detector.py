"""
Tests for ml/detection/yolo_detector.py — Part 5b.

Two groups, same split the module itself uses:

- Pure-logic tests (parse_yolo_result, BoundingBox, detect_frame,
  YOLODetector) run against lightweight fakes that mimic the shape of
  ultralytics' `Results`/model API, so they exercise this module's own
  control flow — box parsing, class filtering, error wrapping — without
  ultralytics installed and without downloading any real weights. These
  always run.

- `load_model`/`get_cached_model` integration tests need the real
  ultralytics package installed and, the first time, network access to
  download yolov8n.pt — neither is guaranteed in every environment this
  suite runs in (e.g. an offline CI worker). They're skipped via
  `pytest.importorskip` / a try-except-skip around the actual load rather
  than mocked, following the same "prefer the real thing when it's
  available" approach test_frame_extraction.py takes with real ffmpeg.

Run with: pytest backend/tests/test_yolo_detector.py -v
"""

from __future__ import annotations

import sys

import pytest

from app.core.ml_path import ensure_ml_importable

ensure_ml_importable()
from ml.detection.yolo_detector import (  # noqa: E402
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    DEFAULT_CLASS_IDS,
    BoundingBox,
    Detection,
    DetectionError,
    FrameDetections,
    ModelLoadError,
    YOLODetector,
    detect_frame,
    detect_frames,
    load_model,
    parse_yolo_result,
)


# --- fakes: shaped like ultralytics' Results/model API, nothing more -------


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


class _FakeEmptyResult:
    """Mimics ultralytics when a frame has zero detections: boxes exists but is empty."""

    def __init__(self):
        self.boxes = _FakeBoxes([], [], [])
        self.names = {}


_COCO_LIKE_NAMES = {0: "person", 32: "sports ball"}


class _FakeModel:
    """Stands in for a loaded ultralytics model — records the kwargs it was called with."""

    def __init__(self, results):
        self._results = results
        self.calls: list[dict] = []

    def predict(self, frame, **kwargs):
        self.calls.append({"frame": frame, **kwargs})
        return self._results


class _RaisingModel:
    def predict(self, frame, **kwargs):
        raise RuntimeError("simulated inference failure")


# --- BoundingBox -------------------------------------------------------------


def test_bounding_box_derives_width_height_center():
    box = BoundingBox(x1=10.0, y1=20.0, x2=30.0, y2=60.0)
    assert box.width == 20.0
    assert box.height == 40.0
    assert box.center == (20.0, 40.0)


# --- parse_yolo_result: pure, no ultralytics/model required -----------------


def test_parse_yolo_result_builds_detections_from_boxes():
    result = _FakeResult(
        xyxy=[[10.0, 10.0, 50.0, 90.0], [100.0, 100.0, 120.0, 120.0]],
        conf=[0.9, 0.4],
        cls=[0, 32],
        names=_COCO_LIKE_NAMES,
    )

    detections = parse_yolo_result(result)

    assert len(detections) == 2
    assert detections[0] == Detection(
        class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(10.0, 10.0, 50.0, 90.0)
    )
    assert detections[1].class_name == "sports ball"


def test_parse_yolo_result_sorts_by_confidence_descending():
    result = _FakeResult(
        xyxy=[[0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 1, 1]],
        conf=[0.2, 0.95, 0.5],
        cls=[0, 0, 0],
        names=_COCO_LIKE_NAMES,
    )

    detections = parse_yolo_result(result)

    assert [d.confidence for d in detections] == [0.95, 0.5, 0.2]


def test_parse_yolo_result_handles_zero_detections():
    assert parse_yolo_result(_FakeEmptyResult()) == []


def test_parse_yolo_result_falls_back_to_class_id_for_unknown_class():
    result = _FakeResult(xyxy=[[0, 0, 1, 1]], conf=[0.5], cls=[99], names={0: "person"})

    detections = parse_yolo_result(result)

    assert detections[0].class_name == "99"


def test_parse_yolo_result_accepts_class_names_override():
    result = _FakeResult(xyxy=[[0, 0, 1, 1]], conf=[0.5], cls=[0], names={})

    detections = parse_yolo_result(result, class_names={0: "player"})

    assert detections[0].class_name == "player"


def test_parse_yolo_result_works_with_plain_lists_no_tensor_needed():
    """Confirms _as_list doesn't require a torch/numpy-style .tolist() — plain lists work too."""
    result = _FakeResult(xyxy=[[1, 2, 3, 4]], conf=[0.77], cls=[32], names=_COCO_LIKE_NAMES)
    detections = parse_yolo_result(result)
    assert detections[0].bbox == BoundingBox(1.0, 2.0, 3.0, 4.0)


# --- detect_frame -------------------------------------------------------------


def test_detect_frame_returns_frame_detections_for_path_input():
    model = _FakeModel([_FakeResult(xyxy=[[0, 0, 10, 10]], conf=[0.8], cls=[0], names=_COCO_LIKE_NAMES)])

    result = detect_frame(model, "frames/frame_000001.jpg")

    assert isinstance(result, FrameDetections)
    assert result.frame_path == "frames/frame_000001.jpg"
    assert len(result) == 1
    assert result.detections[0].class_name == "person"


def test_detect_frame_frame_path_is_none_for_array_input():
    model = _FakeModel([_FakeEmptyResult()])
    fake_ndarray_stand_in = [[0, 0, 0], [0, 0, 0]]  # anything that isn't a str/Path

    result = detect_frame(model, fake_ndarray_stand_in)

    assert result.frame_path is None
    assert result.detections == []


def test_detect_frame_passes_confidence_and_classes_to_model_predict():
    model = _FakeModel([_FakeEmptyResult()])

    detect_frame(model, "f.jpg", confidence_threshold=0.6, classes=(COCO_PERSON_CLASS_ID,))

    assert model.calls[0]["conf"] == 0.6
    assert model.calls[0]["classes"] == [COCO_PERSON_CLASS_ID]
    assert model.calls[0]["verbose"] is False


def test_detect_frame_omits_classes_kwarg_when_filter_disabled():
    model = _FakeModel([_FakeEmptyResult()])

    detect_frame(model, "f.jpg", classes=None)

    assert "classes" not in model.calls[0]


def test_detect_frame_default_classes_are_person_and_sports_ball():
    assert DEFAULT_CLASS_IDS == (COCO_PERSON_CLASS_ID, COCO_SPORTS_BALL_CLASS_ID)


def test_detect_frame_raises_detection_error_on_inference_failure():
    with pytest.raises(DetectionError, match="simulated inference failure"):
        detect_frame(_RaisingModel(), "f.jpg")


def test_detect_frame_handles_empty_results_list_from_model():
    model = _FakeModel([])  # model ran but returned no Results objects at all

    result = detect_frame(model, "f.jpg")

    assert result.detections == []


# --- detect_frames (batch) ----------------------------------------------------


def test_detect_frames_preserves_order_one_result_per_frame():
    model = _FakeModel([_FakeResult(xyxy=[[0, 0, 1, 1]], conf=[0.5], cls=[0], names=_COCO_LIKE_NAMES)])

    results = detect_frames(model, ["a.jpg", "b.jpg", "c.jpg"])

    assert [r.frame_path for r in results] == ["a.jpg", "b.jpg", "c.jpg"]
    assert all(len(r) == 1 for r in results)


def test_detect_frames_propagates_error_from_a_single_bad_frame():
    with pytest.raises(DetectionError):
        detect_frames(_RaisingModel(), ["a.jpg", "b.jpg"])


# --- FrameDetections.of_class -------------------------------------------------


def test_frame_detections_of_class_filters_by_class_id():
    frame = FrameDetections(
        frame_path="f.jpg",
        detections=[
            Detection(0, "person", 0.9, BoundingBox(0, 0, 1, 1)),
            Detection(32, "sports ball", 0.5, BoundingBox(0, 0, 1, 1)),
            Detection(0, "person", 0.7, BoundingBox(0, 0, 1, 1)),
        ],
    )

    players = frame.of_class(COCO_PERSON_CLASS_ID)

    assert len(players) == 2
    assert all(d.class_id == COCO_PERSON_CLASS_ID for d in players)


# --- YOLODetector: stateful wrapper, still using the fake model -------------


def test_yolo_detector_uses_configured_thresholds_and_classes():
    detector = YOLODetector(confidence_threshold=0.4, classes=(COCO_SPORTS_BALL_CLASS_ID,))
    fake_model = _FakeModel([_FakeEmptyResult()])
    detector._model = fake_model  # bypass load_model — this test is about config plumbing, not loading

    detector.detect("f.jpg")

    assert fake_model.calls[0]["conf"] == 0.4
    assert fake_model.calls[0]["classes"] == [COCO_SPORTS_BALL_CLASS_ID]


def test_yolo_detector_detect_many_delegates_to_detect_frames():
    detector = YOLODetector()
    fake_model = _FakeModel([_FakeResult(xyxy=[[0, 0, 1, 1]], conf=[0.5], cls=[0], names=_COCO_LIKE_NAMES)])
    detector._model = fake_model

    results = detector.detect_many(["a.jpg", "b.jpg"])

    assert len(results) == 2
    assert len(fake_model.calls) == 2


def test_yolo_detector_does_not_load_a_model_until_first_use():
    """Constructing a YOLODetector must not require ultralytics to be installed."""
    detector = YOLODetector()
    assert detector._model is None


# --- load_model: real ultralytics, best-effort ------------------------------


def test_load_model_raises_model_load_error_when_ultralytics_missing(monkeypatch):
    """
    Forces `import ultralytics` to raise ImportError regardless of whether
    it's actually installed in this environment (`sys.modules[name] = None`
    is the standard trick for that) so this test is deterministic either
    way. load_model must turn that into our own ModelLoadError — specific
    and catchable — not let a raw ImportError escape ml/ internals.
    """
    monkeypatch.setitem(sys.modules, "ultralytics", None)

    with pytest.raises(ModelLoadError, match="ultralytics is not installed"):
        load_model()


def test_load_model_real_checkpoint_or_skip():
    """
    Best-effort integration test: if ultralytics is installed and the
    yolov8n.pt checkpoint can be obtained (bundled or downloaded), loads a
    real model and confirms load_model()/get_cached_model() hand back
    something detect_frame can call .predict() on. Skips (does not fail)
    when either precondition isn't met, since neither ultralytics nor
    network access is guaranteed in every environment this suite runs in.
    """
    pytest.importorskip("ultralytics")
    try:
        model = load_model()
    except ModelLoadError as exc:
        pytest.skip(f"could not load a real YOLO checkpoint in this environment: {exc}")

    assert hasattr(model, "predict")
