"""
YOLO model setup & inference wrapper — Part 5b.

The `detect` pipeline stage (Part 5c/5d) needs to run court/player/ball
detection against every sampled frame of a match — potentially thousands
of frames per video (see ml/common/frame_extraction.py). This module is
the reusable core both of those sub-parts call: load a model once, then
frame in -> detections out, with a plain, versioned data shape in between
so nothing downstream needs to know anything about ultralytics' own
`Results` objects.

Like ml/common/frame_extraction.py (Part 5a), this module has no
Celery/DB/app dependency on purpose — it's usable from a one-off
debugging script or a notebook, not just from inside the worker
container. The eventual `app/services/detection_stage.py` glue (mirroring
app/services/frame_extraction_stage.py) is what will wire this to a
specific Video row, its extracted frames directory, and settings from
app/core/config.py.

Model choice for now: a pretrained (COCO) Ultralytics YOLO checkpoint,
not yet fine-tuned on padel footage (PRD Section 7 lists this as a named
component; PRD Section 13 Risks calls out "Dataset availability" as the
reason it's pretrained-first). COCO doesn't have "court" as a class at
all, and its only ball class is the generic "sports ball" — which is why
DEFAULT_CLASS_IDS restricts inference to person + sports ball rather than
returning COCO's other 78 classes as noise. `court` detection needs a
different approach entirely (calibration, not object detection — PRD 5.2
"Court detection and calibration") and isn't this module's job.

Two layers, same split as frame_extraction.py's "pure command builder vs.
subprocess runner":
  - `parse_yolo_result` is pure — no model, no I/O, no ultralytics import
    needed to call it — so the numeric/shape logic (box parsing, class-name
    lookup, confidence sorting) is unit-testable against a lightweight
    stand-in for ultralytics' `Results`, without ultralytics installed.
  - `load_model` / `detect_frame` do the real work, and only import
    ultralytics inside the function body (deferred, same reasoning as
    app/core/ml_path.py: this module must stay importable — for the pure
    functions and for tests — even in an environment where ultralytics
    isn't installed yet).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

# Small pretrained COCO checkpoint — good enough to prototype the
# frame-in/detections-out contract without waiting on a padel-specific
# fine-tune. Swappable per-call; nothing here hardcodes this beyond the
# default.
DEFAULT_MODEL_NAME = "yolov8n.pt"
DEFAULT_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_DEVICE = "cpu"

# COCO class ids this platform cares about today. person -> players,
# sports ball -> the padel ball (closest COCO analogue; a fine-tuned
# padel-specific model will replace this mapping entirely, not extend it).
COCO_PERSON_CLASS_ID = 0
COCO_SPORTS_BALL_CLASS_ID = 32
DEFAULT_CLASS_IDS: tuple[int, ...] = (COCO_PERSON_CLASS_ID, COCO_SPORTS_BALL_CLASS_ID)


class ModelLoadError(Exception):
    """Raised when the YOLO model/weights can't be loaded or placed on a device."""


class DetectionError(Exception):
    """Raised when a loaded model fails to run inference on a given frame."""


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in the source frame's pixel coordinates (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


@dataclass(frozen=True)
class Detection:
    """One detected object: what it is, how confident the model is, and where it is."""

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox


@dataclass(frozen=True)
class FrameDetections:
    """
    Every detection found in one frame. `frame_path` is None when the input
    was an already-decoded array (e.g. read via cv2.VideoCapture) rather
    than a path on disk — kept on the result so downstream code (tracking,
    Part 6) can trace a detection back to its source frame when one exists,
    without forcing every caller to have a path in the first place.
    """

    frame_path: str | None
    detections: list[Detection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.detections)

    def of_class(self, class_id: int) -> list[Detection]:
        """Convenience filter — e.g. `frame_detections.of_class(COCO_PERSON_CLASS_ID)`."""
        return [d for d in self.detections if d.class_id == class_id]


def parse_yolo_result(result: Any, class_names: dict[int, str] | None = None) -> list[Detection]:
    """
    Turns one ultralytics `Results` object — or anything shaped like one,
    which is what makes this testable without ultralytics installed — into
    a list of Detection dataclasses, sorted by confidence, highest first.

    Expects `result.boxes` with `.xyxy`, `.conf`, `.cls` (each a torch
    Tensor in real usage, but only needs to support `.tolist()` or plain
    iteration — see `_as_list`), and `result.names` (a {class_id: name}
    dict) unless `class_names` is passed explicitly to override it.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    names = class_names if class_names is not None else getattr(result, "names", {})

    xyxy = _as_list(boxes.xyxy)
    confs = _as_list(boxes.conf)
    cls_ids = _as_list(boxes.cls)

    detections = [
        Detection(
            class_id=int(cls_id),
            class_name=_lookup_class_name(names, int(cls_id)),
            confidence=float(conf),
            bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2)),
        )
        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids)
    ]
    detections.sort(key=lambda d: d.confidence, reverse=True)
    return detections


def _as_list(tensor_like: Any) -> list:
    """
    Accepts a torch.Tensor, numpy array, or plain nested list and returns a
    plain Python list either way — keeps parse_yolo_result callable from
    tests with plain lists, with no torch import required just to exercise
    the parsing logic.
    """
    if hasattr(tensor_like, "tolist"):
        return tensor_like.tolist()
    return list(tensor_like)


def _lookup_class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    try:
        return str(names[class_id])
    except (IndexError, TypeError, KeyError):
        return str(class_id)


def load_model(model_path: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE):
    """
    Loads an ultralytics YOLO model from `model_path` (a bundled checkpoint
    name like "yolov8n.pt", or a path to a local/fine-tuned .pt file — both
    are accepted directly by ultralytics' own `YOLO(...)` constructor) and
    places it on `device` ("cpu", "cuda", "cuda:0", ...).

    ultralytics is imported here, not at module level, so the rest of this
    module — including every pure function above — stays importable in an
    environment where ultralytics isn't installed yet (see module
    docstring). Calling *this* function is what actually requires it to be
    present.

    Raises ModelLoadError with a specific reason on any failure: missing
    dependency, unreadable/corrupt weights file, or a device that doesn't
    exist on this machine — never returns a partially-initialized model.
    """
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ModelLoadError(
            "ultralytics is not installed. Add it to backend/requirements.txt "
            "and install it in this environment before loading a YOLO model."
        ) from exc

    try:
        model = YOLO(model_path)
    except Exception as exc:
        raise ModelLoadError(f"failed to load YOLO model from {model_path!r}: {exc}") from exc

    try:
        model.to(device)
    except Exception as exc:
        raise ModelLoadError(f"failed to place YOLO model on device {device!r}: {exc}") from exc

    return model


@lru_cache(maxsize=4)
def get_cached_model(model_path: str = DEFAULT_MODEL_NAME, device: str = DEFAULT_DEVICE):
    """
    Process-wide cache around load_model, keyed on (model_path, device).
    Loading a checkpoint means reading weights off disk and placing them on
    a device — worth doing once per worker process, not once per frame, for
    the same reason app/core/config.py's get_settings() is @lru_cache'd
    rather than re-parsed on every call. `detect` (Part 5c/5d) is expected
    to call this rather than `load_model` directly whenever it just wants
    "the model", and to call `load_model` directly only when it deliberately
    wants a fresh, uncached instance (e.g. a test).
    """
    return load_model(model_path, device)


def detect_frame(
    model,
    frame,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    classes: Sequence[int] | None = DEFAULT_CLASS_IDS,
) -> FrameDetections:
    """
    The reusable core: frame in -> detections out.

    `model` is anything with a `.predict(frame, conf=..., classes=...,
    verbose=False)` method returning a sequence of Results-shaped objects —
    in production, a real ultralytics model from load_model()/
    get_cached_model(); in tests, a lightweight fake (see
    tests/test_yolo_detector.py), so this function's control flow is
    testable without ever loading real weights.

    `frame` is either a path to an image on disk (str/Path) or an
    already-decoded frame (e.g. a numpy ndarray from cv2.VideoCapture) —
    both are accepted as-is by ultralytics' own `predict()`, so no
    conversion happens here.

    `classes` restricts detection to specific class ids (default: person +
    sports ball, see DEFAULT_CLASS_IDS) so the detect stage isn't sifting
    padel-irrelevant COCO classes out of every frame's result downstream.
    Pass None to disable the filter and get every class the model knows.

    Raises DetectionError if inference itself fails (corrupt frame data, a
    device mismatch, etc.) rather than returning an empty result — the same
    "don't silently succeed at nothing" stance extract_frames takes on zero
    frames (Part 5a), for the same reason: a quietly-empty detection result
    looks identical to "this frame genuinely has nothing in it" to every
    caller downstream, and those are very different failure conditions.
    """
    frame_path = str(frame) if isinstance(frame, (str, Path)) else None

    predict_kwargs: dict[str, Any] = {"conf": confidence_threshold, "verbose": False}
    if classes is not None:
        predict_kwargs["classes"] = list(classes)

    try:
        results = model.predict(frame, **predict_kwargs)
    except Exception as exc:
        raise DetectionError(f"YOLO inference failed for {frame_path or 'frame'}: {exc}") from exc

    if not results:
        return FrameDetections(frame_path=frame_path, detections=[])

    return FrameDetections(frame_path=frame_path, detections=parse_yolo_result(results[0]))


def detect_frames(
    model,
    frames: Sequence,
    *,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    classes: Sequence[int] | None = DEFAULT_CLASS_IDS,
) -> list[FrameDetections]:
    """
    Runs detect_frame across a sequence of frames (e.g. every path returned
    by ml.common.frame_extraction.extract_frames) and returns one
    FrameDetections per frame, in the same order.

    Deliberately a plain per-frame loop rather than a single batched
    `model.predict(list_of_frames)` call — ultralytics supports batching,
    but correctness of the frame-in/detections-out contract matters more
    than throughput at this stage (Part 5b). Batching is a follow-up once
    real match footage shows it's worth the added complexity (chunking,
    partial-batch failure handling, ...). Each frame stays independent: one
    bad frame raises DetectionError immediately rather than silently
    dropping it and shifting every later frame's index in the result.
    """
    return [
        detect_frame(model, frame, confidence_threshold=confidence_threshold, classes=classes)
        for frame in frames
    ]


class YOLODetector:
    """
    Stateful convenience wrapper around load_model + detect_frame, for
    callers that want to load a model once and run many frames through it —
    the normal case, since `detect` processes every sampled frame of a
    match (potentially thousands), not just one. This is the shape Part
    5c/5d and the eventual detect_stage glue (mirroring
    app/services/frame_extraction_stage.py) are expected to use directly.

    Layered on top of the plain functions above rather than replacing them:
    load_model/detect_frame/parse_yolo_result stay usable standalone (a
    one-off debugging script doesn't need to instantiate this class), and
    everything this class does is just those calls plus remembering the
    loaded model and the detection config between calls.

    The model is loaded lazily, on first use, not in __init__ — so
    constructing a YOLODetector (e.g. to read its default config in a test)
    never requires ultralytics to be installed unless detection is actually
    run.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_NAME,
        device: str = DEFAULT_DEVICE,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        classes: Sequence[int] | None = DEFAULT_CLASS_IDS,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.classes = tuple(classes) if classes is not None else None
        self._model: Any = None

    @property
    def model(self):
        """Loads (once) and returns the underlying ultralytics model, via the process-wide cache."""
        if self._model is None:
            self._model = get_cached_model(self.model_path, self.device)
        return self._model

    def detect(self, frame) -> FrameDetections:
        """Runs detection on a single frame (path or decoded array) using this detector's config."""
        return detect_frame(
            self.model,
            frame,
            confidence_threshold=self.confidence_threshold,
            classes=self.classes,
        )

    def detect_many(self, frames: Sequence) -> list[FrameDetections]:
        """Runs detection on a sequence of frames using this detector's config. See detect_frames."""
        return detect_frames(
            self.model,
            frames,
            confidence_threshold=self.confidence_threshold,
            classes=self.classes,
        )
