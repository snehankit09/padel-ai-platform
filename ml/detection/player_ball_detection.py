"""
Player & ball detection across a video's extracted frames — Part 5d.

Layered on top of ml/detection/yolo_detector.py (Part 5b) rather than
duplicating it: this module doesn't run its own model or make its own
inference call — it runs yolo_detector.detect_frame once per frame and
splits *that* frame's detections into players vs. ball.

Why split at all, rather than leaving callers to call
FrameDetections.of_class() twice at the call site: the two targets aren't
equally easy to find, and a single confidence threshold shared between
them forces a bad tradeoff. A padel ball is small, fast, and easily
motion-blurred or occluded by a player, the net, or the court's glass
walls (PRD Section 13 risks: ball detection accuracy, occlusion). Players
are large and comparatively slow-moving, and rarely fully hidden.
Requiring the SAME confidence threshold for both means either:
  - a threshold high enough to keep player detections clean, which misses
    real ball detections the model was only ever moderately confident
    about (the common case, per the PRD's own framing of this as the
    hardest part), or
  - a threshold low enough to catch those ball detections, which lets
    noisy, low-confidence "player" boxes through too.
Two independent thresholds avoid that tradeoff, at no extra inference
cost: detect_players_and_ball still makes exactly one
yolo_detector.detect_frame call per frame, at the *lower* of the two
thresholds (so ultralytics never drops something either class threshold
would have wanted to keep), then applies each class's real threshold
afterward as a plain filter — see detect_players_and_ball and
split_player_ball_detections.

Also deliberately keeps every ball candidate that clears the ball
threshold, not just the single highest-confidence one. A frame can have
zero, one, or (false positive) more than one candidate. Collapsing to
"best guess per frame" here would throw away exactly the information
Part 6's tracker needs to tell a real ball from a false positive using
motion continuity across frames — that's tracking's job, not detection's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ml.detection.yolo_detector import (
    COCO_PERSON_CLASS_ID,
    COCO_SPORTS_BALL_CLASS_ID,
    DEFAULT_CLASS_IDS,
    Detection,
    FrameDetections,
    detect_frame,
)

DEFAULT_PLAYER_CONFIDENCE_THRESHOLD = 0.25
# Lower than the player threshold on purpose — see module docstring. A
# missed ball detection loses that frame's trajectory data outright; a
# spurious low-confidence one is just noise Part 6's tracker can filter
# using motion continuity across frames. The asymmetric cost of the two
# mistakes is why this isn't the same number as the player threshold.
DEFAULT_BALL_CONFIDENCE_THRESHOLD = 0.10


@dataclass(frozen=True)
class FramePlayerBallDetections:
    """One frame's detections, already split into players vs. ball candidates."""

    frame_path: str | None
    players: list[Detection] = field(default_factory=list)
    balls: list[Detection] = field(default_factory=list)

    @property
    def player_count(self) -> int:
        return len(self.players)

    @property
    def ball_count(self) -> int:
        return len(self.balls)

    @property
    def has_ball(self) -> bool:
        return self.ball_count > 0


def split_player_ball_detections(
    frame_detections: FrameDetections,
    *,
    player_confidence_threshold: float = DEFAULT_PLAYER_CONFIDENCE_THRESHOLD,
    ball_confidence_threshold: float = DEFAULT_BALL_CONFIDENCE_THRESHOLD,
) -> FramePlayerBallDetections:
    """
    Splits one frame's already-run detections by class and per-class
    confidence threshold. Pure — no model, no inference — so this half of
    the module's logic is unit-testable against plain Detection objects,
    the same way yolo_detector.parse_yolo_result is testable without a
    real model (see that module's docstring for the same reasoning).
    """
    players = [
        d for d in frame_detections.of_class(COCO_PERSON_CLASS_ID)
        if d.confidence >= player_confidence_threshold
    ]
    balls = [
        d for d in frame_detections.of_class(COCO_SPORTS_BALL_CLASS_ID)
        if d.confidence >= ball_confidence_threshold
    ]
    return FramePlayerBallDetections(frame_path=frame_detections.frame_path, players=players, balls=balls)


def detect_players_and_ball(
    model,
    frame,
    *,
    player_confidence_threshold: float = DEFAULT_PLAYER_CONFIDENCE_THRESHOLD,
    ball_confidence_threshold: float = DEFAULT_BALL_CONFIDENCE_THRESHOLD,
) -> FramePlayerBallDetections:
    """
    One frame in -> players + ball out. Makes exactly one
    yolo_detector.detect_frame call, at the lower of the two thresholds, so
    nothing either class needs gets dropped by ultralytics before this
    function ever sees it — then splits and re-filters per class. See
    module docstring for why this is one call at the min threshold rather
    than two calls at two thresholds.
    """
    inference_threshold = min(player_confidence_threshold, ball_confidence_threshold)
    frame_detections = detect_frame(
        model, frame, confidence_threshold=inference_threshold, classes=DEFAULT_CLASS_IDS
    )
    return split_player_ball_detections(
        frame_detections,
        player_confidence_threshold=player_confidence_threshold,
        ball_confidence_threshold=ball_confidence_threshold,
    )


def detect_players_and_ball_many(
    model,
    frames: Sequence,
    *,
    player_confidence_threshold: float = DEFAULT_PLAYER_CONFIDENCE_THRESHOLD,
    ball_confidence_threshold: float = DEFAULT_BALL_CONFIDENCE_THRESHOLD,
) -> list[FramePlayerBallDetections]:
    """
    Runs detect_players_and_ball across every frame, in order. Plain loop,
    not a batched predict() call — same reasoning as
    yolo_detector.detect_frames: one bad frame raises immediately rather
    than silently dropping it and shifting every later frame's index.
    """
    return [
        detect_players_and_ball(
            model, frame,
            player_confidence_threshold=player_confidence_threshold,
            ball_confidence_threshold=ball_confidence_threshold,
        )
        for frame in frames
    ]


def summarize_ball_detection_rate(results: Sequence[FramePlayerBallDetections]) -> float:
    """
    Fraction of frames with at least one ball candidate. Not a quality
    metric by itself (a wrong-but-present detection still counts), but a
    cheap, immediately-loggable signal for exactly the failure mode PRD
    Section 13 calls out as the main risk here — a rate that's
    unexpectedly low for a given match points straight at ball
    detection/occlusion, not at players or the pipeline plumbing, without
    inspecting individual frames.
    """
    if not results:
        return 0.0
    frames_with_ball = sum(1 for r in results if r.has_ball)
    return frames_with_ball / len(results)


def to_serializable(results: Sequence[FramePlayerBallDetections]) -> list[dict]:
    """
    Plain-dict/JSON-safe form of a list of FramePlayerBallDetections, for a
    caller (the detect_stage glue, app/services/detection_stage.py) that
    needs to persist them to disk for Part 6's tracker to read back later.
    Kept here, next to the dataclasses it serializes, rather than in the
    stage glue, so the on-disk shape stays in one place if either
    dataclass's fields change.
    """
    return [
        {
            "frame_path": r.frame_path,
            "players": [_detection_to_dict(d) for d in r.players],
            "balls": [_detection_to_dict(d) for d in r.balls],
        }
        for r in results
    ]


def _detection_to_dict(d: Detection) -> dict:
    return {
        "class_id": d.class_id,
        "class_name": d.class_name,
        "confidence": d.confidence,
        "bbox": {"x1": d.bbox.x1, "y1": d.bbox.y1, "x2": d.bbox.x2, "y2": d.bbox.y2},
    }
