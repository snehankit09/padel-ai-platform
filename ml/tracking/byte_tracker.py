"""
ByteTrack-style multi-object tracker — Part 6a.

`detect` (Part 5d) already answers "what's in this frame, and where" —
independently, frame by frame. Nothing about a `Detection` says whether
it's the same player as the box two frames ago. That's this module's job:
per-frame detections in -> per-frame *tracked* objects, each carrying a
`track_id` that stays stable across frames, out. Same "reusable core +
stateful convenience wrapper" split as ml/detection/yolo_detector.py (Part
5b), and this is deliberately the same layer of that split — the generic,
class-agnostic tracking engine, not a padel-specific players+ball wrapper
(that's a later part, the way player_ball_detection.py (5d) sits a layer
above yolo_detector.py (5b) rather than being part of it).

Algorithm, following Zhang et al.'s ByteTrack (2022) at the level that
matters here — a working prototype, not a paper reproduction:

  1. **Predict.** Every existing track's next position is predicted with a
     constant-velocity Kalman filter (`_KalmanBoxTracker`) — the same
     motion model SORT/DeepSORT/ByteTrack all use: object centers,
     scale, and aspect ratio move roughly linearly frame-to-frame at
     30fps, so a track's last-known velocity is a good one-step
     prediction even *before* this frame's detections arrive.
  2. **Match, high-confidence first.** Detections at or above
     `track_thresh` are matched to predicted track positions by IoU, via
     the Hungarian algorithm (`scipy.optimize.linear_sum_assignment`) —
     the globally optimal assignment, not a greedy nearest-match, so one
     ambiguous pair doesn't steal a box that clearly belongs to a
     different track.
  3. **Match again, low-confidence.** This is BYTE's actual contribution:
     detections *below* track_thresh (but above `match_thresh_low`)
     aren't discarded outright — they're matched against whatever tracks
     stage 2 left unmatched. A real player who's motion-blurred or
     half-occluded by another player often produces exactly this kind of
     low-confidence box; recovering the match here is what keeps that
     player's track alive instead of losing and re-assigning a new ID a
     few frames later. (Detection thresholds already applied upstream —
     see app/services/detection_stage.py — mean this module only ever
     sees what already cleared *those*; `track_thresh` is a second,
     tracker-internal split on top, not a re-application of the same
     one.)
  4. **Age and cull.** Tracks still unmatched after both stages get older
     (`time_since_update += 1`); once that exceeds `max_age` frames, the
     track is dropped rather than kept alive indefinitely. Leftover
     high-confidence detections that matched nothing become new
     tentative tracks — reported once they've matched `min_hits`
     consecutive frames, so a single spurious detection doesn't mint a
     real track ID.

**Known limitation worth flagging before this gets wired to real footage
(Part 6b): bbox-IoU association has no way to associate a detection with
its predecessor once per-frame displacement exceeds the object's own box
size** — the Kalman filter has no velocity estimate to extrapolate from
until it's seen at least two observations, so on a track's first
prediction the "predicted" box is just its last observed box. A small,
fast object (the ball) sampled at a low frame_sample_rate_fps (Part 5a)
can easily move farther between *sampled* frames than its own size —
exactly the risk PRD Section 13 names for ball tracking specifically. See
test_fast_small_object_loses_identity_when_displacement_exceeds_iou_gate
in test_byte_tracker.py for a verified reproduction, and
test_realistic_object_motion_keeps_identity_across_occlusion for the
displacement range this design does handle correctly. Two directions
worth considering for 6b, neither implemented here: track the ball at a
denser temporal sampling than players, or widen the association gate
(size- or distance-based, not pure IoU) specifically for young tracks
that don't have a velocity estimate yet.

Like yolo_detector.py, this module has no Celery/DB/app dependency — a
`ByteTracker` is constructed and fed frames directly, usable from a
one-off script or a test as easily as from the eventual tracking stage
glue (mirroring app/services/detection_stage.py). Unlike yolo_detector.py,
there's no deferred import here: numpy and scipy are lightweight, already
part of this project's dependency set (numpy transitively via
ultralytics/opencv already; scipy is the one new addition, for
`linear_sum_assignment`), and don't carry the "requires real model
weights / a real device" concerns ultralytics' deferred import exists to
avoid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count
from typing import Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ml.detection.yolo_detector import BoundingBox, Detection, FrameDetections

# Detections at or above this confidence are matched first, against every
# active track (stage 2 above). Distinct from — and typically higher than
# — the per-class thresholds detect_players_and_ball already applied (see
# module docstring): this is the tracker deciding "confident enough to
# match on its own", not the detector deciding "confident enough to keep
# at all".
DEFAULT_TRACK_THRESH = 0.5
# Detections between this and DEFAULT_TRACK_THRESH are only used in stage
# 3, to rescue tracks stage 2 couldn't match — never to start a new track
# on their own (a low-confidence box with no history could just as easily
# be a false positive as an occluded real object).
DEFAULT_MATCH_THRESH_LOW = 0.1
# Minimum IoU for two boxes to be considered the same object at all,
# regardless of confidence. Below this, the Hungarian algorithm would
# happily pair a track with the nearest unrelated detection just because
# nothing better exists this frame — this floor turns "worst available
# match" into "no match", which is what leaves a track briefly unmatched
# rather than silently drifting onto the wrong object.
DEFAULT_IOU_THRESHOLD = 0.3
# Frames a track can go unmatched before it's dropped. At a sampled rate
# of ~5fps (see settings.frame_sample_rate_fps), 5 frames is roughly one
# second of real match time — long enough to survive a genuine brief
# occlusion (a player crossing behind another, the ball behind the net),
# short enough that a track that's actually left the frame doesn't linger.
DEFAULT_MAX_AGE = 5
# Consecutive matched frames a new track needs before it's reported to
# callers at all. 1 means "report immediately" — every new tentative
# track becomes visible on its very first match; raise this if a
# fine-tuned detector (post-PRD-prototype) turns out to throw enough
# one-frame false positives that tentative tracks are worth hiding until
# they've proven themselves. See PRD Section 13 (ball detection accuracy)
# for why a low bar is preferred while the model is still pretrained/COCO.
DEFAULT_MIN_HITS = 1


class TrackingError(Exception):
    """Raised when the tracker is given malformed input it can't reasonably recover from."""


@dataclass(frozen=True)
class TrackedDetection:
    """One tracked object in one frame: everything a Detection has, plus identity and track health."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    # Frames since this track was first created (including this one).
    age: int
    # Total number of frames this track has been successfully matched, ever.
    hits: int
    # Frames since this track last matched a detection. 0 means it matched
    # *this* frame; >0 means this frame's box is a Kalman prediction,
    # carried forward through a brief gap (see DEFAULT_MAX_AGE) rather
    # than an observed detection.
    time_since_update: int

    @property
    def is_predicted(self) -> bool:
        """True when this frame's box is a motion prediction, not an observed detection."""
        return self.time_since_update > 0


@dataclass(frozen=True)
class FrameTracks:
    """
    Every tracked object in one frame — the tracking-stage analogue of
    yolo_detector.FrameDetections, same shape on purpose so downstream
    code (event detection, Part 7) that already knows how to iterate one
    doesn't need a second mental model for the other.
    """

    frame_path: str | None
    tracks: list[TrackedDetection] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.tracks)

    def of_class(self, class_id: int) -> list[TrackedDetection]:
        return [t for t in self.tracks if t.class_id == class_id]


def iou(a: BoundingBox, b: BoundingBox) -> float:
    """
    Intersection-over-union of two boxes, in [0.0, 1.0]. Pure — no numpy,
    no tracker state — so it's directly assertable in tests, the same
    role BoundingBox's own properties play in test_yolo_detector.py.
    """
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, a.width) * max(0.0, a.height)
    area_b = max(0.0, b.width) * max(0.0, b.height)
    union = area_a + area_b - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def iou_matrix(boxes_a: Sequence[BoundingBox], boxes_b: Sequence[BoundingBox]) -> np.ndarray:
    """
    All pairwise IoUs between two box lists, shape (len(boxes_a),
    len(boxes_b)). Thin numpy wrapper around `iou` — kept as its own
    function so match_by_iou's assignment logic doesn't also have to know
    how the matrix gets built.
    """
    matrix = np.zeros((len(boxes_a), len(boxes_b)), dtype=float)
    for i, a in enumerate(boxes_a):
        for j, b in enumerate(boxes_b):
            matrix[i, j] = iou(a, b)
    return matrix


def match_by_iou(
    track_boxes: Sequence[BoundingBox],
    detection_boxes: Sequence[BoundingBox],
    *,
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """
    Optimal (Hungarian algorithm) assignment of tracks to detections by
    IoU. Returns (matches, unmatched_track_indices,
    unmatched_detection_indices), where `matches` is a list of (track_idx,
    detection_idx) pairs — indices into the input sequences, not track IDs
    or Detection objects, so this stays pure and reusable for both match
    stages in ByteTracker.update() (high-confidence and low-confidence)
    without knowing anything about confidence, class, or track bookkeeping
    at all.

    linear_sum_assignment minimizes cost, so it runs against `-iou`
    (maximizing IoU = minimizing negative IoU). Pairs below
    `iou_threshold` are assigned back to "unmatched" after the fact —
    Hungarian assignment alone would happily pair a track with the least
    bad of several unrelated detections; the threshold is what turns that
    into "no match" instead (see DEFAULT_IOU_THRESHOLD).
    """
    if not track_boxes or not detection_boxes:
        return [], list(range(len(track_boxes))), list(range(len(detection_boxes)))

    cost = -iou_matrix(track_boxes, detection_boxes)
    track_idxs, det_idxs = linear_sum_assignment(cost)

    matches: list[tuple[int, int]] = []
    matched_tracks: set[int] = set()
    matched_dets: set[int] = set()
    for t_idx, d_idx in zip(track_idxs, det_idxs):
        if -cost[t_idx, d_idx] >= iou_threshold:
            matches.append((int(t_idx), int(d_idx)))
            matched_tracks.add(int(t_idx))
            matched_dets.add(int(d_idx))

    unmatched_tracks = [i for i in range(len(track_boxes)) if i not in matched_tracks]
    unmatched_dets = [i for i in range(len(detection_boxes)) if i not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets


class _KalmanBoxTracker:
    """
    Constant-velocity Kalman filter over one track's box, parameterized as
    SORT/ByteTrack conventionally do: state = [cx, cy, s, r, vcx, vcy, vs]
    — box center (cx, cy), scale/area (s = width * height), aspect ratio
    (r = width / height, assumed constant — a player or ball's aspect
    ratio doesn't meaningfully change frame to frame, so only position and
    scale get a velocity term), and the first three's velocities.

    Deliberately hand-rolled rather than pulling in `filterpy` (the
    library the reference SORT/ByteTrack implementations use): this is a
    fixed 7-state linear model with no need for filterpy's more general
    machinery, and keeping it here means one fewer dependency for a ~15
    line predict/update pair.
    """

    _id_counter = count(1)

    def __init__(self, bbox: BoundingBox):
        cx, cy = bbox.center
        s = max(bbox.width, 0.0) * max(bbox.height, 0.0)
        r = bbox.width / bbox.height if bbox.height > 0 else 1.0

        # State: [cx, cy, s, r, vcx, vcy, vs]. Aspect ratio r has no
        # velocity term (see class docstring).
        self.state = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=float)

        # Constant-velocity transition: position/scale += velocity each step.
        self._F = np.eye(7)
        self._F[0, 4] = 1.0  # cx += vcx
        self._F[1, 5] = 1.0  # cy += vcy
        self._F[2, 6] = 1.0  # s  += vs

        # Observe cx, cy, s, r directly (identity on the first 4 states).
        self._H = np.zeros((4, 7))
        self._H[0, 0] = self._H[1, 1] = self._H[2, 2] = self._H[3, 3] = 1.0

        # Fixed, moderate process/measurement noise — a hand-tuned
        # starting point (matching the rough magnitudes the reference SORT
        # implementation uses), not fit to real padel footage yet. Good
        # enough for a prototype tracker; worth revisiting once real match
        # video shows whether tracks are too jumpy (noise too low) or too
        # sluggish to follow a fast ball (noise too high).
        self._P = np.eye(7) * 10.0
        self._Q = np.eye(7) * 1.0
        self._R = np.eye(4) * 1.0

    def predict(self) -> BoundingBox:
        """Advances the filter one frame (no observation yet) and returns the predicted box."""
        self.state = self._F @ self.state
        self._P = self._F @ self._P @ self._F.T + self._Q
        return self._state_to_bbox()

    def update(self, bbox: BoundingBox) -> None:
        """Corrects the filter's prediction with an observed detection box."""
        cx, cy = bbox.center
        s = max(bbox.width, 0.0) * max(bbox.height, 0.0)
        r = bbox.width / bbox.height if bbox.height > 0 else 1.0
        z = np.array([cx, cy, s, r], dtype=float)

        y = z - self._H @ self.state
        S = self._H @ self._P @ self._H.T + self._R
        K = self._P @ self._H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self._P = (np.eye(7) - K @ self._H) @ self._P

    def current_bbox(self) -> BoundingBox:
        return self._state_to_bbox()

    def _state_to_bbox(self) -> BoundingBox:
        cx, cy, s, r, *_ = self.state
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = (s * r) ** 0.5
        h = s / w if w > 0 else 0.0
        return BoundingBox(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


@dataclass
class _Track:
    """Internal bookkeeping for one tracked object — not exposed outside this module (see TrackedDetection)."""

    track_id: int
    class_id: int
    class_name: str
    kalman: _KalmanBoxTracker
    confidence: float
    age: int = 1
    hits: int = 1
    time_since_update: int = 0

    def to_tracked_detection(self) -> TrackedDetection:
        return TrackedDetection(
            track_id=self.track_id,
            class_id=self.class_id,
            class_name=self.class_name,
            confidence=self.confidence,
            bbox=self.kalman.current_bbox(),
            age=self.age,
            hits=self.hits,
            time_since_update=self.time_since_update,
        )


class ByteTracker:
    """
    Stateful multi-object tracker: construct once per video (persistent
    IDs only mean something within one video's frame sequence), then call
    `update` / `update_frame` once per frame, in order.

    Tracks *all* classes in one shared ID space by default — a track's
    `class_id` is fixed at creation from its first detection and never
    changes, so a player is never matched against a ball-class detection
    even though they share the same matching pool; separating them into
    per-class matching would only matter if two different classes could
    plausibly overlap enough in IoU to compete for the same track, which
    isn't the case for a padel player vs. a padel ball. A caller that
    wants fully independent ID spaces per class (e.g. players numbered
    separately from the ball) can run two ByteTracker instances, one per
    class, and pre-filter each frame's detections accordingly — this
    class doesn't need to know about that to support it.
    """

    def __init__(
        self,
        *,
        track_thresh: float = DEFAULT_TRACK_THRESH,
        match_thresh_low: float = DEFAULT_MATCH_THRESH_LOW,
        iou_threshold: float = DEFAULT_IOU_THRESHOLD,
        max_age: int = DEFAULT_MAX_AGE,
        min_hits: int = DEFAULT_MIN_HITS,
    ) -> None:
        self.track_thresh = track_thresh
        self.match_thresh_low = match_thresh_low
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits

        self._tracks: list[_Track] = []
        self._id_counter = count(1)

    @property
    def active_track_count(self) -> int:
        """Tracks currently being carried forward — includes tentative ones not yet reported (see min_hits)."""
        return len(self._tracks)

    def update(self, detections: Sequence[Detection]) -> list[TrackedDetection]:
        """
        Runs one full ByteTrack step (see module docstring for the 4
        stages) for a single frame's detections and returns every track
        that's both currently matched-or-recently-seen (within max_age)
        and has cleared min_hits — in other words, exactly what a caller
        should treat as "the tracked objects in this frame".

        `detections` is every detection this frame regardless of
        confidence — this function does the high/low split internally
        (via track_thresh), matching ByteTrack's own design: the caller
        shouldn't have to pre-filter, since part of the point is using
        low-confidence boxes the caller might otherwise have discarded.

        Matching (stages 2 and 3) is done separately per class_id: a
        track's candidate pool is only ever detections that share its
        class, so a ball detection can never be assigned to a player
        track (or vice versa) no matter how well their boxes overlap —
        `match_by_iou` itself has no notion of class, so this method is
        what actually enforces that invariant, by only ever calling it
        with same-class track/detection subsets.
        """
        for t in self._tracks:
            t.kalman.predict()

        high = [d for d in detections if d.confidence >= self.track_thresh]
        low = [d for d in detections if self.match_thresh_low <= d.confidence < self.track_thresh]

        unmatched_track_idxs: set[int] = set(range(len(self._tracks)))
        matched_high_idxs: set[int] = set()

        class_ids_present = (
            {t.class_id for t in self._tracks} | {d.class_id for d in high} | {d.class_id for d in low}
        )

        for class_id in class_ids_present:
            track_idxs = [i for i, t in enumerate(self._tracks) if t.class_id == class_id]
            high_idxs = [i for i, d in enumerate(high) if d.class_id == class_id]

            # Stage 2: this class's high-confidence detections against this class's active tracks.
            track_boxes = [self._tracks[i].kalman.current_bbox() for i in track_idxs]
            high_boxes = [high[i].bbox for i in high_idxs]
            matches, unmatched_local, _ = match_by_iou(track_boxes, high_boxes, iou_threshold=self.iou_threshold)
            for t_local, d_local in matches:
                t_idx, d_idx = track_idxs[t_local], high_idxs[d_local]
                self._apply_match(self._tracks[t_idx], high[d_idx])
                unmatched_track_idxs.discard(t_idx)
                matched_high_idxs.add(d_idx)

            # Stage 3: this class's low-confidence detections against whatever
            # stage 2 left unmatched for this class — never spawns a track on
            # its own (see DEFAULT_MATCH_THRESH_LOW).
            remaining_track_idxs = [track_idxs[i] for i in unmatched_local]
            low_idxs = [i for i, d in enumerate(low) if d.class_id == class_id]
            remaining_boxes = [self._tracks[i].kalman.current_bbox() for i in remaining_track_idxs]
            low_boxes = [low[i].bbox for i in low_idxs]
            low_matches, _, _ = match_by_iou(remaining_boxes, low_boxes, iou_threshold=self.iou_threshold)
            for t_local, d_local in low_matches:
                t_idx, d_idx = remaining_track_idxs[t_local], low_idxs[d_local]
                self._apply_match(self._tracks[t_idx], low[d_idx])
                unmatched_track_idxs.discard(t_idx)

        # Stage 4a: age out tracks nothing matched this frame; drop any
        # that have exceeded max_age.
        still_alive: list[_Track] = []
        for i, t in enumerate(self._tracks):
            if i in unmatched_track_idxs:
                t.time_since_update += 1
                t.age += 1
            if t.time_since_update <= self.max_age:
                still_alive.append(t)
        self._tracks = still_alive

        # Stage 4b: unmatched high-confidence detections spawn new tracks.
        # (Low-confidence detections never do — see DEFAULT_MATCH_THRESH_LOW.)
        for d_idx in range(len(high)):
            if d_idx not in matched_high_idxs:
                self._tracks.append(self._new_track(high[d_idx]))

        return [
            t.to_tracked_detection()
            for t in self._tracks
            if t.hits >= self.min_hits and t.time_since_update == 0
        ]

    def update_frame(self, frame_detections: FrameDetections) -> FrameTracks:
        """
        Frame-shaped convenience over `update`, mirroring
        yolo_detector.detect_frame's naming — takes a FrameDetections
        (Part 5b/5d's own output shape) in, returns a FrameTracks out,
        carrying `frame_path` through unchanged so a caller iterating
        frames doesn't have to track that separately alongside track
        results.
        """
        tracks = self.update(frame_detections.detections)
        return FrameTracks(frame_path=frame_detections.frame_path, tracks=tracks)

    def update_many(self, frames: Sequence[FrameDetections]) -> list[FrameTracks]:
        """
        Runs update_frame across a sequence of frames' detections, in
        order — order matters here in a way it doesn't for
        yolo_detector.detect_frames: this tracker is stateful, so calling
        it out of frame order (or on a frame twice) produces meaningless
        track identities. Same "plain loop, not batched" shape as
        detect_frames/detect_players_and_ball_many for consistency, even
        though the reason (statefulness vs. independence) is different.
        """
        return [self.update_frame(fd) for fd in frames]

    def _apply_match(self, track: _Track, detection: Detection) -> None:
        track.kalman.update(detection.bbox)
        track.confidence = detection.confidence
        track.hits += 1
        track.time_since_update = 0

    def _new_track(self, detection: Detection) -> _Track:
        return _Track(
            track_id=next(self._id_counter),
            class_id=detection.class_id,
            class_name=detection.class_name,
            kalman=_KalmanBoxTracker(detection.bbox),
            confidence=detection.confidence,
        )


def to_serializable(results: Sequence[FrameTracks]) -> list[dict]:
    """
    Plain-dict/JSON-safe form of a list of FrameTracks, mirroring
    player_ball_detection.to_serializable — for a future tracking-stage
    glue module (the Part 6 analogue of app/services/detection_stage.py)
    that needs to persist tracks to disk for Part 7's event detection to
    read back later.
    """
    return [
        {
            "frame_path": r.frame_path,
            "tracks": [
                {
                    "track_id": t.track_id,
                    "class_id": t.class_id,
                    "class_name": t.class_name,
                    "confidence": t.confidence,
                    "bbox": {"x1": t.bbox.x1, "y1": t.bbox.y1, "x2": t.bbox.x2, "y2": t.bbox.y2},
                    "age": t.age,
                    "hits": t.hits,
                    "time_since_update": t.time_since_update,
                }
                for t in r.tracks
            ],
        }
        for r in results
    ]
