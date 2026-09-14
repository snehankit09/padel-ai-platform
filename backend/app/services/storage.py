"""
Storage abstraction.

Why this exists: the upload route needs to persist a file somewhere and
later hand back something that can be turned into a URL. If it called
`open(...)` / `boto3` directly, switching STORAGE_BACKEND from local to s3
later would mean touching every caller. Instead callers only ever see
`StorageService.save()` / `.get_url()`; the config value
`settings.storage_backend` picks the implementation once, at startup.

This mirrors the association-object lesson from Part 2: don't hand-roll
the simple version now and pay for a rewrite later — pay a small amount
of indirection now instead.
"""

from __future__ import annotations

import abc
import os
import shutil
import uuid

from app.core.config import Settings, get_settings


class StorageError(Exception):
    """Raised when a file can't be persisted or resolved by the storage backend."""


class StorageService(abc.ABC):
    @abc.abstractmethod
    def save(self, source_path: str, destination_key: str) -> str:
        """
        Move/copy the file at `source_path` (e.g. a temp upload) into permanent
        storage under `destination_key`. Returns the value that should be
        stored in Video.file_path — backend-specific (a local path or an S3
        key), never a public URL.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_url(self, file_path: str) -> str:
        """
        Resolve a stored Video.file_path into something a client can use to
        fetch the file. For local storage that's a static-mount path; for S3
        it would be a signed URL. Callers never need to know which.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def get_local_path(self, key: str) -> str:
        """
        Resolve a storage key to an absolute path on *this machine's* local
        filesystem, for callers that need direct file access rather than a
        servable URL — the frame-extraction stage (Part 5a) reading the
        source video, and writing extracted frames, being the first case.
        For local storage this is just the key under the storage root; an
        S3 backend would need to download to a local temp path first (not
        implemented yet — see S3StorageService below).
        """
        raise NotImplementedError


class LocalStorageService(StorageService):
    """
    Development backend. Files live under `settings.local_storage_path` on
    disk (the `storage/` folder from the Part 1 scaffold). `destination_key`
    is used as a relative path under that root, so callers don't need to
    know the root location.
    """

    def __init__(self, settings: Settings):
        self._root = settings.local_storage_path

    def save(self, source_path: str, destination_key: str) -> str:
        full_destination = os.path.join(self._root, destination_key)
        os.makedirs(os.path.dirname(full_destination), exist_ok=True)
        try:
            shutil.move(source_path, full_destination)
        except OSError as exc:
            raise StorageError(f"Could not save file to local storage: {exc}") from exc
        return destination_key

    def get_url(self, file_path: str) -> str:
        # In development this is served by a static mount at /media (wired
        # up alongside the routes); no signing needed for local disk.
        return f"/media/{file_path}"

    def get_local_path(self, key: str) -> str:
        # Local backend: the key already *is* a path relative to the
        # storage root, so this is the same join `save()` uses internally
        # — no download step needed, unlike S3.
        return os.path.join(self._root, key)


class S3StorageService(StorageService):
    """
    Placeholder for the production backend. Not implemented yet — Part 3
    only needs local storage to work end-to-end. When we're ready to add
    S3, boto3 goes in requirements.txt and this class fills in using
    settings.aws_* — no changes needed anywhere that calls StorageService.
    """

    def __init__(self, settings: Settings):
        self._settings = settings

    def save(self, source_path: str, destination_key: str) -> str:
        raise NotImplementedError("S3 storage backend not implemented yet")

    def get_url(self, file_path: str) -> str:
        raise NotImplementedError("S3 storage backend not implemented yet")

    def get_local_path(self, key: str) -> str:
        raise NotImplementedError("S3 storage backend not implemented yet")


def get_storage_service() -> StorageService:
    """
    FastAPI dependency / plain-call factory. Reads settings.storage_backend
    once per call — cheap, since get_settings() is itself cached.
    """
    settings = get_settings()
    if settings.storage_backend == "local":
        return LocalStorageService(settings)
    if settings.storage_backend == "s3":
        return S3StorageService(settings)
    raise StorageError(f"Unknown storage backend: {settings.storage_backend!r}")


def make_video_destination_key(match_id: uuid.UUID, original_filename: str) -> str:
    """
    Builds a destination key like `videos/<match_id>/<original_filename>`.
    Namespacing by match_id keeps files from colliding and makes it obvious
    which video belongs to which match just from the path, which is handy
    when eyeballing the storage directory during development.
    """
    safe_name = os.path.basename(original_filename)
    return f"videos/{match_id}/{safe_name}"


def make_frames_destination_dir(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `frames/<video_id>/` — the directory the
    `validate` stage (Part 5a) extracts a video's sampled frames into, and
    that `detect` (Part 5b+) later reads them back out of. Namespaced by
    video_id (not match_id, unlike make_video_destination_key) since frames
    are a pipeline artifact of one specific processing run of one video,
    not of the match as a whole.
    """
    return f"frames/{video_id}"


def make_detections_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `detections/<video_id>/detections.json` —
    where the `detect` stage (Part 5d) writes its per-frame player/ball
    detection results, for `track` (Part 6) to read back instead of
    re-running YOLO. A single JSON file, not a directory of many small
    files like make_frames_destination_dir: one video's detections are one
    coherent artifact, not something later code needs to read frame-by-
    frame off disk the way the raw frame images themselves are.
    """
    return f"detections/{video_id}/detections.json"


def make_court_calibration_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `courts/<video_id>/calibration.json` —
    where the `detect` stage (Part 5e) writes the court's corners +
    homography from ml/detection/court_detector.py, for Part 6+ to read
    back and convert a tracked pixel position into real-world court
    coordinates (PRD 5.2, 5.5) without re-running line detection. Kept as
    its own file, separate from make_detections_destination_path's
    detections.json, since it's a per-video constant (one calibration for
    the whole match) rather than a per-frame result — conflating the two
    would make detections.json's shape depend on whether calibration
    happened to succeed.
    """
    return f"courts/{video_id}/calibration.json"


def make_player_tracks_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `tracks/<video_id>/player_tracks.json` —
    where the `track` stage (Part 6b) writes per-frame player tracking
    results (ml/tracking/byte_tracker.py's output, filtered to the
    player class and run through its own ByteTracker instance), for
    Part 7 (rally/event detection) to read back. Separate from
    make_ball_tracks_destination_path (Part 6c): players and the ball are
    tracked by two independent ByteTracker instances with their own ID
    spaces and tuning (see app/core/config.py's player_track_*/ball_track_*
    settings), so their results are two independent files rather than one
    combined one — a caller that only needs player movement (most stats)
    shouldn't have to load and filter out ball data to get it, and vice
    versa.
    """
    return f"tracks/{video_id}/player_tracks.json"


def make_ball_tracks_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `tracks/<video_id>/ball_tracks.json` —
    where the `track` stage's ball half (Part 6c) writes the ball's tracked
    trajectory after gap interpolation (ml/tracking/byte_tracker.py's
    ByteTracker output, run through ml/tracking/ball_interpolation.py to
    bridge short occlusion/motion-blur gaps), for Part 7 to read back.
    Same file-per-object-class split as make_player_tracks_destination_path,
    and for the same reason: independent ByteTracker instances with
    independent tuning and independent ID spaces (see
    app/core/config.py's player_track_*/ball_track_* settings) naturally
    produce two independent result files rather than one combined one.
    """
    return f"tracks/{video_id}/ball_tracks.json"


def make_rally_segments_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `rallies/<video_id>/rallies.json` —
    where the `analyze` stage's rally-boundary half (Part 7a) writes the
    match's rally segments (ml/pipeline/rally_detection.py's output,
    derived from ball_tracks.json), for Part 7b+ (shot/event
    classification within each rally) and Part 8/9 (highlight clips,
    stats) to read back. Its own file, namespaced by video_id like the
    tracks files rather than folded into ball_tracks.json: a rally
    segment list is a *derived*, match-level summary of the per-frame
    ball tracks, not another per-frame artifact, so keeping it separate
    means Part 8/9 can read just the (small) rally list without also
    loading every frame's raw ball track data to get it.
    """
    return f"rallies/{video_id}/rallies.json"


def make_serve_events_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `serves/<video_id>/serves.json` — where
    the `analyze` stage's serve-identification half (Part 7b) writes one
    ServeEvent per rally (ml/pipeline/serve_detection.py's output), for
    later analyze sub-parts (shot classification, point outcomes) and
    Part 9 (serve percentage / who-served-what stats) to read back. Own
    file for the same reason make_rally_segments_destination_path is its
    own file rather than folded into rallies.json: this is a second,
    independent derived artifact (needs ball_tracks.json AND
    player_tracks.json AND rallies.json, plus optionally a court
    calibration, to produce), not a per-frame one — a caller that only
    wants rally boundaries shouldn't have to load serve data to get them,
    and vice versa.
    """
    return f"serves/{video_id}/serves.json"


def make_shots_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `shots/<video_id>/shots.json` — where
    the `analyze` stage's shot-classification half (Part 7c) writes its
    flat list of classified Shots (ml/pipeline/shot_classification.py's
    output), for Part 8 (highlight clips keyed on shot type) and Part 9
    (shot-type stats) to read back. Own file, same reasoning as
    make_serve_events_destination_path: an independent derived artifact
    (needs ball_tracks.json AND player_tracks.json AND rallies.json, plus
    optionally a court calibration) that a caller wanting only rally or
    serve data shouldn't have to load to get those instead.
    """
    return f"shots/{video_id}/shots.json"


def make_point_outcomes_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `outcomes/<video_id>/outcomes.json` —
    where the `analyze` stage's point-outcome half (Part 7d) writes one
    PointOutcome per rally (ml/pipeline/point_outcome.py's output), for
    Part 9 (win/loss and error-rate stats) to read back. Own file, same
    reasoning as make_shots_destination_path / make_serve_events_destination_path:
    an independent derived artifact (needs rallies.json, ball_tracks.json,
    and a court calibration) that a caller wanting only rally, serve, or
    shot data shouldn't have to load to get those instead.
    """
    return f"outcomes/{video_id}/outcomes.json"


def make_highlights_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `highlights/<video_id>/highlights.json` —
    where the `analyze` stage's highlight-tagging half (Part 7e) writes its
    flat list of candidate HighlightEvents (ml/pipeline/highlight_tagging.py's
    output, derived from rallies.json and shots.json). Part 7f
    (app/services/analyze_persistence_stage.py) reads this back and turns it
    into `Highlight` DB rows (clip_file_path left NULL); Part 8 (clip
    generation) reads those rows back and UPDATEs clip_file_path once it has
    trimmed a real, padded clip. Own file, same reasoning as every other
    Part 7 output path: an independent derived artifact that a caller
    wanting only rally/serve/shot/outcome data shouldn't have to load to
    get those instead, and vice versa.
    """
    return f"highlights/{video_id}/highlights.json"


def make_clip_boundaries_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `clips/<video_id>/clip_boundaries.json` —
    where the `analyze` stage's clip-boundary half (Part 8a) writes its
    flat list of padded, clamped ClipBoundary objects
    (ml/pipeline/clip_boundaries.py's output, derived from highlights.json
    plus the video's own probed duration), for Part 8's later steps (the
    actual FFmpeg trim, and updating each Highlight row's start/end +
    clip_file_path once a real clip file exists) to read back. Own file,
    same reasoning as make_highlights_destination_path: an independent
    derived artifact a caller wanting only the tight, un-padded event
    windows shouldn't have to load to get those instead, and vice versa.
    Namespaced under `clips/` (not `highlights/`) since Part 8's later
    steps will also write actual trimmed clip *files* under that same
    prefix, once they exist.
    """
    return f"clips/{video_id}/clip_boundaries.json"


def make_player_statistics_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `stats/<video_id>/player_stats.json` —
    where the `analyze` stage's player-statistics half (Part 9f) writes
    ml.pipeline.stats_aggregation.player_level_stat_values' track_id-keyed
    output, alongside ml.pipeline.player_identity.assign_court_sides'
    SIDE_A/SIDE_B grouping for the same tracks (Part 9b) — see
    app/services/player_statistics_stage.py's own module docstring for why
    both live in one artifact rather than two. Namespaced under `stats/`
    (a new top-level prefix, not folded into an existing one): unlike
    every other Part 7/8 artifact this is NOT keyed by match_id anywhere
    downstream today, since app/services/player_statistics_persistence_stage.py
    (Part 9e) can't yet turn a track_id into a real Player.id — this file
    is what a future human-in-the-loop identity-confirmation step would
    read to build that mapping, not something Part 9's own automated
    pipeline run re-reads itself.
    """
    return f"stats/{video_id}/player_stats.json"


def make_clip_file_destination_path(video_id: uuid.UUID, index: int) -> str:
    """
    Builds a destination key like `clips/<video_id>/clip_000.mp4` — where
    Part 8b's actual FFmpeg trim (app/services/clip_extraction_stage.py,
    ml/common/clip_extraction.py) writes each highlight's real, playable
    clip file, for `Highlight.clip_file_path` (Part 8b's UPDATE of the rows
    Part 7f inserted with clip_file_path=NULL) and later reel/dashboard
    layers to serve back. Namespaced under `clips/`, the same prefix
    make_clip_boundaries_destination_path already uses for
    clip_boundaries.json — that function's own docstring anticipated this
    exact file living alongside it once Part 8's later steps existed.

    `index` (this stage's own enumeration position over the ClipBoundary
    list it's extracting, not rally_index or highlight_type) is what keeps
    filenames simple and guaranteed collision-free within one video, even
    if two boundaries ever ended up with identical timing. It's not a
    lookup key: the real link from a clip file back to the Highlight row
    it belongs to is the DB UPDATE that stage performs directly (matched by
    event_type + the row's own tight start/end time — see that module's
    docstring for why), not anything derivable from this path alone.
    """
    return f"clips/{video_id}/clip_{index:03d}.mp4"


def make_clip_thumbnail_destination_path(video_id: uuid.UUID, index: int) -> str:
    """
    Builds a destination key like `clips/<video_id>/clip_000_thumb.jpg` —
    Highlights Improvement Roadmap Tier 1a's poster-frame image for one
    highlight clip. Deliberately namespaced and indexed exactly like
    make_clip_file_destination_path above (same `clips/` prefix, same
    zero-padded `index`, same one-per-ClipBoundary-position meaning) since
    a thumbnail is a second artifact *of* the same clip, not an
    independent one — app/services/clip_extraction_stage.py calls both
    this and make_clip_file_destination_path with the same `index` for a
    given boundary, so the two files sit side by side and are trivially
    recognizable as a pair on disk even without the DB in front of you.
    """
    return f"clips/{video_id}/clip_{index:03d}_thumb.jpg"


def make_reel_file_destination_path(video_id: uuid.UUID) -> str:
    """
    Builds a destination key like `reels/<video_id>/reel.mp4` — where the
    `done` stage's reel-assembly half (Part 10f,
    app/services/reel_generation_stage.py) writes the one real, playable
    reel file ml.common.reel_assembly.assemble_reel produces by
    concatenating Part 8's already-cut clips per Part 10a/10b's selection
    and ordering, for `Reel.file_path` (Part 10e's own row, inserted with
    file_path=NULL, gets UPDATEd here once the file actually exists — same
    split as make_clip_file_destination_path's relationship to
    `Highlight.clip_file_path`). Namespaced under its own `reels/` prefix,
    not `clips/`: a reel is a distinct artifact composed FROM clips, not
    another entry in the same per-highlight collection, and PRD Module 4
    is one reel per match/video so `reel.mp4` (no index) is enough to stay
    collision-free the same way a video only ever gets one entry here.
    Keyed by video_id, not match_id, for the same reason every other
    per-processing-run artifact path in this module is: it's this specific
    `done` run's output, not an inherent property of the match itself.
    """
    return f"reels/{video_id}/reel.mp4"
