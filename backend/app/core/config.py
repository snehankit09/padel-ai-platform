"""
Centralized application configuration.

Why this exists: every other module (database, storage, upload validation)
needs config values. Instead of scattering `os.getenv()` calls everywhere,
we define one Settings object, load it once, and import it wherever needed.
pydantic-settings also gives us free validation — e.g. MAX_UPLOAD_SIZE_MB
must actually be an int, or the app fails fast at startup instead of
failing confusingly mid-upload.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    app_name: str = "padel-ai-platform"
    environment: str = "development"
    debug: bool = True
    # Comma-separated list of origins the browser-based frontend (Part 11)
    # is allowed to call this API from — needed because the Next.js dev
    # server (localhost:3000 by default) and this API (localhost:8000) are
    # different origins, so the browser enforces CORS on every request
    # unless the API explicitly allows it. Kept as a plain string (not a
    # list) because that's what a .env file can hold; cors_allowed_origins_list
    # below does the actual splitting, same pattern as
    # allowed_video_formats/allowed_video_formats_list.
    cors_allowed_origins: str = "http://localhost:3000"

    # --- Database ---
    database_url: str

    # --- Redis / Celery ---
    redis_url: str
    celery_broker_url: str
    celery_result_backend: str

    # --- Storage ---
    storage_backend: str = "local"
    local_storage_path: str = "/storage"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_s3_bucket: str = ""
    aws_region: str = "us-east-1"

    # --- Upload limits ---
    max_upload_size_mb: int = 2048
    max_video_duration_minutes: int = 120
    allowed_video_formats: str = "mp4,mov,avi"

    # --- Frame extraction (Part 5a) ---
    # Frames-per-second of *video content* sampled by the `validate` stage,
    # not a frame-skip count — see ml/common/frame_extraction.py. Trades off
    # detection input size against how easily a fast exchange could be
    # missed between samples; revisit once Part 6 (ball tracking) shows
    # whether trajectory reconstruction needs a denser sample.
    frame_sample_rate_fps: float = 5.0

    # --- YOLO detection (Part 5b) ---
    # Bundled checkpoint name ("yolov8n.pt") or a path to a local/fine-tuned
    # .pt file — both are accepted directly by ultralytics' YOLO(...). CPU
    # by default so this runs the same in dev as it will in the worker
    # container until GPU workers (PRD Section 9/13) are provisioned;
    # override with "cuda" once they are.
    yolo_model_path: str = "yolov8n.pt"
    yolo_device: str = "cpu"

    # --- Player & ball detection (Part 5d) ---
    # Two separate thresholds, not one, because the two targets aren't
    # equally easy to find — see ml/detection/player_ball_detection.py's
    # module docstring. The ball threshold is deliberately much lower: a
    # missed ball detection loses that frame's trajectory data outright,
    # while a spurious low-confidence one is noise Part 6's tracker can
    # filter out using motion continuity across frames.
    yolo_player_confidence_threshold: float = 0.25
    yolo_ball_confidence_threshold: float = 0.10

    # --- Court detection & calibration (Part 5c) ---
    # Padel courts are internationally standardized to one size (FIP
    # regulations) — not a per-venue setting in practice — but exposed
    # here rather than hardcoded in ml/detection/court_detector.py so a
    # non-standard/training court doesn't require a code change.
    court_length_m: float = 20.0
    court_width_m: float = 10.0
    # A single frame can fail to calibrate (player standing on a boundary
    # line, motion blur, a stray shadow breaking Canny/Hough) without the
    # match footage itself being a genuinely hard case — so this tries up
    # to N evenly-spaced frames before treating calibration as having
    # failed for the video, rather than betting everything on frame 0.
    court_calibration_max_frame_attempts: int = 5

    # --- Player tracking (Part 6b) ---
    # Separate ByteTracker parameterization from the ball's (Part 6c),
    # tuned for how players actually move: slower and larger than the
    # ball, so a tighter IoU gate and a shorter max_age than the ball's
    # settings are appropriate here (see ml/tracking/byte_tracker.py's
    # module docstring on why bbox-IoU association struggles once
    # per-frame displacement exceeds the object's own box size — that
    # risk is much lower for a player-sized box than a ball-sized one at
    # the same frame_sample_rate_fps). min_hits=2 (vs. ByteTracker's own
    # default of 1) because a stray person in the background — a
    # spectator, someone walking past — is a more plausible false
    # positive to guard against here than it is for the ball, where any
    # detection at all is comparatively rare and worth trusting sooner.
    player_track_thresh: float = 0.6
    player_track_match_thresh_low: float = 0.15
    player_track_iou_threshold: float = 0.3
    player_track_max_age: int = 5
    player_track_min_hits: int = 2

    # --- Ball tracking (Part 6c) ---
    # Own ByteTracker parameterization, separate from player_track_* — see
    # app/services/player_tracking_stage.py's module docstring for why
    # players and the ball run as two independent ByteTracker instances at
    # all. Deliberately looser than the player settings in every direction:
    # a small, fast, easily-motion-blurred ball produces lower-confidence
    # and lower-IoU detections even when it's genuinely there, so the same
    # thresholds tuned for a player-sized box would throw real ball
    # detections away. ball_track_max_age is longer than player's
    # specifically so a track survives a real occlusion/motion-blur run
    # long enough for ml/tracking/ball_interpolation.py to bridge the gap
    # afterward — keep ball_track_max_interpolation_gap_frames <=
    # ball_track_max_age, since a gap ByteTracker itself already gave up on
    # (new track_id on the other side) can never be filled by interpolation
    # regardless of how large the gap setting is.
    ball_track_thresh: float = 0.35
    ball_track_match_thresh_low: float = 0.05
    ball_track_iou_threshold: float = 0.15
    ball_track_max_age: int = 6
    ball_track_min_hits: int = 1
    # Consecutive missing frames interpolate_ball_gaps will bridge with a
    # straight-line estimate between the two real detections on either
    # side. Below this: almost certainly motion blur or a brief occlusion
    # (behind a player, the net) — the ball didn't stop existing, the model
    # just missed it, so a linear guess is a reasonable stand-in. Above
    # this: treated as "no evidence", not interpolated — most commonly a
    # lob carrying the ball out of frame entirely, where a straight line
    # from its last position would be fabricating a trajectory the ball
    # may never have taken (PRD Section 13 risk: ball leaving frame).
    ball_track_max_interpolation_gap_frames: int = 4

    # --- Rally boundary detection (Part 7a) ---
    # See ml/pipeline/rally_detection.py's module docstring for why these
    # are two distinct, deliberately different-purpose knobs rather than
    # one shared threshold.
    #
    # A coarser, position-agnostic gap tolerance layered *on top of*
    # ball_track_max_interpolation_gap_frames (Part 6c) — bridges a
    # stretch of consecutive frames with literally no ball track (real or
    # interpolated) that's still probably mid-rally, e.g. a fast smash
    # the tracker/interpolation couldn't safely follow. Larger than
    # ball_track_max_interpolation_gap_frames on purpose: this module
    # isn't asserting a position across the gap (interpolation already
    # decided that wasn't safe), only "is this still the same point",
    # which can tolerate more uncertainty.
    rally_activity_gap_tolerance_frames: int = 8
    # Below this, a surviving run of ball activity is dropped as noise
    # rather than reported as a rally — guards against a single stray
    # ball-class false positive during a real break becoming its own
    # spurious "rally".
    rally_min_duration_frames: int = 5

    # --- Serve detection (Part 7b) ---
    # See ml/pipeline/serve_detection.py's module docstring for why this
    # prefers court-calibrated meters (Part 5e's homography) over raw
    # pixel distance whenever a video has one, and why the two thresholds
    # below are genuinely different kinds of number (a real physical
    # distance vs. a resolution-dependent pixel count) rather than the
    # same value in two units.
    serve_detection_window_frames: int = 3
    serve_max_ball_player_distance_m: float = 2.5
    serve_max_ball_player_distance_px: float = 150.0

    # --- Shot classification (Part 7c) ---
    # See ml/pipeline/shot_classification.py's module docstring for the
    # full reasoning; short version, each knob maps to one of that
    # module's three signals:
    #   - shot_max_contact_player_distance_{m,px}: same pairing-distance
    #     role as serve_max_ball_player_distance_{m,px} above, just for
    #     every in-rally contact rather than only the first one.
    #   - shot_smash_height_ratio: how far above the contacting player's
    #     own bbox top counts as "overhead" (1.0 == exactly at bbox top;
    #     see the module constant's own comment for why this defaults
    #     slightly above 1.0).
    #   - shot_lob_min_airborne_frames: how many consecutive sampled
    #     frames of flight after a contact, with no next contact found
    #     yet, counts as "unusually long" rather than a normal rally
    #     exchange.
    #   - shot_net_proximity_m: how close (in real court meters) to the
    #     net line a contact needs to land to count as a volley rather
    #     than a groundstroke — only ever checked when a court
    #     calibration exists for the video.
    shot_max_contact_player_distance_m: float = 2.5
    shot_max_contact_player_distance_px: float = 150.0
    shot_smash_height_ratio: float = 1.05
    shot_lob_min_airborne_frames: int = 6
    shot_net_proximity_m: float = 3.0

    # --- Point outcome detection (Part 7d) ---
    # See ml/pipeline/point_outcome.py's module docstring for the full
    # reasoning, including why this — unlike Parts 7b/7c — has no
    # meaningful pixel-only fallback and reports OUTCOME_UNDETERMINED for
    # every rally in an uncalibrated video rather than a coarser guess.
    #   - point_outcome_out_of_bounds_margin_m: how far past the
    #     calibrated court rectangle's edge the ball's last position can
    #     be before this stops calling it in bounds (homography +
    #     detection noise tolerance, not a real line-call margin).
    #   - point_outcome_net_zone_m: how close to the net LINE (not just
    #     "near the net" the way shot_net_proximity_m means for volley
    #     positioning) the ball's last position needs to be to even be
    #     considered for OUTCOME_NET.
    #   - point_outcome_net_deceleration_ratio: how sharply the ball's
    #     speed has to collapse, right at that position, before this
    #     calls it a genuine net hit rather than the ball simply
    #     rallying through that part of the court.
    point_outcome_out_of_bounds_margin_m: float = 0.3
    point_outcome_net_zone_m: float = 1.0
    point_outcome_net_deceleration_ratio: float = 0.35

    # --- Highlight event tagging (Part 7e) ---
    # See ml/pipeline/highlight_tagging.py's module docstring for the full
    # reasoning, including why WINNING_SHOT/MATCH_POINT/BREAK_POINT have no
    # corresponding settings here at all (no threshold would fix a missing
    # signal, not a mistuned one).
    #   - highlight_long_rally_min_duration_s / _score_saturation_s: how
    #     long a rally (Part 7a) needs to run before it's tagged LONG_RALLY,
    #     and the duration past that at which its score maxes out.
    #   - highlight_fast_exchange_max_interval_s: max seconds between two
    #     consecutive in-rally shots (Part 7c) for them to count as part of
    #     one fast-exchange run.
    #   - highlight_fast_exchange_min_shot_count / _score_saturation_count:
    #     how many shots a qualifying run needs before it's tagged at all,
    #     and the count at which its score maxes out.
    #   - highlight_powerful_smash_score_ceiling_ratio: the contact_height_
    #     ratio (Part 7c) at which a POWERFUL_SMASH's score maxes out —
    #     shot_smash_height_ratio (already defined above) is reused as the
    #     floor, not duplicated here, since it has to match whatever value
    #     7c's classify_shot actually ran with for a given video.
    #   - highlight_spectacular_save_max_response_s: max seconds between an
    #     opponent's smash and a different player's return (Part 7c) for
    #     the return to count as a SPECTACULAR_SAVE.
    highlight_long_rally_min_duration_s: float = 15.0
    highlight_long_rally_score_saturation_s: float = 35.0
    highlight_fast_exchange_max_interval_s: float = 1.2
    highlight_fast_exchange_min_shot_count: int = 4
    highlight_fast_exchange_score_saturation_count: int = 8
    highlight_powerful_smash_score_ceiling_ratio: float = 1.6
    highlight_spectacular_save_max_response_s: float = 0.8

    # --- Clip boundary calculation (Part 8a) ---
    # See ml/pipeline/clip_boundaries.py's module docstring for the full
    # reasoning, including per-HighlightType padding (Part 8a's "1c"
    # follow-up — see the highlight-clips roadmap).
    #   - clip_pre_roll_s / clip_post_roll_s: fallback seconds of lead-up
    #     added before, and follow-through added after, each
    #     HighlightEvent's own tight start_time_s/end_time_s (Part 7e)
    #     before it's clamped to the video's own duration — used for any
    #     HighlightType without its own override below (currently
    #     fast_exchange and spectacular_save).
    #   - clip_min_duration_s: no longer force-applied uniformly — the
    #     floor on each clip's final, padded-and-clamped duration is now
    #     derived per-event from whichever pre/post-roll actually applied
    #     to it (see clip_boundaries.compute_clip_boundary). Left here,
    #     unused by the stage, only so the historical 3.0 + 2.0 = 5.0
    #     relationship stays documented in one place.
    #   - clip_<type>_pre_roll_s / clip_<type>_post_roll_s: per-
    #     HighlightType overrides. A smash gets a shorter lead-in (the
    #     contact itself is the payoff) and longer follow-through (let the
    #     reaction land); a long rally gets a longer lead-in so the
    #     buildup before the point turns highlight-worthy has room to
    #     register.
    clip_pre_roll_s: float = 3.0
    clip_post_roll_s: float = 2.0
    clip_min_duration_s: float = 5.0
    clip_powerful_smash_pre_roll_s: float = 2.0
    clip_powerful_smash_post_roll_s: float = 3.0
    clip_long_rally_pre_roll_s: float = 4.0
    clip_long_rally_post_roll_s: float = 2.0

    # --- Highlight clip slow-motion (Highlights Improvement Roadmap Tier 1b) ---
    # Defaults to OFF on purpose. Whether this looks good depends entirely
    # on the source video's actual native frame rate: a 25-30fps source
    # (typical for a phone/court-side camera, not broadcast-grade high-
    # frame-rate capture) has no extra real frames to reveal when slowed
    # down via setpts — it just holds each existing frame longer, which
    # can read as choppy/stuttery rather than smooth, especially on the
    # fastest-moving thing in frame (a smash, a diving save) — exactly the
    # two highlight types this targets. Turn this on and watch a handful
    # of real output clips before deciding it should stay on; don't assume
    # it's a clear win the way the thumbnail feature (Tier 1a) is.
    #   - enable_highlight_slowmo: master on/off switch.
    #   - highlight_slowmo_window_s: total width of the slowed window,
    #     centered on the highlight's decisive frame (HighlightEvent.
    #     source_frame_index — only set on POWERFUL_SMASH/SPECTACULAR_SAVE,
    #     see ml/pipeline/highlight_tagging.py). The clip plays at normal
    #     speed everywhere outside this window.
    #   - highlight_slowmo_factor: playback speed during that window, as a
    #     fraction of real-time (0.45 = roughly 2.2x slower than normal).
    enable_highlight_slowmo: bool = False
    highlight_slowmo_window_s: float = 1.0
    highlight_slowmo_factor: float = 0.45

    # --- Reel clip selection (Part 10a) ---
    # See ml/pipeline/reel_selection.py's module docstring for the full
    # reasoning. Both are optional caps applied together by
    # select_clips_for_reel — a video generates one reel per PRD Module 4,
    # so these are flat defaults rather than per-match settings; a
    # caller can still override either at call time.
    #   - reel_max_clips: hard cap on clip count (0 means an empty reel,
    #     no exemption — see module docstring on why this differs from
    #     the target-duration cap).
    #   - reel_target_duration_s: soft cap on total selected duration;
    #     the single highest-importance clip is always kept even if it
    #     alone exceeds this.
    reel_max_clips: int = 10
    reel_target_duration_s: float = 120.0

    # --- Reel ordering & pacing (Part 10b) ---
    # See ml/pipeline/reel_ordering.py's module docstring for the full
    # reasoning.
    #   - reel_ordering_strategy: "chronological" (default — coherent,
    #     in-context viewing order) or "importance" (best-first, for a
    #     teaser-style reel rather than a narrative one).
    #   - reel_transition_gap_s: seconds reserved *between* consecutive
    #     clips on the reel timeline (never before the first clip or
    #     after the last). 0.0 means a hard cut with no reserved gap.
    #     This module only reserves the time — a later FFmpeg assembly
    #     stage decides how to fill it (cut, crossfade, music sting).
    #     Reel Insta-Level Roadmap Tier 1b: defaults to 0.0 (a true hard
    #     cut, no black flash between clips) as a stopgap — the *actual*
    #     right treatment here is a real crossfade (Tier 3a), not a
    #     reserved black segment, which is why this stays at 0.0 rather
    #     than some nonzero "nicer" gap in the meantime: a hard cut is
    #     honest about being unfinished; a black flash just looks broken.
    reel_ordering_strategy: str = "chronological"
    reel_transition_gap_s: float = 0.0

    # --- Pipeline retry (Part 4d) ---
    # Per-stage retry, not whole-video retry — a stage that fails (e.g.
    # `detect` hitting a transient GPU OOM) retries on its own without
    # re-running earlier stages. Backoff doubles each attempt:
    # attempt 0 -> pipeline_retry_backoff_seconds, attempt 1 -> 2x, etc.,
    # capped by pipeline_retry_backoff_max_seconds.
    pipeline_max_retries: int = 3
    pipeline_retry_backoff_seconds: int = 10
    pipeline_retry_backoff_max_seconds: int = 300

    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    @property
    def allowed_video_formats_list(self) -> list[str]:
        return [fmt.strip().lower() for fmt in self.allowed_video_formats.split(",")]

    @property
    def cors_allowed_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """
    Cached so we parse the .env file once per process, not on every request.
    Import this function (not Settings directly) everywhere else in the app.
    """
    return Settings()
