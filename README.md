# Padel AI Platform

AI-powered padel match analysis and highlight generation, built from the PRD
module by module. This is a learning build — architecture mirrors the real
PRD (cloud, GPU workers, S3) but runs locally with Docker Compose.

## Why this structure

Video processing is slow and CPU/GPU-heavy. A user uploading a 90-minute
match cannot sit and wait for an HTTP response while it's analyzed. So the
system is split into three independently-runnable pieces:

```
padel-ai-platform/
├── backend/            FastAPI app: upload endpoint, status, serves results
│   └── app/
│       ├── main.py          FastAPI entrypoint
│       ├── core/             config + database setup
│       ├── models/           SQLAlchemy ORM models (Part 2)
│       ├── schemas/          Pydantic request/response schemas
│       ├── api/routes/       route handlers, one file per resource
│       ├── services/         business logic (storage, validation, etc.)
│       └── workers/          Celery app + background task definitions
│
├── ml/                  CV/ML pipeline code — imported by workers, not by the API directly
│   ├── detection/            yolo_detector.py (Part 5b), court_detector.py (Part 5c),
│   │                          player_ball_detection.py (Part 5d)
│   ├── tracking/              byte_tracker.py — ByteTrack-style engine (Part 6a)
│   ├── pipeline/              orchestrates detection -> tracking -> events -> stats
│   └── common/                shared utilities — frame_extraction.py (Part 5a), coordinate transforms
│
├── frontend/            Next.js dashboard (Part 11-12)
│
├── storage/              local stand-in for S3 during development
│   ├── uploads/               raw uploaded match videos
│   ├── frames/                 sampled frames extracted per video (Part 5a), one subfolder per video_id
│   ├── detections/             per-frame player/ball detection JSON (Part 5d), one subfolder per video_id
│   ├── clips/                 generated highlight clips
│   └── reels/                 generated reels
│
└── docker-compose.yml    wires Postgres + Redis + backend + worker together
```

## The flow, end to end

1. User uploads a video → `backend` validates it and saves to `storage/uploads/`
2. `backend` writes a `Video` row (status=`pending`) and enqueues a Celery job
3. `worker` picks up the job, runs the `ml/` pipeline: detection → tracking →
   event/highlight detection → stats → reel generation
4. Each stage updates the video's status in Postgres as it completes
5. `backend` API serves the results (highlights, stats, reels) once ready
6. `frontend` dashboard polls status, then renders everything

This mirrors PRD Section 9 (System Architecture) — each stage is a
separately scalable unit, GPU-bound stages (detection, tracking) can scale
independently from CPU-bound ones (stats, API serving).

## Why these tech choices

- **FastAPI**: async-native, matches PRD Section 8, great for I/O-bound
  upload/status endpoints.
- **Postgres**: relational data (Users, Matches, Videos, Highlights, Reels,
  Statistics, Predictions) has real foreign-key relationships — a good fit
  for SQL over a document store.
- **Redis + Celery**: the task queue that lets slow CV work happen outside
  the request/response cycle, with retry-per-stage instead of
  retry-the-whole-video (PRD Reliability NFR).
- **Local storage now, S3-shaped interface**: `STORAGE_BACKEND=local` writes
  to `storage/`; the service layer will be written against an interface so
  swapping to S3 later is a config change, not a rewrite.

## Running it (once Part 2+ adds real routes)

```bash
cp .env.example .env
docker-compose up --build
# API:    http://localhost:8000
# Health: http://localhost:8000/health
```

## Database schema (Part 2)

Models live in `backend/app/models/`, one file per entity, matching PRD
Section 11. Two design points worth knowing before you touch this code:

- **Association objects, not plain join tables.** `match_players` (links
  Match↔Player) and `reel_highlights` (links Reel↔Highlight) both carry
  extra data (`team_number`, `position`) — so they're mapped as their own
  classes (`MatchPlayer`, `ReelHighlight`), not bare SQLAlchemy `Table`
  objects. A plain `secondary=` many-to-many can only add/remove links; it
  can't set extra columns on the join row. See `match_player.py` and
  `reel_highlight.py`. Each parent model exposes a read-only
  `association_proxy` (e.g. `match.players`) for convenient reads; writes
  go through the association object directly (see
  `scripts/sanity_check_models.py` for the pattern).
- **`Statistic` is one row per (match, player, stat_type)**, not a wide
  table with a column per stat — chosen because the PRD explicitly plans
  to add more stat types post-MVP, and this way that's a new enum value,
  not a migration.

Run migrations:
```bash
cd backend
alembic upgrade head          # apply all migrations
alembic revision --autogenerate -m "description"   # after changing models
```

Verify the models work against a real DB:
```bash
cd backend
python -m scripts.sanity_check_models
```

## Build roadmap

- [x] Part 1 — Project setup & architecture
- [x] Part 2 — Database schema & models
- [x] Part 3 — Video upload module (Module 1)
- [ ] Part 4 — Async job pipeline skeleton
  - [x] 4a — Trigger: enqueue on upload
  - [x] 4b — Pipeline task chain with placeholder stages
  - [x] 4c — Status transitions and progress tracking
- [ ] Part 5 — CV prototype: court/player/ball detection (YOLO)
  - [x] 5a — Frame extraction (fills the `validate` stage)
  - [x] 5b — YOLO model setup & inference wrapper (`ml/detection/yolo_detector.py`)
  - [x] 5c — Court detection & calibration (`ml/detection/court_detector.py`)
  - [x] 5d — Player & ball detection (fills the `detect` stage)
- [x] Part 6 — Player/ball tracking (ByteTrack)
  - [x] 6a — Tracking algorithm integration (`ml/tracking/byte_tracker.py`) — generic ByteTrack-style
        engine, mirrors yolo_detector.py's layer: per-frame detections in, per-frame tracked objects
        (persistent track_id) out. Known limitation documented in-module: bbox-IoU association can lose
        identity for a small/fast object (the ball) whose per-frame displacement exceeds its own box
        size — see the module docstring and test_fast_small_object_loses_identity_... in
        test_byte_tracker.py.
  - [x] 6b — Player tracking stage (`app/services/player_tracking_stage.py`) — dedicated ByteTracker
        instance for players only, tuned via `player_track_*` settings, writes `player_tracks.json`.
  - [x] 6c — Ball tracking stage (`app/services/ball_tracking_stage.py`,
        `ml/tracking/ball_interpolation.py`) — a second, independently-tuned (`ball_track_*`)
        ByteTracker instance for the ball, whose raw output is then run through
        `interpolate_ball_gaps`: short gaps (occlusion, motion blur) get linearly interpolated between
        the real detections bracketing them; gaps longer than `ball_track_max_interpolation_gap_frames`
        are left alone rather than fabricated — most commonly a lob carrying the ball out of frame
        entirely. Writes `ball_tracks.json`. Both halves are now wired into `track_stage`
        (`app/workers/tasks.py`).
- [x] Part 7 — Rally & event detection
  - [x] 7a — Rally boundary detection (`ml/pipeline/rally_detection.py`,
        `app/services/rally_detection_stage.py`) — turns `track`'s per-frame ball tracks into a
        list of discrete rally segments (start/end frame + real-world start/end timestamps),
        fills in the first piece of the `analyze` stage. No serve/swing classifier exists yet
        (PRD Section 13: future work), so this uses the best available proxy instead: sustained
        ball trackability. Short gaps (below `rally_activity_gap_tolerance_frames`, a coarser
        tolerance layered on top of Part 6c's own interpolation gap cap) are bridged so a fast
        smash doesn't fracture one rally into two; gaps beyond it — and any surviving activity run
        shorter than `rally_min_duration_frames` — are treated as real between-point dead time.
        Known limitation documented in-module: a very long fully-airborne lob can misread as a
        rally boundary — see the module docstring and the Part 7b+ note about a real swing
        classifier being the more direct fix. Writes `rallies.json`.
  - [x] 7b — Serve identification (`ml/pipeline/serve_detection.py`,
        `app/services/serve_detection_stage.py`) — for each rally from 7a, scans the first
        `serve_detection_window_frames` frames for the closest ball-to-player approach and calls
        that player the server; same "no swing classifier exists yet" posture as 7a, so this reuses
        Parts 5-6's position data instead of new model weights. Uses court-calibrated meters (Part
        5e's homography) when available, raw pixels otherwise, and reports `identified=False` rather
        than guessing when no player is found within the distance threshold. Writes `serves.json`.
  - [x] 7c — Shot/stroke classification (`ml/pipeline/shot_classification.py`,
        `app/services/shot_classification_stage.py`) — finds every in-rally "contact" (a local
        reversal in the ball's vertical motion near a player) and classifies it as smash / lob /
        volley / groundstroke / unknown from three bbox-and-trajectory-derived signals: contact
        height relative to the contacting player's own bounding box (smash), how long the ball
        stays airborne before the next contact (lob), and — only when court calibration is
        available — real-world distance from the net line (volley vs. groundstroke). No pose
        estimation or swing classifier exists in this codebase (PRD Section 13: future work), so a
        contact that isn't a smash or a lob and has no calibration to check is reported `unknown`
        rather than guessed at. Writes `shots.json`.
  - [x] 7d — Point outcome detection (`ml/pipeline/point_outcome.py`,
        `app/services/point_outcome_stage.py`) — per rally, looks at the last real ball detection
        in its frame window and classifies it as out of bounds, at the net (near the net line AND a
        sharp deceleration, checked together), or an honestly-scoped "in bounds end" that covers
        both a winner and a receiving-side unforced error (indistinguishable from position data
        alone — no winner/unforced-error split is attempted). No court calibration for a video means
        every rally comes back `undetermined` rather than a coarser pixel-based guess — unlike 7b/7c,
        "in/out of a rectangle" has no meaningful pixel-only fallback. Writes `outcomes.json`.
  - [x] 7e — Highlight-worthy event tagging (`ml/pipeline/highlight_tagging.py`,
        `app/services/highlight_tagging_stage.py`) — layers the PRD's HighlightType vocabulary on
        top of 7a/7c's output with rule-based thresholds, no new model: `long_rally` (rally duration
        past a threshold), `fast_exchange` (a run of shots with short inter-contact gaps),
        `powerful_smash` (any shot 7c classified as a smash, scored by contact height), and
        `spectacular_save` (a different player returning a smash within a tight response window).
        `winning_shot`, `match_point`, and `break_point` are deliberately NOT tagged: the first needs
        the same winner-vs-unforced-error signal 7d's own module docstring already says this
        pipeline can't determine from position data, and the latter two need live game/set/match
        score state, which no stage anywhere in this codebase tracks yet (PRD Section 13 territory).
        Writes `highlights.json` for Part 8 to turn into padded, trimmed clips.
  - [x] 7f — Wire 7a-7e into `analyze_stage` + persist results
        (`app/services/analyze_persistence_stage.py`) — the last piece of `_analyze`
        (`app/workers/tasks.py`): reads 7a's rally summary and 7e's tagged events back
        off the payload and writes them to Postgres — one `Highlight` row per tagged
        event (`clip_file_path` left NULL for Part 8 to fill in once a real clip
        exists) and a first, deliberately small slice of `Statistic` rows
        (`total_points`, `rally_length_avg`, `longest_rally`, all match-level). Every
        other `StatType` needs either a track_id -> `Player` identity mapping or a
        winner/unforced-error split that don't exist anywhere in this codebase yet —
        real Part 9 work, not faked here. Deletes-then-reinserts only the rows this
        stage owns before writing, so an `analyze` retry (which re-runs every
        sub-stage from scratch) never double-inserts.
  - [x] 7g — Verification run against a real match video
        (`backend/scripts/verify_analyze_pipeline.py`) — same role for 7a-7e as
        `verify_detection_pipeline.py` (5f) / `verify_tracking_pipeline.py` (6e) have for
        their own parts: runs the real pipeline (extraction → detection → tracking →
        rally/serve/shot/outcome/highlight detection) against a real video with no
        Postgres/Celery dependency, then writes labeled frame snapshots at every rally
        boundary and every classified shot plus a markdown checklist
        (`human_review_checklist.md`) so a person can manually confirm correct start/end,
        plausible shot labels, and correct point outcomes for a handful of rallies before
        trusting the pipeline at scale. No ground truth exists anywhere in this codebase for
        a real match video, so this deliberately isn't a pass/fail check — it auto-flags
        rallies worth checking first (very short/long duration, zero shots, mostly-unknown
        shot types, undetermined outcome) and leaves the actual verdict to a human looking
        at the snapshots next to the real footage.
- [x] Part 8 — Highlight clip generation (FFmpeg)
  - [x] 8a — Clip boundary calculation (`ml/pipeline/clip_boundaries.py`,
        `app/services/clip_boundary_stage.py`) — pads each 7e HighlightEvent's tight
        `start_time_s`/`end_time_s` with pre/post-roll (`settings.clip_pre_roll_s` /
        `clip_post_roll_s`), clamps the result to `[0, video_duration_s]`, then tops
        up to `clip_min_duration_s` if clamping left it short. No video file is
        produced yet — writes a flat `clip_boundaries.json` of padded, clamped
        `ClipBoundary` objects for 8b to actually cut.
  - [x] 8b — FFmpeg extraction service (`ml/common/clip_extraction.py`,
        `app/services/clip_extraction_stage.py`) — the actual cutting logic: given a
        source video path and in/out timestamps, produces a real clip file via
        FFmpeg (`-ss` before `-i` + re-encode for frame-accurate boundaries, not
        keyframe-snapped `-c copy`). Kept as its own service, independently
        testable from what calls it, same "swappable backend" shape as
        `app/services/storage.py` (Part 3). The stage half reads 8a's
        `clip_boundaries.json`, cuts one clip per boundary, and UPDATEs the
        matching `Highlight` row (matched back by `event_type` + tight
        start/end, since `Highlight` has no `rally_index` column) with the
        padded start/end times and the new clip file's storage key —
        `clip_file_path` is no longer NULL once `analyze` finishes.
  - [x] 8c — Clip encoding & quality settings (`ml/common/clip_extraction.py`) —
        normalizes every clip regardless of what the source camera/upload
        happened to shoot: downscale-only 1080p resolution cap (never
        upscales — see `_scale_filter`), forced `yuv420p` chroma subsampling
        for broad player compatibility (Safari/iOS in particular), audio
        normalized to AAC/48kHz/stereo/128kbps, and `-movflags +faststart` so
        a clip can start playing/seeking as it streams in rather than only
        after a full download. All fixed module constants, not per-clip
        settings — see the module docstring's "Part 8c" section for why.
  - [x] 8d — Highlight-type label overlay (`ml/common/clip_extraction.py`,
        `app/services/clip_extraction_stage.py`) — burns the clip's
        HighlightType (e.g. "LONG RALLY") into the bottom-left corner via
        FFmpeg's `drawtext`, using a fontfile path rather than a fontconfig
        family lookup so it doesn't depend on the worker container happening
        to have a matching font at runtime (`backend/Dockerfile` installs
        `fonts-dejavu-core` specifically for this). No PRD document is
        included in this codebase to check Module 3's overlay acceptance
        criteria against, and two of the three overlay ideas a "score
        overlays, player names, highlight-type labels" brief would suggest
        are blocked on data that doesn't exist anywhere in this codebase yet:
        `Highlight` has no player/track identity linkage at all (only
        `match_id`), and no stage tracks live game/set/match score state
        (the same gap `ml/pipeline/highlight_tagging.py` already cites for
        why `MATCH_POINT`/`BREAK_POINT` aren't tagged). Player-name and score
        overlays stay deferred until that identity/score data exists for
        real (Part 9+ territory) rather than being faked here; the
        highlight-type label has no such gap (`Highlight.event_type` is
        already populated by Part 7e) so it ships now.
  - [x] 8g — Verification run against a real Part 7 output
        (`backend/scripts/verify_clip_extraction_pipeline.py`) — same role for
        8a/8b/8c/8d as `verify_analyze_pipeline.py` (7g) has for Parts 5-7:
        runs the real pipeline (clip boundary calculation → FFmpeg cutting,
        encoding, and label overlay) with no Postgres/Celery dependency,
        normally against a real `verify_analyze_pipeline.py` (7g) output
        directory rather than re-deriving highlights.json itself (see the
        module docstring on why 8g deliberately doesn't re-run detection/
        tracking/analysis — that's 7g's job). Writes real, playable clip
        files plus a `human_review_checklist.md` so a person can actually
        watch a handful and confirm in/out points feel right, no abrupt
        cuts, and audio/video sync holds up. Layers two things under that
        human step, neither a substitute for it: `flag_boundary` singles out
        clips whose pre/post-roll got silently clamped away by the video's
        own edges (the ones most likely to start or end mid-action) before
        any clip is even cut, and `technical_check` ffprobes each rendered
        clip (has a video/audio stream, duration within tolerance, height
        under the 1080p cap) to catch a broken output file before it wastes
        anyone's time watching it. As with 7g, no ground truth exists here
        for a real match video, so this isn't a pass/fail check — the
        actual verdict on cut quality and A/V sync is left to a human.
- [ ] Part 9 — Statistics engine (Module 5)
- [x] Part 10 — Reel generator (Module 4)
  - [x] 10a — Clip selection logic (`ml/pipeline/reel_selection.py`) — pure logic,
        no Celery/DB/FFmpeg: decides which of Part 8's already-cut `Highlight`
        clips make a reel. One function, `select_clips_for_reel`, covers all
        three cases the brief calls out — top-N (`max_clips` only), a target
        total duration (`target_duration_s` only), or a mix of both — by
        ranking cuttable clips (`clip_file_path` populated) by
        `importance_score` (ties broken by original match order) and greedily
        filling under whichever cap(s) are supplied. Clips without a cut file
        yet are excluded before ranking rather than silently scored. Returns
        the selected subset back in chronological order, not importance
        order — selection and reel ordering are kept as separate concerns, so
        a later ordering/assembly step is free to resequence without this
        module changing. New settings: `reel_max_clips`, `reel_target_duration_s`.
        DB/stage wiring (turning selected `ClipCandidate`s into `ReelHighlight`
        rows) is later Part 10 work, not this piece.
  - [x] 10b — Ordering & pacing (`ml/pipeline/reel_ordering.py`) — pure logic,
        no Celery/DB/FFmpeg: takes 10a's selected clips and decides what
        order they play in and how much gap sits between them.
        `order_clips_for_reel` supports "chronological" (default — a
        rally has to play before the smash it set up makes sense) and
        "importance" (best-first, for a teaser-style reel) strategies,
        both tie-broken by original match order like 10a. `build_reel_timeline`
        then lays the ordered clips out on the reel's own timeline,
        reserving `transition_gap_s` seconds between consecutive clips
        only (never before the first or after the last) — it reserves
        the time but doesn't render the transition; a later FFmpeg
        assembly stage decides whether that gap becomes a hard cut, a
        crossfade, or a music sting. New settings: `reel_ordering_strategy`,
        `reel_transition_gap_s`. DB/stage wiring (writing `ReelHighlight.position`
        rows from a `ReelTimelineEntry` list) is later Part 10 work, not
        this piece.
  - [x] 10c — FFmpeg assembly (`ml/common/reel_assembly.py`) — pure
        ffmpeg-touching logic, no Postgres/Celery: turns an ordered list of
        already-cut Part 8 clip files plus 10b's timeline into one real,
        playable reel file. Concatenates with a hard cut via ffmpeg's
        concat DEMUXER (`-c copy`, no re-encode) rather than the concat
        FILTER, safe only because every clip and every freshly-generated
        black gap segment (filling 10b's reserved `gap_before_s`) is
        encoded to the exact same codec/resolution/pixel-format spec as
        Part 8c's own normalization. No crossfades, no music, no score
        overlays — a black-segment hard cut is the honest floor a richer
        treatment could later replace, same posture Part 8's own overlay
        work already takes toward player-name/score text it has no data
        for.
  - [x] 10d — Title/outro cards (`ml/common/reel_assembly.py`,
        `generate_title_card`/`build_ffmpeg_title_card_command`) — same
        module as 10c: renders a plain black-background, centered-text
        card (reusing 8d's own font file for a consistent look) to the
        identical spec every clip/gap already uses, so it concatenates via
        stream copy right alongside them. A caller opts in by generating
        one and handing its path to `assemble_reel` as the first/last
        entry — this module has no opinion on whether a given reel gets
        one.
  - [x] 10e — Persistence (`app/services/reel_persistence_stage.py`) —
        turns an already-selected, already-ordered `ReelTimelineEntry`
        list into one `Reel` row and one `ReelHighlight` row per entry
        (`position` copied 1:1) — Part 2's association-object pattern,
        same reasoning as `MatchPlayer`. Every `highlight_id` is checked
        against the match's real `Highlight` rows before anything is
        written (refuses, doesn't guess, on a mismatch — same posture as
        9e's `MatchPlayer` check); an empty timeline still writes a real,
        empty `Reel` row rather than being treated as an error, since
        `reel_max_clips=0` is 10a's own honest "no reel" case. Delete-
        then-reinsert idempotency per match, same shape as every earlier
        persistence stage.
  - [x] 10f — Wiring + verification
        (`app/services/reel_generation_stage.py`,
        `backend/scripts/verify_reel_generation_pipeline.py`) — the glue
        that finally plugs Part 10 into the real pipeline: `done_stage`
        (`app/workers/tasks.py`) now calls `run_reel_generation`, which
        reads back every `clip_file_path`-populated `Highlight` row for
        the video's match, runs 10a selection → 10b ordering/pacing → 10e
        persistence (written at `ReelStatus.GENERATING` before ffmpeg
        runs, so a caller polling mid-assembly sees real progress, not
        nothing) → 10c ffmpeg assembly, then UPDATEs the `Reel` row's
        `file_path`/`status` once a real file exists (`READY`) or doesn't
        (`FAILED`, and `done` itself retries via the same backoff every
        other stage uses). No title card or music track gets attached
        automatically — nothing in this codebase picks either today (PRD
        Module 4's "swappable" framing is a human/product decision to
        wire in later, not to fabricate here). The verification half
        mirrors 8g's own role for Part 8: run against a real 8g output
        directory, it runs the real 10a/10b/10c pipeline over real,
        already-cut clip files (no Postgres dependency) and writes one
        real `reel.mp4` plus a `human_review_checklist.md`, so a person
        can actually open and watch the assembled reel start to finish
        and confirm pacing, transitions, and A/V sync — the same "real
        output over a synthetic pass/fail" posture every verify_*_
        pipeline.py script already takes.
- [ ] Part 11 — Frontend: upload UI (11a done — Next.js scaffold, layout, env config, thin API client)
  - [x] 11a — Next.js scaffold, layout, env config, thin API client (`lib/api-client.ts`)
  - [x] 11b — Upload form (`app/upload-form.tsx`)
  - [x] 11c — Upload progress (XHR `onProgress`, `app/upload-form.tsx`)
  - [x] 11d — Post-upload status polling & stage tracker (`app/video-status-panel.tsx`)
  - [x] 11e — Failure detail (`FailureDetail`, `app/video-status-panel.tsx`)
- [ ] Part 12 — Frontend: dashboard
  - [x] 12a — (folded into 11a's scaffold)
  - [x] 12b — Matches overview page (`app/matches/page.tsx`) — every match from
        `GET /matches`, most-recently-played first, linking to `/matches/[id]`.
  - [x] 12c — Match detail: highlights & clips viewer (`app/matches/[id]/page.tsx`)
        — one card per `Highlight` row (Part 7e's tagging, persisted by Part 7f),
        each playing its real trimmed clip inline once Part 8b has extracted one.
  - [x] 12d — Match detail: statistics section (`StatisticsSection`,
        `app/matches/[id]/page.tsx`) — whatever match-level `Statistic` rows
        Part 7f has actually persisted (`GET /matches/{id}/statistics`), plus
        the backend's fixed explanation of why player-level stats aren't
        there yet (Part 9b).
  - [x] 12e — Match detail: reel playback (`ReelSection`,
        `app/matches/[id]/page.tsx`; `GET /matches/{id}/reel`,
        `ReelResponse` in `app/schemas/match.py`) — plays the one assembled
        reel file Part 10f produces per match, once ready. Sits alongside
        12c's individual-clip grid rather than replacing it: Part 10a's
        selection is a top-N/target-duration/mixed subset of a match's
        highlights, not all of them (`ml/pipeline/reel_selection.py`), so
        clips the reel left out would otherwise be unreachable from this
        page. `status: null` on the response (no `Reel` row yet — distinct
        from a real, empty `pending` reel) is a normal state, not an error;
        the section reads `video_status` only to explain that one case
        ("still processing" vs. "failed before reaching the reel stage").
        No polling: `run_reel_generation` runs synchronously inside the
        `done` stage, so by the time `video_status` is `"done"` the reel's
        own status has already resolved to `ready`/`failed`/`pending` — see
        `ReelResponse`'s own docstring for the one narrow window where that
        isn't quite true and why this route doesn't special-case it.
  - [x] 12f — Verification (`backend/scripts/verify_frontend_integration.py`)
        — unlike every earlier verify_*_pipeline.py script (all
        deliberately Postgres-free — see their own docstrings), this one
        IS DB-backed on purpose: it exists to check the frontend-against-
        real-API contract, which only exists once real rows are behind a
        running `app.main:app`. Seeds three real matches through the real
        upload route — "ready" (real FFmpeg-cut clips via
        `ml/common/clip_extraction.extract_clip`, real Statistic rows,
        a real reel assembled by actually running Part 10's
        select → order/pace → persist → assemble pipeline), "still
        processing" (a real upload left untouched — no worker consumes
        it, so it's honestly `pending`/`queued` with zero highlights/
        stats/reel), and "failed" (`Video.status=FAILED` with a real
        `error_message`, no `Reel` row, same as a pipeline that never
        reached `done`) — then asserts every `/matches...` route Part 12
        calls returns exactly the shape `lib/types.ts` expects, and
        ffprobes every real clip/reel file (fetched back over the API's
        own `/media` mount, not just off disk) to confirm each one is
        actually playable before pointing a person at it. Writes
        `verification_report.json` (this run's actual results — see
        `backend/scripts/verification_artifacts_12f/` for the run this
        was verified against) plus `human_review_checklist.md`: exact
        match URLs and a per-page checklist for the one thing this
        script can't do itself — execute the frontend's own client-side
        JS in a real browser and look at the result. Same "real output,
        human confirms the rest" posture 7g/8g/10f already take for
        perceptual judgments; applied here to on-screen rendering.
  - [x] 12g — Match detail: player movement section (`PlayerMovementStatsSection`,
        `app/matches/[id]/page.tsx`; `GET /matches/{id}/player-movement-stats`,
        `PlayerMovementStatsResponse` in `app/schemas/match.py`) — reads back
        what `app/services/player_statistics_stage.py` (9a/9b/9f) already
        computes per tracked player (distance, speed, reaction time,
        smash/net success), instead of leaving that already-computed data
        invisible until a real track_id -> `Player.id` mapping exists. No
        `Statistic` rows exist for these yet (9e is only ever called with an
        empty mapping — see that module's own docstring for why no signal in
        this codebase can tell two teammates on the same court side apart),
        so this section labels each card by court side (`side_a`/`side_b`) or
        track_id rather than a real player name, with the same fixed,
        honest explanation 12d's `player_statistics_pending_reason` already
        gives for the gap. Empty (not an error) until `analyze` reaches 9f
        for this video; not polled, same reasoning as 12e.
