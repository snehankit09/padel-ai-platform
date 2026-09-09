"""
Part 12f — end-to-end verification of the frontend (Parts 11-12) against a
real, running backend: real Postgres rows, real FFmpeg-cut clip files, a
real assembled reel, served over the real API, for a person to actually
click through in a browser afterward.

**Why this script is DB-backed, unlike every verify_*_pipeline.py before
it.** 7g/8g/10f are all deliberately Postgres-free (see their own module
docstrings) — they exist to answer "does this ML/FFmpeg stage's output
look right", a question a database has no bearing on. Part 12f's question
is different: "does the frontend, talking to the real API, render what
Part 2's schema + Part 12's routes actually produce" — the request/
response contract itself, which only exists once real `Match`/`Video`/
`Highlight`/`Reel`/`Statistic` rows are behind a running `app.main:app`.
There's no lighter-weight way to check that than seeding the real tables
through the real ORM models and hitting the real routes.

**What this seeds, and why three matches.** No Celery worker runs here
(this script writes rows directly, the same shortcut test_match_*.py's
own fixtures already take for Statistic/Highlight/Reel rows) — the point
isn't to re-verify the ML pipeline (Parts 5-10 already have their own
verify_*_pipeline.py scripts for that), it's to exercise every rendered
state Part 12's pages actually branch on:

  - Match A ("ready"): a real uploaded video, `VideoStatus.DONE`, four
    `Highlight` rows each with a REAL clip file (cut from the uploaded
    source with ml/common/clip_extraction.py's own extract_clip — the
    exact function app/services/clip_extraction_stage.py calls), four
    match-level `Statistic` rows (the same four
    app/services/analyze_persistence_stage.py actually persists), and a
    real `Reel` row assembled by running the real Part 10 pipeline
    (select_clips_for_reel -> order_clips_for_reel -> build_reel_timeline
    -> persist_reel -> assemble_reel, the exact sequence
    app/services/reel_generation_stage.py's run_reel_generation calls) over
    those four real clips.
  - Match B ("still processing"): a real upload and nothing else —
    `VideoStatus.PENDING` (Celery enqueues `process_video`, but with no
    worker consuming the queue it never advances). This is what a match
    honestly looks like before analysis reaches it: no highlights, no
    statistics, no `Reel` row at all — every blank-state message Part 12b/
    12c/12d/12e's own components render is exercised here, for real, not
    simulated by omitting a seed step.
  - Match C ("failed"): a real upload with `Video.status` set to
    `VideoStatus.FAILED` and a specific `error_message`, mirroring what
    run_stage (app/workers/tasks.py) actually writes when a stage
    exhausts its retries. No `Reel` row exists for it either — a pipeline
    that failed before `done` never reached Part 10 — which is exactly
    the case `ReelResponse.status: None` plus `video_status: "failed"`
    is for (see app/schemas/match.py's own docstring).

**What this checks, and what it can't.** Every GET route Part 12's pages
call (`/matches`, `/matches/{id}`, `/matches/{id}/statistics`,
`/matches/{id}/reel`) is fetched via httpx and asserted against the exact
shape lib/types.ts declares for it, and every real media file (each cut
clip, the assembled reel) is ffprobed to confirm it's actually playable
before anyone is asked to click on it — same `probe_media` shape
scripts/verify_clip_extraction_pipeline.py already uses, reused here
rather than reinvented. The frontend's own pages are fetched too, but
only to confirm the Next.js dev server serves each route without a
server-side exception — every page under app/matches is a "use client"
component (see app/matches/[id]/page.tsx's own docstring), so its actual
data-filled render only happens after browser-side JS fetches from the
API; nothing in this sandbox can execute that JS and look at the result.
That gap is exactly what human_review_checklist.md is for — the same
"real output, human confirms the rest" posture 7g/8g/10f already take for
perceptual judgments (pacing, A/V sync) nothing here has ground truth
for, applied to on-screen rendering instead.

Run with (Postgres + Redis + `uvicorn app.main:app` + `npm run dev` all
already running — this script starts none of them):

    cd backend
    python -m scripts.verify_frontend_integration \\
        --api-base http://localhost:8000 --frontend-base http://localhost:3000
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

import httpx
from sqlalchemy import select

from app.core.database import get_sync_db
from app.core.ml_path import ensure_ml_importable
from app.models.enums import HighlightType, ReelStatus, StatType, VideoStatus
from app.models.highlight import Highlight
from app.models.reel import Reel
from app.models.statistic import Statistic
from app.models.video import Video
from app.services.storage import (
    get_storage_service,
    make_clip_file_destination_path,
    make_reel_file_destination_path,
)


def probe_media(path: str) -> dict:
    """
    ffprobe summary of a media file — same shape and same "trust ffprobe,
    not the file existing" posture as
    scripts/verify_clip_extraction_pipeline.py's own probe_media.
    """
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "stream=codec_type,width,height",
                "-show_entries", "format=duration",
                "-of", "json",
                path,
            ],
            capture_output=True,
        )
    except FileNotFoundError:
        return {"ok": False, "has_video": False, "has_audio": False, "duration_s": None}

    if result.returncode != 0:
        return {"ok": False, "has_video": False, "has_audio": False, "duration_s": None}

    data = json.loads(result.stdout or "{}")
    streams = data.get("streams", [])
    duration_raw = data.get("format", {}).get("duration")
    return {
        "ok": True,
        "has_video": any(s.get("codec_type") == "video" for s in streams),
        "has_audio": any(s.get("codec_type") == "audio" for s in streams),
        "duration_s": float(duration_raw) if duration_raw is not None else None,
    }


def make_sample_video(path: Path, duration_s: int = 60) -> None:
    """A real, valid mp4 (not a mock) — same ffmpeg-lavfi convention test_match_*.py's own fixtures use, just longer so several highlight windows fit inside it."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi", "-i", f"testsrc=duration={duration_s}:size=320x240:rate=15",
            "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration_s}",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(path),
            "-loglevel", "error",
        ],
        check=True,
    )


def upload_match(api_base: str, video_path: Path, venue: str, played_at: str) -> dict:
    with open(video_path, "rb") as f:
        response = httpx.post(
            f"{api_base}/videos/upload",
            files={"file": ("sample.mp4", f, "video/mp4")},
            data={"venue": venue, "format": "singles", "played_at": played_at},
            timeout=60.0,
        )
    response.raise_for_status()
    return response.json()


# Four highlight windows inside the 60s sample video, spaced so none of
# them (plus Part 8's own pre/post-roll padding) run past the source's
# actual duration — real clip_extraction.extract_clip below would raise
# ClipExtractionError otherwise, the same guard it applies to real footage.
_HIGHLIGHT_WINDOWS: list[tuple[HighlightType, float, float, float]] = [
    # (event_type, start_s, end_s, importance_score)
    (HighlightType.LONG_RALLY, 4.0, 14.0, 0.72),
    (HighlightType.POWERFUL_SMASH, 20.0, 24.0, 0.91),
    (HighlightType.FAST_EXCHANGE, 30.0, 36.0, 0.65),
    (HighlightType.SPECTACULAR_SAVE, 44.0, 49.0, 0.83),
]


def seed_ready_match(match_id: str, video_id: str) -> dict:
    """
    Match A. Cuts four real clips from the uploaded source with the real
    ml/common/clip_extraction.extract_clip, persists Highlight rows
    pointing at them, adds the four real match-level Statistic rows Part
    7f actually writes, then runs the real Part 10 pipeline
    (select -> order/pace -> persist -> assemble) over those clips —
    mirroring app/services/reel_generation_stage.py's run_reel_generation
    exactly, just fed hand-seeded Highlight rows instead of real Part 7
    detection output (the same substitution 8g/10f make deliberately for
    their own synthetic-fallback runs).
    """
    ensure_ml_importable()
    from ml.common.clip_extraction import extract_clip
    from ml.common.reel_assembly import assemble_reel
    from ml.pipeline.reel_ordering import build_reel_timeline, order_clips_for_reel
    from ml.pipeline.reel_selection import ClipCandidate, select_clips_for_reel

    storage = get_storage_service()

    with get_sync_db() as db:
        video = db.get(Video, uuid.UUID(video_id))
        source_local_path = storage.get_local_path(video.file_path)

        highlight_ids: list[uuid.UUID] = []
        for index, (event_type, start_s, end_s, score) in enumerate(_HIGHLIGHT_WINDOWS):
            highlight = Highlight(
                match_id=uuid.UUID(match_id),
                event_type=event_type,
                start_time_seconds=start_s,
                end_time_seconds=end_s,
                importance_score=score,
                clip_file_path=None,
            )
            db.add(highlight)
            db.flush()

            clip_key = make_clip_file_destination_path(uuid.UUID(video_id), index)
            clip_local_path = storage.get_local_path(clip_key)
            extract_clip(
                source_local_path, clip_local_path, start_s, end_s,
                label_text=event_type.value.replace("_", " ").upper(),
            )
            highlight.clip_file_path = clip_key
            highlight_ids.append(highlight.id)

        for stat_type, value in [
            (StatType.TOTAL_POINTS, 38.0),
            (StatType.RALLY_LENGTH_AVG, 9.4),
            (StatType.LONGEST_RALLY, 22.1),
            (StatType.ERRORS, 6.0),
        ]:
            db.add(Statistic(match_id=uuid.UUID(match_id), player_id=None, stat_type=stat_type, value=value))

        video.status = VideoStatus.DONE
        video.current_stage = None
        db.commit()

        # Re-read what was actually persisted (not the in-memory objects
        # above) so ClipCandidate is built from the same data the API
        # will later serve — same "don't trust memory, trust the row"
        # posture reel_generation_stage.py itself follows.
        highlights = db.execute(
            select(Highlight).where(Highlight.match_id == uuid.UUID(match_id))
        ).scalars().all()

        candidates = [
            ClipCandidate(
                highlight_id=row.id,
                event_type=row.event_type.value,
                start_time_s=row.start_time_seconds,
                end_time_s=row.end_time_seconds,
                importance_score=row.importance_score,
                clip_file_path=row.clip_file_path,
            )
            for row in highlights
        ]

    selected = select_clips_for_reel(candidates, max_clips=10, target_duration_s=120.0)
    ordered = order_clips_for_reel(selected, strategy="chronological")
    timeline = build_reel_timeline(ordered, transition_gap_s=0.5)

    from app.services.reel_persistence_stage import persist_reel

    reel_id = persist_reel(uuid.UUID(match_id), timeline, status=ReelStatus.GENERATING)

    ordered_clip_paths = [storage.get_local_path(c.clip_file_path) for c in ordered]
    gap_before_s = [entry.gap_before_s for entry in timeline]
    reel_key = make_reel_file_destination_path(uuid.UUID(video_id))
    reel_local_path = storage.get_local_path(reel_key)
    result = assemble_reel(ordered_clip_paths, gap_before_s, reel_local_path)

    with get_sync_db() as db:
        reel = db.get(Reel, reel_id)
        reel.file_path = reel_key
        reel.status = ReelStatus.READY
        db.commit()

    return {
        "highlight_count": len(highlight_ids),
        "clip_paths": ordered_clip_paths,
        "reel_path": reel_local_path,
        "reel_duration_s": result.total_duration_s,
    }


def seed_failed_match(video_id: str) -> None:
    """Match C. No Reel row — a pipeline that fails before `done` never reaches Part 10, same as a match still mid-pipeline (Match B)."""
    with get_sync_db() as db:
        video = db.get(Video, uuid.UUID(video_id))
        video.status = VideoStatus.FAILED
        video.current_stage = "detect"
        video.error_message = (
            "Stage 'detect' failed after 3 attempt(s): "
            "RuntimeError: YOLO inference device 'cuda' is not available in this environment"
        )
        db.commit()


def check_json(label: str, response: httpx.Response, checks: list[dict]) -> dict:
    ok = response.status_code == 200
    body = response.json() if ok else None
    entry = {"label": label, "url": str(response.request.url), "status": response.status_code, "ok": ok}
    if ok:
        entry["body"] = body
    checks.append(entry)
    return entry


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://localhost:8000")
    parser.add_argument("--frontend-base", default="http://localhost:3000")
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="padel_verify_12f_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    api_base = args.api_base.rstrip("/")
    frontend_base = args.frontend_base.rstrip("/")

    print(f"[12f] output dir: {out_dir}")
    print(f"[12f] API base:      {api_base}")
    print(f"[12f] Frontend base: {frontend_base}")

    # --- health check ---
    health = httpx.get(f"{api_base}/health", timeout=10.0)
    health.raise_for_status()
    assert health.json()["database"] is True, "backend /health reports database not reachable"
    print("[12f] backend /health OK, database reachable")

    # --- build + upload real videos for all three matches ---
    sample_video = out_dir / "sample.mp4"
    make_sample_video(sample_video, duration_s=60)
    print(f"[12f] built sample source video: {sample_video} ({probe_media(str(sample_video))})")

    match_a = upload_match(api_base, sample_video, "Court 1 — Ready", "2026-08-10T18:00:00Z")
    match_b = upload_match(api_base, sample_video, "Court 2 — Still Processing", "2026-08-15T18:00:00Z")
    match_c = upload_match(api_base, sample_video, "Court 3 — Failed", "2026-08-18T18:00:00Z")
    print(f"[12f] uploaded match A={match_a['match_id']} B={match_b['match_id']} C={match_c['match_id']}")

    seed_result = seed_ready_match(match_a["match_id"], match_a["video_id"])
    seed_failed_match(match_c["video_id"])
    print(f"[12f] seeded match A: {seed_result['highlight_count']} highlights, reel={seed_result['reel_path']}")
    # match_b intentionally untouched -- see module docstring.

    # --- probe every real media file before asking a human to click on it ---
    media_checks = []
    for clip_path in seed_result["clip_paths"]:
        probe = probe_media(clip_path)
        media_checks.append({"path": clip_path, **probe})
        assert probe["ok"] and probe["has_video"], f"clip failed ffprobe: {clip_path}"
    reel_probe = probe_media(seed_result["reel_path"])
    media_checks.append({"path": seed_result["reel_path"], **reel_probe})
    assert reel_probe["ok"] and reel_probe["has_video"], "assembled reel failed ffprobe"
    print(f"[12f] ffprobed {len(media_checks)} real media files — all playable")

    # --- hit every API route the frontend pages actually call ---
    api_checks: list[dict] = []
    with httpx.Client() as client:
        check_json("GET /matches", client.get(f"{api_base}/matches"), api_checks)

        for label, match_id in [("A (ready)", match_a["match_id"]), ("B (blank)", match_b["match_id"]), ("C (failed)", match_c["match_id"])]:
            check_json(f"GET /matches/{{{label}}}", client.get(f"{api_base}/matches/{match_id}"), api_checks)
            check_json(f"GET /matches/{{{label}}}/statistics", client.get(f"{api_base}/matches/{match_id}/statistics"), api_checks)
            check_json(f"GET /matches/{{{label}}}/reel", client.get(f"{api_base}/matches/{match_id}/reel"), api_checks)

    # --- assert the contract each page's TypeScript types expect ---
    by_label = {c["label"]: c for c in api_checks}

    detail_a = by_label["GET /matches/{A (ready)}"]["body"]
    assert detail_a["video_status"] == "done"
    assert len(detail_a["highlights"]) == 4
    assert all(h["clip_url"] is not None for h in detail_a["highlights"]), "match A: every highlight should have a real clip_url"

    stats_a = by_label["GET /matches/{A (ready)}/statistics"]["body"]
    assert len(stats_a["statistics"]) == 4
    assert stats_a["player_statistics_pending_reason"]

    reel_a = by_label["GET /matches/{A (ready)}/reel"]["body"]
    assert reel_a["status"] == "ready"
    assert reel_a["reel_url"] is not None
    assert reel_a["clip_count"] >= 1

    detail_b = by_label["GET /matches/{B (blank)}"]["body"]
    assert detail_b["video_status"] in ("pending", "queued")
    assert detail_b["highlights"] == [], "match B should have no highlights yet -- honest blank state"

    stats_b = by_label["GET /matches/{B (blank)}/statistics"]["body"]
    assert stats_b["statistics"] == []

    reel_b = by_label["GET /matches/{B (blank)}/reel"]["body"]
    assert reel_b["status"] is None, "match B should have no Reel row yet"
    assert reel_b["reel_url"] is None

    detail_c = by_label["GET /matches/{C (failed)}"]["body"]
    assert detail_c["video_status"] == "failed"

    reel_c = by_label["GET /matches/{C (failed)}/reel"]["body"]
    assert reel_c["status"] is None, "match C failed before reaching the reel stage -- no Reel row"

    print("[12f] every API contract check passed")

    # --- confirm the frontend's own routes come back without a server exception ---
    # (This only proves the Next.js dev server can serve the route shell --
    # every page below is "use client" and fetches its real data in the
    # browser, which nothing here can execute. See module docstring.)
    frontend_checks = []
    frontend_urls = {
        "home": f"{frontend_base}/",
        "matches list": f"{frontend_base}/matches",
        "match A detail": f"{frontend_base}/matches/{match_a['match_id']}",
        "match B detail": f"{frontend_base}/matches/{match_b['match_id']}",
        "match C detail": f"{frontend_base}/matches/{match_c['match_id']}",
    }
    with httpx.Client(follow_redirects=True) as client:
        for label, url in frontend_urls.items():
            response = client.get(url, timeout=30.0)
            html = response.text
            has_error_overlay = "Unhandled Runtime Error" in html or "__next_error__" in html
            ok = response.status_code == 200 and not has_error_overlay
            frontend_checks.append({"label": label, "url": url, "status": response.status_code, "ok": ok})
            assert ok, f"frontend route {label} ({url}) failed: status={response.status_code}, error_overlay={has_error_overlay}"
    print(f"[12f] all {len(frontend_checks)} frontend routes served without a server exception")

    # --- write the machine report + the human checklist ---
    report = {
        "match_a_ready": match_a,
        "match_b_blank": match_b,
        "match_c_failed": match_c,
        "media_checks": media_checks,
        "api_checks": [{k: v for k, v in c.items() if k != "body"} | {"body_summary": _summarize(c.get("body"))} for c in api_checks],
        "frontend_checks": frontend_checks,
    }
    (out_dir / "verification_report.json").write_text(json.dumps(report, indent=2, default=str))

    checklist = _build_checklist(frontend_base, match_a["match_id"], match_b["match_id"], match_c["match_id"])
    (out_dir / "human_review_checklist.md").write_text(checklist)

    print(f"[12f] wrote verification_report.json and human_review_checklist.md to {out_dir}")
    print("[12f] PASS — all automated checks succeeded. Open human_review_checklist.md for the remaining manual pass.")
    return 0


def _summarize(body):
    if body is None:
        return None
    if isinstance(body, dict) and "matches" in body:
        return {"match_count": len(body["matches"])}
    if isinstance(body, dict) and "highlights" in body:
        return {"video_status": body.get("video_status"), "highlight_count": len(body["highlights"])}
    if isinstance(body, dict) and "statistics" in body:
        return {"statistic_count": len(body["statistics"])}
    if isinstance(body, dict) and "clip_count" in body:
        return {"status": body.get("status"), "clip_count": body.get("clip_count")}
    return body


def _build_checklist(frontend_base: str, match_a: str, match_b: str, match_c: str) -> str:
    return f"""# Part 12f — human review checklist

Every automated check in this run passed (see `verification_report.json`
in this same directory): the API returns the exact JSON shape each page's
TypeScript types expect, and every clip/reel file it points at is a real,
ffprobed-playable video. What's left is the part nothing in this sandbox
can do — actually load these pages in a browser and look at them.

Open these with the backend and frontend dev servers both still running:

## 1. Matches list — {frontend_base}/matches
- [ ] All three matches appear, most-recently-played first.
- [ ] "Court 1 — Ready" shows a **Done** status badge.
- [ ] "Court 2 — Still Processing" shows a **Pending** or **Queued** badge.
- [ ] "Court 3 — Failed" shows a **Failed** status badge (in the danger color).

## 2. Match A (ready) — {frontend_base}/matches/{match_a}
- [ ] Statistics section shows 4 stat cards (Total Points, Avg Rally
      Length, Longest Rally, Errors) with real numbers, plus the
      player-stats-pending note underneath.
- [ ] Reel section shows a **Ready** badge and a clip count, and the
      `<video>` element actually plays when you hit play — real footage
      with a burned-in label in the corner of the source clips.
- [ ] Highlights section shows 4 cards, each with its own playable clip,
      correct highlight-type label, time range, and importance bar.
- [ ] Every clip plays without a broken-video icon or a stalled spinner.

## 3. Match B (still processing) — {frontend_base}/matches/{match_b}
- [ ] Statistics section reads "No match statistics yet — these are
      computed once analysis finishes." — not an error, not a blank gap.
- [ ] Reel section reads "The reel will appear here once processing
      finishes." — not "Failed", not a broken player.
- [ ] Highlights section reads "Highlights will appear here once analysis
      finds them."
- [ ] No console errors in the browser devtools on this page.

## 4. Match C (failed) — {frontend_base}/matches/{match_c}
- [ ] Status badge reads **Failed**.
- [ ] Reel section reads "Processing failed before a reel could be
      generated." (not the "still processing" copy — this is the one
      case that tells those two apart).
- [ ] Highlights section reads "No highlights were tagged for this
      match." (Part 12c's own copy for a terminal video_status with zero
      highlights — same message a genuinely-done-but-empty match would
      show, which is an existing, pre-12f wording choice worth knowing
      about rather than being surprised by here.)

## 5. General
- [ ] No hydration warnings in the browser console on any of the four
      pages above.
- [ ] Network tab shows each page's requests going to the expected
      `/matches...` API routes and returning 200.
"""


if __name__ == "__main__":
    sys.exit(main())
