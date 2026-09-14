"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ApiError, getMatch, getMatchPlayerMovementStats, getMatchReel, getMatchStatistics } from "@/lib/api-client";
import { API_BASE_URL } from "@/lib/config";
import {
  formatFormat,
  formatPlayedAt,
  formatStatValue,
  formatTimestamp,
  reelStatusLabel,
  statLabel,
  videoStatusLabel,
} from "@/lib/format";
import type {
  HighlightItem,
  HighlightType,
  MatchDetailResponse,
  ReelResponse,
  StatisticItem,
  TrackMovementStats,
  VideoStatus,
} from "@/lib/types";

/**
 * Match detail page — Part 12c's per-match highlights & clips viewer (one
 * card per `Highlight` row, each playing its real trimmed clip inline once
 * Part 8b has extracted one), extended in Part 12e with a reel player
 * above it.
 *
 * Part 12e deliberately sits ALONGSIDE 12c's grid, not in place of it: a
 * reel is Part 10's *selection* of the match's best clips assembled into
 * one file (see ml/pipeline/reel_selection.py's own module docstring —
 * top-N / target-duration / a mix, never "all of them"), so a match can
 * easily have highlights the reel left out. Replacing the grid with just
 * the reel would make those clips unreachable from this page entirely.
 *
 * This is the route Part 12b's match cards already link to
 * (`/matches/[id]`) — first thing built against it, not just a stub.
 */
export default function MatchDetailPage({ params }: { params: { id: string } }) {
  return <MatchDetail matchId={params.id} />;
}

type DetailState =
  | { phase: "loading" }
  | { phase: "ok"; match: MatchDetailResponse }
  | { phase: "not-found" }
  | { phase: "error"; message: string };

function MatchDetail({ matchId }: { matchId: string }) {
  const [state, setState] = useState<DetailState>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getMatch(matchId, controller.signal)
      .then((match) => setState({ phase: "ok", match }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted before the request finished — not a real error
        }
        if (error instanceof ApiError && error.status === 404) {
          setState({ phase: "not-found" });
          return;
        }
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setState({ phase: "error", message });
      });

    return () => controller.abort();
  }, [matchId]);

  if (state.phase === "loading") {
    return <p className="status-detail">Loading match…</p>;
  }

  if (state.phase === "not-found") {
    return (
      <div className="card empty-state">
        <p>No match found with that id.</p>
        <Link href="/matches" className="button-secondary" style={{ marginTop: 0 }}>
          Back to matches
        </Link>
      </div>
    );
  }

  if (state.phase === "error") {
    return <div className="error-panel">{state.message}</div>;
  }

  return <MatchDetailView match={state.match} />;
}

function MatchDetailView({ match }: { match: MatchDetailResponse }) {
  return (
    <>
      <Link href="/matches" className="back-link">
        ← Matches
      </Link>
      <div className="match-detail-heading">
        <div>
          <h1 className="page-title">{match.venue ?? "Venue not recorded"}</h1>
          <p className="match-detail-subtitle">
            {formatPlayedAt(match.played_at)} · {formatFormat(match.format)}
          </p>
        </div>
        {match.video_status ? (
          <span className={`match-status-badge is-${match.video_status}`}>
            <span className="status-dot" aria-hidden="true" />
            {videoStatusLabel(match.video_status)}
          </span>
        ) : null}
      </div>

      <section className="meta-section" style={{ marginTop: 32 }}>
        <h2 className="meta-heading">Statistics</h2>
        <StatisticsSection matchId={match.id} />
      </section>

      <section className="meta-section">
        <h2 className="meta-heading">Player movement</h2>
        <PlayerMovementStatsSection matchId={match.id} />
      </section>

      <section className="meta-section">
        <h2 className="meta-heading">Reel</h2>
        <ReelSection matchId={match.id} videoStatus={match.video_status} />
      </section>

      <section className="meta-section">
        <h2 className="meta-heading">Highlights</h2>
        <HighlightsSection highlights={match.highlights} videoStatus={match.video_status} />
      </section>
    </>
  );
}

type StatsState =
  | { phase: "loading" }
  | { phase: "ok"; statistics: StatisticItem[]; pendingReason: string }
  | { phase: "error"; message: string };

function StatisticsSection({ matchId }: { matchId: string }) {
  const [state, setState] = useState<StatsState>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getMatchStatistics(matchId, controller.signal)
      .then((response) =>
        setState({
          phase: "ok",
          statistics: response.statistics,
          pendingReason: response.player_statistics_pending_reason,
        })
      )
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted before the request finished — not a real error
        }
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setState({ phase: "error", message });
      });

    return () => controller.abort();
  }, [matchId]);

  if (state.phase === "loading") {
    return <p className="status-detail">Loading statistics…</p>;
  }

  if (state.phase === "error") {
    return <div className="error-panel">{state.message}</div>;
  }

  return (
    <>
      {state.statistics.length === 0 ? (
        <p className="status-detail">
          No match statistics yet — these are computed once analysis finishes.
        </p>
      ) : (
        <ul className="stat-grid">
          {state.statistics.map((stat) => (
            <li key={stat.stat_type} className="stat-card">
              <span className="stat-card-value">{formatStatValue(stat.stat_type, stat.value)}</span>
              <span className="stat-card-label">{statLabel(stat.stat_type)}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="stat-pending-note">{state.pendingReason}</p>
    </>
  );
}

type PlayerMovementState =
  | { phase: "loading" }
  | { phase: "ok"; tracks: TrackMovementStats[]; usedCourtCalibration: boolean; pendingReason: string }
  | { phase: "error"; message: string };

/**
 * Part 9g. GET /matches/{id}/player-movement-stats reads back what
 * app/services/player_statistics_stage.py (9a/9b/9f) already computes
 * per tracked player, keyed by ByteTrack track_id/court_side rather than
 * a real Player name — no `Statistic` row exists for these yet because
 * nothing in this codebase can tell two teammates on the same court side
 * apart (see that response schema's own docstring). `tracks: []` means
 * the pipeline hasn't reached that stage for this video yet, same
 * "empty is a normal, non-error state" posture as StatisticsSection
 * above — not polled, for the same reason ReelSection isn't: by the time
 * a viewer is looking at this page there's nothing left to catch mid-run.
 */
function PlayerMovementStatsSection({ matchId }: { matchId: string }) {
  const [state, setState] = useState<PlayerMovementState>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getMatchPlayerMovementStats(matchId, controller.signal)
      .then((response) =>
        setState({
          phase: "ok",
          tracks: response.tracks,
          usedCourtCalibration: response.used_court_calibration,
          pendingReason: response.identity_pending_reason,
        })
      )
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted before the request finished — not a real error
        }
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setState({ phase: "error", message });
      });

    return () => controller.abort();
  }, [matchId]);

  if (state.phase === "loading") {
    return <p className="status-detail">Loading player movement…</p>;
  }

  if (state.phase === "error") {
    return <div className="error-panel">{state.message}</div>;
  }

  if (state.tracks.length === 0) {
    return (
      <p className="status-detail">
        No player movement stats yet — these are computed once analysis finishes.
      </p>
    );
  }

  return (
    <>
      <div className="player-track-grid">
        {state.tracks.map((track) => (
          <div key={track.track_id} className="player-track-card">
            <h3 className="player-track-heading">
              {track.court_side ? courtSideLabel(track.court_side) : `Tracked player ${track.track_id}`}
            </h3>
            <ul className="stat-grid">
              {track.stats.map((stat) => (
                <li key={stat.stat_type} className="stat-card">
                  <span className="stat-card-value">{formatStatValue(stat.stat_type, stat.value)}</span>
                  <span className="stat-card-label">{statLabel(stat.stat_type)}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>
      {!state.usedCourtCalibration ? (
        <p className="stat-pending-note">
          No court calibration for this video — distance, speed, and court-side grouping aren't available,
          only reaction time.
        </p>
      ) : null}
      <p className="stat-pending-note">{state.pendingReason}</p>
    </>
  );
}

function courtSideLabel(courtSide: string): string {
  return courtSide === "side_a" ? "Near side" : "Far side";
}

type ReelState =
  | { phase: "loading" }
  | { phase: "ok"; reel: ReelResponse }
  | { phase: "error"; message: string };

/**
 * Part 12e. GET /matches/{id}/reel resolves the reel state in one shot —
 * unlike VideoStatusPanel (Part 11d), this never polls. ReelResponse's
 * own docstring explains why that's sound rather than lazy: run_reel_generation
 * runs synchronously inside the `done` stage, so by the time video_status
 * is "done" the reel has already fully resolved to ready/failed/pending
 * (a real, empty reel) — there's nothing further for a poll to catch.
 * `videoStatus` is only used for the one case a single fetch can't
 * explain on its own: status=null (no Reel row yet) means either
 * "still processing" or "failed before reaching the reel stage", and
 * those two read very differently to someone looking at this page.
 */
function ReelSection({ matchId, videoStatus }: { matchId: string; videoStatus: VideoStatus | null }) {
  const [state, setState] = useState<ReelState>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getMatchReel(matchId, controller.signal)
      .then((reel) => setState({ phase: "ok", reel }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted before the request finished — not a real error
        }
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setState({ phase: "error", message });
      });

    return () => controller.abort();
  }, [matchId]);

  if (state.phase === "loading") {
    return <p className="status-detail">Loading reel…</p>;
  }

  if (state.phase === "error") {
    return <div className="error-panel">{state.message}</div>;
  }

  const { reel } = state;

  if (reel.status === null) {
    const message =
      videoStatus === "failed"
        ? "Processing failed before a reel could be generated."
        : "The reel will appear here once processing finishes.";
    return <p className="status-detail">{message}</p>;
  }

  return (
    <div className="reel-player">
      <div className="reel-player-heading">
        <span className={`reel-status-badge is-${reel.status}`}>
          <span className="status-dot" aria-hidden="true" />
          {reelStatusLabel(reel.status)}
        </span>
        {reel.status === "ready" ? (
          <span className="reel-clip-count">
            {reel.clip_count} {reel.clip_count === 1 ? "clip" : "clips"}
          </span>
        ) : null}
      </div>

      {reel.status === "ready" && reel.reel_url ? (
        <video controls preload="metadata" src={`${API_BASE_URL}${reel.reel_url}`} className="reel-player-media" />
      ) : (
        <p className="status-detail">{reelStatusPlaceholder(reel.status)}</p>
      )}
    </div>
  );
}

/**
 * Copy for every ReelStatus that isn't "ready" (which renders the actual
 * player instead). "pending" reads differently from "failed" here even
 * though both leave reel_url null: pending is Part 10a's own honest
 * zero-clip outcome (no highlights were cut, or reel_max_clips=0), not
 * an error — see ml/pipeline/reel_selection.py's module docstring.
 */
function reelStatusPlaceholder(status: Exclude<ReelResponse["status"], null>): string {
  switch (status) {
    case "pending":
      return "No highlights were available to build a reel for this match.";
    case "generating":
      return "The reel is being assembled — check back shortly.";
    case "ready":
      return ""; // unreachable — handled by the video element above
    case "failed":
      return "Reel generation failed for this match.";
  }
}

function HighlightsSection({
  highlights,
  videoStatus,
}: {
  highlights: HighlightItem[];
  videoStatus: MatchDetailResponse["video_status"];
}) {
  if (highlights.length === 0) {
    const message =
      videoStatus === "done" || videoStatus === null
        ? "No highlights were tagged for this match."
        : "Highlights will appear here once analysis finds them.";
    return <p className="status-detail">{message}</p>;
  }

  return (
    <ul className="highlight-grid">
      {highlights.map((highlight) => (
        <li key={highlight.id}>
          <HighlightCard highlight={highlight} />
        </li>
      ))}
    </ul>
  );
}

function HighlightCard({ highlight }: { highlight: HighlightItem }) {
  return (
    <div className="highlight-card">
      <div className="highlight-card-media">
        {highlight.clip_url ? (
          // poster (Highlights Improvement Roadmap Tier 1a) is simply
          // omitted, not defaulted to anything, when thumbnail_url is
          // null — the browser falls back to its own "first decoded
          // frame" behavior, same as before this field existed, rather
          // than this needing its own placeholder-within-a-placeholder.
          <video
            controls
            preload="metadata"
            poster={highlight.thumbnail_url ? `${API_BASE_URL}${highlight.thumbnail_url}` : undefined}
            src={`${API_BASE_URL}${highlight.clip_url}`}
          />
        ) : (
          <div className="highlight-card-media-placeholder">Clip not extracted yet</div>
        )}
      </div>
      <div className="highlight-card-body">
        <div className="highlight-card-heading">
          <span className="highlight-type-tag">{highlightTypeLabel(highlight.event_type)}</span>
          <span className="highlight-card-time">
            {formatTimestamp(highlight.start_time_seconds)}–{formatTimestamp(highlight.end_time_seconds)}
          </span>
        </div>
        <div className="highlight-score">
          <span className="highlight-score-bar-track">
            <span
              className="highlight-score-bar-fill"
              style={{ width: `${Math.round(highlight.importance_score * 100)}%` }}
            />
          </span>
          <span className="highlight-score-value">{highlight.importance_score.toFixed(2)}</span>
        </div>
        <p className="highlight-score-caption">Importance, relative to other {highlightTypeLabel(highlight.event_type).toLowerCase()} moments</p>
      </div>
    </div>
  );
}

/**
 * "long_rally" -> "Long Rally". Frontend-only display formatting, kept
 * distinct from app/services/clip_extraction_stage.py's own
 * `_format_highlight_label` (which produces "LONG RALLY" for the clip's
 * burned-in overlay text) — two different surfaces with two different
 * casing conventions, not something worth sharing across the
 * frontend/backend boundary for.
 */
function highlightTypeLabel(eventType: HighlightType): string {
  return eventType
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}
