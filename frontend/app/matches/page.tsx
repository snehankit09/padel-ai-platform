"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { ApiError, getMatches } from "@/lib/api-client";
import { formatFormat, formatPlayedAt, videoStatusLabel } from "@/lib/format";
import type { MatchListItem, VideoStatus } from "@/lib/types";

/**
 * Matches overview page — Part 12b. Lists every match from GET /matches
 * (most-recently-played first, per that route's own ordering) as cards
 * linking into `/matches/[id]`, the match detail page a later Part adds —
 * same "link to a route before it exists yet" pattern the root layout's
 * nav already used to get here in the first place.
 */
export default function MatchesPage() {
  return (
    <>
      <h1 className="page-title">Matches</h1>
      <p className="page-lede">Every match that's been uploaded, most recently played first.</p>
      <MatchList />
    </>
  );
}

type ListState =
  | { phase: "loading" }
  | { phase: "ok"; matches: MatchListItem[] }
  | { phase: "error"; message: string };

function MatchList() {
  const [state, setState] = useState<ListState>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getMatches(controller.signal)
      .then((response) => setState({ phase: "ok", matches: response.matches }))
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted before the request finished — not a real error
        }
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setState({ phase: "error", message });
      });

    return () => controller.abort();
  }, []);

  if (state.phase === "loading") {
    return <p className="status-detail">Loading matches…</p>;
  }

  if (state.phase === "error") {
    return <div className="error-panel">{state.message}</div>;
  }

  if (state.matches.length === 0) {
    return (
      <div className="card empty-state">
        <p>No matches yet.</p>
        <Link href="/" className="button-secondary" style={{ marginTop: 0 }}>
          Upload your first match
        </Link>
      </div>
    );
  }

  return (
    <ul className="match-grid">
      {state.matches.map((match) => (
        <li key={match.id}>
          <MatchCard match={match} />
        </li>
      ))}
    </ul>
  );
}

function MatchCard({ match }: { match: MatchListItem }) {
  return (
    <Link href={`/matches/${match.id}`} className="match-card">
      <div className="match-card-thumb">
        {match.thumbnail_url ? (
          // eslint-disable-next-line @next/next/no-img-element -- thumbnail source is
          // backend-resolved (local static mount or, later, a signed S3 URL); next/image's
          // domain allow-list would need updating for either, so a plain <img> avoids that
          // coupling for what's currently a rarely-populated field.
          <img src={match.thumbnail_url} alt="" />
        ) : (
          <BallPlaceholder />
        )}
      </div>
      <div className="match-card-body">
        <div className="match-card-heading">
          <span className="match-card-date">{formatPlayedAt(match.played_at)}</span>
          <StatusBadge status={match.video_status} />
        </div>
        <p className="match-card-venue">{match.venue ?? "Venue not recorded"}</p>
        <p className="match-card-format">{formatFormat(match.format)}</p>
      </div>
    </Link>
  );
}

function StatusBadge({ status }: { status: VideoStatus | null }) {
  if (status === null) {
    return null;
  }
  return (
    <span className={`match-status-badge is-${status}`}>
      <span className="status-dot" aria-hidden="true" />
      {statusLabel(status)}
    </span>
  );
}

function statusLabel(status: VideoStatus): string {
  return videoStatusLabel(status);
}

/** Same restrained two-seam ball mark as the header's BallMark (layout.tsx), scaled up as a thumbnail placeholder. */
function BallPlaceholder() {
  return (
    <svg viewBox="0 0 18 18" aria-hidden="true" className="match-card-thumb-ball">
      <circle cx="9" cy="9" r="8" fill="var(--color-line)" />
      <path
        d="M2.5 5.5C5 7 6.8 9.6 6.2 15.5M15.5 12.5C13 11 11.2 8.4 11.8 2.5"
        stroke="var(--color-surface)"
        strokeWidth="1.1"
        fill="none"
        strokeLinecap="round"
      />
    </svg>
  );
}
