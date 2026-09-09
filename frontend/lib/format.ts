import type { ReelStatus, StatType, VideoStatus } from "./types";

/**
 * Small formatting helpers shared between the matches list page
 * (app/matches/page.tsx, Part 12b) and the match detail page
 * (app/matches/[id]/page.tsx, Part 12c) — pulled out here rather than
 * duplicated across both once a second page needed the same "how do we
 * show a played_at / format / video status" logic.
 */

export function formatFormat(format: string): string {
  return format.charAt(0).toUpperCase() + format.slice(1);
}

export function formatPlayedAt(playedAt: string): string {
  return new Date(playedAt).toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

export function videoStatusLabel(status: VideoStatus): string {
  switch (status) {
    case "pending":
      return "Pending";
    case "queued":
      return "Queued";
    case "processing":
      return "Processing";
    case "done":
      return "Done";
    case "failed":
      return "Failed";
  }
}

/** Display label for Part 12e's reel status badge. Doesn't handle null — a null status (no Reel row yet) isn't a ReelStatus value at all, see lib/types.ts's ReelResponse docstring, and the reel section renders its own copy for that case instead of calling this. */
export function reelStatusLabel(status: ReelStatus): string {
  switch (status) {
    case "pending":
      return "No reel yet";
    case "generating":
      return "Generating";
    case "ready":
      return "Ready";
    case "failed":
      return "Failed";
  }
}

/** mm:ss for a clip timeline — highlight durations here are always well under an hour. */
export function formatTimestamp(totalSeconds: number): string {
  const wholeSeconds = Math.max(0, Math.round(totalSeconds));
  const minutes = Math.floor(wholeSeconds / 60);
  const seconds = wholeSeconds % 60;
  return `${minutes}:${seconds.toString().padStart(2, "0")}`;
}

/**
 * Display label + formatted value for the four match-level StatTypes
 * GET /matches/{id}/statistics ever actually returns (see
 * app/schemas/match.py's StatisticItem docstring) — the other eight
 * StatType values are player-level and not persisted anywhere yet
 * (Part 9b), so they're covered by player_statistics_pending_reason on
 * that response, not by this map. A stat_type this map doesn't recognize
 * falls back to its raw enum value rather than throwing, so a future new
 * StatType showing up here doesn't break the page before this map is
 * updated for it.
 */
const STAT_DISPLAY: Partial<Record<StatType, { label: string; format: (value: number) => string }>> = {
  total_points: { label: "Total Points", format: (v) => Math.round(v).toString() },
  rally_length_avg: { label: "Avg Rally Length", format: (v) => `${v.toFixed(1)}s` },
  longest_rally: { label: "Longest Rally", format: (v) => `${v.toFixed(1)}s` },
  errors: { label: "Errors", format: (v) => Math.round(v).toString() },
};

export function statLabel(statType: StatType): string {
  return STAT_DISPLAY[statType]?.label ?? statType;
}

export function formatStatValue(statType: StatType, value: number): string {
  const entry = STAT_DISPLAY[statType];
  return entry ? entry.format(value) : value.toString();
}
