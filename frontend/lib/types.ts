/**
 * Types mirroring backend/app/schemas/video.py and
 * backend/app/models/enums.py's VideoStatus exactly, field for field —
 * kept in sync by hand for now (no shared schema generation between
 * FastAPI and this frontend yet). If the backend schema changes, this
 * file needs the matching edit; there's no build-time check that catches
 * drift, so treat a change to those Pydantic models as also touching
 * this file.
 */

// Mirrors app/models/enums.py's VideoStatus(str, enum.Enum) values exactly.
export type VideoStatus = "pending" | "queued" | "processing" | "done" | "failed";

// Mirrors app/schemas/video.py's VideoUploadResponse.
export interface VideoUploadResponse {
  video_id: string;
  match_id: string;
  status: VideoStatus;
  original_filename: string;
}

// Mirrors app/schemas/video.py's VideoStatusResponse.
export interface VideoStatusResponse {
  id: string;
  match_id: string;
  status: VideoStatus;
  current_stage: string | null;
  error_message: string | null;
  original_filename: string;
  duration_seconds: number | null;
  resolution_width: number | null;
  resolution_height: number | null;
  fps: number | null;
  file_size_bytes: number | null;
  created_at: string;
  updated_at: string;
}

// Mirrors app/main.py's GET /health response shape.
export interface HealthResponse {
  status: "ok" | "degraded";
  database: boolean;
}

// Mirrors app/schemas/match.py's MatchListItem — one row of GET /matches.
// video_id/video_status/thumbnail_url are all nullable because Match.video
// is nullable at the DB level, and thumbnail_url has no producing stage
// yet regardless (see that schema's own docstring for both).
export interface MatchListItem {
  id: string;
  played_at: string;
  venue: string | null;
  format: string;
  video_id: string | null;
  video_status: VideoStatus | null;
  thumbnail_url: string | null;
}

// Mirrors app/schemas/match.py's MatchListResponse.
export interface MatchListResponse {
  matches: MatchListItem[];
}

// Mirrors app/models/enums.py's HighlightType(str, enum.Enum) values exactly.
export type HighlightType =
  | "long_rally"
  | "winning_shot"
  | "spectacular_save"
  | "match_point"
  | "break_point"
  | "fast_exchange"
  | "powerful_smash";

// Mirrors app/schemas/match.py's HighlightItem — one row of
// MatchDetailResponse.highlights. importance_score is in [0, 1] and only
// meaningful for comparing highlights of the *same* event_type, not
// across types — see that schema's own docstring. clip_url is null
// whenever the clip hasn't been extracted yet (or failed to extract) for
// this highlight, not just when the match overall isn't done processing.
// thumbnail_url follows the same null-until-extracted convention
// independently of clip_url — a clip can have one set without the other,
// since thumbnail extraction is a soft failure server-side (see
// HighlightItem's own docstring in schemas/match.py).
export interface HighlightItem {
  id: string;
  event_type: HighlightType;
  start_time_seconds: number;
  end_time_seconds: number;
  importance_score: number;
  clip_url: string | null;
  thumbnail_url: string | null;
}

// Mirrors app/schemas/match.py's MatchDetailResponse — GET /matches/{id}.
export interface MatchDetailResponse {
  id: string;
  played_at: string;
  venue: string | null;
  format: string;
  video_id: string | null;
  video_status: VideoStatus | null;
  highlights: HighlightItem[];
}

// Mirrors app/models/enums.py's StatType(str, enum.Enum) values exactly —
// only the four this app can actually persist today are ever returned by
// GET /matches/{id}/statistics (see StatisticItem below), but the full
// set is listed here so this type stays traceable to the real enum.
export type StatType =
  | "total_points"
  | "winners"
  | "errors"
  | "serve_percentage"
  | "rally_length_avg"
  | "net_success_rate"
  | "smash_success_rate"
  | "distance_covered"
  | "movement_speed_avg"
  | "longest_rally"
  | "reaction_time_avg"
  | "momentum_possession";

// Mirrors app/schemas/match.py's StatisticItem — one row of GET
// /matches/{id}/statistics. In practice stat_type is always one of
// total_points/rally_length_avg/longest_rally/errors, the only StatTypes
// Part 7f persists — see that schema's own docstring.
export interface StatisticItem {
  stat_type: StatType;
  value: number;
}

// Mirrors app/schemas/match.py's MatchStatisticsResponse.
export interface MatchStatisticsResponse {
  match_id: string;
  statistics: StatisticItem[];
  player_statistics_pending_reason: string;
}

// Mirrors app/schemas/match.py's TrackStatValue — one stat within
// TrackMovementStats. sample_size mirrors ml.pipeline.stats_aggregation's
// StatValue field: how many underlying observations produced `value`, for
// flagging a rate computed from very few attempts.
export interface TrackStatValue {
  stat_type: StatType;
  value: number;
  sample_size: number;
}

// Mirrors app/schemas/match.py's TrackMovementStats — every computed stat
// for one ByteTrack track_id (not yet a real Player.id — see
// PlayerMovementStatsResponse below). court_side/side_confidence are null
// when the video has no court calibration.
export interface TrackMovementStats {
  track_id: number;
  court_side: "side_a" | "side_b" | null;
  side_confidence: number | null;
  stats: TrackStatValue[];
}

// Mirrors app/schemas/match.py's PlayerMovementStatsResponse — GET
// /matches/{id}/player-movement-stats. tracks is empty (not an error)
// until `analyze` reaches Part 9f for this video; identity_pending_reason
// explains why entries are labeled by track_id/court_side rather than a
// real player name — see that schema's own docstring.
export interface PlayerMovementStatsResponse {
  match_id: string;
  video_status: VideoStatus | null;
  used_court_calibration: boolean;
  tracks: TrackMovementStats[];
  identity_pending_reason: string;
}

// Mirrors app/models/enums.py's ReelStatus(str, enum.Enum) values exactly.
export type ReelStatus = "pending" | "generating" | "ready" | "failed";

// Mirrors app/schemas/match.py's ReelResponse — GET /matches/{id}/reel.
// status is null when no Reel row exists yet for this match at all —
// see that schema's own docstring for why that's distinct from every
// real ReelStatus value, including "pending" (a real, empty zero-clip
// reel). reel_url is null whenever status isn't "ready".
export interface ReelResponse {
  match_id: string;
  status: ReelStatus | null;
  reel_url: string | null;
  clip_count: number;
}

// Options for uploadVideo — mirrors the multipart form fields
// POST /videos/upload actually accepts (app/api/routes/videos.py).
// `format` matches the Match model's own field name and default
// ("doubles") — not renamed to something friendlier here, so it stays
// obviously traceable back to that one call site if the backend's
// accepted values ever change.
export interface UploadVideoOptions {
  playedAt?: Date;
  venue?: string;
  format?: string;
}
