/**
 * Thin client for the backend API — Part 11a.
 *
 * Deliberately thin: one function per real endpoint
 * (backend/app/api/routes/videos.py, backend/app/main.py), each doing
 * exactly one request (fetch for everything except uploadVideo, which
 * needs XMLHttpRequest instead — see its own docstring), typed with
 * lib/types.ts, throwing one shared ApiError shape on any non-2xx
 * response. No caching, no retry logic, no request de-duplication —
 * those are concerns for whatever later Part actually needs them (Part
 * 11d's status polling, for one), not something to bake into the client
 * every caller has to work around.
 *
 * Every function accepts an optional AbortSignal so a caller — Part 11d's
 * polling loop in particular — can cancel an in-flight request (e.g. the
 * component unmounted, or a newer poll superseded an older one) without
 * this file needing to know anything about polling itself.
 */

import { API_BASE_URL } from "./config";
import type {
  HealthResponse,
  MatchDetailResponse,
  MatchListResponse,
  MatchStatisticsResponse,
  PlayerMovementStatsResponse,
  ReelResponse,
  UploadVideoOptions,
  VideoStatusResponse,
  VideoUploadResponse,
} from "./types";

/**
 * One error shape for every non-2xx response this client can produce.
 * `detail` is always a human-readable string, regardless of which of
 * FastAPI's two real error shapes produced it: an HTTPException this
 * backend raises itself (detail is already a plain string — see
 * app/api/routes/videos.py's own 422/404/500 raises) or FastAPI's own
 * built-in request-validation error (detail is an array of
 * {loc, msg, type} objects — e.g. an invalid UUID in the /status path
 * parameter) — parseErrorDetail below normalizes both into one string so
 * every caller only ever has to handle one shape.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Shared by both the fetch-based `request` below and uploadVideo's
 * XMLHttpRequest path (fetch has no upload-progress event, so that one
 * can't use `request` — see uploadVideo's own docstring). Takes an
 * already-parsed body rather than a Response so both call sites can
 * share it despite reading the body two different ways (response.json()
 * vs. JSON.parse(xhr.responseText)).
 */
function detailFromParsedBody(body: unknown, status: number): string {
  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") {
      return detail;
    }
    if (Array.isArray(detail)) {
      // FastAPI's built-in validation error shape: an array of
      // {loc, msg, type}. Join every msg so nothing is silently
      // dropped when there's more than one validation issue.
      return detail
        .map((item) => (item && typeof item === "object" && "msg" in item ? String((item as { msg: unknown }).msg) : JSON.stringify(item)))
        .join("; ");
    }
  }
  return `Request failed with status ${status}`;
}

async function parseErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    return detailFromParsedBody(body, response.status);
  } catch {
    // Response body wasn't JSON (or was empty) — fall through to the generic message below.
    return `Request failed with status ${response.status}`;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, init);
  if (!response.ok) {
    throw new ApiError(response.status, await parseErrorDetail(response));
  }
  return (await response.json()) as T;
}

/** GET /health — proves the API process and its database connection are both reachable. */
export async function getHealth(signal?: AbortSignal): Promise<HealthResponse> {
  return request<HealthResponse>("/health", { signal });
}

/**
 * POST /videos/upload — multipart form upload. Mirrors
 * app/api/routes/videos.py's accepted fields exactly: `file` is
 * required, `played_at`/`venue`/`format` are optional (format defaults
 * server-side to "doubles" if omitted here, same as the route's own
 * default).
 *
 * Built on XMLHttpRequest rather than the `request` helper above (and
 * therefore rather than fetch) specifically for `onProgress` — Part
 * 11c's whole reason for existing: fetch has no upload-progress event at
 * all, only XHR's `upload.onprogress` does. Everything else about this
 * function (error shape, ApiError, abort support) still matches
 * `request`'s behavior so callers don't have to treat this one
 * differently — see detailFromParsedBody, shared by both paths.
 */
export async function uploadVideo(
  file: File,
  options: UploadVideoOptions = {},
  onProgress?: (fraction: number) => void,
  signal?: AbortSignal
): Promise<VideoUploadResponse> {
  const formData = new FormData();
  formData.append("file", file);
  if (options.playedAt) {
    formData.append("played_at", options.playedAt.toISOString());
  }
  if (options.venue) {
    formData.append("venue", options.venue);
  }
  if (options.format) {
    formData.append("format", options.format);
  }

  return new Promise<VideoUploadResponse>((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${API_BASE_URL}/videos/upload`);

    if (signal) {
      if (signal.aborted) {
        reject(new DOMException("Upload aborted.", "AbortError"));
        return;
      }
      signal.addEventListener("abort", () => xhr.abort());
    }

    // lengthComputable is false only if the browser couldn't determine
    // the total size up front (rare for a FormData file upload) — skip
    // reporting rather than emit a meaningless fraction.
    xhr.upload.onprogress = (event) => {
      if (onProgress && event.lengthComputable) {
        onProgress(event.loaded / event.total);
      }
    };

    xhr.onabort = () => reject(new DOMException("Upload aborted.", "AbortError"));
    xhr.onerror = () => reject(new ApiError(0, "Network error during upload."));

    xhr.onload = () => {
      let body: unknown = null;
      try {
        body = xhr.responseText ? JSON.parse(xhr.responseText) : null;
      } catch {
        body = null;
      }
      if (xhr.status >= 200 && xhr.status < 300) {
        resolve(body as VideoUploadResponse);
      } else {
        reject(new ApiError(xhr.status, detailFromParsedBody(body, xhr.status)));
      }
    };

    xhr.send(formData);
  });
}

/** GET /videos/{video_id}/status — the shape Part 11d's polling loop reads `current_stage`/`error_message` from. */
export async function getVideoStatus(videoId: string, signal?: AbortSignal): Promise<VideoStatusResponse> {
  return request<VideoStatusResponse>(`/videos/${encodeURIComponent(videoId)}/status`, { signal });
}

/** GET /matches — the list Part 12b's matches overview page renders, most-recently-played first. */
export async function getMatches(signal?: AbortSignal): Promise<MatchListResponse> {
  return request<MatchListResponse>("/matches", { signal });
}

/** GET /matches/{match_id} — match metadata plus its highlights, for Part 12c's highlights & clips viewer. Throws ApiError(404, ...) if the match doesn't exist. */
export async function getMatch(matchId: string, signal?: AbortSignal): Promise<MatchDetailResponse> {
  return request<MatchDetailResponse>(`/matches/${encodeURIComponent(matchId)}`, { signal });
}

/** GET /matches/{match_id}/statistics — whatever match-level stats Part 9 has actually persisted, for Part 12d's stats section. Throws ApiError(404, ...) if the match doesn't exist. */
export async function getMatchStatistics(
  matchId: string,
  signal?: AbortSignal
): Promise<MatchStatisticsResponse> {
  return request<MatchStatisticsResponse>(`/matches/${encodeURIComponent(matchId)}/statistics`, { signal });
}

/** GET /matches/{match_id}/player-movement-stats — per-tracked-player movement stats (distance, speed, reaction time, smash/net success), not yet matched to a real Player name — see PlayerMovementStatsResponse's own docstring. `tracks: []` means the pipeline hasn't reached that stage yet, not an error. Throws ApiError(404, ...) if the match doesn't exist. */
export async function getMatchPlayerMovementStats(
  matchId: string,
  signal?: AbortSignal
): Promise<PlayerMovementStatsResponse> {
  return request<PlayerMovementStatsResponse>(
    `/matches/${encodeURIComponent(matchId)}/player-movement-stats`,
    { signal }
  );
}

/** GET /matches/{match_id}/reel — the match's assembled reel, for Part 12e's reel player. `status: null` means no Reel row exists yet (not an error — see ReelResponse's own docstring). Throws ApiError(404, ...) if the match doesn't exist. */
export async function getMatchReel(matchId: string, signal?: AbortSignal): Promise<ReelResponse> {
  return request<ReelResponse>(`/matches/${encodeURIComponent(matchId)}/reel`, { signal });
}
