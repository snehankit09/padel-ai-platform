/**
 * Environment config — Part 11a, extended in Part 11c.
 *
 * One read of each NEXT_PUBLIC_* var, in one place, with a real error if
 * a *required* one is missing rather than every call site silently
 * falling back to some guessed default. A wrong or missing API URL
 * should fail loudly and immediately (at import time, in dev), not show
 * up later as a confusing network error on whatever page happens to
 * call the API first.
 *
 * MAX_UPLOAD_SIZE_MB / ALLOWED_VIDEO_FORMATS (Part 11c) are different:
 * unlike the API URL there's a real, correct default for both — the
 * same default backend/app/core/config.py's own Settings ships
 * (max_upload_size_mb=2048, allowed_video_formats="mp4,mov,avi") — so a
 * missing env var here isn't a misconfiguration, it's just "using the
 * default", and only gets overridden if this frontend is ever pointed
 * at a backend that was deployed with different limits. lib/validation.ts
 * is what actually mirrors the backend's checks; this file only owns
 * where the numbers come from.
 */

function readApiBaseUrl(): string {
  const value = process.env.NEXT_PUBLIC_API_BASE_URL;
  if (!value) {
    throw new Error(
      "NEXT_PUBLIC_API_BASE_URL is not set. Copy frontend/.env.local.example to " +
        "frontend/.env.local and set it to your backend's URL (e.g. http://localhost:8000)."
    );
  }
  // A trailing slash would double up when api-client.ts joins this with a
  // path like "/videos/upload" -> "http://host//videos/upload" — stripped
  // once here so every call site can join paths the same simple way.
  return value.replace(/\/+$/, "");
}

// Mirrors backend/app/core/config.py's Settings.max_upload_size_mb default.
const DEFAULT_MAX_UPLOAD_SIZE_MB = 2048;

// Mirrors backend/app/core/config.py's Settings.allowed_video_formats default.
const DEFAULT_ALLOWED_VIDEO_FORMATS = "mp4,mov,avi";

function readMaxUploadSizeMb(): number {
  const raw = process.env.NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB;
  if (!raw) {
    return DEFAULT_MAX_UPLOAD_SIZE_MB;
  }
  const parsed = Number(raw);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(
      `NEXT_PUBLIC_MAX_UPLOAD_SIZE_MB must be a positive number, got "${raw}".`
    );
  }
  return parsed;
}

function readAllowedVideoFormats(): string[] {
  const raw = process.env.NEXT_PUBLIC_ALLOWED_VIDEO_FORMATS || DEFAULT_ALLOWED_VIDEO_FORMATS;
  // Same split/trim/lowercase shape as the backend's own
  // allowed_video_formats_list property, so the two stay comparable
  // line-for-line if either default ever changes.
  return raw
    .split(",")
    .map((format) => format.trim().toLowerCase())
    .filter(Boolean);
}

export const API_BASE_URL = readApiBaseUrl();
export const MAX_UPLOAD_SIZE_MB = readMaxUploadSizeMb();
export const ALLOWED_VIDEO_FORMATS = readAllowedVideoFormats();
