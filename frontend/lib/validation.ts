/**
 * Client-side upload validation — Part 11c.
 *
 * Mirrors two of backend/app/services/video_validation.py's checks —
 * allowed extension (backend/app/api/routes/videos.py's own extension
 * check, actually; video_validation.py itself never looks at the
 * filename) and max size (video_validation.py's _check_size) — so a bad
 * file gets rejected before a multi-GB upload even starts, instead of
 * after however long that upload takes.
 *
 * This is a courtesy, not a security boundary: nothing stops a request
 * bypassing this file entirely, so the backend re-runs both checks
 * itself on every upload regardless of what this file decided. Kept
 * separate from lib/config.ts's MAX_UPLOAD_SIZE_MB / ALLOWED_VIDEO_FORMATS
 * (the actual limits) so this file is pure logic, easy to compare
 * line-for-line against the backend checks it mirrors.
 *
 * Deliberately does NOT check duration, unlike video_validation.py's own
 * _check_duration: a File object exposes a name and a byte size, nothing
 * about its contents, so there's no client-side equivalent available
 * without reading the file — that's what ffprobe is for, and why
 * video_validation.py trusts ffprobe over any browser API in the first
 * place (see that module's own docstring). Left to the backend's 422.
 */

import { ALLOWED_VIDEO_FORMATS, MAX_UPLOAD_SIZE_MB } from "./config";

/** Returns a user-facing error message, or null if the file passes both checks. */
export function validateFileClientSide(file: File): string | null {
  // Mirrors app/api/routes/videos.py's `extension not in
  // settings.allowed_video_formats_list` check, message included.
  const extension = file.name.includes(".") ? file.name.split(".").pop()!.toLowerCase() : "";
  if (!ALLOWED_VIDEO_FORMATS.includes(extension)) {
    return (
      `Unsupported file type '.${extension}'. ` +
      `Allowed formats: ${ALLOWED_VIDEO_FORMATS.join(", ")}.`
    );
  }

  // Mirrors video_validation.py's _check_size, message included.
  const maxBytes = MAX_UPLOAD_SIZE_MB * 1024 * 1024;
  if (file.size > maxBytes) {
    return (
      `File is too large (${(file.size / (1024 * 1024)).toFixed(1)} MB). ` +
      `The maximum allowed size is ${MAX_UPLOAD_SIZE_MB} MB.`
    );
  }
  if (file.size === 0) {
    return "The uploaded file is empty.";
  }

  return null;
}
