"use client";

import { ChangeEvent, FormEvent, useRef, useState } from "react";
import { ApiError, uploadVideo } from "@/lib/api-client";
import { validateFileClientSide } from "@/lib/validation";
import type { VideoUploadResponse } from "@/lib/types";
import { VideoStatusPanel } from "./video-status-panel";

/**
 * Upload form — Part 11b, extended in Part 11c and 11d.
 *
 * Wires the real form fields to POST /videos/upload (via
 * lib/api-client.ts's uploadVideo) and shows whatever the backend
 * actually returns or rejects with.
 *
 * Part 11c adds two things on top of that:
 *   - Immediate file-type/size feedback, via lib/validation.ts, the
 *     moment a file is chosen — instead of only finding out after a
 *     full upload that the backend's 422 (video_validation.py) was
 *     always going to reject it. This is a courtesy check, not a
 *     replacement for the backend's: the submit handler still lets a
 *     request through to the server and shows whatever 422 comes back,
 *     since the backend re-validates regardless of what this form
 *     decided (real duration and file-integrity checks in particular —
 *     video_validation.py's ffprobe checks — have no client-side
 *     equivalent at all; see validateFileClientSide's own docstring).
 *   - A real upload-progress indicator, via uploadVideo's onProgress
 *     callback (lib/api-client.ts), so a multi-GB video doesn't just
 *     sit on an indeterminate "Uploading…" for however long that takes.
 *
 * Part 11d replaces this form's success view (previously the raw POST
 * response and nothing else) with ./video-status-panel.tsx, which polls
 * GET /videos/{id}/status and renders the same current_stage state
 * machine Part 4c's run_stage wrapper writes to Postgres. This file only
 * hands that panel the new video_id — see that file's own docstring for
 * the polling, stage-progress, and (Part 11e) failure-detail logic.
 *
 * The two "doubles"/"singles" options mirror the exact values
 * backend/app/models/match.py's own comment names for `format` — not an
 * enum on the backend, just the two the PRD's format field describes.
 */

const ACCEPTED_EXTENSIONS = ".mp4,.mov,.avi";

type SubmitState =
  | { phase: "idle" }
  | { phase: "submitting"; progress: number }
  | { phase: "success"; result: VideoUploadResponse }
  | { phase: "error"; message: string };

export function UploadForm() {
  const [state, setState] = useState<SubmitState>({ phase: "idle" });
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  // Set the moment a file is chosen (see handleFileChange), separate from
  // `state`'s "error" phase — that one's for a rejected *submission*
  // (client- or server-side); this is a standing note next to the field
  // itself, cleared as soon as a valid file replaces the bad one.
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0] ?? null;
    setSelectedFile(file);
    setFileError(file ? validateFileClientSide(file) : null);
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!selectedFile) {
      setState({ phase: "error", message: "Choose a video file first." });
      return;
    }
    // Belt-and-suspenders: the submit button is already disabled while
    // fileError is set, but a disabled check alone doesn't stop a
    // keyboard/programmatic form submit from reaching here.
    const clientError = validateFileClientSide(selectedFile);
    if (clientError) {
      setFileError(clientError);
      return;
    }

    const form = event.currentTarget;
    const venue = (form.elements.namedItem("venue") as HTMLInputElement).value.trim();
    const format = (form.elements.namedItem("format") as HTMLSelectElement).value;
    const playedAtValue = (form.elements.namedItem("playedAt") as HTMLInputElement).value;

    setState({ phase: "submitting", progress: 0 });
    try {
      const result = await uploadVideo(
        selectedFile,
        {
          venue: venue || undefined,
          format,
          playedAt: playedAtValue ? new Date(playedAtValue) : undefined,
        },
        (fraction) => setState({ phase: "submitting", progress: fraction })
      );
      setState({ phase: "success", result });
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Upload failed unexpectedly.";
      setState({ phase: "error", message });
    }
  }

  if (state.phase === "success") {
    return (
      <>
        <VideoStatusPanel videoId={state.result.video_id} />
        <button
          type="button"
          className="button-secondary"
          onClick={() => {
            setState({ phase: "idle" });
            setSelectedFile(null);
            setFileError(null);
            if (fileInputRef.current) {
              fileInputRef.current.value = "";
            }
          }}
        >
          Upload another
        </button>
      </>
    );
  }

  const isSubmitting = state.phase === "submitting";
  const progressPercent = isSubmitting ? Math.round(state.progress * 100) : 0;

  return (
    <form className="card upload-form" onSubmit={handleSubmit}>
      <label className="field">
        <span className="field-label">Match video</span>
        <input
          ref={fileInputRef}
          type="file"
          name="file"
          accept={ACCEPTED_EXTENSIONS}
          required
          disabled={isSubmitting}
          onChange={handleFileChange}
        />
        {fileError ? <span className="field-error">{fileError}</span> : null}
      </label>

      <div className="field-row">
        <label className="field">
          <span className="field-label">Venue (optional)</span>
          <input type="text" name="venue" placeholder="Central Padel Club" disabled={isSubmitting} />
        </label>

        <label className="field">
          <span className="field-label">Format</span>
          <select name="format" defaultValue="doubles" disabled={isSubmitting}>
            <option value="doubles">Doubles</option>
            <option value="singles">Singles</option>
          </select>
        </label>
      </div>

      <label className="field">
        <span className="field-label">Played on (optional)</span>
        <input type="datetime-local" name="playedAt" disabled={isSubmitting} />
      </label>

      {isSubmitting ? (
        <div className="upload-progress" role="progressbar" aria-valuenow={progressPercent} aria-valuemin={0} aria-valuemax={100}>
          <div className="upload-progress-track">
            <div className="upload-progress-fill" style={{ width: `${progressPercent}%` }} />
          </div>
          <span className="upload-progress-label">
            Uploading… {progressPercent}%
          </span>
        </div>
      ) : null}

      {state.phase === "error" ? <div className="error-panel">{state.message}</div> : null}

      <button type="submit" className="button-primary" disabled={isSubmitting || Boolean(fileError)}>
        {isSubmitting ? "Uploading…" : "Upload match"}
      </button>
    </form>
  );
}
