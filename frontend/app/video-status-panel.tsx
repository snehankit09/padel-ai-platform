"use client";

import { useEffect, useState } from "react";
import { ApiError, getVideoStatus } from "@/lib/api-client";
import type { VideoStatus, VideoStatusResponse } from "@/lib/types";

/**
 * Post-upload pipeline progress — Part 11d, extended in Part 11e.
 *
 * Polls GET /videos/{id}/status (lib/api-client.ts's getVideoStatus) and
 * renders it as a five-step stage tracker. The five steps and their
 * order are NOT invented here — they're exactly the five Celery tasks
 * backend/app/workers/tasks.py chains (validate -> detect -> track ->
 * analyze -> done), and `current_stage` is the same field Part 4c's
 * run_stage wrapper writes before each one runs (see that module's own
 * docstring for the full status/current_stage state machine, including
 * why a stage name, not a percentage, is the unit of progress here).
 *
 * Part 11e is the FailureDetail component below: when status is FAILED,
 * it surfaces the actual persisted error_message Part 4d made specific
 * per-stage — not a generic "something went wrong" — see that
 * component's own docstring for why the message is shown verbatim
 * rather than reworded.
 */

const POLL_INTERVAL_MS = 2000;

const STAGES: { key: string; label: string }[] = [
  { key: "validate", label: "Validate" },
  { key: "detect", label: "Detect" },
  { key: "track", label: "Track" },
  { key: "analyze", label: "Analyze" },
  { key: "done", label: "Done" },
];

function isTerminal(status: VideoStatus): boolean {
  return status === "done" || status === "failed";
}

type StageVisual = "complete" | "active" | "failed" | "pending";

/**
 * A stage's visual state given the video's overall status and
 * current_stage. Mirrors run_stage's own state machine (tasks.py's
 * module docstring) rather than inventing a separate notion of
 * progress:
 *   - DONE means every stage succeeded, current_stage having already
 *     been cleared to null by done_stage's own on_success_status write
 *     — so "done" overall short-circuits straight to "every stage
 *     complete" without needing current_stage at all.
 *   - FAILED means current_stage is frozen on whichever stage
 *     ultimately exhausted its retries (run_stage never clears it on
 *     failure) — stages before it succeeded, stages after it never ran.
 *   - Otherwise (pending/queued/processing), current_stage is either
 *     null (pending/queued — the chain hasn't dispatched its first
 *     stage's start-of-task write yet) or the stage presently running.
 */
function stageVisual(
  stageKey: string,
  index: number,
  status: VideoStatus,
  currentStage: string | null
): StageVisual {
  if (status === "done") {
    return "complete";
  }
  const currentIndex = currentStage ? STAGES.findIndex((s) => s.key === currentStage) : -1;
  if (status === "failed") {
    if (stageKey === currentStage) {
      return "failed";
    }
    return currentIndex >= 0 && index < currentIndex ? "complete" : "pending";
  }
  if (currentIndex < 0) {
    return "pending";
  }
  if (index < currentIndex) {
    return "complete";
  }
  return index === currentIndex ? "active" : "pending";
}

function stageLabel(stageKey: string | null): string {
  return STAGES.find((s) => s.key === stageKey)?.label ?? "unknown stage";
}

function overallDotClass(status: VideoStatus): string {
  switch (status) {
    case "done":
      return "is-ok";
    case "failed":
      return "is-error";
    default:
      return "is-pending";
  }
}

function overallLabel(status: VideoStatusResponse): string {
  switch (status.status) {
    case "pending":
      return "Waiting to be queued…";
    case "queued":
      return "Queued for processing…";
    case "processing":
      return `Processing — ${stageLabel(status.current_stage)}`;
    case "done":
      return "Processing complete";
    case "failed":
      return "Processing failed";
  }
}

/**
 * Failure detail — Part 11e.
 *
 * The stepper above already marks the failed stage red; this is what
 * actually answers "why" — the specific, per-stage error_message Part
 * 4d's run_stage wrapper wrote (`Stage '<name>' failed after N
 * attempt(s): <the real exception>`, truncated to 1024 chars — see that
 * module's own docstring), not a generic "something went wrong". Shown
 * verbatim rather than summarized or reworded: this string is often the
 * one piece of information that tells someone whether a failure is worth
 * re-uploading over (a truncated/corrupt file) or reporting as a bug
 * (something the pipeline itself choked on), and paraphrasing it risks
 * losing exactly the part that distinguishes those two cases.
 *
 * error_message is nullable in the schema even though run_stage always
 * sets it before marking a video FAILED (see _set_video_progress) — this
 * still handles a null defensively, with a message that's honest about
 * the gap rather than pretending there's a generic explanation to fall
 * back on.
 */
function FailureDetail({ status }: { status: VideoStatusResponse }) {
  return (
    <div className="failure-panel" role="alert">
      <p className="failure-panel-heading">
        Failed during the <strong>{stageLabel(status.current_stage)}</strong> stage
      </p>
      <p className="failure-panel-message">
        {status.error_message ?? "No error details were recorded for this failure."}
      </p>
    </div>
  );
}

export function VideoStatusPanel({ videoId }: { videoId: string }) {
  const [status, setStatus] = useState<VideoStatusResponse | null>(null);
  // Distinct from a real error state: a poll that fails (network blip,
  // backend momentarily unreachable) shouldn't nuke the last-known
  // status the user is looking at — just note it and keep retrying on
  // the same interval, same as SystemStatusCard's own health check does
  // for a single failed request.
  const [pollWarning, setPollWarning] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    let timeoutId: ReturnType<typeof setTimeout> | undefined;
    let cancelled = false;

    async function poll() {
      try {
        const next = await getVideoStatus(videoId, controller.signal);
        if (cancelled) return;
        setPollWarning(null);
        setStatus(next);
        if (!isTerminal(next.status)) {
          timeoutId = setTimeout(poll, POLL_INTERVAL_MS);
        }
        // Terminal status (done/failed): no further timeout scheduled —
        // that's what actually stops polling.
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted (or videoId changed) mid-request — not a real error
        }
        if (cancelled) return;
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setPollWarning(`Having trouble checking status (${message}) — retrying…`);
        timeoutId = setTimeout(poll, POLL_INTERVAL_MS);
      }
    }

    poll(); // check immediately on mount rather than waiting a full interval

    return () => {
      cancelled = true;
      controller.abort();
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [videoId]);

  if (!status) {
    return (
      <div className="card">
        <div className="status-row">
          <span className="status-dot is-pending" aria-hidden="true" />
          <strong>Checking status…</strong>
        </div>
      </div>
    );
  }

  return (
    <div className="card">
      <div className="status-row">
        <span className={`status-dot ${overallDotClass(status.status)}`} aria-hidden="true" />
        <strong>{overallLabel(status)}</strong>
      </div>

      <ol className="pipeline-stages">
        {STAGES.map((stage, index) => {
          const visual = stageVisual(stage.key, index, status.status, status.current_stage);
          return (
            <li key={stage.key} className={`pipeline-stage is-${visual}`}>
              <span className="pipeline-stage-marker" aria-hidden="true" />
              <span className="pipeline-stage-label">{stage.label}</span>
            </li>
          );
        })}
      </ol>

      {status.status === "failed" ? <FailureDetail status={status} /> : null}

      {pollWarning ? (
        <p className="status-detail" style={{ marginTop: 8 }}>
          {pollWarning}
        </p>
      ) : null}

      <p className="status-detail" style={{ marginTop: 12 }}>
        Video ID: <code>{status.id}</code>
      </p>
    </div>
  );
}
