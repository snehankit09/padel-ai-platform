"use client";

import { useEffect, useState } from "react";
import { ApiError, getHealth } from "@/lib/api-client";
import type { HealthResponse } from "@/lib/types";
import { UploadForm } from "./upload-form";

/**
 * Home page. Part 11a's own connectivity check now sits below the real
 * upload form (Part 11b) as a secondary "is the backend up" signal
 * rather than the page's main content — the upload form is what this
 * page is actually for.
 */
export default function HomePage() {
  return (
    <>
      <h1 className="page-title">Turn match footage into highlights.</h1>
      <p className="page-lede">
        Upload a padel match and the pipeline finds the rallies, tags the highlight-worthy
        moments, and puts together the stats and a reel.
      </p>
      <UploadForm />
      <section className="meta-section">
        <h2 className="meta-heading">Backend connection</h2>
        <SystemStatusCard />
      </section>
    </>
  );
}

type CheckState =
  | { phase: "loading" }
  | { phase: "ok"; health: HealthResponse }
  | { phase: "degraded"; health: HealthResponse }
  | { phase: "error"; message: string };

function SystemStatusCard() {
  const [state, setState] = useState<CheckState>({ phase: "loading" });

  useEffect(() => {
    const controller = new AbortController();

    getHealth(controller.signal)
      .then((health) => {
        setState({ phase: health.status === "ok" ? "ok" : "degraded", health });
      })
      .catch((error: unknown) => {
        if (error instanceof DOMException && error.name === "AbortError") {
          return; // component unmounted before the request finished — not a real error
        }
        const message = error instanceof ApiError ? error.message : "Could not reach the API.";
        setState({ phase: "error", message });
      });

    return () => controller.abort();
  }, []);

  return (
    <div className="card">
      <div className="status-row">
        <span className={`status-dot ${dotClassFor(state.phase)}`} aria-hidden="true" />
        <strong>{labelFor(state)}</strong>
      </div>
      {state.phase === "ok" || state.phase === "degraded" ? (
        <p className="status-detail">Database connection: {state.health.database ? "reachable" : "unreachable"}</p>
      ) : null}
      {state.phase === "error" ? (
        <div className="error-panel" style={{ marginTop: 12 }}>
          {state.message} — is the backend running at the URL in{" "}
          <code>NEXT_PUBLIC_API_BASE_URL</code>?
        </div>
      ) : null}
    </div>
  );
}

function dotClassFor(phase: CheckState["phase"]): string {
  switch (phase) {
    case "ok":
      return "is-ok";
    case "degraded":
      return "is-degraded";
    case "error":
      return "is-error";
    case "loading":
      return "is-pending";
  }
}

function labelFor(state: CheckState): string {
  switch (state.phase) {
    case "loading":
      return "Checking backend connection…";
    case "ok":
      return "Backend connected";
    case "degraded":
      return "Backend reachable, but degraded";
    case "error":
      return "Backend unreachable";
  }
}
