# Padel AI Platform — Frontend

Next.js (App Router, TypeScript) frontend for the padel highlight
platform. This is Part 11a of the build: project scaffold, shared
layout, environment config, and a thin client wrapping the backend API.
The upload form itself is Part 11b — the home page here is intentionally
just a landing spot plus a live backend connectivity check for now, not
the finished upload flow.

## Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local   # then edit NEXT_PUBLIC_API_BASE_URL if needed
npm run dev
```

Requires the backend running (see the repo root README) and reachable at
`NEXT_PUBLIC_API_BASE_URL` — defaults to `http://localhost:8000`, the
backend's own default dev port. The backend also needs
`CORS_ALLOWED_ORIGINS` (in its own `.env`) to include this frontend's
origin — defaults to `http://localhost:3000`, matching Next.js's own
default dev port, so the two defaults line up out of the box.

## Structure

```
app/
  layout.tsx      Root layout — fonts, global nav, page shell
  page.tsx         Home page — landing + live backend health check
  globals.css      Design tokens and shared page primitives
lib/
  config.ts        Reads and validates NEXT_PUBLIC_API_BASE_URL
  types.ts         TypeScript types mirroring the backend's Pydantic schemas
  api-client.ts    One function per backend endpoint (getHealth, uploadVideo, getVideoStatus)
```

`lib/types.ts` is kept in sync with `backend/app/schemas/video.py` and
`backend/app/models/enums.py` by hand — there's no shared schema
generation between FastAPI and this frontend yet, so a backend schema
change needs the matching edit here too.

## Verification status

The three files in `lib/` (the actual "thin client" deliverable) were
type-checked with `tsc --strict` against real DOM types — zero errors.
The `.tsx` components (`layout.tsx`, `page.tsx`) could not be fully
type-checked in the environment this was built in: no network access to
install `@types/react` or Next.js's own generated types, the same
limitation noted elsewhere in this project's build log for anything
needing a package registry. They were reviewed carefully by hand and
follow standard Next.js App Router conventions, but running

```bash
npm install
npm run typecheck
npm run build
```

locally is the first thing worth doing before building on top of this —
it hasn't been possible to confirm those two commands succeed from
inside the build environment itself.
