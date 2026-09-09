"""
Application entrypoint.

Route modules (videos, matches, highlights, ...) will be added as
`app.include_router(...)` calls here as we build each module of the PRD.
For now: app setup + a health check that proves Postgres is reachable.
"""

import os

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.routes.matches import router as matches_router
from app.api.routes.videos import router as videos_router
from app.core.config import get_settings
from app.core.database import get_db

settings = get_settings()

app = FastAPI(
    title="Padel AI Platform API",
    description="AI-powered padel match analysis and highlight generation",
    version="0.1.0",
)

# Part 11's frontend runs on a different origin (Next.js dev server,
# localhost:3000) than this API (localhost:8000) — without this, every
# browser-issued request (including the multipart upload) is blocked by
# the browser itself before it even reaches a route. cors_allowed_origins
# is env-configurable (see config.py) so a deployed frontend's real origin
# can replace the local dev default without a code change.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(videos_router, prefix="/videos", tags=["videos"])
app.include_router(matches_router, prefix="/matches", tags=["matches"])

if settings.storage_backend == "local":
    # Serves whatever LocalStorageService.get_url() points at (see
    # app/services/storage.py). Only relevant in development — the S3
    # backend resolves to signed URLs instead and needs no mount.
    os.makedirs(settings.local_storage_path, exist_ok=True)
    app.mount("/media", StaticFiles(directory=settings.local_storage_path), name="media")


@app.get("/")
async def root():
    return {"service": settings.app_name, "status": "running"}


@app.get("/health")
async def health(db: AsyncSession = Depends(get_db)):
    """
    Confirms both the API process and its database connection are alive.
    Useful for docker-compose healthchecks and, later, load balancer probes.
    """
    result = await db.execute(text("SELECT 1"))
    db_ok = result.scalar() == 1
    return {"status": "ok" if db_ok else "degraded", "database": db_ok}

# Future routers, added module by module as we build the PRD.
