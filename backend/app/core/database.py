"""
Database engine and session management.

We use SQLAlchemy's async engine because FastAPI is async end-to-end —
using a blocking DB driver here would stall the event loop under load.
This file has no models yet; those come in Part 2. For now it just proves
the connection works via the /health endpoint.
"""

from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings

settings = get_settings()

engine = create_async_engine(settings.database_url, echo=settings.debug, future=True)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class every ORM model (Part 2: Users, Matches, Videos, ...) will inherit from."""
    pass


async def get_db():
    """
    FastAPI dependency that yields a DB session per-request and closes it
    afterward. Routes will use this via: db: AsyncSession = Depends(get_db)
    """
    async with AsyncSessionLocal() as session:
        yield session


def _sync_database_url(url: str) -> str:
    """
    Celery tasks (app/workers/) run synchronously — there's no event loop
    to await into, so the async engine/session above (asyncpg) isn't usable
    there. Rather than run an asyncio event loop inside every task just to
    do a couple of ORM writes, workers get their own plain sync engine
    (psycopg2) against the same database. Only the driver differs.
    """
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://")


sync_engine = create_engine(_sync_database_url(settings.database_url), echo=settings.debug, future=True)

SyncSessionLocal = sessionmaker(bind=sync_engine, class_=Session, expire_on_commit=False)


@contextmanager
def get_sync_db():
    """
    Sync counterpart to get_db(), for use in Celery tasks — not a FastAPI
    dependency, just a plain context manager:

        with get_sync_db() as db:
            ...
            db.commit()
    """
    db = SyncSessionLocal()
    try:
        yield db
    finally:
        db.close()
