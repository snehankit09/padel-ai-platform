"""
Alembic environment, adapted to:
  1. Pull the DB URL from our app's Settings (app.core.config) instead of
     duplicating it in alembic.ini — one source of truth for DATABASE_URL.
  2. Use SQLAlchemy's async engine via run_sync, since our app uses
     asyncpg everywhere and we don't want a second, sync-only driver
     just for migrations.
  3. Point target_metadata at Base.metadata (via app.models, which
     imports every model) so `alembic revision --autogenerate` can diff
     our models against the real schema.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

# Ensure every model is imported and registered on Base.metadata
from app.core.database import Base
from app.core.config import get_settings
import app.models  # noqa: F401 — import side effect registers all tables

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override whatever's in alembic.ini with our real app settings
settings = get_settings()
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    asyncio.run(run_migrations_online())
