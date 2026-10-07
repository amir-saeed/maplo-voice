"""Async database engine, unit-of-work sessions and FastAPI wiring."""

from __future__ import annotations

from collections.abc import AsyncIterator  # noqa: TC003 - FastAPI resolves at runtime
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Annotated

import sqlalchemy as sa
from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from maplo_voice.observability import get_logger, instrument_engine

if TYPE_CHECKING:
    from fastapi import FastAPI

    from maplo_voice.config import Settings

log = get_logger(__name__)


class Database:
    """Owns the connection pool. One instance per process."""

    def __init__(self, settings: Settings) -> None:
        cfg = settings.database
        self.engine: AsyncEngine = create_async_engine(
            str(cfg.url),
            pool_size=cfg.pool_size,
            max_overflow=cfg.max_overflow,
            pool_timeout=cfg.pool_timeout_s,
            pool_pre_ping=True,
            pool_recycle=1800,
            echo=cfg.echo,
            connect_args={"server_settings": {"application_name": settings.app_name}},
        )
        instrument_engine(self.engine, settings)
        self.sessionmaker = async_sessionmaker(self.engine, expire_on_commit=False, autoflush=False)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """Unit of work: commit on success, roll back on any exception."""
        async with self.sessionmaker() as session:
            try:
                yield session
                await session.commit()
            except BaseException:
                await session.rollback()
                raise

    async def ping(self) -> None:
        async with self.engine.connect() as conn:
            await conn.execute(sa.text("SELECT 1"))

    async def dispose(self) -> None:
        await self.engine.dispose()
        log.info("database_disposed")


def register_database(app: FastAPI, settings: Settings) -> Database:
    """Attach the database to the app: state, readiness probe and shutdown hook."""
    db = Database(settings)
    app.state.db = db
    app.state.readiness_checks["database"] = db.ping
    app.state.shutdown_hooks.append(db.dispose)
    return db


# --------------------------------------------------------------------------- dependencies
def get_database(request: Request) -> Database:
    db: Database = request.app.state.db
    return db


async def get_session(
    db: Annotated[Database, Depends(get_database)],
) -> AsyncIterator[AsyncSession]:
    async with db.session() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(get_session)]
