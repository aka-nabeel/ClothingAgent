"""SQLAlchemy engine, ORM base, and request-scoped database sessions.

Only the catalog and inventory modules use PostgreSQL. Temporary demo carts are
kept by the cart module and do not depend on the legacy runtime tables.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import MetaData
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_config

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Declarative base for read-only mappings to the existing catalog."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


import asyncio

_engine: AsyncEngine | None = None
_engine_loop: asyncio.AbstractEventLoop | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def get_engine() -> AsyncEngine:
    """Create or reuse the async PostgreSQL engine bound to the current running event loop."""
    global _engine, _engine_loop, _session_factory
    try:
        current_loop = asyncio.get_running_loop()
    except RuntimeError:
        current_loop = None

    if _engine is None or (_engine_loop is not None and current_loop is not None and current_loop is not _engine_loop):
        config = get_config()
        _engine = create_async_engine(
            config.database_url,
            pool_pre_ping=True,
            pool_size=config.database_pool_size,
            max_overflow=config.database_max_overflow,
        )
        _engine_loop = current_loop
        _session_factory = None

    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Return the shared async-session factory bound to the current engine."""
    global _session_factory
    engine = get_engine()
    if _session_factory is None:
        _session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return _session_factory


async def get_db() -> AsyncIterator[AsyncSession]:
    """Provide one SQLAlchemy session for an API request."""

    async with get_session_factory()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def close_database() -> None:
    """Dispose the PostgreSQL pool during application shutdown."""
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None
