"""Database engine and session factory.

One engine per process, one session per unit of work. That is the SQLAlchemy
contract, and violating it -- sharing sessions across threads, holding a
session open across the frame loop -- is the source of most SQLAlchemy bugs
in production.

The module exposes two things:

``build_engine(config)``
    Creates (or returns a cached) SQLAlchemy engine from the
    :class:`~alpr.config.schema.DatabaseConfig`. Applies WAL mode for SQLite
    automatically. Call this once at startup.

``session_factory(engine)``
    Returns a context manager that opens a session, commits on clean exit,
    and rolls back on exception. Use it with ``with``:

    .. code-block:: python

        with session_factory(engine) as session:
            session.add(vehicle)
        # committed here; session is closed

Nothing in this module imports from the rest of the application -- the session
factory knows nothing about vehicles or plates. That is what keeps it testable
in isolation with an in-memory SQLite database.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from alpr.config.schema import DatabaseConfig
from alpr.database.models import Base
from alpr.utils.logging import get_logger

logger = get_logger(__name__)

# In-process engine cache: one engine per URL.
_engines: dict[str, Engine] = {}


def build_engine(config: DatabaseConfig) -> Engine:
    """Create (or return a cached) SQLAlchemy engine.

    For SQLite, WAL journal mode is enabled when ``config.wal`` is True.
    WAL is applied via a ``connect`` event so it fires for every new
    connection in the pool, not just the first one.

    Args:
        config: Database settings from :class:`~alpr.config.schema.AppConfig`.

    Returns:
        A :class:`sqlalchemy.engine.Engine` with all tables created.
    """
    if config.url in _engines:
        return _engines[config.url]

    logger.info("creating database engine: %s (echo=%s)", config.url, config.echo)
    engine = create_engine(config.url, echo=config.echo)

    if config.url.startswith("sqlite") and config.wal:
        @event.listens_for(engine, "connect")
        def _set_wal(dbapi_conn, _connection_record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA journal_mode=WAL")
            dbapi_conn.execute("PRAGMA synchronous=NORMAL")  # safe with WAL
            logger.debug("SQLite WAL mode enabled")

    Base.metadata.create_all(engine)
    _engines[config.url] = engine
    return engine


def clear_engine_cache() -> None:
    """Drop the engine cache. For tests that need a fresh engine per test."""
    _engines.clear()


@contextmanager
def session_factory(engine: Engine) -> Generator[Session, None, None]:
    """Context manager that opens, commits, and closes a session.

    Rolls back on any exception so partial writes never reach the database.

    Usage::

        engine = build_engine(config)
        with session_factory(engine) as session:
            session.add(SomeModel(...))
        # committed; session closed

    Raises:
        Any exception raised inside the ``with`` block -- after rolling back.
    """
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
