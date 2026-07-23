"""Database bootstrap and sessions."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from .config import Settings
from .models import Actor, Base

SAFE_SQLITE_MINIMUM = (3, 51, 3)
SAFE_SQLITE_BACKPORTS = {(3, 44, 6), (3, 50, 7)}


def sqlite_is_patched(version: tuple[int, int, int]) -> bool:
    return version >= SAFE_SQLITE_MINIMUM or version in SAFE_SQLITE_BACKPORTS


def create_database_engine(settings: Settings) -> Engine:
    settings.ensure_directories()
    connect_args = {"check_same_thread": False} if settings.resolved_database_url.startswith("sqlite") else {}
    engine = create_engine(settings.resolved_database_url, connect_args=connect_args, future=True)
    if settings.resolved_database_url.startswith("sqlite"):
        version = sqlite3.sqlite_version_info
        if not sqlite_is_patched(version) and not settings.allow_unsafe_sqlite:
            raise RuntimeError(
                "SQLite WAL requires SQLite >= 3.51.3 (or 3.50.7/3.44.6 backports). "
                f"Found {sqlite3.sqlite_version}. Set SHAKEPI_ALLOW_UNSAFE_SQLITE=true only for development."
            )

        @event.listens_for(engine, "connect")
        def _configure_sqlite(connection: sqlite3.Connection, _record: object) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute(f"PRAGMA busy_timeout={settings.sqlite_busy_timeout_ms}")
            cursor.close()
    return engine


def initialize_database(engine: Engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    with factory.begin() as session:
        if session.query(Actor).filter_by(display_name="anonymous").one_or_none() is None:
            session.add(Actor(display_name="anonymous"))
    return factory


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
