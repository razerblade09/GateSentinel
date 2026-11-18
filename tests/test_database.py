"""Tests for the database layer.

Every test uses an in-memory SQLite database (``url="sqlite://"``). There is
no test database file, no teardown fixture that deletes things, and no shared
state between tests -- each fixture creates a fresh engine. That is what makes
these tests fast (< 1 s total) and completely hermetic.

The repository is tested through its public API. We never query the ORM models
directly in assertions, because the whole point of a repository is that the
caller only knows about the public API. If a future refactor renames a column,
the tests should not break -- only the repository implementation changes.
"""

from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from alpr.config.schema import DatabaseConfig
from alpr.database import (
    AccessEvent,
    PlateRead,
    Repository,
    Vehicle,
    build_engine,
    clear_engine_cache,
    session_factory,
)
from alpr.database.session import build_engine as _build_engine
from alpr.ocr.types import PlateRead as OcrPlateRead, TextLine
from alpr.tracking.tracker import VoteResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def engine() -> Engine:
    """Fresh in-memory SQLite engine for each test."""
    clear_engine_cache()
    cfg = DatabaseConfig(url="sqlite://", echo=False, wal=False)
    return build_engine(cfg)


@pytest.fixture
def repo(engine: Engine) -> Repository:
    return Repository(engine, session_id="test-session")


def _ocr_read(
    text: str = "ABC1234",
    confidence: float = 0.9,
    valid: bool = True,
) -> OcrPlateRead:
    return OcrPlateRead(
        raw_text=text,
        text=text if valid else "",
        confidence=confidence,
        format_name="car" if valid else None,
        corrections=0,
        lines=(),
        engine="fake",
    )


def _vote(
    track_id: int = 1,
    text: str = "ABC1234",
    confidence: float = 0.9,
    n_reads: int = 5,
    n_valid: int = 4,
) -> VoteResult:
    return VoteResult(
        track_id=track_id,
        text=text,
        confidence=confidence,
        n_reads=n_reads,
        n_valid_reads=n_valid,
        format_name="car",
    )


# ===========================================================================
# Engine and session
# ===========================================================================


class TestBuildEngine:
    def test_creates_tables(self, engine: Engine) -> None:
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = set(inspector.get_table_names())
        assert {"vehicles", "plate_reads", "access_events"}.issubset(tables)

    def test_returns_cached_engine(self) -> None:
        clear_engine_cache()
        cfg = DatabaseConfig(url="sqlite:///:memory:", echo=False, wal=False)
        e1 = build_engine(cfg)
        e2 = build_engine(cfg)
        assert e1 is e2

    def test_wal_mode_applied_for_sqlite(self, tmp_path) -> None:
        clear_engine_cache()
        db_path = tmp_path / "test.db"
        cfg = DatabaseConfig(url=f"sqlite:///{db_path}", echo=False, wal=True)
        engine = build_engine(cfg)
        # Verify WAL is active by querying the pragma.
        with engine.connect() as conn:
            result = conn.execute(__import__("sqlalchemy").text("PRAGMA journal_mode"))
            mode = result.scalar()
        assert mode == "wal"


class TestSessionFactory:
    def test_commit_on_clean_exit(self, engine: Engine) -> None:
        with session_factory(engine) as s:
            s.add(Vehicle(
                track_id=1, session_id="x",
                first_seen_at=datetime.datetime.now(tz=datetime.timezone.utc),
                last_seen_at=datetime.datetime.now(tz=datetime.timezone.utc),
            ))
        # Data should be visible in a new session.
        with session_factory(engine) as s:
            from sqlalchemy import select
            count = s.execute(
                select(__import__("sqlalchemy").func.count()).select_from(Vehicle)
            ).scalar()
        assert count == 1

    def test_rollback_on_exception(self, engine: Engine) -> None:
        try:
            with session_factory(engine) as s:
                s.add(Vehicle(
                    track_id=99, session_id="x",
                    first_seen_at=datetime.datetime.now(tz=datetime.timezone.utc),
                    last_seen_at=datetime.datetime.now(tz=datetime.timezone.utc),
                ))
                raise ValueError("simulated failure")
        except ValueError:
            pass

        with session_factory(engine) as s:
            from sqlalchemy import select
            count = s.execute(
                select(__import__("sqlalchemy").func.count())
                .select_from(Vehicle)
                .where(Vehicle.track_id == 99)
            ).scalar()
        assert count == 0


# ===========================================================================
# Repository — Vehicle
# ===========================================================================


class TestVehicleRepository:
    def test_open_vehicle_assigns_id(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        assert v.id is not None
        assert v.track_id == 1

    def test_get_vehicle_roundtrip(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=42)
        fetched = repo.get_vehicle(v.id)
        assert fetched is not None
        assert fetched.track_id == 42

    def test_get_vehicle_not_found_returns_none(self, repo: Repository) -> None:
        assert repo.get_vehicle(99999) is None

    def test_find_vehicle_by_track(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=7)
        found = repo.find_vehicle_by_track(7)
        assert found is not None
        assert found.id == v.id

    def test_find_vehicle_wrong_session_returns_none(self, engine: Engine) -> None:
        repo_a = Repository(engine, session_id="session-A")
        repo_b = Repository(engine, session_id="session-B")
        repo_a.open_vehicle(track_id=1)
        assert repo_b.find_vehicle_by_track(1) is None

    def test_touch_vehicle_updates_last_seen(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=3)
        original = repo.get_vehicle(v.id)
        assert original is not None
        original_ts = original.last_seen_at

        # Small sleep to ensure timestamp changes.
        import time
        time.sleep(0.01)
        repo.touch_vehicle(v.id)

        updated = repo.get_vehicle(v.id)
        assert updated is not None
        # last_seen_at should be >= original (may be equal if clock resolution
        # is low, but must not be before).
        assert updated.last_seen_at >= original_ts

    def test_session_id_stored(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=5)
        fetched = repo.get_vehicle(v.id)
        assert fetched is not None
        assert fetched.session_id == "test-session"


# ===========================================================================
# Repository — PlateRead
# ===========================================================================


class TestPlateReadRepository:
    def test_add_read_persists(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        read = repo.add_read(v.id, _ocr_read("ABC1234"))
        assert read.id is not None
        assert read.text == "ABC1234"
        assert read.is_valid is True

    def test_invalid_read_stored(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        read = repo.add_read(v.id, _ocr_read("", valid=False))
        assert read.is_valid is False
        assert read.text == ""

    def test_get_reads_for_vehicle(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        for text in ["ABC1234", "ABC1234", "XYZ9999"]:
            repo.add_read(v.id, _ocr_read(text))
        reads = repo.get_reads_for_vehicle(v.id)
        assert len(reads) == 3
        assert [r.text for r in reads] == ["ABC1234", "ABC1234", "XYZ9999"]

    def test_reads_isolated_between_vehicles(self, repo: Repository) -> None:
        v1 = repo.open_vehicle(track_id=1)
        v2 = repo.open_vehicle(track_id=2)
        repo.add_read(v1.id, _ocr_read("AAA0001"))
        repo.add_read(v2.id, _ocr_read("BBB0002"))
        assert len(repo.get_reads_for_vehicle(v1.id)) == 1
        assert len(repo.get_reads_for_vehicle(v2.id)) == 1

    def test_engine_name_stored(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        read = repo.add_read(v.id, _ocr_read())
        assert read.engine == "fake"

    def test_confidence_stored(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        read = repo.add_read(v.id, _ocr_read(confidence=0.75))
        assert abs(read.confidence - 0.75) < 1e-6


# ===========================================================================
# Repository — AccessEvent
# ===========================================================================


class TestAccessEventRepository:
    def test_record_access_persists(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        event = repo.record_access(
            v.id, _vote(), authorised=True, gate_opened=True
        )
        assert event.id is not None
        assert event.plate == "ABC1234"
        assert event.authorised is True
        assert event.gate_opened is True

    def test_unauthorised_event(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        event = repo.record_access(
            v.id, _vote(text="UNKNOWN"), authorised=False, gate_opened=False
        )
        assert event.authorised is False
        assert event.gate_opened is False

    def test_image_path_stored(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        event = repo.record_access(
            v.id, _vote(), authorised=True, gate_opened=True,
            image_path="/tmp/frame_001.jpg",
        )
        assert event.image_path == "/tmp/frame_001.jpg"

    def test_recent_events_newest_first(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        for i in range(5):
            repo.record_access(
                v.id, _vote(text=f"AAA000{i}"), authorised=True, gate_opened=True
            )
        events = repo.recent_events(limit=5)
        assert len(events) == 5
        # Newest first: last inserted has highest id.
        assert events[0].id > events[-1].id

    def test_recent_events_limit(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        for i in range(10):
            repo.record_access(
                v.id, _vote(text=f"AAA000{i}"), authorised=True, gate_opened=True
            )
        assert len(repo.recent_events(limit=3)) == 3

    def test_authorised_only_filter(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        repo.record_access(v.id, _vote(), authorised=True, gate_opened=True)
        repo.record_access(v.id, _vote(text="DENY"), authorised=False, gate_opened=False)
        events = repo.recent_events(authorised_only=True)
        assert all(e.authorised for e in events)
        assert len(events) == 1

    def test_count_events(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        for _ in range(3):
            repo.record_access(v.id, _vote(), authorised=True, gate_opened=True)
        repo.record_access(v.id, _vote(), authorised=False, gate_opened=False)
        assert repo.count_events() == 4
        assert repo.count_events(authorised_only=True) == 3

    def test_n_reads_stored(self, repo: Repository) -> None:
        v = repo.open_vehicle(track_id=1)
        result = _vote(n_reads=10, n_valid=7)
        event = repo.record_access(v.id, result, authorised=True, gate_opened=True)
        assert event.n_reads == 10
        assert event.n_valid_reads == 7
