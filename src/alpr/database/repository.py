"""Repository layer: typed CRUD operations over the database models.

The repository pattern puts all database queries in one place. The pipeline
and API layers call repository methods and get back domain objects -- they
never write SQL directly. That boundary has two benefits:

1. **Testability.** The pipeline can be tested against an in-memory SQLite
   database by injecting a repository. No mocks, no patches -- just a fast,
   real database.

2. **Portability.** Moving to PostgreSQL means adding a connection string.
   Moving to a different ORM means rewriting this file, not the pipeline.

The repository is not a God object: it covers only the operations the system
actually needs. YAGNI applies. Add a method when you need it, not before.
"""

from __future__ import annotations

import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.engine import Engine

from alpr.database.models import AccessEvent, PlateRead, Vehicle
from alpr.database.session import session_factory
from alpr.ocr.types import PlateRead as OcrPlateRead
from alpr.tracking.tracker import VoteResult
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


class Repository:
    """CRUD operations over the ALPR database.

    Args:
        engine: A SQLAlchemy engine, typically from
            :func:`~alpr.database.session.build_engine`.
        session_id: Restart-safe label attached to every ``Vehicle`` row.
            Use an ISO timestamp from startup so rows from different runs
            can be distinguished even though ByteTrack IDs reset.
    """

    def __init__(self, engine: Engine, session_id: str = "") -> None:
        self._engine = engine
        self._session_id = session_id

    # ------------------------------------------------------------------
    # Vehicle
    # ------------------------------------------------------------------

    def open_vehicle(self, track_id: int) -> Vehicle:
        """Create and persist a new vehicle track row.

        Args:
            track_id: ByteTrack identifier assigned to this vehicle.

        Returns:
            The persisted :class:`~alpr.database.models.Vehicle` with its
            database ``id`` populated.
        """
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        vehicle = Vehicle(
            track_id=track_id,
            session_id=self._session_id,
            first_seen_at=now,
            last_seen_at=now,
        )
        with session_factory(self._engine) as session:
            session.add(vehicle)
        logger.debug("opened vehicle: track=%d db_id=%d", track_id, vehicle.id)
        return vehicle

    def touch_vehicle(self, vehicle_id: int) -> None:
        """Update ``last_seen_at`` for an active vehicle.

        Call this once per frame that the vehicle is visible. The tracking
        timeout uses wall-clock time, so this is only needed when you want
        the database row to reflect actual observation times.

        Args:
            vehicle_id: Database primary key of the vehicle.
        """
        now = datetime.datetime.now(tz=datetime.timezone.utc)
        with session_factory(self._engine) as session:
            stmt = select(Vehicle).where(Vehicle.id == vehicle_id)
            vehicle = session.scalars(stmt).one_or_none()
            if vehicle is not None:
                vehicle.last_seen_at = now

    def get_vehicle(self, vehicle_id: int) -> Vehicle | None:
        """Fetch a vehicle by database primary key.

        Args:
            vehicle_id: Database primary key.

        Returns:
            The :class:`~alpr.database.models.Vehicle`, or ``None`` if not found.
        """
        with session_factory(self._engine) as session:
            stmt = select(Vehicle).where(Vehicle.id == vehicle_id)
            return session.scalars(stmt).one_or_none()

    def find_vehicle_by_track(
        self, track_id: int, session_id: str | None = None
    ) -> Vehicle | None:
        """Find the most recent vehicle with this ByteTrack ID.

        Args:
            track_id: ByteTrack identifier.
            session_id: If provided, restricts the search to this session.
                Defaults to the repository's own ``session_id``.

        Returns:
            The :class:`~alpr.database.models.Vehicle`, or ``None``.
        """
        sid = session_id if session_id is not None else self._session_id
        with session_factory(self._engine) as session:
            stmt = (
                select(Vehicle)
                .where(Vehicle.track_id == track_id, Vehicle.session_id == sid)
                .order_by(Vehicle.id.desc())
                .limit(1)
            )
            return session.scalars(stmt).one_or_none()

    # ------------------------------------------------------------------
    # PlateRead
    # ------------------------------------------------------------------

    def add_read(self, vehicle_id: int, ocr_read: OcrPlateRead) -> PlateRead:
        """Persist one OCR frame result.

        Args:
            vehicle_id: Database primary key of the owning vehicle.
            ocr_read: Output of :class:`~alpr.ocr.base.PlateRecognizer.read`.

        Returns:
            The persisted :class:`~alpr.database.models.PlateRead`.
        """
        row = PlateRead(
            vehicle_id=vehicle_id,
            read_at=datetime.datetime.now(tz=datetime.timezone.utc),
            raw_text=ocr_read.raw_text,
            text=ocr_read.text,
            confidence=ocr_read.confidence,
            format_name=ocr_read.format_name,
            corrections=ocr_read.corrections,
            engine=ocr_read.engine,
            is_valid=ocr_read.is_valid,
        )
        with session_factory(self._engine) as session:
            session.add(row)
        logger.debug(
            "added read: vehicle_id=%d text=%r valid=%s",
            vehicle_id, ocr_read.text, ocr_read.is_valid,
        )
        return row

    def get_reads_for_vehicle(self, vehicle_id: int) -> Sequence[PlateRead]:
        """Fetch all plate reads for a vehicle, oldest first.

        Args:
            vehicle_id: Database primary key of the vehicle.

        Returns:
            Sequence of :class:`~alpr.database.models.PlateRead` rows.
        """
        with session_factory(self._engine) as session:
            stmt = (
                select(PlateRead)
                .where(PlateRead.vehicle_id == vehicle_id)
                .order_by(PlateRead.id)
            )
            return list(session.scalars(stmt))

    # ------------------------------------------------------------------
    # AccessEvent
    # ------------------------------------------------------------------

    def record_access(
        self,
        vehicle_id: int,
        result: VoteResult,
        *,
        authorised: bool,
        gate_opened: bool,
        image_path: str | None = None,
        notes: str | None = None,
    ) -> AccessEvent:
        """Persist the final gate decision for a vehicle.

        Args:
            vehicle_id: Database primary key of the vehicle.
            result: Output of :class:`~alpr.tracking.tracker.TrackVoter`.
            authorised: Whether the plate was in the allowed list.
            gate_opened: Whether the barrier actually moved.
            image_path: Optional path to a saved frame crop.
            notes: Optional free-text annotation.

        Returns:
            The persisted :class:`~alpr.database.models.AccessEvent`.
        """
        event = AccessEvent(
            vehicle_id=vehicle_id,
            decided_at=datetime.datetime.now(tz=datetime.timezone.utc),
            plate=result.text,
            confidence=result.confidence,
            n_reads=result.n_reads,
            n_valid_reads=result.n_valid_reads,
            format_name=result.format_name,
            authorised=authorised,
            gate_opened=gate_opened,
            image_path=image_path,
            notes=notes,
        )
        with session_factory(self._engine) as session:
            session.add(event)
        logger.info(
            "access event: plate=%r authorised=%s gate=%s",
            result.text, authorised, gate_opened,
        )
        return event

    def recent_events(
        self, limit: int = 50, *, authorised_only: bool = False
    ) -> Sequence[AccessEvent]:
        """Fetch recent access events, newest first.

        Args:
            limit: Maximum number of rows to return.
            authorised_only: When True, only return events where the plate
                was authorised.

        Returns:
            Sequence of :class:`~alpr.database.models.AccessEvent` rows.
        """
        with session_factory(self._engine) as session:
            stmt = select(AccessEvent).order_by(AccessEvent.id.desc()).limit(limit)
            if authorised_only:
                stmt = stmt.where(AccessEvent.authorised.is_(True))
            return list(session.scalars(stmt))

    def count_events(self, *, authorised_only: bool = False) -> int:
        """Return the total number of access events.

        Args:
            authorised_only: When True, count only authorised events.

        Returns:
            Integer count.
        """
        from sqlalchemy import func

        with session_factory(self._engine) as session:
            stmt = select(func.count()).select_from(AccessEvent)
            if authorised_only:
                stmt = stmt.where(AccessEvent.authorised.is_(True))
            result = session.execute(stmt).scalar()
            return int(result or 0)
