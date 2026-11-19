"""SQLAlchemy ORM models for the ALPR database layer.

Three tables, one relationship chain:

    Vehicle ──< PlateRead       one vehicle, many per-frame reads
    Vehicle ──< AccessEvent     one vehicle, one final gate decision

**Why three tables and not one?**

The gate decision (``AccessEvent``) is the atomic business fact: a plate was
read, authorised (or not), and the barrier moved. ``PlateRead`` rows are
evidence -- they feed the vote aggregator and then get archived. Keeping them
separate means:

- The access log stays compact and fast to query even as millions of reads
  accumulate over months.
- Individual reads can be audited after the fact ("why did it open for X?")
  without cluttering the access log.
- Vehicle rows hold per-track metadata (first seen, last seen) independently
  of how many times the plate was read.

**SQLite WAL mode** is set by the session factory when ``config.wal`` is True.
WAL lets the capture loop INSERT while an API client SELECTs -- without it,
any read that overlaps a write serialises both, adding latency to the frame
loop. This is not premature optimisation; it is what makes SQLite viable at
all for a real-time gate.
"""

from __future__ import annotations

import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Vehicle(Base):
    """One continuous track (a vehicle passing the gate).

    A new row is created when ByteTrack assigns a fresh ``track_id``. The row
    is updated as the vehicle lingers in frame and closed when the track times
    out or is explicitly finalised.

    ``track_id`` is the ByteTrack identifier for the session; it resets on
    restart and is not globally unique -- that is what the ``id`` primary key
    is for.
    """

    __tablename__ = "vehicles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    track_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    session_id: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="",
        doc="Restart-safe session label (e.g. ISO timestamp at startup).",
    )
    first_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_seen_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    reads: Mapped[list[PlateRead]] = relationship(
        "PlateRead", back_populates="vehicle", cascade="all, delete-orphan"
    )
    access_events: Mapped[list[AccessEvent]] = relationship(
        "AccessEvent", back_populates="vehicle", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Vehicle id={self.id} track={self.track_id}>"


class PlateRead(Base):
    """One OCR result from one frame.

    Every call to ``PlateRecognizer.read()`` that is attached to a tracked
    vehicle produces one row here. These accumulate until the vote aggregator
    closes the track, at which point an ``AccessEvent`` is written and reads
    can be archived.

    ``is_valid`` mirrors ``PlateRead.is_valid`` from the OCR types: True when
    the read matched a known plate format above the confidence threshold.
    """

    __tablename__ = "plate_reads"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    read_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # OCR result fields -- mirrors alpr.ocr.types.PlateRead
    raw_text: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    text: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    format_name: Mapped[str | None] = mapped_column(String(32), nullable=True)
    corrections: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    engine: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    is_valid: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    vehicle: Mapped[Vehicle] = relationship("Vehicle", back_populates="reads")

    def __repr__(self) -> str:
        return f"<PlateRead id={self.id} text={self.text!r} valid={self.is_valid}>"


class AccessEvent(Base):
    """One gate decision — the final voted result for a vehicle track.

    Written once per vehicle when the vote aggregator closes the track. The
    ``plate`` field is the winning string (empty when no valid read was
    collected). ``authorised`` records whether the plate was in the allowed
    list at the time of the decision; it is stored even for unreadable plates
    so the access log is complete.

    ``image_path`` optionally stores the path to a saved frame crop. At a
    real gate this gives the operator a photo to review when a dispute arises.
    """

    __tablename__ = "access_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    vehicle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("vehicles.id", ondelete="CASCADE"), nullable=False, index=True
    )
    decided_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Vote result
    plate: Mapped[str] = mapped_column(String(32), nullable=False, default="")
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    n_reads: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    n_valid_reads: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    format_name: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # Gate decision
    authorised: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    gate_opened: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    # Optional evidence
    image_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    notes: Mapped[str | None] = mapped_column(Text, nullable=True)

    vehicle: Mapped[Vehicle] = relationship("Vehicle", back_populates="access_events")

    def __repr__(self) -> str:
        return (
            f"<AccessEvent id={self.id} plate={self.plate!r} "
            f"authorised={self.authorised}>"
        )
