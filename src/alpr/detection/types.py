"""Types shared across the detection stage.

Keeping types in their own module prevents circular imports: the vehicle
detector, the plate detector, and the pipeline can all import from here
without importing each other.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Detection:
    """One bounding-box prediction from a YOLO model.

    All coordinates are in the pixel space of the image that was passed to the
    model -- they must be offset before use if the model was run on a crop.

    Attributes:
        x1, y1, x2, y2: Box corners, float pixels.
        confidence: Model confidence in [0, 1].
        class_id: COCO (or custom) class index.
        track_id: ByteTrack identifier. ``None`` until tracking is applied.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    track_id: int | None = None

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    def with_track(self, track_id: int) -> Detection:
        """Return a copy with ``track_id`` set."""
        return Detection(
            x1=self.x1, y1=self.y1, x2=self.x2, y2=self.y2,
            confidence=self.confidence, class_id=self.class_id,
            track_id=track_id,
        )

    def with_offset(self, dx: float, dy: float) -> Detection:
        """Return a copy translated by ``(dx, dy)``."""
        return Detection(
            x1=self.x1 + dx, y1=self.y1 + dy,
            x2=self.x2 + dx, y2=self.y2 + dy,
            confidence=self.confidence, class_id=self.class_id,
            track_id=self.track_id,
        )

    def __str__(self) -> str:
        tid = f" track={self.track_id}" if self.track_id is not None else ""
        return (
            f"Detection(cls={self.class_id}{tid}, "
            f"conf={self.confidence:.2f}, "
            f"box=[{self.x1:.0f},{self.y1:.0f},{self.x2:.0f},{self.y2:.0f}])"
        )
