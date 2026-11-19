"""Types for the recognition stage.

The important design choice here is that a read carries its *provenance*, not
just a string: which engine produced it, which lines it came from, how many
characters had to be corrected, and how confident the engine was. A bare string
cannot be voted on, thresholded, or debugged -- and all three are needed before
a plate is allowed to open a barrier.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextLine:
    """One line of text as returned by an OCR engine.

    ``box`` is ``(x1, y1, x2, y2)`` in the coordinate space of the crop that was
    passed to the engine, and is what lets multi-line plates be reassembled in
    reading order rather than in whatever order the engine happened to emit.
    """

    text: str
    confidence: float
    box: tuple[float, float, float, float] | None = None

    @property
    def y_center(self) -> float:
        return (self.box[1] + self.box[3]) / 2 if self.box else 0.0

    @property
    def x_start(self) -> float:
        return self.box[0] if self.box else 0.0

    @property
    def height(self) -> float:
        return (self.box[3] - self.box[1]) if self.box else 0.0


@dataclass(frozen=True, slots=True)
class PlateRead:
    """The result of recognising one plate crop.

    Attributes:
        text: Final plate string after correction. Empty when the read failed
            validation -- callers check :attr:`is_valid` rather than truthiness
            of the string, so that "failed" and "empty" stay distinguishable.
        raw_text: Normalised concatenation before correction, kept for
            debugging and for measuring how much correction is doing.
        confidence: Aggregate in [0, 1], already penalised for corrections.
        format_name: Which plate format matched, e.g. "car".
        corrections: Number of positional substitutions applied.
        lines: The raw engine output, in reading order.
        engine: Name of the engine that produced this read.
    """

    raw_text: str
    text: str
    confidence: float
    format_name: str | None
    corrections: int
    lines: tuple[TextLine, ...]
    engine: str

    @property
    def is_valid(self) -> bool:
        """Whether this read matched a known plate format."""
        return bool(self.text) and self.format_name is not None

    @property
    def line_count(self) -> int:
        return len(self.lines)

    def __str__(self) -> str:
        if not self.is_valid:
            return f"<invalid: {self.raw_text!r} via {self.engine}>"
        suffix = f" ({self.corrections} corrected)" if self.corrections else ""
        return f"{self.text} [{self.confidence:.2f}]{suffix}"


@dataclass(frozen=True, slots=True)
class CorrectionResult:
    """Outcome of positional character correction against a format template."""

    text: str
    format_name: str
    corrections: int
