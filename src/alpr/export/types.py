"""Shared types for the export subsystem.

Using an ``ExportFormat`` enum rather than bare strings means a typo in a
CLI flag is caught at argument-parse time, not discovered when the exporter
tries to open a non-existent directory an hour later.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ExportFormat(str, Enum):
    """Output format for a converted model.

    ``ONNX`` is the universal interchange format; most inference runtimes can
    consume it.  ``OPENVINO`` produces an IR directory (``model.xml`` +
    ``model.bin``) that the OpenVINO runtime reads directly — no ONNX
    intermediate step needed on the inference machine.
    """

    ONNX = "onnx"
    OPENVINO = "openvino"

    def __str__(self) -> str:  # noqa: D105
        return self.value


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Record of a completed or failed export.

    Keeping success and failure in the same type rather than raising on
    failure means the caller can collect results from multiple exports and
    report them together instead of bailing out on the first problem.

    Attributes:
        source: The original ``.pt`` weights file.
        format: The requested output format.
        output_path: Where the exported model was written.  ``None`` if the
            export failed.
        duration_s: Wall-clock seconds taken by the conversion.
        ok: Whether the export succeeded.
        error: Human-readable failure reason, or empty string on success.
    """

    source: Path
    format: ExportFormat
    output_path: Path | None
    duration_s: float
    ok: bool
    error: str = ""

    @classmethod
    def success(
        cls,
        source: Path,
        fmt: ExportFormat,
        output_path: Path,
        duration_s: float,
    ) -> ExportResult:
        """Convenience constructor for a successful export."""
        return cls(
            source=source,
            format=fmt,
            output_path=output_path,
            duration_s=duration_s,
            ok=True,
        )

    @classmethod
    def failure(
        cls,
        source: Path,
        fmt: ExportFormat,
        error: str,
        duration_s: float,
    ) -> ExportResult:
        """Convenience constructor for a failed export."""
        return cls(
            source=source,
            format=fmt,
            output_path=None,
            duration_s=duration_s,
            ok=False,
            error=error,
        )

    def __str__(self) -> str:  # noqa: D105
        status = "OK" if self.ok else f"FAILED ({self.error})"
        dest = str(self.output_path) if self.output_path else "—"
        return (
            f"[{self.format}] {self.source.name} → {dest} "
            f"({self.duration_s:.1f}s) {status}"
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Latency measurement for a single inference format.

    Attributes:
        format: Which runtime was benchmarked (or ``"pytorch"`` for the
            original weights).
        weights_path: Path to the weights / IR directory that was loaded.
        mean_ms: Mean inference latency in milliseconds.
        p95_ms: 95th-percentile latency in milliseconds.
        n_runs: Number of timed inference passes (warm-up excluded).
        imgsz: Image size used for the benchmark (square side in pixels).
    """

    format: str
    weights_path: Path
    mean_ms: float
    p95_ms: float
    n_runs: int
    imgsz: int

    def __str__(self) -> str:  # noqa: D105
        return (
            f"[{self.format}] {self.weights_path.name}: "
            f"mean={self.mean_ms:.1f}ms  p95={self.p95_ms:.1f}ms  "
            f"({self.n_runs} runs @ {self.imgsz}px)"
        )
