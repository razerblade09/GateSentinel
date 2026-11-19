"""ONNX export: convert Ultralytics YOLO ``.pt`` weights to ``.onnx``.

Why ONNX first?
---------------
ONNX is the intermediate format that most runtimes (TensorRT, OpenVINO,
ONNX Runtime, CoreML) can read.  Exporting to ONNX is therefore step one
of *any* deployment path, even when the final target is OpenVINO IR.

Ultralytics handles the actual tracing; this module is a thin wrapper that:

1. Validates inputs before invoking Ultralytics (fail fast, clear error).
2. Captures the output path the exporter chose (Ultralytics derives it
   from the source path, so the caller does not have to guess).
3. Wraps success and failure in an ``ExportResult`` rather than raising,
   so the CLI can report all requested exports together.

The ``opset`` default is 17.  OpenVINO 2024.x supports opsets 1-21, and
opset 17 is what Ultralytics uses by default.  Lower opsets may be needed
for very old runtimes; higher opsets may expose ops that edge SDKs do not
implement yet.
"""

from __future__ import annotations

import time
from pathlib import Path

from alpr.export.types import ExportFormat, ExportResult
from alpr.utils.logging import get_logger

logger = get_logger(__name__)

# Ultralytics appends this suffix to the stem of the source weights.
_ONNX_SUFFIX = ".onnx"


def export_onnx(
    weights: Path,
    *,
    imgsz: int = 640,
    opset: int = 17,
    dynamic: bool = False,
    simplify: bool = True,
    half: bool = False,
) -> ExportResult:
    """Export a YOLO ``.pt`` file to ONNX.

    Args:
        weights: Path to the trained ``.pt`` file.
        imgsz: Square image size for which the graph is traced.  Must match
            the size used at inference time (or use ``dynamic=True``).
        opset: ONNX opset version.  17 is Ultralytics' default and is
            supported by all recent runtimes.
        dynamic: If ``True``, the batch and spatial dimensions are marked
            dynamic, which enables variable-resolution inference at the cost
            of slightly reduced optimisation.  For a gate camera with a fixed
            resolution, ``False`` is correct.
        simplify: Run ``onnxsim`` to fold constants and remove dead nodes.
            Saves ~15% model size and speeds up graph loading.
        half: Export FP16 weights.  Only useful when the target runtime
            supports FP16 (most GPU paths do; most ARM/CPU paths do not).

    Returns:
        An :class:`ExportResult` describing the outcome.
    """
    t0 = time.perf_counter()

    if not weights.exists():
        return ExportResult.failure(
            weights,
            ExportFormat.ONNX,
            f"weights file not found: {weights}",
            time.perf_counter() - t0,
        )
    if weights.suffix != ".pt":
        return ExportResult.failure(
            weights,
            ExportFormat.ONNX,
            f"expected a .pt file, got {weights.suffix!r}",
            time.perf_counter() - t0,
        )

    logger.info(
        "exporting %s → ONNX (imgsz=%d, opset=%d, dynamic=%s, half=%s)",
        weights.name,
        imgsz,
        opset,
        dynamic,
        half,
    )

    try:
        # Import lazily — ultralytics is a heavy dependency and importing it at
        # module level would slow down every CLI invocation that imports this
        # module.
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(str(weights))
        exported = model.export(
            format="onnx",
            imgsz=imgsz,
            opset=opset,
            dynamic=dynamic,
            simplify=simplify,
            half=half,
        )
        output_path = Path(exported) if exported else weights.with_suffix(_ONNX_SUFFIX)

    except Exception as exc:  # noqa: BLE001
        logger.error("ONNX export failed: %s", exc)
        return ExportResult.failure(
            weights,
            ExportFormat.ONNX,
            str(exc),
            time.perf_counter() - t0,
        )

    duration = time.perf_counter() - t0
    logger.info("ONNX export complete in %.1fs → %s", duration, output_path)
    return ExportResult.success(weights, ExportFormat.ONNX, output_path, duration)
