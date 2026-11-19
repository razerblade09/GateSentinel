"""OpenVINO export: convert YOLO ``.pt`` weights to OpenVINO IR.

What is OpenVINO IR?
--------------------
Intel OpenVINO IR (Intermediate Representation) is a pair of files:

* ``model.xml``  — the graph structure in XML.
* ``model.bin``  — the weight tensor data as a flat binary blob.

The OpenVINO runtime reads the pair together.  On Intel hardware (including
the Celeron/Pentium CPUs common in gate controllers and NUCs), it is
typically 2-4× faster than PyTorch CPU inference for the same model.

Two export paths are supported
------------------------------
**Direct (via Ultralytics):**  ``YOLO.export(format="openvino")`` handles
both the ONNX tracing *and* the ``mo`` (model optimiser) conversion in one
call.  This is the preferred path because Ultralytics knows the correct
export flags for its own models.

**Two-step (ONNX → OpenVINO):**  If the ONNX file already exists (from a
previous ``export_onnx`` call), it can be fed directly to the OpenVINO
Model Optimizer.  Useful when you want to inspect the ONNX graph before
conversion, or when you exported ONNX on one machine and are converting
on another.

The ``export_openvino`` function tries the direct path first and falls back
to the two-step path only if an ONNX file is explicitly supplied.
"""

from __future__ import annotations

import time
from pathlib import Path

from alpr.export.types import ExportFormat, ExportResult
from alpr.utils.logging import get_logger

logger = get_logger(__name__)

# Ultralytics names the output directory by appending this to the stem.
_OV_DIR_SUFFIX = "_openvino_model"


def export_openvino(
    weights: Path,
    *,
    imgsz: int = 640,
    half: bool = False,
    onnx_path: Path | None = None,
) -> ExportResult:
    """Export a YOLO ``.pt`` file to OpenVINO IR.

    Args:
        weights: Path to the trained ``.pt`` file.
        imgsz: Square image size for the traced graph.  Must match inference
            time (gate cameras are fixed resolution — keep ``dynamic=False``).
        half: Export FP16 weights.  Supported on Intel GPU (iGPU/dGPU) and
            some NPUs; not supported on plain CPU inference.  Leave ``False``
            for maximum compatibility with low-cost gate hardware.
        onnx_path: Optional pre-existing ONNX file to convert instead of
            tracing from ``.pt`` directly.  When supplied the function uses
            the OpenVINO CLI (``ovc``) directly instead of Ultralytics.

    Returns:
        An :class:`ExportResult` describing the outcome.
    """
    t0 = time.perf_counter()

    if not weights.exists():
        return ExportResult.failure(
            weights,
            ExportFormat.OPENVINO,
            f"weights file not found: {weights}",
            time.perf_counter() - t0,
        )

    if onnx_path is not None:
        return _convert_from_onnx(weights, onnx_path, half=half, t0=t0)

    return _export_direct(weights, imgsz=imgsz, half=half, t0=t0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _export_direct(
    weights: Path,
    *,
    imgsz: int,
    half: bool,
    t0: float,
) -> ExportResult:
    """Use Ultralytics to export directly to OpenVINO IR."""
    logger.info(
        "exporting %s → OpenVINO IR (imgsz=%d, half=%s)",
        weights.name,
        imgsz,
        half,
    )

    try:
        from ultralytics import YOLO  # noqa: PLC0415

        model = YOLO(str(weights))
        exported = model.export(
            format="openvino",
            imgsz=imgsz,
            half=half,
        )
        # Ultralytics returns the IR directory path as a string.
        output_path = Path(exported) if exported else _default_ir_dir(weights)

    except Exception as exc:  # noqa: BLE001
        logger.error("OpenVINO direct export failed: %s", exc)
        return ExportResult.failure(
            weights,
            ExportFormat.OPENVINO,
            str(exc),
            time.perf_counter() - t0,
        )

    duration = time.perf_counter() - t0
    logger.info("OpenVINO export complete in %.1fs → %s", duration, output_path)
    return ExportResult.success(weights, ExportFormat.OPENVINO, output_path, duration)


def _convert_from_onnx(
    source_pt: Path,
    onnx_path: Path,
    *,
    half: bool,
    t0: float,
) -> ExportResult:
    """Convert an existing ONNX file to OpenVINO IR using ``ovc``."""
    if not onnx_path.exists():
        return ExportResult.failure(
            source_pt,
            ExportFormat.OPENVINO,
            f"ONNX file not found: {onnx_path}",
            time.perf_counter() - t0,
        )

    output_dir = _default_ir_dir(source_pt)
    logger.info(
        "converting %s → OpenVINO IR via ovc (half=%s) → %s",
        onnx_path.name,
        half,
        output_dir,
    )

    try:
        # ``openvino`` is an optional dependency; import lazily.
        from openvino.tools.ovc import convert_model  # noqa: PLC0415
        from openvino import save_model  # noqa: PLC0415

        compress_fp16 = half
        ov_model = convert_model(str(onnx_path), compress_to_fp16=compress_fp16)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_model(ov_model, str(output_dir / "model.xml"))

    except ImportError:
        return ExportResult.failure(
            source_pt,
            ExportFormat.OPENVINO,
            "openvino package not installed; run: pip install openvino",
            time.perf_counter() - t0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ovc conversion failed: %s", exc)
        return ExportResult.failure(
            source_pt,
            ExportFormat.OPENVINO,
            str(exc),
            time.perf_counter() - t0,
        )

    duration = time.perf_counter() - t0
    logger.info("ovc conversion complete in %.1fs → %s", duration, output_dir)
    return ExportResult.success(
        source_pt, ExportFormat.OPENVINO, output_dir, duration
    )


def _default_ir_dir(weights: Path) -> Path:
    """Return the IR directory Ultralytics would choose for these weights."""
    return weights.parent / (weights.stem + _OV_DIR_SUFFIX)
