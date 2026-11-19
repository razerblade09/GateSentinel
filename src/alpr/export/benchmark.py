"""Inference latency benchmark across model formats.

Purpose
-------
Before deploying to a gate controller you want to know *empirically*
whether OpenVINO is faster than plain PyTorch on that specific board.  The
answer depends on the CPU, OS, and driver version — measuring on the target
machine beats guessing.

This module runs a fixed number of inference passes on a synthetic frame
(all-black by default, which is fine for latency measurement), discards a
warm-up burst, and reports mean and p95 latency.

What the numbers tell you
-------------------------
* Mean latency determines maximum throughput.
* p95 latency determines whether the system can keep up during bursts (e.g.
  two vehicles entering at once).  A gate that averages 30ms but spikes to
  200ms on 1 frame in 20 is worse than one that sits steady at 50ms.

The benchmark does **not** measure accuracy — use the evaluation scripts for
that.  A converted model that is faster but less accurate is not an upgrade.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from alpr.export.types import BenchmarkResult
from alpr.utils.logging import get_logger

logger = get_logger(__name__)

# Warm-up passes that are excluded from the latency measurement.
_WARMUP_RUNS = 5


def benchmark(
    weights: Path,
    *,
    n_runs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
    format_label: str | None = None,
) -> BenchmarkResult:
    """Measure inference latency for a YOLO model (any format).

    The function accepts ``.pt``, ``.onnx``, and OpenVINO IR directories
    because Ultralytics' ``YOLO`` class auto-detects the format from the
    path.

    Args:
        weights: Path to the weights file or IR directory to benchmark.
        n_runs: Number of timed inference passes (after warm-up).
        imgsz: Square image side in pixels fed to the model.
        device: Torch/Ultralytics device string (``"cpu"``, ``"cuda:0"``,
            ``"openvino"``, etc.).
        format_label: Human-readable label for the format column in reports.
            Inferred from the weights path if not supplied.

    Returns:
        A :class:`BenchmarkResult` with mean and p95 latency.
    """
    if not weights.exists():
        raise FileNotFoundError(f"weights not found: {weights}")

    label = format_label or _infer_label(weights)
    logger.info(
        "benchmarking %s (%s) on %s — %d runs @ %dpx",
        weights.name,
        label,
        device,
        n_runs,
        imgsz,
    )

    try:
        from ultralytics import YOLO  # noqa: PLC0415
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is required for benchmarking; run: pip install ultralytics"
        ) from exc

    model = YOLO(str(weights))
    if device not in ("auto", "openvino", ""):
        model.to(device)

    # Synthetic frame — shape is (height, width, channels).
    frame = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # Warm-up: let the runtime JIT-compile kernels, allocate buffers, etc.
    logger.debug("warm-up (%d passes)…", _WARMUP_RUNS)
    for _ in range(_WARMUP_RUNS):
        model(frame, verbose=False)

    # Timed passes.
    latencies: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model(frame, verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    arr = np.array(latencies)
    result = BenchmarkResult(
        format=label,
        weights_path=weights,
        mean_ms=float(arr.mean()),
        p95_ms=float(np.percentile(arr, 95)),
        n_runs=n_runs,
        imgsz=imgsz,
    )

    logger.info("%s", result)
    return result


def compare(
    *paths: Path,
    n_runs: int = 50,
    imgsz: int = 640,
    device: str = "cpu",
) -> list[BenchmarkResult]:
    """Benchmark multiple model paths and return results sorted by mean latency.

    Args:
        *paths: Weight files / IR directories to benchmark.
        n_runs: Timed passes per model.
        imgsz: Benchmark image size.
        device: Inference device.

    Returns:
        List of :class:`BenchmarkResult`, fastest first.
    """
    results = [
        benchmark(p, n_runs=n_runs, imgsz=imgsz, device=device) for p in paths
    ]
    return sorted(results, key=lambda r: r.mean_ms)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _infer_label(path: Path) -> str:
    """Guess a human-readable format label from a weights path."""
    if path.is_dir():
        # OpenVINO IR directories end with ``_openvino_model``.
        return "openvino"
    suffix = path.suffix.lower()
    return {
        ".pt": "pytorch",
        ".onnx": "onnx",
    }.get(suffix, suffix.lstrip("."))
