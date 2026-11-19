"""CLI entry point for model export and benchmarking.

Usage examples
--------------
Export a single model to ONNX::

    python scripts/export.py --weights models/plate_yolo26s.pt --format onnx

Export to OpenVINO IR directly::

    python scripts/export.py --weights models/plate_yolo26s.pt --format openvino

Export both vehicle and plate detectors in one pass::

    python scripts/export.py \\
        --weights models/yolo26n.pt models/plate_yolo26s.pt \\
        --format openvino

Benchmark all three formats after exporting::

    python scripts/export.py \\
        --weights models/plate_yolo26s.pt \\
        --format openvino \\
        --benchmark \\
        --benchmark-runs 100

Export and immediately benchmark on a specific device::

    python scripts/export.py \\
        --weights models/plate_yolo26s.pt \\
        --format openvino \\
        --benchmark \\
        --device openvino
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Project root → sys.path so the script works without ``pip install -e .``
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from alpr.export.benchmark import benchmark, compare
from alpr.export.onnx import export_onnx
from alpr.export.openvino import export_openvino
from alpr.export.types import ExportFormat, ExportResult
from alpr.utils.logging import get_logger, setup_logging

logger = get_logger(__name__)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Export YOLO weights to ONNX or OpenVINO IR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--weights",
        nargs="+",
        required=True,
        type=Path,
        metavar="PATH",
        help="One or more .pt weight files to export.",
    )
    p.add_argument(
        "--format",
        choices=[f.value for f in ExportFormat],
        required=True,
        help="Output format.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=640,
        metavar="N",
        help="Square image size for graph tracing (default: 640).",
    )
    p.add_argument(
        "--opset",
        type=int,
        default=17,
        metavar="N",
        help="ONNX opset version (default: 17, ignored for openvino).",
    )
    p.add_argument(
        "--half",
        action="store_true",
        help="Export FP16 weights (GPU/NPU targets only).",
    )
    p.add_argument(
        "--benchmark",
        action="store_true",
        help="Run latency benchmark after exporting.",
    )
    p.add_argument(
        "--benchmark-runs",
        type=int,
        default=50,
        metavar="N",
        dest="benchmark_runs",
        help="Number of timed inference passes (default: 50).",
    )
    p.add_argument(
        "--device",
        default="cpu",
        help="Inference device for the benchmark (default: cpu).",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        dest="log_level",
        help="Logging verbosity (default: INFO).",
    )
    return p.parse_args(argv)


def _run_export(weights: Path, fmt: ExportFormat, args: argparse.Namespace) -> ExportResult:
    if fmt is ExportFormat.ONNX:
        return export_onnx(
            weights,
            imgsz=args.imgsz,
            opset=args.opset,
            half=args.half,
        )
    elif fmt is ExportFormat.OPENVINO:
        return export_openvino(
            weights,
            imgsz=args.imgsz,
            half=args.half,
        )
    else:  # pragma: no cover
        raise ValueError(f"unhandled format: {fmt}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    setup_logging(level=args.log_level)

    fmt = ExportFormat(args.format)
    results: list[ExportResult] = []

    for weights in args.weights:
        logger.info("— %s", weights.name)
        result = _run_export(weights, fmt, args)
        results.append(result)
        print(result)

    # Summary.
    n_ok = sum(1 for r in results if r.ok)
    n_fail = len(results) - n_ok
    print(f"\n{n_ok}/{len(results)} exports succeeded.")

    # Optional benchmark.
    if args.benchmark:
        exported_paths = [
            r.output_path for r in results if r.ok and r.output_path is not None
        ]
        # Include the original .pt files so the user can compare.
        pt_paths = [w for w in args.weights if w.exists()]
        all_paths = pt_paths + exported_paths

        if all_paths:
            print("\nBenchmark results (fastest first):\n")
            brs = compare(
                *all_paths,
                n_runs=args.benchmark_runs,
                imgsz=args.imgsz,
                device=args.device,
            )
            for br in brs:
                print(f"  {br}")

    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
