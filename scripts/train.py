"""train.py — launch a YOLO plate-detector training run.

Usage
-----
Quick start (all defaults, 100 epochs):

    python scripts/train.py

Override specific hyperparameters:

    python scripts/train.py --epochs 200 --batch 32 --imgsz 1280

Point at a custom dataset YAML or starting weights:

    python scripts/train.py \\
        --dataset outputs/splits/dataset.yaml \\
        --weights yolo26n.pt \\
        --name plate_detector_nano

Resume into the same run directory:

    python scripts/train.py --exist-ok --name plate_detector

The script writes results to ``outputs/runs/<name>/``. The best checkpoint is
at ``outputs/runs/<name>/weights/best.pt`` -- this is the file to copy into
``models/`` and reference in ``configs/default.yaml:plate_detector.weights``.

Training flow reminder
----------------------
1. ``python scripts/audit_dataset.py --root <data>``  — always audit first
2. ``python scripts/build_splits.py --root <data>``   — build leakage-safe splits
3. ``python scripts/train.py``                        — train
4. ``python scripts/evaluate_ocr.py ...``             — measure OCR on results
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from alpr.training.config import TrainingConfig
from alpr.training.trainer import train


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train the YOLO plate detector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # ---- data ----------------------------------------------------------------
    p.add_argument(
        "--dataset",
        dest="dataset_yaml",
        type=Path,
        default=Path("outputs/splits/dataset.yaml"),
        metavar="PATH",
        help="Ultralytics dataset YAML (default: outputs/splits/dataset.yaml).",
    )
    # ---- model ---------------------------------------------------------------
    p.add_argument(
        "--weights",
        dest="base_weights",
        default="yolo26s.pt",
        metavar="CHECKPOINT",
        help="Starting weights (default: yolo26s.pt).",
    )
    # ---- training loop -------------------------------------------------------
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--workers", type=int, default=4)
    # ---- optimiser -----------------------------------------------------------
    p.add_argument("--optimizer", default="auto", choices=["SGD", "Adam", "AdamW", "auto"])
    p.add_argument("--lr0", type=float, default=0.01)
    # ---- output --------------------------------------------------------------
    p.add_argument(
        "--project",
        type=Path,
        default=Path("outputs/runs"),
        metavar="DIR",
        help="Parent directory for run outputs.",
    )
    p.add_argument("--name", default="plate_detector", help="Run sub-directory name.")
    p.add_argument(
        "--exist-ok",
        action="store_true",
        help="Overwrite an existing run directory instead of numbering a new one.",
    )
    # ---- runtime -------------------------------------------------------------
    p.add_argument(
        "--device",
        default="",
        metavar="DEVICE",
        help="Torch device string, e.g. '0', 'cpu'. Empty = auto.",
    )
    p.add_argument("--half", action="store_true", help="FP16 training (CUDA only).")
    p.add_argument(
        "--cache",
        choices=["ram", "disk"],
        default=None,
        help="Cache images in RAM or on disk for faster epochs.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    config = TrainingConfig(
        dataset_yaml=args.dataset_yaml,
        base_weights=args.base_weights,
        epochs=args.epochs,
        patience=args.patience,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        optimizer=args.optimizer,
        lr0=args.lr0,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        device=args.device,
        half=args.half,
        cache=args.cache or False,
        seed=args.seed,
        verbose=args.verbose,
    )

    print(f"Training config:\n  dataset : {config.dataset_yaml}")
    print(f"  weights : {config.base_weights}")
    print(f"  epochs  : {config.epochs}  patience={config.patience}")
    print(f"  batch   : {config.batch}   imgsz={config.imgsz}")
    print(f"  output  : {config.project / config.name}")
    print()

    try:
        best = train(config)
    except FileNotFoundError as exc:
        sys.exit(str(exc))
    except RuntimeError as exc:
        sys.exit(str(exc))

    print(f"\nDone. Best weights: {best}")
    print(f"Update configs/default.yaml: plate_detector.weights: {best.name}")


if __name__ == "__main__":
    main()
