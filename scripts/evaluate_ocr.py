"""evaluate_ocr.py — batch OCR metric report from a CSV or directory tree.

Usage
-----
Given a CSV file with columns ``image_id``, ``prediction``, ``ground_truth``
(and optionally ``engine``):

    python scripts/evaluate_ocr.py --csv results/ocr_runs.csv

Given a directory tree where each sub-directory is named after the engine and
each file is named ``<image_id>.txt`` containing the prediction, with ground
truths in a parallel ``labels/`` directory:

    python scripts/evaluate_ocr.py \\
        --pred-dir results/paddle \\
        --gt-dir   data/labels   \\
        --engine   paddleocr

The script writes a summary table to stdout and optionally a JSON report to
``--out``.

Design note: this script does **not** run inference. It measures an already-
produced result set. Keep model execution and metric reporting separate:
running them together makes it impossible to re-evaluate after a bugfix without
re-running the expensive inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from alpr.config.schema import PlateFormatConfig
from alpr.evaluation.ocr import OCRSample, OCRSummary, ocr_metrics


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def _load_csv(path: Path) -> list[OCRSample]:
    """Load samples from a CSV with columns image_id, prediction, ground_truth[, engine]."""
    samples: list[OCRSample] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        required = {"image_id", "prediction", "ground_truth"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            sys.exit(
                f"CSV must have columns: {required}. "
                f"Found: {list(reader.fieldnames or [])}"
            )
        for row in reader:
            samples.append(
                OCRSample(
                    image_id=row["image_id"],
                    prediction=row["prediction"],
                    ground_truth=row["ground_truth"],
                    engine=row.get("engine", ""),
                )
            )
    return samples


def _load_dir(pred_dir: Path, gt_dir: Path, engine: str) -> list[OCRSample]:
    """Load samples from prediction and ground-truth file trees.

    Each ``.txt`` file in ``pred_dir`` must have a matching file in ``gt_dir``
    with the same name. Files that exist only in one directory are skipped with
    a warning.
    """
    pred_files = {p.stem: p for p in pred_dir.glob("*.txt")}
    gt_files = {p.stem: p for p in gt_dir.glob("*.txt")}

    only_pred = set(pred_files) - set(gt_files)
    only_gt = set(gt_files) - set(pred_files)
    if only_pred:
        print(
            f"WARNING: {len(only_pred)} prediction file(s) have no ground truth; "
            "skipped.",
            file=sys.stderr,
        )
    if only_gt:
        print(
            f"WARNING: {len(only_gt)} ground-truth file(s) have no prediction; "
            "skipped.",
            file=sys.stderr,
        )

    samples: list[OCRSample] = []
    for stem in sorted(set(pred_files) & set(gt_files)):
        pred = pred_files[stem].read_text(encoding="utf-8").strip()
        gt = gt_files[stem].read_text(encoding="utf-8").strip()
        if not gt:
            print(
                f"WARNING: ground-truth file for {stem!r} is empty; skipped.",
                file=sys.stderr,
            )
            continue
        samples.append(OCRSample(image_id=stem, prediction=pred, ground_truth=gt, engine=engine))
    return samples


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _print_summary(summary: OCRSummary, label: str = "Overall") -> None:
    print(f"\n{'='*60}")
    print(f" {label}")
    print(f"{'='*60}")
    print(f"  Samples:          {summary.n_samples}")
    print(f"  Reads (non-empty):{summary.n_reads}  "
          f"({summary.read_rate:.1%})")
    print(f"  Mean CER:         {summary.mean_cer:.4f}")
    print(f"  Exact match:      {summary.exact_match_rate:.4f}  "
          f"({summary.exact_match_rate:.1%})")
    if summary.format_match_rate is not None:
        print(f"  Format match:     {summary.format_match_rate:.4f}  "
              f"({summary.format_match_rate:.1%})")


def _summary_to_dict(s: OCRSummary) -> dict:  # type: ignore[type-arg]
    return {
        "n_samples": s.n_samples,
        "n_reads": s.n_reads,
        "read_rate": s.read_rate,
        "mean_cer": s.mean_cer,
        "exact_match_rate": s.exact_match_rate,
        "format_match_rate": s.format_match_rate,
        "per_engine": {k: _summary_to_dict(v) for k, v in s.per_engine.items()},
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compute OCR metrics from prediction/ground-truth pairs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--csv",
        type=Path,
        metavar="PATH",
        help="CSV file with image_id, prediction, ground_truth[, engine] columns.",
    )
    source.add_argument(
        "--pred-dir",
        type=Path,
        metavar="DIR",
        help="Directory of .txt prediction files (requires --gt-dir).",
    )
    p.add_argument(
        "--gt-dir",
        type=Path,
        metavar="DIR",
        help="Directory of .txt ground-truth files (required with --pred-dir).",
    )
    p.add_argument(
        "--engine",
        default="",
        help="Engine label attached to samples from --pred-dir.",
    )
    p.add_argument(
        "--out",
        type=Path,
        metavar="PATH",
        help="Write JSON report to this file.",
    )
    p.add_argument(
        "--no-format",
        action="store_true",
        help="Skip plate-format validation (useful for non-PH datasets).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.pred_dir and not args.gt_dir:
        sys.exit("--pred-dir requires --gt-dir")

    if args.csv:
        samples = _load_csv(args.csv)
    else:
        samples = _load_dir(args.pred_dir, args.gt_dir, args.engine)

    if not samples:
        sys.exit("No samples loaded. Check your input files.")

    fmt = None if args.no_format else PlateFormatConfig()
    summary = ocr_metrics(samples, fmt)

    _print_summary(summary, label="Overall")
    for engine, sub in sorted(summary.per_engine.items()):
        _print_summary(sub, label=f"Engine: {engine}")

    print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(_summary_to_dict(summary), indent=2), encoding="utf-8"
        )
        print(f"JSON report written to {args.out}")


if __name__ == "__main__":
    main()
