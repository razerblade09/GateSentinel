"""Build leakage-safe train/val/test splits from a flat YOLO dataset.

    python scripts/build_splits.py --root path/to/data --out outputs/splits
    python scripts/build_splits.py --root path/to/data --out /kaggle/working/ds --mode copy

Use ``--mode copy`` for Kaggle: /kaggle/input is read-only and Ultralytics needs
to write a labels cache next to the labels directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpr.config import load_config  # noqa: E402
from alpr.data.audit import format_audit  # noqa: E402
from alpr.data.discovery import DatasetError, discover  # noqa: E402
from alpr.data.audit import audit_records  # noqa: E402
from alpr.data.materialize import materialize, save_manifest  # noqa: E402
from alpr.data.splitting import SplitError, format_split, split_records  # noqa: E402
from alpr.utils import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subset", default="train")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--mode", choices=["lists", "copy"], default="lists")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--class-name", default="License_Plate")
    parser.add_argument("--no-audit", action="store_true")
    args = parser.parse_args()

    config = load_config(args.profile)
    configure_logging(config.logging)

    root = (args.root or config.paths.data_dir).resolve()
    out = (args.out or config.paths.outputs_dir / "splits").resolve()

    try:
        records, issues = discover(root, args.subset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not args.no_audit:
        print(format_audit(audit_records(root, records, issues)))
        print()

    try:
        manifest = split_records(
            records,
            dataset_root=root,
            ratios={"train": args.train, "val": args.val, "test": args.test},
            seed=args.seed,
        )
    except SplitError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(format_split(manifest))
    print()

    save_manifest(manifest, out / "split_manifest.json")
    yaml_path = materialize(
        manifest, out, mode=args.mode, class_names={0: args.class_name}
    )

    print(f"train with:  yolo detect train data={yaml_path} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
