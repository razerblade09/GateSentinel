"""Audit a YOLO-format dataset.

Run this before every training run. It takes seconds and answers the questions
that otherwise get answered by a wasted five-hour run.

    python scripts/audit_dataset.py --root path/to/data
    python scripts/audit_dataset.py --root path/to/data --subset train --json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpr.config import load_config  # noqa: E402
from alpr.data.audit import audit_dataset, format_audit  # noqa: E402
from alpr.data.discovery import DatasetError  # noqa: E402
from alpr.utils import configure_logging  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Dataset root containing images/ and labels/. "
        "Defaults to paths.data_dir from config.",
    )
    parser.add_argument("--subset", default="train")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")
    args = parser.parse_args()

    config = load_config(args.profile)
    configure_logging(config.logging)

    root = args.root or config.paths.data_dir

    try:
        audit = audit_dataset(root, args.subset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        payload = dataclasses.asdict(audit)
        payload["root"] = str(payload["root"])
        payload["issues"] = {
            key: [str(v) for v in value]
            for key, value in payload["issues"].items()
        }
        payload["duplication_factor"] = audit.duplication_factor
        print(json.dumps(payload, indent=2, default=str))
    else:
        print(format_audit(audit))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
