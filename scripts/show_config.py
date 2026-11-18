"""Print the fully resolved configuration.

Run this whenever behaviour does not match what you think you configured. It
shows the *result* of merging every layer, which is almost always where the
surprise is hiding.

    python scripts/show_config.py
    python scripts/show_config.py --profile kaggle
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running directly from a checkout without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpr.config import ConfigError, load_config  # noqa: E402
from alpr.utils import configure_logging, get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=None, help="e.g. kaggle")
    args = parser.parse_args()

    try:
        config = load_config(args.profile)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 1

    configure_logging(config.logging)
    logger.info("configuration loaded (profile=%s)", args.profile or "none")

    print(json.dumps(config.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
