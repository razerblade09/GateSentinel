"""Compare OCR engines on real plate crops cut from ground-truth boxes.

    python scripts/benchmark_ocr.py --root path/to/data --limit 40
    python scripts/benchmark_ocr.py --root path/to/data --engines tesseract --save-crops out/

What this measures and what it does not
---------------------------------------
It measures whether an engine finds text at all, how confident it is, and how
long it takes -- all of which are engine properties independent of region.

It does **not** measure accuracy, because that needs ground-truth plate strings
and the detection dataset only has boxes. Format-validation rate is reported but
is only meaningful when the crops and the configured plate formats come from the
same country; on a Vietnamese dataset with Philippine formats configured, a low
rate is the expected and correct outcome, not a defect.

Accuracy measurement arrives with the evaluation milestone, once there are
Philippine crops with known strings to compare against.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from alpr.config import load_config  # noqa: E402
from alpr.data.discovery import DatasetError, discover  # noqa: E402
from alpr.data.types import ImageRecord  # noqa: E402
from alpr.ocr import PlateRead, build_recognizer  # noqa: E402
from alpr.utils import configure_logging  # noqa: E402


def crop_plate(
    image: np.ndarray, record: ImageRecord, index: int = 0, padding: float = 0.04
) -> np.ndarray | None:
    """Cut the plate region out of an image using its normalised box."""
    if index >= len(record.boxes):
        return None
    box = record.boxes[index]
    height, width = image.shape[:2]

    half_w = box.w * (1 + padding) / 2
    half_h = box.h * (1 + padding) / 2
    x1 = max(0, int((box.cx - half_w) * width))
    y1 = max(0, int((box.cy - half_h) * height))
    x2 = min(width, int((box.cx + half_w) * width))
    y2 = min(height, int((box.cy + half_h) * height))

    if x2 <= x1 or y2 <= y1:
        return None
    return image[y1:y2, x1:x2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--subset", default="train")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--engines", nargs="+", default=["tesseract", "paddleocr"])
    parser.add_argument("--save-crops", type=Path, default=None)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--show", type=int, default=12, help="Rows of raw output.")
    args = parser.parse_args()

    config = load_config(args.profile)
    configure_logging(config.logging)

    root = (args.root or config.paths.data_dir).resolve()

    try:
        records, _ = discover(root, args.subset)
    except DatasetError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    with_boxes = [r for r in records if r.boxes]
    random.Random(args.seed).shuffle(with_boxes)
    sample = with_boxes[: args.limit]

    crops: list[tuple[str, np.ndarray]] = []
    for record in sample:
        image = cv2.imread(str(record.image_path))
        if image is None:
            continue
        crop = crop_plate(image, record)
        if crop is not None and crop.size:
            crops.append((record.stem, crop))

    if not crops:
        print("error: no usable crops", file=sys.stderr)
        return 1

    print(f"{len(crops)} plate crops from {root.name}")
    sizes = [f"{c.shape[1]}x{c.shape[0]}" for _, c in crops[:6]]
    print(f"sample crop sizes: {', '.join(sizes)}")
    print()

    if args.save_crops:
        args.save_crops.mkdir(parents=True, exist_ok=True)
        for stem, crop in crops:
            cv2.imwrite(str(args.save_crops / f"{stem[:40]}.png"), crop)
        print(f"crops written to {args.save_crops}\n")

    results: dict[str, list[PlateRead]] = {}
    timings: dict[str, float] = {}

    for engine_name in args.engines:
        ocr_config = config.ocr.model_copy(update={"engine": engine_name})
        try:
            recognizer = build_recognizer(ocr_config, config.plate_format)
        except Exception as exc:
            print(f"{engine_name}: unavailable ({type(exc).__name__}: {exc})\n")
            continue

        recognizer._engine.warmup()  # noqa: SLF001 - benchmark needs a fair first call

        start = time.perf_counter()
        reads = [recognizer.read(crop) for _, crop in crops]
        elapsed = time.perf_counter() - start

        results[engine_name] = reads
        timings[engine_name] = elapsed / len(crops) * 1000

    if not results:
        print("no engines available", file=sys.stderr)
        return 1

    print(f"{'engine':<22}{'got text':>10}{'valid fmt':>11}{'mean conf':>11}{'ms/crop':>10}")
    print("-" * 64)
    for engine_name, reads in results.items():
        got_text = sum(1 for r in reads if r.raw_text)
        valid = sum(1 for r in reads if r.is_valid)
        confident = [r.confidence for r in reads if r.raw_text]
        mean_conf = sum(confident) / len(confident) if confident else 0.0
        print(
            f"{engine_name:<22}{got_text / len(reads) * 100:>9.0f}%"
            f"{valid / len(reads) * 100:>10.0f}%"
            f"{mean_conf:>11.2f}{timings[engine_name]:>10.0f}"
        )

    print()
    print(f"raw output, first {args.show} crops")
    print("-" * 64)
    header = f"{'#':<4}" + "".join(f"{name:<26}" for name in results)
    print(header)
    for index in range(min(args.show, len(crops))):
        row = f"{index:<4}"
        for engine_name in results:
            read = results[engine_name][index]
            shown = read.raw_text or "-"
            row += f"{shown[:22]:<26}"
        print(row)

    print()
    print(
        "note: 'valid fmt' checks against configured "
        f"{config.plate_format.region} formats. Crops from another country will "
        "correctly score near zero."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
