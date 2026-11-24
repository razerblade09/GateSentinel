"""GateSentinel — live webcam demonstration script.

Captures one frame from the webcam, runs the two-stage detection pipeline
(vehicle → plate), saves an annotated image, and prints what was found.
No trained plate model needed — it downloads yolo11n.pt automatically for
the vehicle stage and uses it in single-stage mode for the demo.

Usage::

    python scripts/demo_webcam.py              # uses camera 0
    python scripts/demo_webcam.py --source 1   # camera index 1
    python scripts/demo_webcam.py --source path/to/image.jpg
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from ultralytics import YOLO

from alpr.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("demo")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLORS = {
    "vehicle": (0, 200, 0),    # green
    "plate":   (0, 100, 255),  # orange
    "label":   (255, 255, 255),
}


def grab_frame(source: str) -> np.ndarray | None:
    """Grab a single frame from camera index or image file."""
    p = Path(source)
    if p.exists() and p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
        frame = cv2.imread(str(p))
        logger.info("Loaded image: %s (%dx%d)", p.name, frame.shape[1], frame.shape[0])
        return frame

    index = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        logger.error("Cannot open camera source: %s", source)
        return None

    # Drain a few frames so the sensor auto-exposure settles.
    for _ in range(5):
        cap.read()
    ret, frame = cap.read()
    cap.release()
    if not ret:
        logger.error("Failed to read frame from camera %s", source)
        return None

    logger.info("Captured frame from camera %s: %dx%d", source, frame.shape[1], frame.shape[0])
    return frame


def draw_box(
    img: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    label: str,
    color: tuple[int, int, int],
) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLORS["label"], 1, cv2.LINE_AA)


def run_demo(source: str, out_path: Path) -> None:
    frame = grab_frame(source)
    if frame is None:
        sys.exit(1)

    annotated = frame.copy()

    # -----------------------------------------------------------------------
    # Stage 1: Vehicle detection (auto-downloads yolo11n.pt ~6 MB on first run)
    # -----------------------------------------------------------------------
    logger.info("Loading vehicle detector (yolo11n.pt — downloads if not cached)…")
    t0 = time.perf_counter()
    vehicle_model = YOLO("yolo11n.pt")
    results = vehicle_model(
        frame,
        classes=list(VEHICLE_CLASSES.keys()),
        conf=0.35,
        verbose=False,
    )
    vehicle_ms = (time.perf_counter() - t0) * 1000

    vehicles_found = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label  = f"{VEHICLE_CLASSES.get(cls_id, 'vehicle')} {conf:.0%}"
            draw_box(annotated, x1, y1, x2, y2, label, COLORS["vehicle"])
            vehicles_found.append({"class": VEHICLE_CLASSES.get(cls_id, "?"), "conf": conf,
                                   "box": (x1, y1, x2, y2)})

    # -----------------------------------------------------------------------
    # Stage 2: Plate detection stub (needs trained plate_yolo26s.pt — skipped)
    # -----------------------------------------------------------------------
    plate_model_path = Path("models/plate_yolo26s.pt")
    plates_found = []
    if plate_model_path.exists():
        logger.info("Loading plate detector…")
        plate_model = YOLO(str(plate_model_path))
        for v in vehicles_found:
            x1, y1, x2, y2 = v["box"]
            # Pad 10 %
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)
            crop = frame[max(0, y1-pad_y):y2+pad_y, max(0, x1-pad_x):x2+pad_x]
            if crop.size == 0:
                continue
            pr = plate_model(crop, conf=0.30, verbose=False)
            if pr and pr[0].boxes is not None:
                for pb in pr[0].boxes:
                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                    # Translate back to full-frame coords
                    fx1 = x1 - pad_x + px1
                    fy1 = y1 - pad_y + py1
                    fx2 = x1 - pad_x + px2
                    fy2 = y1 - pad_y + py2
                    draw_box(annotated, fx1, fy1, fx2, fy2,
                             f"plate {float(pb.conf[0]):.0%}", COLORS["plate"])
                    plates_found.append({"conf": float(pb.conf[0])})
    else:
        logger.info("Plate detector not available yet (need to train plate_yolo26s.pt)")
        # Draw a placeholder banner on the image
        cv2.putText(annotated,
                    "Plate detector: not trained yet",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 100, 255), 2, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Overlay pipeline info
    # -----------------------------------------------------------------------
    info_lines = [
        f"GateSentinel Demo  |  {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Vehicle detection : {vehicle_ms:.0f} ms  |  {len(vehicles_found)} vehicle(s) found",
        f"Plate detection   : {'skipped — model not trained' if not plate_model_path.exists() else f'{len(plates_found)} plate(s) found'}",
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(annotated, line, (10, 24 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

    # -----------------------------------------------------------------------
    # Save output
    # -----------------------------------------------------------------------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), annotated)

    # -----------------------------------------------------------------------
    # Print summary
    # -----------------------------------------------------------------------
    print("\n" + "="*60)
    print("  GateSentinel — Demo Results")
    print("="*60)
    print(f"  Frame size       : {frame.shape[1]}x{frame.shape[0]}")
    print(f"  Vehicle detector : {vehicle_ms:.0f} ms  ({len(vehicles_found)} found)")
    for v in vehicles_found:
        print(f"    • {v['class']}  conf={v['conf']:.0%}  box={v['box']}")
    if plate_model_path.exists():
        print(f"  Plate detector   : {len(plates_found)} plate(s) found")
    else:
        print(f"  Plate detector   : NOT TRAINED YET")
        print(f"    → Train on Kaggle with notebooks/train_plate_detector.ipynb")
        print(f"    → Then place plate_yolo26s.pt in models/")
    print(f"\n  Annotated image  : {out_path}")
    print("="*60 + "\n")


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GateSentinel single-frame webcam demo")
    p.add_argument("--source", default="0", help="Camera index or image path (default: 0)")
    p.add_argument("--out", default="outputs/demo_result.jpg", help="Output image path")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    run_demo(args.source, Path(args.out))
