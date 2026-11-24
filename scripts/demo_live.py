"""GateSentinel — Live GUI demo with real-time plate OCR.

Opens an OpenCV window showing the real-time camera feed with vehicle
and license plate detection boxes, and reads plate text using Tesseract.
Press Q or Escape to quit.

What you'll see
---------------
- Green boxes around vehicles (car, motorcycle, bus, truck)
- Orange boxes around license plates
- High-contrast text label on each plate showing the recognised license number
- FPS counter, detection latency, OCR latency, mode indicator (top HUD)
- Running log of recent plate reads (bottom-right corner)

Detection modes
---------------
Two-stage (default): vehicle detector → plate detector inside each crop.
  Auto-fallback: when no vehicles are found, the plate detector automatically
  runs on the full frame — so pointing the camera at a screen or printed
  plate image immediately detects and reads it.
Single-stage (--single-stage flag or press M): plate detector always runs
  on the full frame, skipping vehicle detection entirely.

Controls
--------
  Q / Escape   Quit
  S            Save current annotated frame to outputs/
  P            Pause / resume
  M            Toggle single-stage / two-stage mode at runtime

Usage::

    python scripts/demo_live.py                        # camera 0, auto-fallback
    python scripts/demo_live.py --source 1             # camera index 1
    python scripts/demo_live.py --single-stage         # plate on full frame always
    python scripts/demo_live.py --no-ocr               # detection only (faster)
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np

from alpr.utils.logging import setup_logging, get_logger

setup_logging("INFO")
logger = get_logger("live_demo")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
VEHICLE_CLASSES   = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLOR_VEHICLE     = (57, 255, 20)      # neon green
COLOR_PLATE       = (0, 140, 255)      # orange
COLOR_TEXT_BG     = (20, 20, 20)      # dark gray pill
COLOR_TEXT_FG     = (0, 255, 255)      # bright yellow text
COLOR_HUD         = (0, 220, 255)      # yellow
COLOR_STATUS_OK   = (57, 255, 20)
COLOR_STATUS_WARN = (0, 140, 255)
FONT              = cv2.FONT_HERSHEY_SIMPLEX
DETECT_EVERY_N    = 3                  # run detector every N frames (perf)
FPS_WINDOW        = 30                 # frames to average FPS over
OCR_REFRESH_N     = 10                 # re-read known plates every N frames
MAX_RECENT        = 6                  # plate reads to show in HUD log

TESS_BINARY = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


# ---------------------------------------------------------------------------
# OCR helpers
# ---------------------------------------------------------------------------

def _init_ocr() -> object | None:
    """Return a configured pytesseract module or None if unavailable."""
    try:
        import pytesseract
        if TESS_BINARY.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(TESS_BINARY)
        # Warm-up inference
        pytesseract.image_to_string(
            np.full((32, 96, 3), 255, dtype=np.uint8),
            config="--psm 7 --oem 3",
        )
        logger.info("Tesseract OCR ready")
        return pytesseract
    except Exception as exc:
        logger.warning("OCR not available (%s) — text recognition disabled", exc)
        return None


def _preprocess_plate(crop: np.ndarray) -> np.ndarray:
    """Prepare a plate crop for OCR: upscale, crop outer border, grayscale."""
    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return crop

    # Upscale to standard height (~130px) for consistent character stroke width
    target_h = 130
    scale = max(1.0, target_h / h)
    up = cv2.resize(crop, (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)

    # Inset by 8% top/bottom and 4% left/right to remove plate borders & screws
    iy = max(1, int(gray.shape[0] * 0.08))
    ix = max(1, int(gray.shape[1] * 0.04))
    inner = gray[iy:-iy, ix:-ix]
    if inner.size == 0:
        return gray

    return inner


def _read_plate(pytesseract, crop: np.ndarray) -> str:
    """Run OCR on a plate crop and extract cleaned plate alphanumeric text."""
    if crop is None or crop.size == 0:
        return ""
    processed = _preprocess_plate(crop)

    # First try PSM 7 (single text line, typical for vehicle plates)
    cfg_base = "-c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    raw = pytesseract.image_to_string(
        processed,
        config=f"--psm 7 --oem 3 {cfg_base}",
    ).strip()

    # If empty, try PSM 6 (uniform block of text, e.g. two-line motorcycle plates)
    if not raw:
        raw = pytesseract.image_to_string(
            processed,
            config=f"--psm 6 --oem 3 {cfg_base}",
        ).strip()

    # If still empty, try Otsu thresholded version
    if not raw:
        _, otsu = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(
            otsu,
            config=f"--psm 7 --oem 3 {cfg_base}",
        ).strip()

    # Clean text: keep uppercase alphanumeric and single spaces
    cleaned = re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)

    # Filter out single-character false positives
    if len(cleaned.replace(" ", "")) < 2:
        return ""

    return cleaned


def _calc_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    """Calculate Intersection-over-Union between two bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    denom = a1 + a2 - inter
    return inter / float(denom) if denom > 0 else 0.0


def _match_previous_text(box: tuple[int, int, int, int], prev_plates: list[dict]) -> str:
    """Find previously recognised text for a box if IoU > 0.35."""
    for p in prev_plates:
        if p.get("text") and _calc_iou(box, p["box"]) > 0.35:
            return p["text"]
    return ""


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_box(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
             label: str, color: tuple) -> None:
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.52, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), FONT, 0.52,
                (10, 10, 10), 1, cv2.LINE_AA)


def draw_plate_badge(img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                     text: str) -> None:
    """Draw a high-contrast pill showing the plate's recognised text."""
    if not text:
        return
    h, w = img.shape[:2]
    fs = 0.85
    thick = 2
    display = f" {text} "
    (tw, th), bl = cv2.getTextSize(display, FONT, fs, thick)

    # Position: below plate box, or above if near bottom edge
    tx = x1 + max(0, ((x2 - x1) - tw) // 2)
    tx = min(tx, w - tw - 6)
    tx = max(4, tx)

    if y2 + th + 14 < h:
        ty = y2 + th + 10
    else:
        ty = max(th + 6, y1 - 10)

    pad_x = 6
    pad_y = 4
    # Black pill background with white outline
    bx1 = tx - pad_x
    by1 = ty - th - pad_y
    bx2 = tx + tw + pad_x
    by2 = ty + bl + pad_y

    cv2.rectangle(img, (bx1, by1), (bx2, by2), COLOR_TEXT_BG, -1)
    cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
    cv2.putText(img, display, (tx, ty), FONT, fs,
                COLOR_TEXT_FG, thick, cv2.LINE_AA)


def draw_hud(img: np.ndarray, fps: float, detect_ms: float, ocr_ms: float,
             n_vehicles: int, n_plates: int, paused: bool,
             single_stage: bool = False, auto_fallback: bool = False,
             recent_reads: list[str] | None = None) -> None:
    h, w = img.shape[:2]

    # Semi-transparent top banner (two lines)
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 48), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    pause_tag = "  [PAUSED]" if paused else ""
    ocr_tag = f"  |  OCR {ocr_ms:.0f}ms" if ocr_ms > 0 else ""
    cv2.putText(img,
        f"GateSentinel  |  {fps:.1f} FPS  |  detect {detect_ms:.0f}ms{ocr_tag}"
        f"  |  vehicles {n_vehicles}  plates {n_plates}{pause_tag}",
        (8, 22), FONT, 0.52, COLOR_HUD, 1, cv2.LINE_AA)

    if single_stage:
        mode_text = "Mode: SINGLE-STAGE — plate detector on full frame  [M=toggle]"
        mode_color = COLOR_PLATE
    elif auto_fallback:
        mode_text = "Mode: AUTO-FALLBACK — no vehicle, scanning full frame  [M=toggle]"
        mode_color = COLOR_PLATE
    else:
        mode_text = "Mode: TWO-STAGE — vehicle then plate  [M=toggle]"
        mode_color = COLOR_STATUS_OK
    cv2.putText(img, mode_text, (8, 42), FONT, 0.40, mode_color, 1, cv2.LINE_AA)

    # Bottom status bar
    overlay2 = img.copy()
    cv2.rectangle(overlay2, (0, h - 28), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.65, img, 0.35, 0, img)

    plate_model_path = Path("models/plate_yolo26s.pt")
    if plate_model_path.exists():
        status = "Plate detector: READY  |  OCR: Tesseract"
        color = COLOR_STATUS_OK
    else:
        status = ("Plate detector: NOT TRAINED  "
                  "( run notebooks/train_plate_detector.ipynb on Kaggle )")
        color = COLOR_STATUS_WARN

    cv2.putText(img, status, (8, h - 10), FONT, 0.45, color, 1, cv2.LINE_AA)
    cv2.putText(img, "Q/Esc=quit  S=snapshot  P=pause  M=mode",
                (w - 310, h - 10), FONT, 0.42, (160, 160, 160), 1, cv2.LINE_AA)

    # Recent reads log (bottom-right panel)
    if recent_reads:
        panel_x = max(10, w - 280)
        # Background card for readability
        n_items = min(len(recent_reads), MAX_RECENT)
        card_h = n_items * 24 + 14
        card_y = h - 34 - card_h
        cv2.rectangle(img, (panel_x - 10, card_y), (w - 10, h - 34), (15, 15, 15), -1)
        cv2.rectangle(img, (panel_x - 10, card_y), (w - 10, h - 34), (70, 70, 70), 1)
        cv2.putText(img, "Recent Plates:", (panel_x - 4, card_y + 16),
                    FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        for i, item in enumerate(reversed(recent_reads[-n_items:])):
            py = card_y + 36 + i * 22
            cv2.putText(img, item, (panel_x, py), FONT, 0.50, COLOR_TEXT_FG, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(source: str, conf: float = 0.35, plate_conf: float = 0.25,
        single_stage: bool = False, enable_ocr: bool = True) -> None:
    # ----- Load models -----
    logger.info("Loading vehicle detector (yolo11n.pt)…")
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed — run: pip install ultralytics")
        sys.exit(1)

    vehicle_model = YOLO("yolo11n.pt")

    plate_model = None
    plate_path = Path("models/plate_yolo26s.pt")
    if plate_path.exists():
        logger.info("Loading plate detector (%s)…", plate_path)
        plate_model = YOLO(str(plate_path))
    else:
        logger.warning("Plate model not found at %s — plate boxes disabled", plate_path)

    pytesseract = _init_ocr() if enable_ocr else None

    # ----- Open camera -----
    src = int(source) if source.isdigit() else source
    backend = cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info("Camera open — %dx%d", w, h)

    win = "GateSentinel — Live Demo  (Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, min(w, 1280), min(h, 720))

    frame_times: deque[float] = deque(maxlen=FPS_WINDOW)
    last_boxes: list[dict] = []
    last_plate_boxes: list[dict] = []
    last_detect_ms = 0.0
    last_ocr_ms = 0.0
    frame_idx = 0
    paused = False
    auto_fallback = False
    annotated = np.zeros((h, w, 3), dtype=np.uint8)
    recent_reads: list[str] = []
    snapshot_dir = Path("outputs")
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Window open — Q/Esc=quit  S=snapshot  P=pause  M=toggle mode")
    logger.info("Starting in %s mode | OCR: %s",
                "SINGLE-STAGE" if single_stage else "TWO-STAGE",
                "Tesseract" if pytesseract else "disabled")

    while True:
        t_frame = time.perf_counter()

        if not paused:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Camera disconnected or end of video — stopping")
                break

            annotated = frame.copy()

            # ---- Detection (every DETECT_EVERY_N frames) ----
            if frame_idx % DETECT_EVERY_N == 0:
                t0 = time.perf_counter()
                prev_plates = list(last_plate_boxes)
                for v in last_boxes:
                    prev_plates.extend(v.get("plates", []))

                if single_stage:
                    last_boxes = []
                    last_plate_boxes = []
                    auto_fallback = False
                    if plate_model is not None:
                        pr = plate_model(frame, conf=plate_conf, verbose=False)
                        if pr and pr[0].boxes is not None:
                            for pb in pr[0].boxes:
                                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                box = (px1, py1, px2, py2)
                                p_text = _match_previous_text(box, prev_plates)
                                last_plate_boxes.append({
                                    "conf": float(pb.conf[0]),
                                    "box": box,
                                    "text": p_text,
                                })
                else:
                    results = vehicle_model(
                        frame,
                        classes=list(VEHICLE_CLASSES.keys()),
                        conf=conf,
                        verbose=False,
                    )
                    last_boxes = []
                    last_plate_boxes = []

                    if results and results[0].boxes is not None:
                        for box in results[0].boxes:
                            cls_id = int(box.cls[0])
                            bconf  = float(box.conf[0])
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            last_boxes.append({
                                "class": VEHICLE_CLASSES.get(cls_id, "vehicle"),
                                "conf": bconf,
                                "box": (x1, y1, x2, y2),
                                "plates": [],
                            })
                            if plate_model is not None:
                                pad_x = int((x2 - x1) * 0.10)
                                pad_y = int((y2 - y1) * 0.10)
                                crop = frame[
                                    max(0, y1 - pad_y): min(h, y2 + pad_y),
                                    max(0, x1 - pad_x): min(w, x2 + pad_x),
                                ]
                                if crop.size > 0:
                                    pr = plate_model(crop, conf=plate_conf, verbose=False)
                                    if pr and pr[0].boxes is not None:
                                        for pb in pr[0].boxes:
                                            px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                            global_box = (
                                                x1 - pad_x + px1,
                                                y1 - pad_y + py1,
                                                x1 - pad_x + px2,
                                                y1 - pad_y + py2,
                                            )
                                            p_text = _match_previous_text(global_box, prev_plates)
                                            last_boxes[-1]["plates"].append({
                                                "conf": float(pb.conf[0]),
                                                "box": global_box,
                                                "text": p_text,
                                            })

                    # Auto-fallback: no vehicles found → scan full frame for plates
                    if not last_boxes and plate_model is not None:
                        auto_fallback = True
                        pr = plate_model(frame, conf=plate_conf, verbose=False)
                        if pr and pr[0].boxes is not None:
                            for pb in pr[0].boxes:
                                px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                box = (px1, py1, px2, py2)
                                p_text = _match_previous_text(box, prev_plates)
                                last_plate_boxes.append({
                                    "conf": float(pb.conf[0]),
                                    "box": box,
                                    "text": p_text,
                                })
                    else:
                        auto_fallback = False

                last_detect_ms = (time.perf_counter() - t0) * 1000

            # ---- OCR: read any unread plate, or refresh periodically ----
            if pytesseract is not None:
                all_current_plates: list[dict] = list(last_plate_boxes)
                for v in last_boxes:
                    all_current_plates.extend(v.get("plates", []))

                # Read immediately if text is missing, or every OCR_REFRESH_N frames
                needs_ocr = [
                    p for p in all_current_plates
                    if not p.get("text") or (frame_idx % OCR_REFRESH_N == 0)
                ]

                if needs_ocr:
                    t_ocr = time.perf_counter()
                    for plate in needs_ocr:
                        px1, py1, px2, py2 = plate["box"]
                        crop = frame[max(0, py1):min(h, py2), max(0, px1):min(w, px2)]
                        text = _read_plate(pytesseract, crop)
                        if text:
                            plate["text"] = text
                            # Add to recent log
                            log_entry = f"{text}"
                            if not recent_reads or recent_reads[-1] != log_entry:
                                recent_reads.append(log_entry)
                                logger.info("Recognised plate: %s", text)
                    last_ocr_ms = (time.perf_counter() - t_ocr) * 1000

            # ---- Draw ----
            n_plates = 0

            # Vehicles + nested plates (two-stage mode)
            for v in last_boxes:
                x1, y1, x2, y2 = v["box"]
                draw_box(annotated, x1, y1, x2, y2,
                         f"{v['class']} {v['conf']:.0%}", COLOR_VEHICLE)
                for plate in v.get("plates", []):
                    px1, py1, px2, py2 = plate["box"]
                    label = f"plate {plate['conf']:.0%}"
                    draw_box(annotated, px1, py1, px2, py2, label, COLOR_PLATE)
                    draw_plate_badge(annotated, px1, py1, px2, py2, plate.get("text", ""))
                    n_plates += 1

            # Direct plates (single-stage or auto-fallback)
            for plate in last_plate_boxes:
                px1, py1, px2, py2 = plate["box"]
                label = f"plate {plate['conf']:.0%}"
                draw_box(annotated, px1, py1, px2, py2, label, COLOR_PLATE)
                draw_plate_badge(annotated, px1, py1, px2, py2, plate.get("text", ""))
                n_plates += 1

            frame_times.append(time.perf_counter() - t_frame)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            draw_hud(annotated, fps, last_detect_ms, last_ocr_ms,
                     len(last_boxes), n_plates, paused,
                     single_stage=single_stage, auto_fallback=auto_fallback,
                     recent_reads=recent_reads)

            cv2.imshow(win, annotated)
            frame_idx += 1

        # Key handler
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key in (ord("s"), ord("S")):
            snap = snapshot_dir / f"snapshot_{int(time.time())}.jpg"
            cv2.imwrite(str(snap), annotated)
            logger.info("Snapshot saved: %s", snap)
        elif key in (ord("p"), ord("P")):
            paused = not paused
            logger.info("Paused: %s", paused)
        elif key in (ord("m"), ord("M")):
            single_stage = not single_stage
            last_boxes = []
            last_plate_boxes = []
            logger.info("Switched to %s mode",
                        "SINGLE-STAGE" if single_stage else "TWO-STAGE")

        if paused:
            time.sleep(0.03)

    cap.release()
    cv2.destroyAllWindows()
    logger.info("Demo closed.")


def _parse(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GateSentinel live GUI demo")
    p.add_argument("--source", default="0",
                   help="Camera index, video file, or RTSP URL (default: 0)")
    p.add_argument("--conf", type=float, default=0.35,
                   help="Vehicle detection confidence threshold (default: 0.35)")
    p.add_argument("--plate-conf", type=float, default=0.25,
                   help="Plate detection confidence threshold (default: 0.25)")
    p.add_argument("--single-stage", action="store_true",
                   help="Run plate detector on full frame (no vehicle needed)")
    p.add_argument("--no-ocr", action="store_true",
                   help="Disable Tesseract OCR (faster, boxes only)")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse()
    run(args.source, args.conf, args.plate_conf,
        args.single_stage, not args.no_ocr)
