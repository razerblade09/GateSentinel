"""GateSentinel — Live Web Dashboard & MJPEG Camera Stream.

Streams live annotated video (vehicles + plates + OCR badges) and system
telemetry directly to your browser without needing an OpenCV desktop window.

Usage::
    python scripts/web_app.py --source 0 --port 8080
"""

from __future__ import annotations

import argparse
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from ultralytics import YOLO

from alpr.utils.logging import get_logger, setup_logging

setup_logging("INFO")
logger = get_logger("web_app")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COLOR_VEHICLE = (57, 255, 20)   # neon green
COLOR_PLATE   = (0, 140, 255)   # orange
COLOR_BADGE_BG = (20, 20, 20)
COLOR_BADGE_FG = (0, 255, 255)

TESS_BINARY = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")

# Global shared frame state
latest_jpeg: bytes = b""
stats = {
    "fps": 0.0,
    "detect_ms": 0.0,
    "vehicles": 0,
    "plates": 0,
    "plate_status": "Checking...",
    "recent_plates": [],
}

HTML_DASHBOARD = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>GateSentinel — Live Dashboard</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: #0f172a; color: #f8fafc; padding: 20px; }
        .header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 20px; border-bottom: 1px solid #334155; padding-bottom: 12px; }
        .title { font-size: 24px; font-weight: 700; color: #38bdf8; display: flex; align-items: center; gap: 10px; }
        .badge { background: #059669; color: white; padding: 4px 10px; border-radius: 9999px; font-size: 12px; font-weight: 600; text-transform: uppercase; }
        .grid { display: grid; grid-template-columns: 1fr 340px; gap: 20px; }
        .stream-card { background: #1e293b; border-radius: 12px; overflow: hidden; border: 1px solid #334155; }
        .stream-card img { width: 100%; height: auto; display: block; }
        .sidebar { display: flex; flex-direction: column; gap: 15px; }
        .stat-card { background: #1e293b; padding: 18px; border-radius: 12px; border: 1px solid #334155; }
        .stat-label { font-size: 13px; color: #94a3b8; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.05em; }
        .stat-value { font-size: 26px; font-weight: 700; color: #f8fafc; }
        .legend { display: flex; flex-direction: column; gap: 8px; font-size: 14px; margin-top: 10px; }
        .legend-item { display: flex; align-items: center; gap: 8px; }
        .legend-dot { width: 12px; height: 12px; border-radius: 3px; }
        .dot-green { background: #39ff14; }
        .dot-orange { background: #ff8c00; }
        .dot-yellow { background: #00ffff; }
        .plate-chip { background: #0f172a; border: 1px solid #38bdf8; color: #38bdf8; font-family: monospace; font-size: 16px; font-weight: 700; padding: 6px 12px; border-radius: 6px; margin-top: 6px; display: inline-block; }
    </style>
</head>
<body>
    <div class="header">
        <div class="title">
            <span>🛡️ GateSentinel ALPR</span>
            <span class="badge">Live Monitoring</span>
        </div>
        <div style="color: #94a3b8; font-size: 14px;">Philippine Gate Access Control</div>
    </div>
    <div class="grid">
        <div class="stream-card">
            <img src="/video_feed" alt="Live Stream">
        </div>
        <div class="sidebar">
            <div class="stat-card">
                <div class="stat-label">Model Pipeline Status</div>
                <div style="font-size: 15px; font-weight: 600; color: #38bdf8;">Vehicle: YOLO11n (COCO)</div>
                <div style="font-size: 15px; font-weight: 500; color: #fb923c; margin-top: 4px;">Plate: plate_yolo26s.pt</div>
                <div style="font-size: 15px; font-weight: 500; color: #4ade80; margin-top: 4px;">OCR: Tesseract Engine (Active)</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Detection Legend</div>
                <div class="legend">
                    <div class="legend-item"><div class="legend-dot dot-green"></div><span>Vehicle Bounding Box</span></div>
                    <div class="legend-item"><div class="legend-dot dot-orange"></div><span>License Plate Box</span></div>
                    <div class="legend-item"><div class="legend-dot dot-yellow"></div><span>Recognised Plate Text</span></div>
                </div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Instructions</div>
                <p style="font-size: 13px; color: #94a3b8; line-height: 1.5;">
                    Point your camera at a car or license plate photo on your screen. The system runs real-time vehicle detection, plate localization, and OCR reading.
                </p>
            </div>
        </div>
    </div>
</body>
</html>
"""


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_DASHBOARD.encode("utf-8"))
        elif self.path == "/video_feed":
            self.send_response(200)
            self.send_header(
                "Content-Type", "multipart/x-mixed-replace; boundary=frame"
            )
            self.end_headers()
            while True:
                if latest_jpeg:
                    try:
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(latest_jpeg)))
                        self.end_headers()
                        self.wfile.write(latest_jpeg)
                        self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        break
                time.sleep(0.04)  # ~25 FPS delivery
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass  # Suppress HTTP request logging spam


def _init_ocr():
    try:
        import pytesseract
        if TESS_BINARY.is_file():
            pytesseract.pytesseract.tesseract_cmd = str(TESS_BINARY)
        pytesseract.image_to_string(
            np.full((32, 96, 3), 255, dtype=np.uint8),
            config="--psm 7 --oem 3",
        )
        logger.info("Tesseract OCR ready in web app")
        return pytesseract
    except Exception as exc:
        logger.warning("OCR unavailable in web app (%s)", exc)
        return None


def _read_plate(pytesseract, crop: np.ndarray) -> str:
    if crop is None or crop.size == 0 or pytesseract is None:
        return ""
    h, w = crop.shape[:2]
    if h < 5 or w < 5:
        return ""
    scale = max(1.0, 130.0 / h)
    up = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
    iy = max(1, int(gray.shape[0] * 0.08))
    ix = max(1, int(gray.shape[1] * 0.04))
    inner = gray[iy:-iy, ix:-ix]
    if inner.size == 0:
        return ""

    cfg = "--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    raw = pytesseract.image_to_string(inner, config=cfg).strip()
    if not raw:
        _, otsu = cv2.threshold(inner, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        raw = pytesseract.image_to_string(otsu, config=cfg).strip()

    cleaned = re.sub(r"[^A-Z0-9 ]", "", raw.upper()).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned if len(cleaned.replace(" ", "")) >= 2 else ""


def video_loop(source: str):
    global latest_jpeg, stats

    logger.info("Loading models for Web App...")
    vehicle_model = YOLO("yolo11n.pt")

    plate_model = None
    plate_path = Path("models/plate_yolo26s.pt")
    if plate_path.exists():
        plate_model = YOLO(str(plate_path))
        stats["plate_status"] = "Model loaded"

    pytesseract = _init_ocr()

    src = int(source) if source.isdigit() else source
    backend = cv2.CAP_DSHOW if isinstance(src, int) else cv2.CAP_ANY
    cap = cv2.VideoCapture(src, backend)
    if not cap.isOpened():
        cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        logger.error("Web app: cannot open camera source %s", source)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    cached_boxes: list[dict] = []
    cached_plates: list[dict] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        annotated = frame.copy()

        # Run inference every 3 frames
        if frame_idx % 3 == 0:
            cached_boxes = []
            cached_plates = []

            results = vehicle_model(
                frame,
                classes=list(VEHICLE_CLASSES.keys()),
                conf=0.35,
                verbose=False,
            )

            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    v_item = {
                        "class": VEHICLE_CLASSES.get(cls_id, "vehicle"),
                        "conf": float(box.conf[0]),
                        "box": (x1, y1, x2, y2),
                        "plates": [],
                    }
                    if plate_model is not None:
                        pad_x = int((x2 - x1) * 0.10)
                        pad_y = int((y2 - y1) * 0.10)
                        crop = frame[
                            max(0, y1 - pad_y):min(h, y2 + pad_y),
                            max(0, x1 - pad_x):min(w, x2 + pad_x),
                        ]
                        if crop.size > 0:
                            pr = plate_model(crop, conf=0.25, verbose=False)
                            if pr and pr[0].boxes is not None:
                                for pb in pr[0].boxes:
                                    px1, py1, px2, py2 = map(int, pb.xyxy[0])
                                    p_crop = crop[py1:py2, px1:px2]
                                    text = _read_plate(pytesseract, p_crop)
                                    v_item["plates"].append({
                                        "box": (
                                            x1 - pad_x + px1,
                                            y1 - pad_y + py1,
                                            x1 - pad_x + px2,
                                            y1 - pad_y + py2,
                                        ),
                                        "conf": float(pb.conf[0]),
                                        "text": text,
                                    })
                    cached_boxes.append(v_item)

            # Auto-fallback: if no vehicle found, scan full frame for plate
            if not cached_boxes and plate_model is not None:
                pr = plate_model(frame, conf=0.25, verbose=False)
                if pr and pr[0].boxes is not None:
                    for pb in pr[0].boxes:
                        px1, py1, px2, py2 = map(int, pb.xyxy[0])
                        p_crop = frame[py1:py2, px1:px2]
                        text = _read_plate(pytesseract, p_crop)
                        cached_plates.append({
                            "box": (px1, py1, px2, py2),
                            "conf": float(pb.conf[0]),
                            "text": text,
                        })

        # Draw annotations
        for v in cached_boxes:
            x1, y1, x2, y2 = v["box"]
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_VEHICLE, 2)
            cv2.putText(
                annotated,
                f"{v['class']} {v['conf']:.0%}",
                (x1 + 3, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_VEHICLE,
                2,
            )
            for p in v["plates"]:
                px1, py1, px2, py2 = p["box"]
                cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_PLATE, 2)
                cv2.putText(
                    annotated,
                    f"Plate {p['conf']:.0%}",
                    (px1 + 3, py1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    COLOR_PLATE,
                    2,
                )
                if p["text"]:
                    cv2.putText(
                        annotated,
                        p["text"],
                        (px1 + 4, py2 + 22),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        COLOR_BADGE_FG,
                        2,
                    )

        for p in cached_plates:
            px1, py1, px2, py2 = p["box"]
            cv2.rectangle(annotated, (px1, py1), (px2, py2), COLOR_PLATE, 2)
            cv2.putText(
                annotated,
                f"Plate {p['conf']:.0%}",
                (px1 + 3, py1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                COLOR_PLATE,
                2,
            )
            if p["text"]:
                cv2.putText(
                    annotated,
                    p["text"],
                    (px1 + 4, py2 + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    COLOR_BADGE_FG,
                    2,
                )

        ret, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ret:
            latest_jpeg = buffer.tobytes()

        frame_idx += 1
        time.sleep(0.01)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index (0 for USB, 1 for laptop)")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    t = threading.Thread(target=video_loop, args=(args.source,), daemon=True)
    t.start()

    server_address = ("", args.port)
    httpd = ThreadingHTTPServer(server_address, StreamingHandler)
    logger.info("Serving GateSentinel Web UI on http://localhost:%d", args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
