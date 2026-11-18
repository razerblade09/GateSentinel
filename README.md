# GateSentinel — License Plate Recognition System

A real-time automatic license plate recognition (ALPR) system for gate access control, built with YOLOv8, Tesseract OCR, and PyQt5.

## Features

- **Live camera feed** with real-time YOLOv8 license plate detection
- **OCR recognition** of Philippine plate formats (4-wheel and motorcycle)
- **Access control** — registered vehicles, guests, and expired registrations handled automatically
- **Admin web panel** (Flask) for managing vehicles, guests, and logs
- **Time In / Time Out logging** with 5-minute cooldown per plate
- **Excel report export** of scan logs
- **Auto-expiry** of guest access after 24 hours (background scheduler)
- **Dark-themed PyQt5 desktop GUI**

## Project Structure

```
system/
├── license_plate_gui.py     # Main desktop application (PyQt5)
├── camera_feed.py           # Webcam capture + YOLO detection + OCR
├── requirements.txt         # Python dependencies
├── admin/
│   ├── server.py            # Flask admin web panel (port 5001)
│   └── templates/           # Admin UI templates
runs/
└── detect/train16/weights/
    └── best.pt              # Trained YOLOv8 model (custom plate detector)
data/
└── images/train/            # Training dataset
```

## Requirements

- Python 3.11
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) installed at `C:\Program Files\Tesseract-OCR\`
- Webcam

## Setup

```bash
# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -r system/requirements.txt
```

## Running

**1. Start the admin server** (in one terminal):
```bash
cd system/admin
python server.py
# Admin panel: http://localhost:5001
```

**2. Start the main GUI** (in another terminal):
```bash
cd system
python license_plate_gui.py
```

## Model

Custom-trained YOLOv8n model on Philippine license plate dataset.  
Training run: `train16` — 16 iterations of fine-tuning on real gate footage.

| Stage | Tool |
|---|---|
| Detection | YOLOv8n (custom fine-tuned) |
| OCR | Tesseract (psm 8, alphanumeric whitelist) |
| Framework | Ultralytics 8.0.203 |

## Tech Stack

| Component | Technology |
|---|---|
| Desktop GUI | PyQt5 |
| Object Detection | Ultralytics YOLOv8 |
| OCR | Tesseract via pytesseract |
| Admin API | Flask |
| Database | SQLite |
| Scheduling | APScheduler |
| Reports | Pandas + XlsxWriter |

## License Plate Formats Supported

- 4-wheel vehicles: `ABC-1234`
- Motorcycles: `123ABC`, `1234AB`, `AB12345`
