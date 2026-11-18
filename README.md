# GateSentinel — Automatic License Plate Recognition for Gate Access Control

> **Upcoming release preview.** Milestones 1–9 of 11 are complete and all
> 221 unit tests pass. Milestones 10–11 (OpenVINO export and deployment
> packaging) are in progress. This tag marks the state of the ALPR core
> ahead of the first full GateSentinel release.

GateSentinel is a Philippine-focused, two-stage ALPR pipeline built for
embedded gate-controller hardware: detect vehicles, find plates within them,
recognise the text across multiple frames, and log the result to a SQLite
database that drives the gate relay.

**Status:** Milestone 9 of 11 complete — real-time inference pipeline
operational. Next: Optimisation and OpenVINO export (milestone 10).

## Recognition

```bash
pip install -e ".[ocr]"
python scripts/benchmark_ocr.py --root path/to/data --limit 40
```

Measured on 40 real plate crops (Vietnamese dataset, CPU):

| engine | produced text | mean confidence | ms/crop |
|---|---|---|---|
| tesseract (psm 6) | 2% | 0.68 | 120 |
| paddleocr | **82%** | 0.82 | 2402 |

Tesseract read 1 crop in 40. That is not a tuning problem — it is a document OCR
engine being asked to read embossed plates at ~25 px character height.

PaddleOCR is ~20× slower. At a gate that is affordable because tracking means
only a few frames per vehicle need reading, but it rules out per-frame OCR.

## Dataset workflow

```bash
python scripts/audit_dataset.py --root path/to/data
python scripts/build_splits.py  --root path/to/data --out outputs/splits
```

Audit first, every time. It takes seconds and answers the questions that
otherwise get answered by a wasted five-hour training run.

`build_splits.py` produces a **leakage-safe** split: every image derived from
the same source photograph lands in the same split. That is not a nicety — a
random split over this data would put augmented copies of the same photo in both
train and validation, and every metric downstream would be inflated.

Use `--mode copy` on Kaggle. `/kaggle/input` is read-only, and Ultralytics writes
a `labels.cache` next to the labels directory on the first epoch.

## Why two stages

A vehicle detector runs first, and plates are searched inside each vehicle crop.
This costs a second inference per frame and buys three things:

1. **A stable object to track.** Reading a plate once is unreliable; reading it
   across 15 frames and taking a majority vote is not. Voting needs a per-vehicle
   identity to attach reads to, and vehicle boxes track far more robustly than
   plate boxes.
2. **False-positive suppression.** Signage, stickers and reflections are
   plate-shaped. Requiring containment within a vehicle eliminates most of them.
3. **Effective resolution.** A plate is a few percent of a full frame but a large
   fraction of a vehicle crop, so the plate detector sees an easier problem.

Stage one needs no training — COCO-pretrained weights already detect car,
motorcycle, bus and truck.

Set `vehicle_detector.enabled: false` to run single-stage instead; the
comparison is worth measuring rather than assuming.

## Layout

```
configs/          YAML configuration (default, per-profile, local)
src/alpr/
  config/         schema + layered loader
  data/           dataset audit, deduplication, leakage-safe splitting
  models/         model registry and loading
  training/       training orchestration
  detection/      vehicle and plate detectors
  tracking/       ByteTrack wrapper and vote aggregation
  preprocessing/  cropping, padding, coordinate transforms
  ocr/            recognition and post-processing
  pipeline/       wires the stages together
  database/       repository layer
  evaluation/     detection and OCR metrics
  utils/          logging, geometry, IO
scripts/          CLI entry points
notebooks/        Kaggle training notebooks
tests/
```

Modules do not import each other laterally. `detection` knows nothing about
`ocr`; `pipeline` wires them. That is what makes a component swappable.

## Setup

The project uses its own virtual environment at `.venv/`, isolated from the
system Python. Upgrading a global library cannot affect this project, and
installing here cannot break anything else on the machine.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

**Always activate before working.** Your prompt shows `(.venv)` when it is
active. Without it, `python` and `pip` refer to the system interpreter and you
are back to a shared, drifting environment.

To reproduce the exact verified set instead of re-resolving:

```bash
pip install -r requirements.lock.txt
```

### On version pinning

`pyproject.toml` declares floors, not invented ceilings. The failure mode that
broke the previous version of this project was not "NumPy 2 is bad" — it was an
*inconsistent* set: NumPy 2.0 against an OpenCV wheel compiled for the 1.x ABI.

Reproducibility comes from `requirements.lock.txt`, which records versions that
were actually installed and verified to import together. Regenerate it after any
dependency change:

```bash
pip freeze --exclude-editable > requirements.lock.txt
```

Verified working: NumPy 2.4.6, OpenCV 5.0.0, Torch 2.13.0+cpu,
Ultralytics 8.4.112 (YOLO26 and YOLO11 both available).

## Configuration

Layers, lowest priority to highest:

| Layer | Committed | Purpose |
|---|---|---|
| `configs/default.yaml` | yes | documented baseline |
| `configs/<profile>.yaml` | yes | e.g. `kaggle` for training runs |
| `configs/local.yaml` | **no** | machine-specific paths, camera source |
| `ALPR__SECTION__KEY` env | n/a | deployment overrides, secrets |
| explicit overrides | n/a | tests and CLI flags |

Each layer states only what it changes. Inspect the merged result with:

```bash
python scripts/show_config.py --profile kaggle
```

Unknown keys are rejected at startup, so a typo fails loudly instead of being
silently ignored.

## Tests

```bash
pytest
```

## Roadmap

1. ~~Foundation and architecture~~
2. ~~Scaffold and configuration~~
3. ~~Dataset: audit, deduplication, leakage-safe splitting~~
4. ~~OCR pipeline~~
5. ~~Evaluation: detection and OCR metrics~~
6. ~~Detection pipeline: two-stage and tracking~~
7. ~~Training pipeline (Kaggle)~~
8. ~~Database layer~~
9. ~~Real-time inference orchestration~~
10. Optimisation and OpenVINO export
11. Deployment preparation
