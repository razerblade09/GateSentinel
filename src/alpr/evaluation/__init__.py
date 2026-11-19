"""Evaluation package — detection and OCR metrics.

Two independent metric families, nothing shared between them except the
principle that every number comes with the sample count it was computed on.
A mean over three images is not the same claim as a mean over three thousand.

Detection metrics (:mod:`alpr.evaluation.detection`):
    IoU, precision/recall/F1 at a threshold, mean Average Precision (mAP).
    These are COCO-style metrics so the numbers are directly comparable to
    published YOLO benchmarks.

OCR metrics (:mod:`alpr.evaluation.ocr`):
    Character Error Rate (CER), exact-match accuracy, format-match rate.
    CER is the edit distance between prediction and ground truth, normalised
    by ground-truth length. It is the standard metric for OCR research and
    the right one here: "ABCX234" is one character wrong, not entirely wrong.
"""

from alpr.evaluation.detection import (
    DetectionResult,
    DetectionSample,
    average_precision,
    compute_iou,
    detection_metrics,
    match_predictions,
)
from alpr.evaluation.ocr import (
    OCRSample,
    OCRSummary,
    character_error_rate,
    ocr_metrics,
)

__all__ = [
    # detection
    "DetectionResult",
    "DetectionSample",
    "average_precision",
    "compute_iou",
    "detection_metrics",
    "match_predictions",
    # ocr
    "OCRSample",
    "OCRSummary",
    "character_error_rate",
    "ocr_metrics",
]
