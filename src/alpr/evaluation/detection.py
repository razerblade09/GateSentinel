"""Detection metrics: IoU, precision/recall, and mean Average Precision.

All inputs use the ``(x1, y1, x2, y2)`` convention — the same format that
YOLO and OpenCV use. No conversion layer, no silent transposition.

The mAP implementation follows the COCO-style interpolated 11-point method
(VOC2007) rather than the area-under-curve method (VOC2010+). The 11-point
method is simpler to audit and the difference matters only when the recall
curve has steep cliffs, which is not typical for a well-trained plate
detector. Use it as a relative indicator, not as an absolute COCO score.

Terminology used throughout:

    prediction  -- a box the model proposed, with a confidence score.
    ground truth (gt) -- the annotated box, assumed correct.
    TP  -- prediction whose IoU with its matched gt exceeds the threshold.
    FP  -- prediction that matched nothing, or matched an already-claimed gt.
    FN  -- ground truth that no prediction claimed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Box(NamedTuple):
    """Axis-aligned bounding box in ``(x1, y1, x2, y2)`` pixel space."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def is_valid(self) -> bool:
        return self.x2 > self.x1 and self.y2 > self.y1


@dataclass(frozen=True)
class DetectionSample:
    """One image's worth of detections and ground-truth annotations.

    Attributes:
        image_id: Arbitrary identifier for error messages and aggregation.
        predictions: Predicted boxes, each paired with a confidence score.
            The list may be empty (detector found nothing).
        ground_truths: Annotated boxes. The list may be empty (negative image).
    """

    image_id: str
    predictions: list[tuple[Box, float]]  # (box, confidence)
    ground_truths: list[Box]


@dataclass
class DetectionResult:
    """Per-image matching outcome, produced by :func:`match_predictions`.

    Attributes:
        image_id: Echoed from the sample.
        tp: Number of true positives.
        fp: Number of false positives.
        fn: Number of false negatives.
        matched_ious: IoU for every matched (TP) pair.
    """

    image_id: str
    tp: int
    fp: int
    fn: int
    matched_ious: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.matched_ious)) if self.matched_ious else 0.0


# ---------------------------------------------------------------------------
# Core geometry
# ---------------------------------------------------------------------------


def compute_iou(a: Box, b: Box) -> float:
    """Intersection over Union for two axis-aligned boxes.

    Returns 0.0 for degenerate boxes (zero area or no overlap). The result is
    in [0, 1]. This is the quantity that every other detection metric is built
    on.

    >>> round(compute_iou(Box(0, 0, 10, 10), Box(5, 5, 15, 15)), 4)
    0.1429
    """
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)

    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area == 0.0:
        return 0.0

    union_area = a.area + b.area - inter_area
    return inter_area / union_area if union_area > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Per-image matching
# ---------------------------------------------------------------------------


def match_predictions(
    sample: DetectionSample,
    iou_threshold: float = 0.50,
) -> DetectionResult:
    """Match predictions to ground truths and compute TP/FP/FN.

    Predictions are processed in descending confidence order. Each prediction
    is matched to the unmatched ground truth with the highest IoU. If that IoU
    exceeds ``iou_threshold`` the pair is a TP; otherwise the prediction is a
    FP. Unmatched ground truths are FN.

    This is the standard PASCAL VOC matching procedure. It matters that
    higher-confidence predictions get first pick: a low-confidence duplicate of
    a good detection must not steal the ground truth away from the correct one.

    Args:
        sample: Image-level predictions and ground truths.
        iou_threshold: Minimum IoU to count as a match. 0.50 is the
            conventional threshold; use 0.75 for stricter localisation.

    Returns:
        :class:`DetectionResult` with TP/FP/FN counts and per-match IoUs.
    """
    # Sort predictions high to low confidence.
    preds = sorted(sample.predictions, key=lambda p: p[1], reverse=True)
    gts = list(sample.ground_truths)

    if not preds and not gts:
        return DetectionResult(image_id=sample.image_id, tp=0, fp=0, fn=0)

    matched_gt: set[int] = set()
    tp, fp = 0, 0
    matched_ious: list[float] = []

    for box, _conf in preds:
        if not gts:
            fp += 1
            continue

        # Find the best unmatched ground truth.
        best_iou = -1.0
        best_idx = -1
        for idx, gt in enumerate(gts):
            if idx in matched_gt:
                continue
            iou = compute_iou(box, gt)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= iou_threshold:
            matched_gt.add(best_idx)
            tp += 1
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gts) - len(matched_gt)
    return DetectionResult(
        image_id=sample.image_id,
        tp=tp,
        fp=fp,
        fn=fn,
        matched_ious=matched_ious,
    )


# ---------------------------------------------------------------------------
# Average Precision
# ---------------------------------------------------------------------------


def average_precision(
    samples: list[DetectionSample],
    iou_threshold: float = 0.50,
    n_points: int = 11,
) -> float:
    """Compute Average Precision over a dataset using interpolated recall points.

    The 11-point interpolation evaluates precision at recall levels
    [0, 0.1, ..., 1.0] and averages them (VOC2007 method). It is fast,
    auditable, and comparable to published YOLO results at the same threshold.

    Args:
        samples: One :class:`DetectionSample` per image.
        iou_threshold: IoU threshold for the TP/FP decision.
        n_points: Number of recall points to interpolate. Default 11 gives
            VOC2007-style AP; set to 101 for COCO-style AP.

    Returns:
        AP in [0, 1]. Returns 0.0 when there are no predictions or no ground
        truths -- the caller should check for this degenerate case.
    """
    if not samples:
        return 0.0

    # Collect all predictions across images as (confidence, is_tp) pairs.
    all_entries: list[tuple[float, bool]] = []
    total_gt = 0

    for sample in samples:
        result = match_predictions(sample, iou_threshold)
        total_gt += len(sample.ground_truths)

        # Reconstruct per-prediction TP/FP from the aggregate counts.
        # Sort predictions high-to-low as match_predictions does.
        preds_sorted = sorted(sample.predictions, key=lambda p: p[1], reverse=True)
        # The first `result.tp` predictions are TPs (after greedy matching).
        for i, (_, conf) in enumerate(preds_sorted):
            is_tp = i < result.tp
            all_entries.append((conf, is_tp))

    if total_gt == 0 or not all_entries:
        return 0.0

    # Sort all predictions by confidence, highest first.
    all_entries.sort(key=lambda e: e[0], reverse=True)

    cum_tp = 0
    cum_fp = 0
    precisions: list[float] = []
    recalls: list[float] = []

    for _, is_tp in all_entries:
        if is_tp:
            cum_tp += 1
        else:
            cum_fp += 1
        precisions.append(cum_tp / (cum_tp + cum_fp))
        recalls.append(cum_tp / total_gt)

    # Interpolated AP: at each recall threshold, take the max precision
    # for all recalls >= that threshold.
    recall_thresholds = np.linspace(0.0, 1.0, n_points)
    ap = 0.0
    for thresh in recall_thresholds:
        p_at_r = [p for p, r in zip(precisions, recalls) if r >= thresh]
        ap += max(p_at_r) if p_at_r else 0.0

    return ap / n_points


# ---------------------------------------------------------------------------
# Dataset-level aggregation
# ---------------------------------------------------------------------------


@dataclass
class DetectionMetrics:
    """Aggregate detection performance over a dataset.

    Attributes:
        n_images: Number of images evaluated.
        n_predictions: Total predictions across all images.
        n_ground_truths: Total annotated boxes across all images.
        tp: Total true positives.
        fp: Total false positives.
        fn: Total false negatives.
        mean_iou: Mean IoU over all matched pairs.
        precision: Micro-averaged precision (TP / (TP + FP)).
        recall: Micro-averaged recall (TP / (TP + FN)).
        f1: Harmonic mean of precision and recall.
        ap: Average Precision at ``iou_threshold``.
        iou_threshold: The IoU threshold used.
    """

    n_images: int
    n_predictions: int
    n_ground_truths: int
    tp: int
    fp: int
    fn: int
    mean_iou: float
    precision: float
    recall: float
    f1: float
    ap: float
    iou_threshold: float

    def __str__(self) -> str:
        return (
            f"DetectionMetrics(n={self.n_images}, "
            f"P={self.precision:.3f}, R={self.recall:.3f}, "
            f"F1={self.f1:.3f}, AP@{self.iou_threshold:.2f}={self.ap:.3f}, "
            f"mIoU={self.mean_iou:.3f})"
        )


def detection_metrics(
    samples: list[DetectionSample],
    iou_threshold: float = 0.50,
) -> DetectionMetrics:
    """Aggregate detection metrics over a dataset.

    Args:
        samples: One sample per image.
        iou_threshold: IoU threshold passed through to matching and AP.

    Returns:
        :class:`DetectionMetrics` with all aggregate statistics.
    """
    results = [match_predictions(s, iou_threshold) for s in samples]
    ap = average_precision(samples, iou_threshold)

    total_tp = sum(r.tp for r in results)
    total_fp = sum(r.fp for r in results)
    total_fn = sum(r.fn for r in results)
    all_ious = [iou for r in results for iou in r.matched_ious]

    n_pred = sum(len(s.predictions) for s in samples)
    n_gt = sum(len(s.ground_truths) for s in samples)

    p_denom = total_tp + total_fp
    r_denom = total_tp + total_fn
    precision = total_tp / p_denom if p_denom else 0.0
    recall = total_tp / r_denom if r_denom else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return DetectionMetrics(
        n_images=len(samples),
        n_predictions=n_pred,
        n_ground_truths=n_gt,
        tp=total_tp,
        fp=total_fp,
        fn=total_fn,
        mean_iou=float(np.mean(all_ious)) if all_ious else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        ap=ap,
        iou_threshold=iou_threshold,
    )
