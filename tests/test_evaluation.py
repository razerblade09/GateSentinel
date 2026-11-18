"""Tests for the evaluation package — detection and OCR metrics.

The same philosophy as test_ocr.py: no real images, no real models. Every
test is a hand-constructed scenario where the right answer is known in advance.
That is the only way to catch a metric implementation that is off-by-one, uses
the wrong normalisation, or silently returns 0.0 on an empty list.
"""

from __future__ import annotations

import pytest

from alpr.config.schema import PlateFormatConfig
from alpr.evaluation.detection import (
    Box,
    DetectionSample,
    average_precision,
    compute_iou,
    detection_metrics,
    match_predictions,
)
from alpr.evaluation.ocr import (
    OCRSample,
    character_error_rate,
    ocr_metrics,
)


# ===========================================================================
# Detection — IoU
# ===========================================================================


class TestComputeIou:
    def test_identical_boxes_give_one(self) -> None:
        assert compute_iou(Box(0, 0, 10, 10), Box(0, 0, 10, 10)) == pytest.approx(1.0)

    def test_no_overlap_gives_zero(self) -> None:
        assert compute_iou(Box(0, 0, 5, 5), Box(10, 10, 20, 20)) == 0.0

    def test_touching_edges_give_zero(self) -> None:
        # Boxes that share only an edge have zero intersection area.
        assert compute_iou(Box(0, 0, 5, 5), Box(5, 0, 10, 5)) == 0.0

    def test_half_overlap(self) -> None:
        # Two 10×10 boxes overlapping by a 5×10 strip.
        # Intersection = 50, Union = 150. IoU = 1/3.
        iou = compute_iou(Box(0, 0, 10, 10), Box(5, 0, 15, 10))
        assert iou == pytest.approx(1 / 3, abs=1e-6)

    def test_known_value(self) -> None:
        # From docstring: Box(0,0,10,10) ∩ Box(5,5,15,15) = 5×5=25.
        # Union = 100 + 100 - 25 = 175. IoU = 25/175 ≈ 0.1429.
        iou = compute_iou(Box(0, 0, 10, 10), Box(5, 5, 15, 15))
        assert iou == pytest.approx(25 / 175, abs=1e-4)

    def test_degenerate_box_gives_zero(self) -> None:
        assert compute_iou(Box(0, 0, 0, 10), Box(0, 0, 10, 10)) == 0.0

    def test_contained_box(self) -> None:
        # Small box fully inside the large one.
        # Intersection = small area = 100. Union = 400+100-100 = 400. IoU = 0.25.
        iou = compute_iou(Box(0, 0, 20, 20), Box(5, 5, 15, 15))
        assert iou == pytest.approx(100 / 400, abs=1e-6)


# ===========================================================================
# Detection — matching
# ===========================================================================


class TestMatchPredictions:
    def _sample(
        self,
        preds: list[tuple[Box, float]],
        gts: list[Box],
        image_id: str = "img",
    ) -> DetectionSample:
        return DetectionSample(image_id=image_id, predictions=preds, ground_truths=gts)

    def test_perfect_match(self) -> None:
        box = Box(0, 0, 10, 10)
        result = match_predictions(self._sample([(box, 0.9)], [box]))
        assert result.tp == 1
        assert result.fp == 0
        assert result.fn == 0
        assert result.matched_ious == pytest.approx([1.0])

    def test_no_overlap_is_fp_and_fn(self) -> None:
        pred = Box(0, 0, 5, 5)
        gt = Box(100, 100, 110, 110)
        result = match_predictions(self._sample([(pred, 0.9)], [gt]))
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 1

    def test_no_predictions_all_fn(self) -> None:
        gt = Box(0, 0, 10, 10)
        result = match_predictions(self._sample([], [gt]))
        assert result.tp == 0
        assert result.fp == 0
        assert result.fn == 1

    def test_no_ground_truths_all_fp(self) -> None:
        pred = Box(0, 0, 10, 10)
        result = match_predictions(self._sample([(pred, 0.9)], []))
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 0

    def test_empty_sample(self) -> None:
        result = match_predictions(self._sample([], []))
        assert result.tp == result.fp == result.fn == 0

    def test_duplicate_prediction_only_one_tp(self) -> None:
        """Two predictions on the same gt: the higher-confidence one wins."""
        gt = Box(0, 0, 10, 10)
        result = match_predictions(
            self._sample([(gt, 0.9), (gt, 0.5)], [gt])
        )
        assert result.tp == 1
        assert result.fp == 1
        assert result.fn == 0

    def test_below_threshold_is_fp(self) -> None:
        # IoU = 1/3 ≈ 0.33, below threshold of 0.5.
        pred = Box(0, 0, 10, 10)
        gt = Box(5, 0, 15, 10)
        result = match_predictions(self._sample([(pred, 0.9)], [gt]), iou_threshold=0.5)
        assert result.tp == 0
        assert result.fp == 1
        assert result.fn == 1

    def test_precision_recall_f1(self) -> None:
        box = Box(0, 0, 10, 10)
        result = match_predictions(self._sample([(box, 0.9)], [box]))
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f1 == pytest.approx(1.0)

    def test_zero_denominator_precision(self) -> None:
        """No predictions and no gt: precision is 0 not a ZeroDivisionError."""
        result = match_predictions(self._sample([], []))
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0


# ===========================================================================
# Detection — average precision
# ===========================================================================


class TestAveragePrecision:
    def test_perfect_detector(self) -> None:
        box = Box(0, 0, 10, 10)
        samples = [
            DetectionSample(str(i), [(box, 0.9)], [box]) for i in range(10)
        ]
        ap = average_precision(samples)
        assert ap == pytest.approx(1.0, abs=1e-6)

    def test_empty_samples_give_zero(self) -> None:
        assert average_precision([]) == 0.0

    def test_no_ground_truths_give_zero(self) -> None:
        box = Box(0, 0, 10, 10)
        samples = [DetectionSample("img", [(box, 0.9)], [])]
        assert average_precision(samples) == 0.0

    def test_no_predictions_give_zero(self) -> None:
        box = Box(0, 0, 10, 10)
        samples = [DetectionSample("img", [], [box])]
        assert average_precision(samples) == 0.0

    def test_ap_bounded(self) -> None:
        box = Box(0, 0, 10, 10)
        far = Box(100, 100, 110, 110)
        samples = [
            DetectionSample("good", [(box, 0.9)], [box]),
            DetectionSample("bad", [(far, 0.5)], [box]),
        ]
        ap = average_precision(samples)
        assert 0.0 <= ap <= 1.0


# ===========================================================================
# Detection — dataset aggregation
# ===========================================================================


class TestDetectionMetrics:
    def test_all_correct(self) -> None:
        box = Box(0, 0, 10, 10)
        samples = [DetectionSample(str(i), [(box, 0.9)], [box]) for i in range(5)]
        m = detection_metrics(samples)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)
        assert m.f1 == pytest.approx(1.0)
        assert m.ap == pytest.approx(1.0, abs=1e-6)
        assert m.mean_iou == pytest.approx(1.0)

    def test_counts(self) -> None:
        box = Box(0, 0, 10, 10)
        far = Box(100, 100, 110, 110)
        samples = [
            DetectionSample("a", [(box, 0.9)], [box]),       # 1 TP
            DetectionSample("b", [(far, 0.9)], [box]),       # 1 FP, 1 FN
            DetectionSample("c", [], [box]),                  # 1 FN
        ]
        m = detection_metrics(samples)
        assert m.tp == 1
        assert m.fp == 1
        assert m.fn == 2
        assert m.n_images == 3
        assert m.n_ground_truths == 3
        assert m.n_predictions == 2


# ===========================================================================
# OCR — character error rate
# ===========================================================================


class TestCharacterErrorRate:
    def test_perfect_match(self) -> None:
        assert character_error_rate("ABC1234", "ABC1234") == 0.0

    def test_empty_prediction_is_one(self) -> None:
        assert character_error_rate("", "ABC1234") == 1.0

    def test_one_substitution(self) -> None:
        # edit distance = 1, len = 7 → CER = 1/7 ≈ 0.1429
        assert character_error_rate("ABC0234", "ABC1234") == pytest.approx(1 / 7, abs=1e-4)

    def test_one_deletion(self) -> None:
        # "ABC123" vs "ABC1234": edit distance = 1, len = 7
        assert character_error_rate("ABC123", "ABC1234") == pytest.approx(1 / 7, abs=1e-4)

    def test_clamped_to_one(self) -> None:
        # Wildly wrong prediction: even if edit distance > length, CER ≤ 1.
        assert character_error_rate("XXXXXXXX", "AB") <= 1.0

    def test_empty_ground_truth_raises(self) -> None:
        with pytest.raises(ValueError, match="ground_truth must be non-empty"):
            character_error_rate("ABC1234", "")

    def test_symmetry_not_assumed(self) -> None:
        # CER is not symmetric: len(gt) is the denominator.
        cer_ab = character_error_rate("A", "AB")   # dist=1, len=2 → 0.5
        cer_ba = character_error_rate("AB", "A")   # dist=1, len=1 → 1.0
        assert cer_ab == pytest.approx(0.5)
        assert cer_ba == pytest.approx(1.0)


# ===========================================================================
# OCR — dataset aggregation
# ===========================================================================


class TestOcrMetrics:
    @pytest.fixture
    def formats(self) -> PlateFormatConfig:
        return PlateFormatConfig()

    def test_perfect_reads(self, formats: PlateFormatConfig) -> None:
        samples = [
            OCRSample("a", "ABC1234", "ABC1234"),
            OCRSample("b", "123ABC", "123ABC"),
        ]
        m = ocr_metrics(samples, formats)
        assert m.n_samples == 2
        assert m.mean_cer == 0.0
        assert m.exact_match_rate == 1.0
        assert m.read_rate == 1.0
        # Both are valid plate formats.
        assert m.format_match_rate == pytest.approx(1.0)

    def test_no_reads(self, formats: PlateFormatConfig) -> None:
        samples = [
            OCRSample("a", "", "ABC1234"),
            OCRSample("b", "", "123ABC"),
        ]
        m = ocr_metrics(samples, formats)
        assert m.read_rate == 0.0
        assert m.exact_match_rate == 0.0
        assert m.mean_cer == 1.0
        # No predictions → format_match_rate is None (undefined).
        assert m.format_match_rate is None

    def test_mixed_reads(self) -> None:
        samples = [
            OCRSample("ok", "ABC1234", "ABC1234"),
            OCRSample("fail", "", "XYZ9999"),
        ]
        m = ocr_metrics(samples)
        assert m.n_samples == 2
        assert m.n_reads == 1
        assert m.read_rate == pytest.approx(0.5)
        assert m.exact_match_rate == pytest.approx(0.5)
        # CER: sample 1 = 0.0, sample 2 = 1.0 → mean = 0.5
        assert m.mean_cer == pytest.approx(0.5)

    def test_per_engine_breakdown(self) -> None:
        samples = [
            OCRSample("a", "ABC1234", "ABC1234", engine="tesseract"),
            OCRSample("b", "123ABC", "123ABC", engine="paddleocr"),
            OCRSample("c", "WRONG00", "ABC1234", engine="tesseract"),
        ]
        m = ocr_metrics(samples)
        assert "tesseract" in m.per_engine
        assert "paddleocr" in m.per_engine

        tess = m.per_engine["tesseract"]
        assert tess.n_samples == 2
        assert tess.exact_match_rate == pytest.approx(0.5)

        paddle = m.per_engine["paddleocr"]
        assert paddle.n_samples == 1
        assert paddle.exact_match_rate == pytest.approx(1.0)

    def test_no_engine_no_per_engine(self) -> None:
        samples = [OCRSample("a", "ABC1234", "ABC1234")]
        m = ocr_metrics(samples)
        assert m.per_engine == {}

    def test_empty_samples(self) -> None:
        m = ocr_metrics([])
        assert m.n_samples == 0
        assert m.mean_cer == 0.0

    def test_format_match_rate_ignores_invalid(
        self, formats: PlateFormatConfig
    ) -> None:
        # "GARBAGE" is not a plate format; "ABC1234" is.
        samples = [
            OCRSample("a", "GARBAGE", "ABC1234"),
            OCRSample("b", "ABC1234", "ABC1234"),
        ]
        m = ocr_metrics(samples, formats)
        # 1 out of 2 non-empty predictions matches a format.
        assert m.format_match_rate == pytest.approx(0.5)

    def test_normalisation_applied(self, formats: PlateFormatConfig) -> None:
        # Raw prediction with separator and lowercase — should still match.
        samples = [OCRSample("a", "abc-1234", "ABC1234")]
        m = ocr_metrics(samples, formats)
        assert m.exact_match_rate == pytest.approx(1.0)
        assert m.mean_cer == pytest.approx(0.0)
