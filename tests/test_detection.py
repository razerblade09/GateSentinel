"""Tests for preprocessing, detection types, and tracking.

YOLO models are faked throughout. The contract being tested is not "does
Ultralytics work" (they test that) but "does our wrapper correctly parse
results, offset coordinates, and feed the vote aggregator".

The vote aggregator is entirely model-free and is tested exhaustively here.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from alpr.config.schema import PlateDetectorConfig, TrackingConfig, VehicleDetectorConfig
from alpr.detection.plate import PlateDetector
from alpr.detection.types import Detection
from alpr.detection.vehicle import VehicleDetector
from alpr.ocr.types import PlateRead, TextLine
from alpr.preprocessing.geometry import clamp_box, crop, offset_box, pad_box
from alpr.tracking.tracker import TrackVoter, VoteResult


# ===========================================================================
# Preprocessing — geometry
# ===========================================================================


class TestClampBox:
    def test_inside_image(self) -> None:
        assert clamp_box(10, 20, 50, 80, 100, 100) == (10, 20, 50, 80)

    def test_clips_negative(self) -> None:
        x1, y1, x2, y2 = clamp_box(-5, -10, 60, 70, 100, 100)
        assert x1 == 0
        assert y1 == 0

    def test_clips_beyond_image(self) -> None:
        x1, y1, x2, y2 = clamp_box(10, 10, 200, 300, 100, 100)
        assert x2 == 100
        assert y2 == 100

    def test_returns_integers(self) -> None:
        result = clamp_box(1.7, 2.3, 50.9, 80.1, 100, 100)
        assert all(isinstance(v, int) for v in result)


class TestPadBox:
    def test_zero_padding_unchanged(self) -> None:
        assert pad_box(10, 10, 50, 50, 0.0, 100, 100) == (10, 10, 50, 50)

    def test_expands_symmetrically(self) -> None:
        # 40×40 box, 10% padding → +4px each side.
        x1, y1, x2, y2 = pad_box(10, 10, 50, 50, 0.10, 200, 200)
        assert x1 == 6
        assert y1 == 6
        assert x2 == 54
        assert y2 == 54

    def test_clips_at_boundary(self) -> None:
        # Box very close to the top-left corner.
        x1, y1, _, _ = pad_box(2, 2, 20, 20, 0.5, 100, 100)
        assert x1 == 0
        assert y1 == 0


class TestCrop:
    def _img(self) -> np.ndarray:
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_valid_crop(self) -> None:
        img = self._img()
        result = crop(img, 10, 20, 50, 60)
        assert result.shape == (40, 40, 3)

    def test_zero_area_returns_empty(self) -> None:
        result = crop(self._img(), 10, 10, 10, 50)
        assert result.size == 0

    def test_crop_is_view(self) -> None:
        img = self._img()
        c = crop(img, 0, 0, 10, 10)
        c[0, 0, 0] = 255
        assert img[0, 0, 0] == 255


class TestOffsetBox:
    def test_positive_offset(self) -> None:
        assert offset_box(5, 10, 20, 30, 100, 150) == (105, 160, 120, 180)

    def test_zero_offset(self) -> None:
        assert offset_box(5, 10, 20, 30, 0, 0) == (5, 10, 20, 30)


# ===========================================================================
# Detection — types
# ===========================================================================


class TestDetectionType:
    def _det(self, **kw: float) -> Detection:
        defaults = dict(x1=10, y1=20, x2=50, y2=80, confidence=0.9, class_id=2)
        defaults.update(kw)
        return Detection(**defaults)  # type: ignore[arg-type]

    def test_dimensions(self) -> None:
        d = self._det(x1=10, y1=20, x2=50, y2=80)
        assert d.width == pytest.approx(40)
        assert d.height == pytest.approx(60)
        assert d.area == pytest.approx(2400)

    def test_centre(self) -> None:
        d = self._det(x1=10, y1=20, x2=50, y2=80)
        assert d.cx == pytest.approx(30)
        assert d.cy == pytest.approx(50)

    def test_with_track(self) -> None:
        d = self._det()
        t = d.with_track(42)
        assert t.track_id == 42
        assert d.track_id is None  # original unchanged

    def test_with_offset(self) -> None:
        d = self._det(x1=10, y1=20, x2=50, y2=80)
        shifted = d.with_offset(100, 200)
        assert shifted.x1 == pytest.approx(110)
        assert shifted.y1 == pytest.approx(220)
        assert shifted.x2 == pytest.approx(150)
        assert shifted.y2 == pytest.approx(280)


# ===========================================================================
# Detection — vehicle and plate detectors with fake YOLO
# ===========================================================================


def _make_fake_yolo_result(boxes_data: list[tuple[float, float, float, float, float, int]]):
    """Build a fake Ultralytics result object."""
    result = MagicMock()
    if not boxes_data:
        result.boxes = None
        return result

    boxes = MagicMock()
    n = len(boxes_data)

    import torch

    xyxy_list = [[x1, y1, x2, y2] for x1, y1, x2, y2, _conf, _cls in boxes_data]
    conf_list = [conf for _x1, _y1, _x2, _y2, conf, _cls in boxes_data]
    cls_list = [cls for _x1, _y1, _x2, _y2, _conf, cls in boxes_data]

    boxes.xyxy = torch.tensor(xyxy_list, dtype=torch.float32)
    boxes.conf = torch.tensor(conf_list, dtype=torch.float32)
    boxes.cls = torch.tensor(cls_list, dtype=torch.float32)
    boxes.id = None
    boxes.__len__ = lambda self: n  # noqa: ARG005

    result.boxes = boxes
    return result


class TestVehicleDetector:
    def _detector(self, results_data=None):
        model = MagicMock()
        if results_data is None:
            results_data = []
        fake_result = _make_fake_yolo_result(results_data)
        model.predict.return_value = [fake_result]
        cfg = VehicleDetectorConfig()
        return VehicleDetector(model, cfg), model

    def test_returns_detections(self) -> None:
        detector, _ = self._detector([(10, 20, 50, 80, 0.9, 2)])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        detections = detector.detect(frame)
        assert len(detections) == 1
        assert detections[0].x1 == pytest.approx(10)
        assert detections[0].confidence == pytest.approx(0.9)
        assert detections[0].class_id == 2

    def test_sorted_by_confidence(self) -> None:
        detector, _ = self._detector([
            (0, 0, 10, 10, 0.5, 2),
            (20, 20, 40, 40, 0.9, 2),
        ])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        dets = detector.detect(frame)
        assert dets[0].confidence > dets[1].confidence

    def test_empty_frame_returns_empty(self) -> None:
        detector, _ = self._detector()
        result = detector.detect(np.zeros((0, 0, 3), np.uint8))
        assert result == []

    def test_no_boxes_returns_empty(self) -> None:
        detector, _ = self._detector([])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.detect(frame) == []

    def test_inference_failure_returns_empty(self) -> None:
        model = MagicMock()
        model.predict.side_effect = RuntimeError("GPU exploded")
        detector = VehicleDetector(model, VehicleDetectorConfig())
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        assert detector.detect(frame) == []


class TestPlateDetector:
    def _detector(self, results_data=None):
        model = MagicMock()
        if results_data is None:
            results_data = []
        fake_result = _make_fake_yolo_result(results_data)
        model.predict.return_value = [fake_result]
        cfg = PlateDetectorConfig(max_plates_per_vehicle=2)
        return PlateDetector(model, cfg), model

    def test_returns_plate_detections(self) -> None:
        detector, _ = self._detector([(5, 10, 60, 30, 0.85, 0)])
        crop_img = np.zeros((60, 100, 3), dtype=np.uint8)
        dets = detector.detect(crop_img)
        assert len(dets) == 1
        assert dets[0].confidence == pytest.approx(0.85)

    def test_caps_at_max_plates(self) -> None:
        # Three plates returned by model, but max is 2.
        detector, _ = self._detector([
            (0, 0, 10, 10, 0.9, 0),
            (20, 0, 30, 10, 0.8, 0),
            (40, 0, 50, 10, 0.7, 0),
        ])
        crop_img = np.zeros((60, 100, 3), dtype=np.uint8)
        assert len(detector.detect(crop_img)) == 2

    def test_empty_crop_returns_empty(self) -> None:
        detector, _ = self._detector()
        assert detector.detect(np.zeros((0, 0, 3), np.uint8)) == []


# ===========================================================================
# Tracking — TrackVoter
# ===========================================================================


def _read(text: str, confidence: float = 0.9, valid: bool = True) -> PlateRead:
    return PlateRead(
        raw_text=text,
        text=text if valid else "",
        confidence=confidence,
        format_name="car" if valid else None,
        corrections=0,
        lines=(),
        engine="fake",
    )


class TestTrackVoter:
    @pytest.fixture
    def cfg(self) -> TrackingConfig:
        return TrackingConfig(
            min_reads_for_vote=3,
            max_reads_per_track=5,
            track_timeout_s=0.1,  # short for testing
        )

    def test_no_result_until_budget_hit(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        # 4 reads, budget is 5 → no immediate result yet.
        for _ in range(4):
            result = voter.record(1, _read("ABC1234"))
        assert result is None

    def test_result_on_budget_hit(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        result = None
        for _ in range(5):
            result = voter.record(1, _read("ABC1234"))
        assert result is not None
        assert result.track_id == 1
        assert result.text == "ABC1234"

    def test_majority_vote(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        reads = [
            _read("ABC1234"),
            _read("ABC1234"),
            _read("XYZ9999"),
            _read("ABC1234"),
            _read("XYZ9999"),
        ]
        result = None
        for r in reads:
            result = voter.record(1, r)
        assert result is not None
        assert result.text == "ABC1234"  # 3 votes vs 2

    def test_tie_broken_by_confidence(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        reads = [
            _read("ABC1234", confidence=0.9),
            _read("ABC1234", confidence=0.9),
            _read("XYZ9999", confidence=0.95),
            _read("XYZ9999", confidence=0.95),
            _read("ABC1234", confidence=0.1),  # 3 for ABC, 2 for XYZ
        ]
        result = None
        for r in reads:
            result = voter.record(1, r)
        # ABC wins by count.
        assert result is not None
        assert result.text == "ABC1234"

    def test_no_valid_reads_gives_empty_result(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        result = None
        for _ in range(5):
            result = voter.record(1, _read("", valid=False))
        assert result is not None
        assert not result.is_valid
        assert result.text == ""

    def test_emitted_track_ignored_on_further_records(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        for _ in range(5):
            voter.record(1, _read("ABC1234"))
        # Track 1 is now emitted. Extra reads should be silently dropped.
        result = voter.record(1, _read("XYZ9999"))
        assert result is None

    def test_flush_expired_emits_timed_out_tracks(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        # 3 reads (= min_reads_for_vote), then wait for timeout.
        for _ in range(3):
            voter.record(1, _read("ABC1234"))

        time.sleep(0.15)  # wait past track_timeout_s = 0.1
        results = voter.flush_expired()
        assert len(results) == 1
        assert results[0].text == "ABC1234"

    def test_flush_does_not_emit_insufficient_reads(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        # Only 2 reads, below min_reads_for_vote=3.
        voter.record(1, _read("ABC1234"))
        voter.record(1, _read("ABC1234"))

        time.sleep(0.15)
        results = voter.flush_expired()
        assert results == []

    def test_independent_tracks(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        result1 = result2 = None
        for _ in range(5):
            result1 = voter.record(1, _read("ABC1234"))
            result2 = voter.record(2, _read("XYZ9999"))
        assert result1 is not None and result1.text == "ABC1234"
        assert result2 is not None and result2.text == "XYZ9999"

    def test_active_track_ids(self, cfg: TrackingConfig) -> None:
        voter = TrackVoter(cfg)
        voter.record(1, _read("ABC1234"))
        voter.record(2, _read("XYZ9999"))
        ids = voter.active_track_ids()
        assert set(ids) == {1, 2}

    def test_vote_result_str(self) -> None:
        r = VoteResult(
            track_id=7, text="ABC1234", confidence=0.88,
            n_reads=5, n_valid_reads=4, format_name="car",
        )
        assert "ABC1234" in str(r)
        assert "7" in str(r)
