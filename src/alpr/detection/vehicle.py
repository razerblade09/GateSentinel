"""Stage-one detector: find vehicles in the full frame.

This detector uses COCO-pretrained weights and needs no training -- cars,
motorcycles, buses, and trucks are all COCO categories. That is a deliberate
architectural choice: the first stage is about finding a stable object to
track, not about reading its plate, and COCO covers everything we need.

The ``detect`` method is intentionally simple. It runs inference and converts
Ultralytics' result format to our ``Detection`` type. All the intelligence
(padding, cropping, feeding the plate detector) lives in the pipeline layer,
which wires stages together without any stage knowing about the others.

The class does **not** own a model instance. The registry manages loading and
caching; the pipeline passes the model in. This keeps the detector testable:
you can inject any callable that returns the right format.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from alpr.config.schema import VehicleDetectorConfig
from alpr.detection.types import Detection
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


class VehicleDetector:
    """Run the COCO-pretrained vehicle detector on a full frame.

    Args:
        model: A loaded Ultralytics YOLO model. Use
            :func:`alpr.models.registry.load_model` to obtain one.
        config: Vehicle detector settings from the application config.
    """

    def __init__(self, model: YOLO, config: VehicleDetectorConfig) -> None:
        self._model = model
        self._cfg = config

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Find vehicles in ``frame``.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            List of :class:`~alpr.detection.types.Detection` objects, one per
            vehicle found, sorted by confidence descending. Empty list if
            nothing was found or inference failed.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self._model.predict(
                source=frame,
                conf=self._cfg.confidence,
                iou=self._cfg.iou,
                imgsz=self._cfg.imgsz,
                classes=self._cfg.classes,
                verbose=False,
            )
        except Exception:
            logger.exception("vehicle detector inference failed")
            return []

        return self._parse(results)

    @staticmethod
    def _parse(results: list) -> list[Detection]:  # type: ignore[type-arg]
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls = int(boxes.cls[i].cpu())
                detections.append(
                    Detection(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls,
                    )
                )
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
