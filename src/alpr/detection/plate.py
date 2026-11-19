"""Stage-two detector: find the licence plate inside a vehicle crop.

The plate detector is a custom-trained YOLO model (produced in the training
milestone). Its input is a vehicle crop, not a full frame. Running it on a
crop rather than the full frame gives three advantages:

1. The plate occupies a much larger fraction of the crop, so the detector
   sees an easier problem.
2. The crop eliminates the other side of the road and background signage,
   reducing false positives.
3. The model only has to handle one vehicle's worth of image, so a small,
   fast checkpoint (``plate_yolo26s``) is sufficient.

After running inference on the crop, all coordinates are translated back to
the original frame's space with :func:`~alpr.detection.types.Detection.with_offset`.
The pipeline layer does that translation; this class only knows about the crop.
"""

from __future__ import annotations

import numpy as np
from ultralytics import YOLO

from alpr.config.schema import PlateDetectorConfig
from alpr.detection.types import Detection
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


class PlateDetector:
    """Find licence plates inside a vehicle crop (or a full frame).

    Args:
        model: A loaded Ultralytics YOLO model (plate-trained).
        config: Plate detector settings from the application config.
    """

    def __init__(self, model: YOLO, config: PlateDetectorConfig) -> None:
        self._model = model
        self._cfg = config

    def detect(self, crop: np.ndarray) -> list[Detection]:
        """Find plates in ``crop``.

        Args:
            crop: BGR image. Coordinates in the returned detections are
                relative to this image's top-left corner.

        Returns:
            Up to ``config.max_plates_per_vehicle`` detections, sorted by
            confidence descending. Coordinates are in ``crop``-space.
        """
        if crop is None or crop.size == 0:
            return []

        try:
            results = self._model.predict(
                source=crop,
                conf=self._cfg.confidence,
                iou=self._cfg.iou,
                imgsz=self._cfg.imgsz,
                verbose=False,
            )
        except Exception:
            logger.exception("plate detector inference failed")
            return []

        detections = self._parse(results)
        return detections[: self._cfg.max_plates_per_vehicle]

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
