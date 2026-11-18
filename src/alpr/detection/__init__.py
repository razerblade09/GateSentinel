"""Detection package — vehicle and plate detectors."""

from alpr.detection.plate import PlateDetector
from alpr.detection.types import Detection
from alpr.detection.vehicle import VehicleDetector

__all__ = ["Detection", "PlateDetector", "VehicleDetector"]
