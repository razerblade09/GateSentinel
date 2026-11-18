"""Pipeline package — real-time ALPR orchestration."""

from alpr.pipeline.capture import open_capture
from alpr.pipeline.factory import build_pipeline
from alpr.pipeline.pipeline import Pipeline
from alpr.pipeline.types import FrameResult, GateDecision, PlateInFrame

__all__ = [
    "open_capture",
    "build_pipeline",
    "Pipeline",
    "FrameResult",
    "GateDecision",
    "PlateInFrame",
]
