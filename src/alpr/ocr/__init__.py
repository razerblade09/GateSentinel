"""Plate text recognition."""

from alpr.ocr.base import OCREngine, PlateRecognizer
from alpr.ocr.engines import (
    available_engines,
    build_engine,
    build_recognizer,
    recognizer_from_app_config,
    register,
)
from alpr.ocr.types import CorrectionResult, PlateRead, TextLine

__all__ = [
    "CorrectionResult",
    "OCREngine",
    "PlateRead",
    "PlateRecognizer",
    "TextLine",
    "available_engines",
    "build_engine",
    "build_recognizer",
    "recognizer_from_app_config",
    "register",
]
