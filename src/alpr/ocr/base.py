"""The OCR engine interface and the shared recognition pipeline.

Engines differ only in how they turn an image into text lines. Everything after
that -- ordering, normalisation, correction, confidence aggregation -- is
identical and lives here, in :class:`PlateRecognizer`.

That split is what makes the engine swappable. Tesseract today, PaddleOCR
tomorrow, a custom character detector after that: the pipeline around them does
not change, and the evaluation harness can compare them on equal terms because
they are all doing exactly the same post-processing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from alpr.config.schema import OCRConfig, PlateFormatConfig
from alpr.ocr.postprocess import correct, join_lines, order_lines
from alpr.ocr.types import PlateRead, TextLine
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


class OCREngine(ABC):
    """Turns an image of a plate into lines of text.

    Implementations must not do any plate-specific reasoning: no format
    validation, no character correction, no case handling. Those belong to the
    shared pipeline so that every engine is measured identically.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier recorded on every read, e.g. "paddleocr"."""

    @abstractmethod
    def read_lines(self, image: np.ndarray) -> list[TextLine]:
        """Recognise text in a BGR image.

        Args:
            image: The plate crop, in OpenCV's BGR channel order.

        Returns:
            Text lines in arbitrary order. Return ``[]`` when nothing is found;
            do not raise for an unreadable image.
        """

    def warmup(self) -> None:
        """Optionally run a throwaway inference to pay initialisation cost.

        Worth calling at startup: the first inference of most engines is an
        order of magnitude slower than the rest, and at a gate that latency
        lands on the first vehicle of the day.
        """


class PlateRecognizer:
    """Engine-agnostic recognition pipeline.

    Do not put engine-specific behaviour in here. If a new engine needs special
    handling, that handling belongs inside the engine.
    """

    def __init__(
        self,
        engine: OCREngine,
        ocr_config: OCRConfig,
        format_config: PlateFormatConfig,
    ) -> None:
        self._engine = engine
        self._ocr = ocr_config
        self._format = format_config

    @property
    def engine_name(self) -> str:
        return self._engine.name

    def read(self, crop: np.ndarray) -> PlateRead:
        """Recognise one plate crop.

        The crop is passed to the engine **unmodified**. Historic ALPR advice to
        binarise and sharpen before OCR predates engines trained on natural
        images, and on a plate it destroys the stroke detail those engines rely
        on -- an adaptive threshold with an 11-pixel block operates at roughly
        the width of the characters themselves.
        """
        if crop is None or crop.size == 0:
            return self._empty("empty crop")

        try:
            lines = self._engine.read_lines(crop)
        except Exception:
            logger.exception("engine %s failed on a crop", self._engine.name)
            return self._empty("engine error")

        lines = [
            line
            for line in lines
            if line.text.strip() and line.confidence >= self._ocr.min_char_confidence
        ]
        if not lines:
            return self._empty("")

        ordered = order_lines(lines)
        raw = join_lines(ordered)

        base_confidence = self._aggregate_confidence(ordered)
        result = correct(raw, self._format)

        if result is None:
            return PlateRead(
                raw_text=raw,
                text="",
                confidence=base_confidence,
                format_name=None,
                corrections=0,
                lines=tuple(ordered),
                engine=self._engine.name,
            )

        confidence = base_confidence * (
            self._format.correction_penalty**result.corrections
        )

        if confidence < self._ocr.min_string_confidence:
            logger.debug(
                "rejecting %s: confidence %.2f below threshold %.2f",
                result.text,
                confidence,
                self._ocr.min_string_confidence,
            )
            return PlateRead(
                raw_text=raw,
                text="",
                confidence=confidence,
                format_name=None,
                corrections=result.corrections,
                lines=tuple(ordered),
                engine=self._engine.name,
            )

        return PlateRead(
            raw_text=raw,
            text=result.text,
            confidence=confidence,
            format_name=result.format_name,
            corrections=result.corrections,
            lines=tuple(ordered),
            engine=self._engine.name,
        )

    @staticmethod
    def _aggregate_confidence(lines: list[TextLine]) -> float:
        """Character-count-weighted mean of line confidences.

        Weighting by length stops a stray high-confidence single character from
        dragging the score of a long, uncertain line upward.
        """
        total_chars = sum(len(line.text) for line in lines)
        if not total_chars:
            return 0.0
        weighted = sum(line.confidence * len(line.text) for line in lines)
        return min(1.0, max(0.0, weighted / total_chars))

    def _empty(self, reason: str) -> PlateRead:
        if reason:
            logger.debug("no read: %s", reason)
        return PlateRead(
            raw_text="",
            text="",
            confidence=0.0,
            format_name=None,
            corrections=0,
            lines=(),
            engine=self._engine.name,
        )
