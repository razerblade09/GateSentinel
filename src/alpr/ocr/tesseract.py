"""Tesseract engine, kept as a measurable baseline.

Not recommended for production. It is here because "the new thing is better"
is a claim that needs a number attached, and this is what that number is
measured against.

Two deliberate differences from the previous system, both of which matter:

``--psm 6`` instead of ``--psm 8``
    ``8`` means "treat the image as a single word" and therefore cannot read a
    two-line plate at all. ``6`` means "a uniform block of text", which permits
    multiple lines. Philippine motorcycle plates are two-line, so ``8`` silently
    discarded them.

``image_to_data`` instead of ``image_to_string``
    Returns per-word confidence and bounding boxes rather than a bare string.
    Without confidence there is nothing to threshold and nothing to vote on, so
    every read has to be trusted equally -- including the wrong ones.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from alpr.ocr.base import OCREngine
from alpr.ocr.types import TextLine
from alpr.utils.logging import get_logger

logger = get_logger(__name__)

WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

_WINDOWS_DEFAULT = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class TesseractEngine(OCREngine):
    """Wraps pytesseract.

    Args:
        binary: Path to the tesseract executable. Discovered from PATH, then
            the standard Windows install location, if not given.
        psm: Page segmentation mode. 6 (uniform block) handles multi-line.
    """

    def __init__(self, binary: Path | str | None = None, psm: int = 6) -> None:
        try:
            import pytesseract
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "pytesseract is not installed. Install the OCR extra:\n"
                '    pip install -e ".[ocr]"'
            ) from exc

        self._pytesseract = pytesseract
        self._psm = psm

        resolved = self._resolve_binary(binary)
        if resolved is not None:
            pytesseract.pytesseract.tesseract_cmd = str(resolved)
        self._binary = resolved

        self._config = (
            f"--psm {psm} --oem 3 -c tessedit_char_whitelist={WHITELIST}"
        )

    @staticmethod
    def _resolve_binary(binary: Path | str | None) -> Path | None:
        if binary:
            path = Path(binary)
            if not path.is_file():
                raise FileNotFoundError(f"tesseract binary not found: {path}")
            return path
        found = shutil.which("tesseract")
        if found:
            return Path(found)
        if _WINDOWS_DEFAULT.is_file():
            return _WINDOWS_DEFAULT
        logger.warning(
            "tesseract binary not found on PATH; relying on pytesseract's default"
        )
        return None

    @property
    def name(self) -> str:
        return f"tesseract-psm{self._psm}"

    def read_lines(self, image: np.ndarray) -> list[TextLine]:
        data = self._pytesseract.image_to_data(
            image,
            config=self._config,
            output_type=self._pytesseract.Output.DICT,
        )

        lines: list[TextLine] = []
        for index, raw_text in enumerate(data["text"]):
            text = raw_text.strip()
            if not text:
                continue
            # Tesseract reports -1 for entries it declines to score.
            raw_conf = float(data["conf"][index])
            if raw_conf < 0:
                continue
            left = float(data["left"][index])
            top = float(data["top"][index])
            lines.append(
                TextLine(
                    text=text,
                    confidence=raw_conf / 100.0,
                    box=(
                        left,
                        top,
                        left + float(data["width"][index]),
                        top + float(data["height"][index]),
                    ),
                )
            )
        return lines

    def warmup(self) -> None:
        self.read_lines(np.full((32, 96, 3), 255, dtype=np.uint8))
