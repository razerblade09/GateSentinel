"""PaddleOCR engine.

Why this over Tesseract
-----------------------
Tesseract is a *document* OCR engine: trained on scanned pages, flat ink,
consistent fonts, high DPI. PaddleOCR's recognition models are trained on
**natural scene text** -- photographs of signs, labels and, yes, plates, with
perspective, glare, motion blur and varying illumination. That distribution
difference is the entire reason to switch; it is not a matter of one model being
generically "better".

It also detects text regions before recognising them, so a two-line plate comes
back as two lines with separate boxes and confidences, rather than requiring the
caller to guess where to split the image.

What it is still not
--------------------
Plate-specialised. It has no notion that ``LLLDDDD`` is a plate, and it will
happily read a bumper sticker. The format validation and positional correction
in :mod:`alpr.ocr.postprocess` supply that, and a dedicated character model
trained on Philippine plates would beat this if the measured accuracy turns out
to be insufficient.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from alpr.ocr.base import OCREngine
from alpr.ocr.types import TextLine
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


class PaddleOCREngine(OCREngine):
    """Wraps PaddleOCR's detection + recognition pipeline.

    The import is deferred to construction time so that the rest of the system
    remains usable without a ~500 MB dependency installed.

    Args:
        lang: Recognition language model. "en" covers Latin letters and digits.
        use_angle_cls: Enable the orientation classifier, which corrects crops
            that arrive rotated 180 degrees.
        det: Run text detection. Disable only when the crop is already a tight
            single line, which is rarely true for plates.
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        det: bool = True,
    ) -> None:
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "paddleocr is not installed. Install the OCR extra:\n"
                '    pip install -e ".[ocr]"'
            ) from exc

        self._lang = lang
        self._det = det
        self._ocr = self._construct(PaddleOCR, lang, use_angle_cls)

    @staticmethod
    def _construct(paddle_cls: Any, lang: str, use_angle_cls: bool) -> Any:
        """Build a PaddleOCR instance across incompatible major versions.

        Two problems are handled here.

        **API drift.** PaddleOCR renamed and removed constructor arguments
        between 2.x and 3.x (``show_log`` disappeared, ``use_angle_cls`` became
        ``use_textline_orientation``). Probing keyword sets is ugly but keeps
        this working across whichever version pip resolves.

        **oneDNN.** paddlepaddle 3.3 raises ``NotImplementedError:
        ConvertPirAttribute2RuntimeAttribute`` at inference time on some CPU
        builds -- its oneDNN kernel path does not support an attribute type the
        PP-OCRv6 graph uses. ``enable_mkldnn=False`` selects the plain CPU
        kernels and works. The failure surfaces on the *first inference*, not at
        construction, which is why :meth:`warmup` matters: better to fail at
        startup than on the first vehicle at the gate.
        """
        attempts: list[dict[str, Any]] = [
            {
                "lang": lang,
                "use_textline_orientation": use_angle_cls,
                "enable_mkldnn": False,
            },
            {"lang": lang, "enable_mkldnn": False},
            {"lang": lang, "use_textline_orientation": use_angle_cls},
            {"lang": lang, "use_angle_cls": use_angle_cls, "show_log": False},
            {"lang": lang, "use_angle_cls": use_angle_cls},
            {"lang": lang},
        ]
        last_error: Exception | None = None
        for kwargs in attempts:
            try:
                instance = paddle_cls(**kwargs)
                logger.debug("PaddleOCR constructed with %s", sorted(kwargs))
                return instance
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError(f"could not construct PaddleOCR: {last_error}")

    @property
    def name(self) -> str:
        return f"paddleocr-{self._lang}"

    def read_lines(self, image: np.ndarray) -> list[TextLine]:
        raw = self._invoke(image)
        return self._parse(raw)

    def _invoke(self, image: np.ndarray) -> Any:
        """Call whichever inference entry point this version exposes."""
        if hasattr(self._ocr, "predict"):
            try:
                return self._ocr.predict(image)
            except (TypeError, AttributeError):
                pass
        try:
            return self._ocr.ocr(image, cls=True)
        except TypeError:
            return self._ocr.ocr(image)

    def _parse(self, raw: Any) -> list[TextLine]:
        """Normalise PaddleOCR output into :class:`TextLine` objects.

        Two shapes are handled:

        2.x  ``[[[polygon, (text, confidence)], ...]]``
        3.x  ``[{"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}]``
        """
        if not raw:
            return []

        lines: list[TextLine] = []

        # 3.x dict-style result.
        first = raw[0] if isinstance(raw, (list, tuple)) and raw else None
        if isinstance(first, dict) or hasattr(first, "get"):
            for page in raw:
                texts = page.get("rec_texts") or []
                scores = page.get("rec_scores") or []
                polys = page.get("rec_polys") or page.get("dt_polys") or []
                for index, text in enumerate(texts):
                    score = float(scores[index]) if index < len(scores) else 0.0
                    poly = polys[index] if index < len(polys) else None
                    lines.append(
                        TextLine(text=str(text), confidence=score, box=_poly_to_box(poly))
                    )
            return lines

        # 2.x nested-list result.
        for page in raw:
            if not page:
                continue
            for entry in page:
                try:
                    poly, payload = entry[0], entry[1]
                    text, score = payload[0], float(payload[1])
                except (IndexError, TypeError, ValueError):
                    logger.debug("skipping unparseable PaddleOCR entry: %r", entry)
                    continue
                lines.append(
                    TextLine(text=str(text), confidence=score, box=_poly_to_box(poly))
                )
        return lines

    def warmup(self) -> None:
        self.read_lines(np.full((48, 144, 3), 255, dtype=np.uint8))


def _poly_to_box(poly: Any) -> tuple[float, float, float, float] | None:
    """Reduce a 4-point polygon to an axis-aligned bounding box.

    Plates can be photographed at an angle, so the detected region is a
    quadrilateral rather than a rectangle. Reading order only needs the extent,
    so the polygon collapses to its bounds.
    """
    if poly is None:
        return None
    try:
        points = np.asarray(poly, dtype=float).reshape(-1, 2)
    except (ValueError, TypeError):
        return None
    if points.size == 0:
        return None
    return (
        float(points[:, 0].min()),
        float(points[:, 1].min()),
        float(points[:, 0].max()),
        float(points[:, 1].max()),
    )
