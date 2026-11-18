"""Engine construction from configuration.

A registry rather than an ``if/elif`` chain: adding an engine means registering
it, not editing a factory that everything else depends on. It also gives the
evaluation harness a way to enumerate what is available and benchmark all of
them without hardcoding a list.
"""

from __future__ import annotations

from collections.abc import Callable

from alpr.config.schema import AppConfig, OCRConfig, PlateFormatConfig
from alpr.ocr.base import OCREngine, PlateRecognizer
from alpr.utils.logging import get_logger

logger = get_logger(__name__)

EngineFactory = Callable[[OCRConfig], OCREngine]

_REGISTRY: dict[str, EngineFactory] = {}


def register(name: str) -> Callable[[EngineFactory], EngineFactory]:
    """Register an engine factory under ``name``."""

    def decorator(factory: EngineFactory) -> EngineFactory:
        _REGISTRY[name] = factory
        return factory

    return decorator


@register("paddleocr")
def _build_paddle(config: OCRConfig) -> OCREngine:
    from alpr.ocr.paddle import PaddleOCREngine

    return PaddleOCREngine(lang=config.lang, use_angle_cls=config.use_angle_cls)


@register("tesseract")
def _build_tesseract(config: OCRConfig) -> OCREngine:
    from alpr.ocr.tesseract import TesseractEngine

    # psm 6 (uniform block) rather than 8 (single word): plates are frequently
    # two-line, and 8 cannot read them at all.
    return TesseractEngine(psm=6)


def available_engines() -> tuple[str, ...]:
    """Names of every registered engine, whether or not its deps are installed."""
    return tuple(sorted(_REGISTRY))


def build_engine(config: OCRConfig) -> OCREngine:
    """Construct the engine named by ``config.engine``."""
    try:
        factory = _REGISTRY[config.engine]
    except KeyError:
        raise ValueError(
            f"unknown OCR engine {config.engine!r}; "
            f"available: {', '.join(available_engines())}"
        ) from None

    engine = factory(config)
    logger.info("OCR engine ready: %s", engine.name)
    return engine


def build_recognizer(
    ocr_config: OCRConfig,
    format_config: PlateFormatConfig,
) -> PlateRecognizer:
    """Construct the full recognition pipeline from config sections."""
    return PlateRecognizer(build_engine(ocr_config), ocr_config, format_config)


def recognizer_from_app_config(config: AppConfig) -> PlateRecognizer:
    """Convenience wrapper taking the root config object."""
    return build_recognizer(config.ocr, config.plate_format)
