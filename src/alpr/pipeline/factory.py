"""Factory: build a fully-wired Pipeline from an AppConfig.

The pipeline constructor takes many injected dependencies. This factory
resolves them all from a single :class:`~alpr.config.schema.AppConfig` so
the run script has one call to make rather than ten.

Keeping the factory separate from the pipeline class preserves testability:
tests that want a fake detector don't go through the factory -- they
construct the pipeline directly with their fake. The factory is only for
production use.
"""

from __future__ import annotations

import datetime

from alpr.config.schema import AppConfig
from alpr.database.repository import Repository
from alpr.database.session import build_engine
from alpr.detection.plate import PlateDetector
from alpr.detection.vehicle import VehicleDetector
from alpr.models.registry import load_model
from alpr.ocr.base import PlateRecognizer
from alpr.ocr.engines import build_engine as build_ocr_engine
from alpr.pipeline.pipeline import Pipeline
from alpr.tracking.tracker import TrackVoter, VehicleTracker
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


def _resolve_device(config: AppConfig) -> str:
    """Resolve 'auto' to a concrete Torch device string."""
    if config.runtime.device != "auto":
        return config.runtime.device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def build_pipeline(
    config: AppConfig,
    *,
    allow_list: set[str] | None = None,
    keep_frames: bool = False,
    with_database: bool = True,
) -> Pipeline:
    """Construct a fully-wired :class:`~alpr.pipeline.pipeline.Pipeline`.

    Args:
        config: Root application config (from the layered YAML loader).
        allow_list: Initial set of authorised plate strings. Can be updated
            later via :meth:`~alpr.pipeline.pipeline.Pipeline.set_allow_list`.
        keep_frames: Attach raw frames to FrameResults (costs RAM).
        with_database: Whether to wire a database repository. Set False for
            offline benchmarking runs that don't need persistence.

    Returns:
        A ready-to-use :class:`~alpr.pipeline.pipeline.Pipeline`.
    """
    device = _resolve_device(config)
    logger.info("pipeline device: %s", device)

    # ---- Models ----------------------------------------------------------
    models_dir = config.paths.models_dir
    vehicle_model = load_model(
        config.vehicle_detector.weights, models_dir, device=device
    )
    plate_model = load_model(
        config.plate_detector.weights, models_dir, device=device
    )

    # ---- Stage 1 ---------------------------------------------------------
    vehicle_detector = VehicleDetector(vehicle_model, config.vehicle_detector)
    vehicle_tracker = (
        VehicleTracker(vehicle_model, config.tracking)
        if config.tracking.enabled
        else None
    )

    # ---- Stage 2 ---------------------------------------------------------
    plate_detector = PlateDetector(plate_model, config.plate_detector)

    # ---- OCR -------------------------------------------------------------
    ocr_engine = build_ocr_engine(config.ocr)
    ocr_engine.warmup()
    recognizer = PlateRecognizer(ocr_engine, config.ocr, config.plate_format)

    # ---- Tracking / voting -----------------------------------------------
    voter = TrackVoter(config.tracking)

    # ---- Database --------------------------------------------------------
    repository: Repository | None = None
    session_id = ""
    if with_database:
        engine = build_engine(config.database)
        session_id = datetime.datetime.now(tz=datetime.timezone.utc).isoformat()
        repository = Repository(engine, session_id=session_id)

    logger.info(
        "pipeline built: vehicle=%s plate=%s ocr=%s db=%s",
        config.vehicle_detector.weights,
        config.plate_detector.weights,
        config.ocr.engine,
        "enabled" if repository else "disabled",
    )

    return Pipeline(
        config=config,
        vehicle_detector=vehicle_detector,
        vehicle_tracker=vehicle_tracker,
        plate_detector=plate_detector,
        recognizer=recognizer,
        voter=voter,
        repository=repository,
        allow_list=allow_list,
        keep_frames=keep_frames,
        session_id=session_id,
    )
