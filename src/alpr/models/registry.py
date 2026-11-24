"""Model registry: resolve a model name to a loaded YOLO instance.

The registry pattern keeps model loading out of the detection classes.
Detection classes describe *what to do with a model*; the registry describes
*which model to load and from where*. Keeping them separate means:

1. Tests can inject a fake model without touching the filesystem.
2. A new weight format (TorchScript, ONNX, OpenVINO) is added in one place.
3. ``configs/default.yaml`` weight names stay symbolic; the registry resolves
   them to real paths at startup.

Loading on first use (lazy) rather than at import time means importing the
package in a test or notebook does not trigger a 300 MB model download.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

from alpr.utils.logging import get_logger

logger = get_logger(__name__)

# Cache so the same weights file is not loaded twice.
_cache: dict[str, YOLO] = {}


def load_model(
    weights: str | Path,
    models_dir: Path | None = None,
    *,
    device: str = "cpu",
) -> YOLO:
    """Load a YOLO model, using an in-process cache for repeated calls.

    Resolution order for ``weights``:
    1. Treat as an absolute path -- use directly if it exists.
    2. Resolve against ``models_dir`` if supplied and the file exists there.
    3. Let Ultralytics handle it (downloads from hub if needed).

    Args:
        weights: Weight file name (``"yolo26n.pt"``) or path.
        models_dir: Directory to search before falling back to Ultralytics hub.
        device: Torch device string -- ``"cpu"``, ``"cuda:0"``, etc.
            Pass ``"auto"`` to let Ultralytics pick.

    Returns:
        A loaded :class:`ultralytics.YOLO` instance.
    """
    resolved = _resolve(weights, models_dir)
    cache_key = f"{resolved}:{device}"

    if cache_key in _cache:
        logger.debug("model cache hit: %s", resolved)
        return _cache[cache_key]

    logger.info("loading model: %s (device=%s)", resolved, device)
    model = YOLO(str(resolved))
    # Move to device now so the first inference is not slower than the rest.
    if device not in ("auto", ""):
        model.to(device)

    _cache[cache_key] = model
    return model


def clear_cache() -> None:
    """Drop all cached models. Useful in long-running test suites."""
    _cache.clear()


def _resolve(weights: str | Path, models_dir: Path | None) -> Path | str:
    path = Path(weights)
    if path.is_absolute() and path.exists():
        return path
    if models_dir is not None:
        candidate = models_dir / path.name
        if candidate.exists():
            return candidate
    # Fall through: let Ultralytics resolve (hub download or cwd).
    return str(weights)
