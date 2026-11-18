"""Training orchestration: prepare the environment and launch YOLO training.

This module's job is to translate our :class:`TrainingConfig` into the
Ultralytics YOLO ``train()`` call and handle the two environments the system
runs in:

**Local (Windows / Linux developer machine)**
    Dataset is already split and the ``outputs/splits/dataset.yaml`` exists.
    Training writes results to ``outputs/runs/<name>/``.

**Kaggle (notebook environment)**
    ``/kaggle/input`` is read-only. The dataset must be materialised in
    ``copy`` mode into ``/kaggle/working`` before training. The notebook calls
    :func:`kaggle_setup` to do this once at the top.

Both paths end at :func:`train`, which takes a :class:`TrainingConfig` and
returns the path to the best weights file.

The class does **not** manage epochs, loss functions, or model architecture --
those are Ultralytics' job and are exposed through config fields. We only
manage the boundary between our config system and Ultralytics' API.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from ultralytics import YOLO

from alpr.training.config import TrainingConfig
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


def train(config: TrainingConfig) -> Path:
    """Run one YOLO training session and return the path to the best weights.

    The dataset YAML must already exist (run ``python scripts/build_splits.py``
    first). On Kaggle, call :func:`kaggle_setup` before this function.

    Args:
        config: Fully populated :class:`TrainingConfig`. Every hyperparameter
            has a reasonable default; override only what you have a reason to.

    Returns:
        Absolute path to ``best.pt`` inside the run directory.

    Raises:
        FileNotFoundError: If ``config.dataset_yaml`` does not exist.
        RuntimeError: If YOLO training fails (wraps Ultralytics' exception).
    """
    dataset_yaml = Path(config.dataset_yaml).resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(
            f"dataset YAML not found: {dataset_yaml}\n"
            "Run: python scripts/build_splits.py --root <data_dir> --out outputs/splits"
        )

    logger.info(
        "starting training: %s, epochs=%d, imgsz=%d, device=%r",
        config.base_weights,
        config.epochs,
        config.imgsz,
        config.device or "auto",
    )

    model = YOLO(config.base_weights)

    # Build the kwargs dict from our config. Only pass fields that Ultralytics
    # train() actually accepts -- unknown kwargs raise a TypeError.
    train_kwargs = {
        "data": str(dataset_yaml),
        "epochs": config.epochs,
        "patience": config.patience,
        "batch": config.batch,
        "imgsz": config.imgsz,
        "workers": config.workers,
        "optimizer": config.optimizer,
        "lr0": config.lr0,
        "lrf": config.lrf,
        "momentum": config.momentum,
        "weight_decay": config.weight_decay,
        "warmup_epochs": config.warmup_epochs,
        "hsv_h": config.hsv_h,
        "hsv_s": config.hsv_s,
        "hsv_v": config.hsv_v,
        "degrees": config.degrees,
        "translate": config.translate,
        "scale": config.scale,
        "shear": config.shear,
        "perspective": config.perspective,
        "flipud": config.flipud,
        "fliplr": config.fliplr,
        "mosaic": config.mosaic,
        "mixup": config.mixup,
        "project": str(config.project),
        "name": config.name,
        "exist_ok": config.exist_ok,
        "save_period": config.save_period,
        "half": config.half,
        "cache": config.cache,
        "seed": config.seed,
        "deterministic": config.deterministic,
        "verbose": config.verbose,
    }
    if config.device:
        train_kwargs["device"] = config.device

    try:
        results = model.train(**train_kwargs)
    except Exception as exc:
        raise RuntimeError(f"YOLO training failed: {exc}") from exc

    # Ultralytics saves weights to <project>/<name>/weights/best.pt
    run_dir = Path(config.project) / config.name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        # Ultralytics may number the dir (name2, name3) on conflict.
        candidates = sorted(Path(config.project).glob(f"{config.name}*/weights/best.pt"))
        best = candidates[-1] if candidates else best

    logger.info("training complete; best weights: %s", best)
    return best.resolve()


def kaggle_setup(
    input_root: str | Path,
    working_dir: str | Path,
    *,
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.80, 0.10, 0.10),
) -> Path:
    """Prepare a Kaggle environment for training.

    ``/kaggle/input`` is read-only. Ultralytics writes a ``labels.cache``
    next to the labels directory on the first epoch; that write fails silently
    and the cache is rebuilt every epoch, making training ~3× slower.

    This function materialises the dataset into ``/kaggle/working`` using
    ``copy`` mode so Ultralytics can write wherever it needs.

    Args:
        input_root: Path to the dataset under ``/kaggle/input``.
        working_dir: Writable output directory (``/kaggle/working``).
        seed: Random seed for split reproducibility.
        ratios: (train, val, test) fractions.

    Returns:
        Path to the ``dataset.yaml`` ready for :func:`train`.
    """
    from alpr.data.audit import audit_dataset
    from alpr.data.splitting import split
    from alpr.data.materialize import materialize

    input_root = Path(input_root)
    working_dir = Path(working_dir)

    logger.info("auditing dataset at %s", input_root)
    report = audit_dataset(input_root)
    logger.info(
        "audit: %d images, %d labelled, %d duplicates",
        report.total_images,
        report.labelled,
        report.duplicate_groups,
    )

    logger.info("splitting dataset (%.0f/%.0f/%.0f)", *(r * 100 for r in ratios))
    manifest = split(input_root, seed=seed, ratios={"train": ratios[0], "val": ratios[1], "test": ratios[2]})

    out_dir = working_dir / "splits"
    dataset_yaml = materialize(manifest, out_dir, mode="copy")
    logger.info("dataset ready: %s", dataset_yaml)
    return dataset_yaml
