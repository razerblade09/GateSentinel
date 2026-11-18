"""Training configuration and hyperparameter types.

Training has its own config section separate from the runtime config because
the values are only needed during the training milestone, not at the gate.
Keeping them here (rather than in schema.py) prevents the runtime schema from
accumulating dead fields.

The TrainingConfig is not part of AppConfig -- it is loaded separately by
the training script and the Kaggle notebook. That is intentional: an operator
at the gate should not have to reason about epoch counts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class TrainingConfig(StrictModel):
    """Hyperparameters for one YOLO training run.

    Every field has a sensible default so the config file only needs to
    specify what deviates from the baseline. Comment every change you make --
    hyperparameter tuning without rationale is archaeology later.
    """

    # ---- data ----------------------------------------------------------------
    dataset_yaml: Path = Path("outputs/splits/dataset.yaml")
    """Path to the Ultralytics dataset descriptor produced by build_splits.py."""

    # ---- model ---------------------------------------------------------------
    base_weights: str = "yolo26s.pt"
    """Starting checkpoint. Use a pretrained checkpoint for transfer learning.
    Use an architecture name (``"yolo26s.yaml"``) to train from scratch.
    Transfer learning from COCO wins on small datasets -- plates are not in
    COCO but the low-level edge and corner features are universal."""

    # ---- training loop -------------------------------------------------------
    epochs: int = Field(default=100, ge=1)
    patience: int = Field(
        default=20,
        ge=1,
        description="Early stopping: halt when val loss does not improve for "
        "this many epochs. 20 is aggressive; raise to 50 for final runs.",
    )
    batch: int = Field(
        default=16,
        ge=1,
        description="Global batch size. Ultralytics scales per-GPU automatically "
        "when using multi-GPU. Set to -1 to let Ultralytics autobatch.",
    )
    imgsz: int = Field(default=640, ge=32, multiple_of=32)
    workers: int = Field(default=4, ge=0)

    # ---- optimiser -----------------------------------------------------------
    optimizer: Literal["SGD", "Adam", "AdamW", "auto"] = "auto"
    lr0: float = Field(default=0.01, gt=0)
    lrf: float = Field(default=0.01, gt=0, description="Final LR = lr0 * lrf.")
    momentum: float = Field(default=0.937, ge=0, le=1)
    weight_decay: float = Field(default=0.0005, ge=0)
    warmup_epochs: float = Field(default=3.0, ge=0)

    # ---- augmentation --------------------------------------------------------
    hsv_h: float = Field(default=0.015, ge=0, le=1)
    hsv_s: float = Field(default=0.7, ge=0, le=1)
    hsv_v: float = Field(default=0.4, ge=0, le=1)
    degrees: float = Field(
        default=0.0,
        ge=0,
        description="Plate text must remain readable after rotation. Keep small "
        "or zero -- real cameras are roughly level.",
    )
    translate: float = Field(default=0.1, ge=0, le=0.5)
    scale: float = Field(default=0.5, ge=0)
    shear: float = Field(default=0.0, ge=0)
    perspective: float = Field(default=0.0, ge=0)
    flipud: float = Field(
        default=0.0,
        ge=0,
        le=1,
        description="Upside-down plates don't exist at a gate. Keep at 0.",
    )
    fliplr: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Mirror-image plates are fine for localisation.",
    )
    mosaic: float = Field(default=1.0, ge=0, le=1)
    mixup: float = Field(default=0.0, ge=0, le=1)

    # ---- output --------------------------------------------------------------
    project: Path = Path("outputs/runs")
    name: str = "plate_detector"
    exist_ok: bool = Field(
        default=False,
        description="Allow overwriting an existing run directory. Set True to "
        "resume or to re-run without bumping the name.",
    )
    save_period: int = Field(
        default=-1,
        description="Save a checkpoint every N epochs (-1 = only best and last).",
    )

    # ---- runtime -------------------------------------------------------------
    device: str = Field(
        default="",
        description="Torch device string. Leave empty for Ultralytics auto-detect "
        "(GPU when available). Override with '0' for first GPU or 'cpu'.",
    )
    half: bool = Field(
        default=False,
        description="FP16 training. Halves VRAM, requires a CUDA device.",
    )
    cache: Literal["ram", "disk", False] = Field(
        default=False,
        description="Cache images in RAM or on disk for faster epochs. "
        "RAM cache on a 24k-image dataset needs ~16 GB free.",
    )
    seed: int = Field(default=42, description="Torch and NumPy random seed.")
    deterministic: bool = Field(
        default=True,
        description="Reproducible training at a ~10% speed cost on CUDA.",
    )
    verbose: bool = False
