"""Typed configuration schema for the ALPR system.

Every tunable value in the system is declared here exactly once. Modules read
their settings from an :class:`AppConfig` instance that is validated at startup,
so a bad value fails immediately with a readable message instead of raising
somewhere deep inside a frame loop.

Nothing in this module reads files or the environment -- that is the loader's
job. Keeping validation separate from loading means the schema can be unit
tested without touching the filesystem.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# COCO class indices for the vehicle categories we care about. These come from
# the pretrained detector's own label set, which is why vehicle detection needs
# no training of its own.
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_BUS = 5
COCO_TRUCK = 7


class StrictModel(BaseModel):
    """Base for every config section.

    ``extra="forbid"`` turns a typo in a YAML key into a startup error. Without
    it, ``confidance: 0.9`` is silently ignored and you spend an afternoon
    wondering why the threshold never changed.
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class PathsConfig(StrictModel):
    """Filesystem locations, all resolved to absolute paths by the loader.

    Relative values in YAML are interpreted against the project root, so the
    same config file works on Windows, on Kaggle, and on the gate machine.
    """

    project_root: Path = Path(".")
    data_dir: Path = Path("data")
    models_dir: Path = Path("models")
    outputs_dir: Path = Path("outputs")


class RuntimeConfig(StrictModel):
    """Where and how inference executes."""

    device: Literal["auto", "cpu", "cuda", "mps", "openvino"] = "auto"
    half: bool = Field(
        default=False,
        description="FP16 inference. Speeds up CUDA, unsupported on most CPUs.",
    )
    max_fps: float = Field(
        default=15.0,
        gt=0,
        description="Upper bound on processing rate. A gate does not need 30fps, "
        "and capping it leaves CPU for the rest of the system.",
    )

    @model_validator(mode="after")
    def _half_requires_accelerator(self) -> RuntimeConfig:
        if self.half and self.device in ("cpu", "openvino"):
            raise ValueError(
                f"half=true is not supported on device={self.device!r}; "
                "use device='cuda' or set half=false"
            )
        return self


class CameraConfig(StrictModel):
    """Video source settings.

    ``source`` is a string so it can hold a device index ("0"), a file path, or
    an RTSP URL without changing type. The capture layer decides how to open it.
    """

    source: str = "0"
    width: int | None = Field(default=None, gt=0)
    height: int | None = Field(default=None, gt=0)
    fps: float | None = Field(default=None, gt=0)
    reconnect_delay_s: float = Field(default=3.0, ge=0)


class VehicleDetectorConfig(StrictModel):
    """Stage one: find vehicles so plates can be searched inside them."""

    enabled: bool = Field(
        default=True,
        description="Disable to run single-stage plate detection on the full "
        "frame. Kept configurable so the two approaches can be compared.",
    )
    weights: str = "yolo26n.pt"
    confidence: float = Field(default=0.40, ge=0.0, le=1.0)
    iou: float = Field(default=0.50, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=32, multiple_of=32)
    classes: list[int] = Field(
        default_factory=lambda: [COCO_CAR, COCO_MOTORCYCLE, COCO_BUS, COCO_TRUCK]
    )
    box_padding: float = Field(
        default=0.10,
        ge=0.0,
        le=0.5,
        description="Fractional expansion of the vehicle box before cropping. "
        "Without padding, plates near the box edge get clipped.",
    )


class PlateDetectorConfig(StrictModel):
    """Stage two: find the plate, either inside a vehicle crop or full-frame."""

    weights: str = "plate_yolo26s.pt"
    confidence: float = Field(default=0.35, ge=0.0, le=1.0)
    iou: float = Field(default=0.50, ge=0.0, le=1.0)
    imgsz: int = Field(default=640, ge=32, multiple_of=32)
    max_plates_per_vehicle: int = Field(
        default=2,
        ge=1,
        description="Trucks and trailers can legitimately show two plates.",
    )


class TrackingConfig(StrictModel):
    """Multi-frame association and vote aggregation.

    This is what turns many noisy single-frame reads into one decision. It is
    the highest-value accuracy mechanism in the pipeline.
    """

    enabled: bool = True
    tracker: Literal["bytetrack.yaml", "botsort.yaml"] = "bytetrack.yaml"
    min_reads_for_vote: int = Field(
        default=3,
        ge=1,
        description="Reads required before a track may open the gate.",
    )
    max_reads_per_track: int = Field(
        default=15,
        ge=1,
        description="Stop re-reading a track once this many reads are collected.",
    )
    track_timeout_s: float = Field(
        default=5.0,
        gt=0,
        description="Idle time after which a track is finalised and discarded.",
    )

    @model_validator(mode="after")
    def _min_not_above_max(self) -> TrackingConfig:
        if self.min_reads_for_vote > self.max_reads_per_track:
            raise ValueError(
                f"min_reads_for_vote ({self.min_reads_for_vote}) cannot exceed "
                f"max_reads_per_track ({self.max_reads_per_track})"
            )
        return self


class OCRConfig(StrictModel):
    """Text recognition settings."""

    engine: Literal["paddleocr", "tesseract"] = "paddleocr"
    lang: str = "en"
    use_angle_cls: bool = Field(
        default=True,
        description="Let PaddleOCR correct 180-degree rotation of the crop.",
    )
    min_char_confidence: float = Field(default=0.50, ge=0.0, le=1.0)
    min_string_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    handle_two_line_plates: bool = Field(
        default=True,
        description="Philippine motorcycle plates are two-line. Single-line-only "
        "recognition silently loses every one of them.",
    )
    two_line_aspect_threshold: float = Field(
        default=2.0,
        gt=0,
        description="Plate crops with width/height below this are treated as "
        "two-line and split before recognition.",
    )


class PlateFormatConfig(StrictModel):
    """Region-specific validation and character correction.

    Formats are declared as **slot templates** rather than regexes: ``L`` for a
    letter position, ``D`` for a digit position. A Philippine car plate is
    ``LLLDDDD``.

    Templates rather than regexes because the position information is the whole
    point. Knowing that slot 4 must be a digit means a glyph read as ``O`` there
    can be corrected to ``0`` deterministically. A regex can only tell you the
    string failed; a template tells you where and how to fix it. The regex form
    is derived from the template, so validation and correction can never
    disagree about what a valid plate looks like.
    """

    region: str = "PH"
    formats: dict[str, str] = Field(
        default_factory=lambda: {
            "car": "LLLDDDD",
            "motorcycle_new": "DDDLLL",
            "motorcycle_alt": "DDDDLL",
            "motorcycle_old": "LLDDDDD",
        }
    )
    letter_to_digit: dict[str, str] = Field(
        default_factory=lambda: {"O": "0", "Q": "0", "I": "1", "L": "1", "Z": "2",
                                 "S": "5", "B": "8", "G": "6"}
    )
    digit_to_letter: dict[str, str] = Field(
        default_factory=lambda: {"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B",
                                 "6": "G"}
    )
    max_corrections: int = Field(
        default=2,
        ge=0,
        description="Reject a read needing more substitutions than this. "
        "Unbounded correction will happily turn noise into a valid-looking "
        "plate, which is worse than admitting failure at a gate.",
    )
    correction_penalty: float = Field(
        default=0.85,
        gt=0.0,
        le=1.0,
        description="Confidence multiplier applied per substitution, so a "
        "corrected read ranks below a clean one during multi-frame voting.",
    )

    @field_validator("formats")
    @classmethod
    def _templates_must_be_slot_strings(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("at least one plate format must be defined")
        for name, template in value.items():
            if not template:
                raise ValueError(f"plate format {name!r} is empty")
            invalid = sorted(set(template) - {"L", "D"})
            if invalid:
                raise ValueError(
                    f"plate format {name!r} contains {invalid}; "
                    "only 'L' (letter) and 'D' (digit) are allowed"
                )
        return value


class DatabaseConfig(StrictModel):
    """Persistence settings.

    ``url`` is a SQLAlchemy connection string rather than a file path. That one
    choice is what makes the later move to MySQL or PostgreSQL a config change
    instead of a rewrite.
    """

    url: str = "sqlite:///data/alpr.db"
    echo: bool = Field(default=False, description="Log every SQL statement.")
    wal: bool = Field(
        default=True,
        description="SQLite write-ahead logging. Lets the capture loop write "
        "while the API reads, instead of the two blocking each other.",
    )


class LoggingConfig(StrictModel):
    """Observability settings."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    file: Path | None = None
    format: Literal["plain", "json"] = "plain"


class AppConfig(StrictModel):
    """Root configuration object passed through the application."""

    paths: PathsConfig = Field(default_factory=PathsConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    vehicle_detector: VehicleDetectorConfig = Field(
        default_factory=VehicleDetectorConfig
    )
    plate_detector: PlateDetectorConfig = Field(default_factory=PlateDetectorConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    plate_format: PlateFormatConfig = Field(default_factory=PlateFormatConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="after")
    def _tracking_requires_vehicle_stage(self) -> AppConfig:
        if self.tracking.enabled and not self.vehicle_detector.enabled:
            raise ValueError(
                "tracking.enabled requires vehicle_detector.enabled: vote "
                "aggregation needs a stable per-vehicle identity to attach "
                "reads to. Disable tracking or enable the vehicle stage."
            )
        return self
