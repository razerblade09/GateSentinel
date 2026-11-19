"""Frame-level result types for the pipeline.

The pipeline returns structured results at two granularities:

``FrameResult``
    Everything found in one camera frame: vehicle boxes with their tracked
    IDs, plate boxes relative to the frame, and any OCR reads that completed
    in this frame. Useful for drawing overlays, logging, and streaming a
    preview feed. Produced on every frame.

``GateDecision``
    The final, committed decision for one vehicle: vote result, authorisation
    status, and whether the barrier was signalled to open. Produced at most
    once per vehicle, when the track closes.

Keeping these two separate prevents the common mistake of acting on a
single-frame read (which is unreliable) instead of the voted result (which is
the whole point of tracking).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from alpr.detection.types import Detection
from alpr.ocr.types import PlateRead
from alpr.tracking.tracker import VoteResult


@dataclass
class PlateInFrame:
    """One plate found in one frame, attached to a vehicle detection.

    Attributes:
        plate_box: The plate detection, coordinates in the original frame.
        ocr_read: The recognition result for this crop. ``is_valid`` may be
            False -- the frame is logged regardless so the voter has evidence.
        vehicle_detection: The vehicle box this plate was found inside.
    """

    plate_box: Detection
    ocr_read: PlateRead
    vehicle_detection: Detection


@dataclass
class FrameResult:
    """Everything the pipeline found in one camera frame.

    Attributes:
        frame_index: Monotonically increasing counter from the capture loop.
        vehicles: Tracked vehicle detections (with track_id set).
        plates: One entry per plate found, matched to its vehicle.
        gate_decisions: Any gate decisions that closed in this frame. Usually
            empty; non-empty when a track timed out or hit its read budget.
        frame: The original BGR image, included only when
            ``Pipeline.keep_frames`` is True (default False, to save RAM).
    """

    frame_index: int
    vehicles: list[Detection] = field(default_factory=list)
    plates: list[PlateInFrame] = field(default_factory=list)
    gate_decisions: list[GateDecision] = field(default_factory=list)
    frame: np.ndarray | None = None


@dataclass
class GateDecision:
    """The committed result for one vehicle pass.

    Attributes:
        vote: The majority-vote outcome from :class:`~alpr.tracking.tracker.TrackVoter`.
        authorised: Whether ``vote.text`` was in the allow-list at decision time.
        gate_opened: Whether the gate signal was sent.
        db_event_id: Primary key of the persisted
            :class:`~alpr.database.models.AccessEvent`, if a repository was
            configured.
    """

    vote: VoteResult
    authorised: bool
    gate_opened: bool
    db_event_id: int | None = None

    @property
    def plate(self) -> str:
        return self.vote.text

    def __str__(self) -> str:
        status = "OPEN" if self.gate_opened else "DENY"
        return (
            f"GateDecision({status} plate={self.plate!r} "
            f"auth={self.authorised} conf={self.vote.confidence:.2f})"
        )
