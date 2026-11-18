"""The main pipeline: wire detection → tracking → OCR → voting → gate.

This module is the top of the dependency graph. Every other module knows
nothing about the ones above it; ``Pipeline`` is the only place that knows
about everything. That is what makes each stage independently testable.

**Frame-level flow** (called per frame by :meth:`Pipeline.process_frame`):

1. Run vehicle detector + ByteTrack to get tracked vehicle boxes.
2. For each vehicle box: pad and crop, run plate detector.
3. For each plate box: crop, run OCR.
4. Record each OCR result to the vote aggregator.
5. Record each OCR result to the database (if configured).
6. Flush the vote aggregator for any timed-out tracks → gate decisions.

**Gate decisions** are emitted by :meth:`Pipeline.process_frame` as part of
the returned :class:`~alpr.pipeline.types.FrameResult`. The caller (the run
script) acts on them: signal the gate hardware, log to stderr, etc. The
pipeline itself does not write to GPIO or serial ports -- that is the
orchestrator's job.

**Allow-list** is a plain ``set[str]`` that can be hot-reloaded between
frames by calling :meth:`Pipeline.set_allow_list`. The pipeline never reads
from a file or a database table itself -- the caller manages that lifecycle.
"""

from __future__ import annotations

import datetime

import numpy as np

from alpr.config.schema import AppConfig
from alpr.database.repository import Repository
from alpr.detection.plate import PlateDetector
from alpr.detection.types import Detection
from alpr.detection.vehicle import VehicleDetector
from alpr.ocr.base import PlateRecognizer
from alpr.pipeline.types import FrameResult, GateDecision, PlateInFrame
from alpr.preprocessing.geometry import clamp_box, crop, pad_box
from alpr.tracking.tracker import TrackVoter, VehicleTracker, VoteResult
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


class Pipeline:
    """Real-time two-stage ALPR pipeline.

    Inject every dependency; the pipeline owns none of them. That is what
    allows tests to swap in fake detectors and an in-memory database without
    patching anything.

    Args:
        config: Full application configuration.
        vehicle_detector: Stage-1 detector (or tracker, used for detection
            when tracking is disabled).
        vehicle_tracker: ByteTrack wrapper. Required when
            ``config.tracking.enabled`` is True.
        plate_detector: Stage-2 plate detector.
        recognizer: OCR pipeline.
        voter: Vote aggregator.
        repository: Optional database repository. When ``None``, gate
            decisions are emitted but not persisted.
        allow_list: Set of authorised plate strings. Hot-reloadable via
            :meth:`set_allow_list`.
        keep_frames: When True, raw frames are attached to
            :class:`~alpr.pipeline.types.FrameResult`. Costs RAM; useful
            for debugging and recording.
    """

    def __init__(
        self,
        config: AppConfig,
        vehicle_detector: VehicleDetector,
        vehicle_tracker: VehicleTracker | None,
        plate_detector: PlateDetector,
        recognizer: PlateRecognizer,
        voter: TrackVoter,
        *,
        repository: Repository | None = None,
        allow_list: set[str] | None = None,
        keep_frames: bool = False,
        session_id: str = "",
    ) -> None:
        self._cfg = config
        self._vehicle_detector = vehicle_detector
        self._vehicle_tracker = vehicle_tracker
        self._plate_detector = plate_detector
        self._recognizer = recognizer
        self._voter = voter
        self._repo = repository
        self._allow_list: set[str] = allow_list or set()
        self._keep_frames = keep_frames
        self._session_id = session_id

        # Map ByteTrack track_id → database vehicle primary key.
        self._db_vehicle_ids: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_allow_list(self, plates: set[str]) -> None:
        """Replace the current allow-list (thread-safe for CPython GIL).

        Args:
            plates: Normalised plate strings (uppercase, no separators).
        """
        self._allow_list = plates
        logger.info("allow-list updated: %d plates", len(plates))

    def process_frame(self, frame: np.ndarray, frame_index: int) -> FrameResult:
        """Run the full pipeline on one camera frame.

        Args:
            frame: BGR image from the capture loop.
            frame_index: Monotonically increasing frame counter.

        Returns:
            :class:`~alpr.pipeline.types.FrameResult` with all detections,
            OCR reads, and any gate decisions emitted this frame.
        """
        result = FrameResult(
            frame_index=frame_index,
            frame=frame if self._keep_frames else None,
        )

        h, w = frame.shape[:2]

        # ---- Stage 1: vehicle detection / tracking -------------------------
        if self._cfg.tracking.enabled and self._vehicle_tracker is not None:
            vehicles = self._vehicle_tracker.track(frame)
        else:
            vehicles = self._vehicle_detector.detect(frame)

        result.vehicles = vehicles

        # ---- Stage 2: plate detection + OCR --------------------------------
        for vehicle in vehicles:
            plates_in_vehicle = self._process_vehicle(frame, vehicle, w, h)
            result.plates.extend(plates_in_vehicle)

        # ---- Flush timed-out tracks ----------------------------------------
        for vote in self._voter.flush_expired():
            decision = self._make_decision(vote)
            result.gate_decisions.append(decision)
            logger.info(
                "frame %d: %s", frame_index, decision
            )

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _process_vehicle(
        self,
        frame: np.ndarray,
        vehicle: Detection,
        frame_w: int,
        frame_h: int,
    ) -> list[PlateInFrame]:
        """Crop vehicle, run plate detector, run OCR on each plate."""
        # Pad and crop the vehicle region.
        px1, py1, px2, py2 = pad_box(
            vehicle.x1, vehicle.y1, vehicle.x2, vehicle.y2,
            self._cfg.vehicle_detector.box_padding,
            frame_w, frame_h,
        )
        vehicle_crop = crop(frame, px1, py1, px2, py2)
        if vehicle_crop.size == 0:
            return []

        plate_dets = self._plate_detector.detect(vehicle_crop)
        plates: list[PlateInFrame] = []

        for plate_det in plate_dets:
            # Translate plate coords from crop-space → frame-space.
            plate_in_frame = plate_det.with_offset(px1, py1)

            # Crop the plate region from the original frame.
            cx1, cy1, cx2, cy2 = clamp_box(
                plate_in_frame.x1, plate_in_frame.y1,
                plate_in_frame.x2, plate_in_frame.y2,
                frame_w, frame_h,
            )
            plate_crop = crop(frame, cx1, cy1, cx2, cy2)
            if plate_crop.size == 0:
                continue

            ocr_read = self._recognizer.read(plate_crop)

            # Feed vote aggregator if the vehicle has a track ID.
            if vehicle.track_id is not None:
                self._record_read(vehicle.track_id, ocr_read)

            plates.append(PlateInFrame(
                plate_box=plate_in_frame,
                ocr_read=ocr_read,
                vehicle_detection=vehicle,
            ))

        return plates

    def _record_read(self, track_id: int, ocr_read) -> None:  # type: ignore[no-untyped-def]
        """Feed a read to the voter and optionally the database."""
        # Ensure a database vehicle row exists for this track.
        if self._repo is not None and track_id not in self._db_vehicle_ids:
            vehicle_row = self._repo.open_vehicle(track_id)
            self._db_vehicle_ids[track_id] = vehicle_row.id

        # Persist the individual read.
        if self._repo is not None and track_id in self._db_vehicle_ids:
            self._repo.add_read(self._db_vehicle_ids[track_id], ocr_read)

        # Feed the voter; handle immediate emission (budget hit).
        vote = self._voter.record(track_id, ocr_read)
        if vote is not None:
            decision = self._make_decision(vote)
            # Return value is not used here; decisions surface via
            # flush_expired() on the next process_frame() call.
            # We log immediately so the event is not silently lost.
            logger.info("immediate gate decision: %s", decision)

    def _make_decision(self, vote: VoteResult) -> GateDecision:
        """Translate a vote result into an authorisation decision."""
        authorised = bool(vote.text) and vote.text in self._allow_list
        gate_opened = authorised  # extend here for time-windows, anti-passback, etc.

        db_id: int | None = None
        if self._repo is not None:
            vehicle_db_id = self._db_vehicle_ids.get(vote.track_id)
            if vehicle_db_id is None:
                # Vote arrived for a track the repo hasn't seen (e.g. single-
                # stage mode without tracking). Create a stub vehicle row.
                v = self._repo.open_vehicle(vote.track_id)
                vehicle_db_id = v.id
                self._db_vehicle_ids[vote.track_id] = vehicle_db_id

            event = self._repo.record_access(
                vehicle_db_id,
                vote,
                authorised=authorised,
                gate_opened=gate_opened,
            )
            db_id = event.id

        return GateDecision(
            vote=vote,
            authorised=authorised,
            gate_opened=gate_opened,
            db_event_id=db_id,
        )
