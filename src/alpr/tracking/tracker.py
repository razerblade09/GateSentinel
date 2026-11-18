"""ByteTrack wrapper and multi-frame vote aggregation.

Tracking is what turns many noisy single-frame reads into one reliable
decision. Without it, the pipeline would need to read a plate correctly on
the very first frame it appears -- which at real-world plate quality and
resolution is unreliable.

With tracking:
1. The vehicle is assigned a stable ``track_id`` the moment it enters the
   frame.
2. The plate is read on each frame up to ``max_reads_per_track`` times.
3. When ``min_reads_for_vote`` reads have been collected, the most common
   read (majority vote, tie-broken by mean confidence) is accepted.

**ByteTrack wrapper** (``VehicleTracker``)
    Wraps Ultralytics' built-in ByteTrack, which is exposed through
    ``model.track()``. This class manages the per-session tracker state so
    the pipeline does not have to.

**Vote aggregator** (``TrackVoter``)
    Accumulates plate reads per track ID and decides when a track has enough
    evidence to emit a result. It is intentionally decoupled from the tracker:
    the same aggregator logic works for mock tracks in tests and real tracks
    in production.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from ultralytics import YOLO

from alpr.config.schema import TrackingConfig
from alpr.detection.types import Detection
from alpr.ocr.types import PlateRead
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ByteTrack wrapper
# ---------------------------------------------------------------------------


class VehicleTracker:
    """Run ByteTrack on vehicle detections to assign persistent track IDs.

    Ultralytics exposes ByteTrack through ``model.track()``, which runs
    inference *and* tracking together. We separate them: detection already ran
    (possibly on a cropped image), so we use ``model.track()`` with
    ``persist=True`` to maintain state across frames.

    Args:
        model: The vehicle YOLO model (same instance used for detection --
            Ultralytics' tracker state is attached to the model, not to this
            class).
        config: Tracking settings from the application config.
    """

    def __init__(self, model: YOLO, config: TrackingConfig) -> None:
        self._model = model
        self._cfg = config

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Run detection + tracking on ``frame``.

        Unlike :class:`~alpr.detection.vehicle.VehicleDetector`, this method
        calls ``model.track()`` rather than ``model.predict()``, so ByteTrack
        state is updated and each result carries a ``track_id``.

        Args:
            frame: BGR image, shape ``(H, W, 3)``.

        Returns:
            Detections with ``track_id`` set. Same coordinate space as input.
        """
        if frame is None or frame.size == 0:
            return []

        try:
            results = self._model.track(
                source=frame,
                tracker=self._cfg.tracker,
                conf=self._model.predictor.args.conf
                if hasattr(self._model, "predictor") and self._model.predictor
                else 0.40,
                persist=True,
                verbose=False,
            )
        except Exception:
            logger.exception("ByteTrack inference failed")
            return []

        return self._parse(results)

    @staticmethod
    def _parse(results: list) -> list[Detection]:  # type: ignore[type-arg]
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes
            has_ids = boxes.id is not None
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu())
                cls = int(boxes.cls[i].cpu())
                tid = int(boxes.id[i].cpu()) if has_ids else None
                detections.append(
                    Detection(
                        x1=float(xyxy[0]),
                        y1=float(xyxy[1]),
                        x2=float(xyxy[2]),
                        y2=float(xyxy[3]),
                        confidence=conf,
                        class_id=cls,
                        track_id=tid,
                    )
                )
        return detections


# ---------------------------------------------------------------------------
# Vote aggregator
# ---------------------------------------------------------------------------


@dataclass
class _TrackState:
    """Internal state for one active track."""

    track_id: int
    reads: list[PlateRead] = field(default_factory=list)
    last_seen: float = field(default_factory=time.monotonic)
    emitted: bool = False

    def record(self, read: PlateRead) -> None:
        self.reads.append(read)
        self.last_seen = time.monotonic()

    def age_s(self) -> float:
        return time.monotonic() - self.last_seen


@dataclass(frozen=True)
class VoteResult:
    """The decided plate string for one vehicle track.

    Attributes:
        track_id: ByteTrack identifier.
        text: Winning plate string (empty if no valid read won the vote).
        confidence: Mean confidence of the reads that voted for ``text``.
        n_reads: Total reads collected for this track.
        n_valid_reads: Reads that were valid (non-empty plate string).
        format_name: Plate format of the winning read, if any.
    """

    track_id: int
    text: str
    confidence: float
    n_reads: int
    n_valid_reads: int
    format_name: str | None

    @property
    def is_valid(self) -> bool:
        return bool(self.text)

    def __str__(self) -> str:
        return (
            f"VoteResult(track={self.track_id}, "
            f"plate={self.text!r}, conf={self.confidence:.2f}, "
            f"reads={self.n_reads})"
        )


class TrackVoter:
    """Accumulate plate reads per track and emit a majority-vote result.

    Call :meth:`record` for every frame a plate is read. Call
    :meth:`flush_expired` periodically (once per frame is enough) to collect
    results from tracks that have timed out or filled their read budget.

    Args:
        config: Tracking settings from the application config.
    """

    def __init__(self, config: TrackingConfig) -> None:
        self._cfg = config
        self._tracks: dict[int, _TrackState] = {}

    def record(self, track_id: int, read: PlateRead) -> VoteResult | None:
        """Add a plate read to a track.

        Args:
            track_id: ByteTrack identifier of the vehicle.
            read: Output of :class:`~alpr.ocr.base.PlateRecognizer`.

        Returns:
            A :class:`VoteResult` if the track just reached its read budget
            and a decision can be made immediately. ``None`` otherwise --
            the result will be emitted by :meth:`flush_expired` when the
            track times out.
        """
        state = self._tracks.setdefault(track_id, _TrackState(track_id=track_id))
        if state.emitted or len(state.reads) >= self._cfg.max_reads_per_track:
            return None

        state.record(read)

        if len(state.reads) >= self._cfg.max_reads_per_track:
            return self._emit(state)

        return None

    def flush_expired(self) -> list[VoteResult]:
        """Collect results from tracks that have been idle too long.

        Call once per frame. Any track that has accumulated at least
        ``min_reads_for_vote`` reads and has not been seen for longer than
        ``track_timeout_s`` is finalised and removed.

        Returns:
            Zero or more :class:`VoteResult` objects.
        """
        now = time.monotonic()
        expired = [
            state
            for state in self._tracks.values()
            if not state.emitted
            and state.age_s() >= self._cfg.track_timeout_s
            and len(state.reads) >= self._cfg.min_reads_for_vote
        ]
        results = [self._emit(state) for state in expired]
        # Prune emitted and timed-out-but-insufficient tracks.
        to_remove = [
            tid
            for tid, state in self._tracks.items()
            if state.emitted or (
                state.age_s() >= self._cfg.track_timeout_s * 2
            )
        ]
        for tid in to_remove:
            del self._tracks[tid]
        return results

    def active_track_ids(self) -> list[int]:
        """IDs of tracks that have not yet been emitted."""
        return [tid for tid, s in self._tracks.items() if not s.emitted]

    @staticmethod
    def _emit(state: _TrackState) -> VoteResult:
        state.emitted = True
        valid = [r for r in state.reads if r.is_valid]
        n_reads = len(state.reads)

        if not valid:
            return VoteResult(
                track_id=state.track_id,
                text="",
                confidence=0.0,
                n_reads=n_reads,
                n_valid_reads=0,
                format_name=None,
            )

        # Majority vote by text, tie-broken by mean confidence.
        vote_counts: dict[str, int] = defaultdict(int)
        vote_conf: dict[str, list[float]] = defaultdict(list)

        for read in valid:
            vote_counts[read.text] += 1
            vote_conf[read.text].append(read.confidence)

        winner = max(
            vote_counts,
            key=lambda t: (vote_counts[t], float(np.mean(vote_conf[t]))),
        )
        mean_conf = float(np.mean(vote_conf[winner]))
        fmt = next(
            (r.format_name for r in valid if r.text == winner), None
        )

        logger.debug(
            "track %d voted: %r (%d/%d valid reads, conf=%.2f)",
            state.track_id, winner, len(valid), n_reads, mean_conf,
        )

        return VoteResult(
            track_id=state.track_id,
            text=winner,
            confidence=mean_conf,
            n_reads=n_reads,
            n_valid_reads=len(valid),
            format_name=fmt,
        )
