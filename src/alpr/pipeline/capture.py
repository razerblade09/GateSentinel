"""Camera capture: open a video source and yield BGR frames.

The capture layer isolates everything that varies between environments:

- A USB webcam on the developer machine (``source="0"``).
- A test video clip (``source="path/to/clip.mp4"``).
- An IP camera over RTSP (``source="rtsp://192.168.1.10/stream"``).

The pipeline never calls ``cv2.VideoCapture`` directly. It calls
:func:`open_capture` and iterates the returned generator. That boundary means:

1. Tests can inject any iterable of BGR frames without touching a camera.
2. Reconnect logic lives in one place, not scattered across callers.
3. The pipeline is testable without hardware.

Frame rate limiting (``max_fps``) is enforced here, not in the pipeline. The
gate does not need 30 fps. At 15 fps a plate travelling at 10 km/h through a
3-metre gate zone is visible for ~16 frames, which is plenty for the voter.
Capping the rate leaves CPU for the OCR thread.
"""

from __future__ import annotations

import time
from collections.abc import Generator

import cv2
import numpy as np

from alpr.config.schema import CameraConfig, RuntimeConfig
from alpr.utils.logging import get_logger

logger = get_logger(__name__)


def open_capture(
    camera: CameraConfig,
    runtime: RuntimeConfig,
    *,
    max_frames: int | None = None,
) -> Generator[tuple[int, np.ndarray], None, None]:
    """Open ``camera.source`` and yield ``(frame_index, bgr_frame)`` pairs.

    Args:
        camera: Camera settings (source, resolution, reconnect delay).
        runtime: Runtime settings (max_fps).
        max_frames: Stop after this many frames. ``None`` = run forever.
            Use in tests and benchmarks to bound execution time.

    Yields:
        ``(frame_index, frame)`` where ``frame_index`` starts at 0 and
        increments on every successfully captured frame.

    Notes:
        On a failed read the generator sleeps ``camera.reconnect_delay_s``
        seconds and tries to reopen the source. For a file source this means
        the loop terminates cleanly when the file ends (failed read on EOF)
        rather than raising.
    """
    min_interval = 1.0 / runtime.max_fps if runtime.max_fps > 0 else 0.0
    frame_index = 0
    last_frame_time = 0.0

    while True:
        cap = _open(camera)
        if cap is None:
            logger.warning(
                "could not open source %r; retrying in %.1f s",
                camera.source,
                camera.reconnect_delay_s,
            )
            time.sleep(camera.reconnect_delay_s)
            continue

        logger.info("capture opened: %s", camera.source)

        while True:
            if max_frames is not None and frame_index >= max_frames:
                cap.release()
                return

            # Rate limiting: sleep until the next frame slot.
            now = time.monotonic()
            elapsed = now - last_frame_time
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)

            ok, frame = cap.read()
            if not ok:
                logger.info("capture ended or failed: %s", camera.source)
                cap.release()
                if _is_file_source(camera.source):
                    return  # end of file — stop cleanly
                # Network / device source: try to reconnect.
                time.sleep(camera.reconnect_delay_s)
                break  # outer while: reopen

            last_frame_time = time.monotonic()
            yield frame_index, frame
            frame_index += 1


def _open(camera: CameraConfig) -> cv2.VideoCapture | None:
    """Open a VideoCapture for ``camera.source``, configure resolution."""
    src: int | str
    try:
        src = int(camera.source)
    except ValueError:
        src = camera.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None

    if camera.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.width)
    if camera.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.height)
    if camera.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, camera.fps)

    return cap


def _is_file_source(source: str) -> bool:
    """True when ``source`` looks like a local file path rather than a device."""
    if source.isdigit():
        return False
    if source.lower().startswith(("rtsp://", "http://", "https://")):
        return False
    return True
