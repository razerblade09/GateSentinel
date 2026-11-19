"""Preprocessing: crop extraction, padding, and coordinate transforms.

This module handles everything that happens to an image between reading a frame
and passing a region to a detector or OCR engine. Three responsibilities:

**Coordinate transforms**
    Detections are returned in the coordinate space of whatever image was
    passed to the model (which may be a cropped sub-image). Before cropping
    the next stage's input, boxes must be mapped back to the original frame's
    coordinate system. Getting this wrong silently shifts every downstream
    crop.

**Safe cropping**
    Clamp to image bounds before slicing. An out-of-bounds crop raises an
    unhandled exception inside the frame loop, which is worse than a small
    boundary error.

**Box padding**
    Expanding a vehicle box before passing it to the plate detector gives the
    detector more context and reduces edge clipping. The amount is fractional
    so it scales correctly across different resolutions.
"""

from __future__ import annotations

import numpy as np


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Clamp a box to image bounds and convert to integer pixel coordinates.

    Args:
        x1, y1, x2, y2: Box corners, in any numeric type.
        img_w, img_h: Image dimensions in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` as integers, guaranteed within
        ``[0, img_w] × [0, img_h]``.
    """
    cx1 = int(max(0.0, min(float(x1), float(img_w))))
    cy1 = int(max(0.0, min(float(y1), float(img_h))))
    cx2 = int(max(0.0, min(float(x2), float(img_w))))
    cy2 = int(max(0.0, min(float(y2), float(img_h))))
    return cx1, cy1, cx2, cy2


def pad_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    padding: float,
    img_w: int,
    img_h: int,
) -> tuple[int, int, int, int]:
    """Expand a box by ``padding`` fraction of its own dimensions, then clamp.

    A padding of 0.10 on a 100×50 box adds 10 px on each horizontal side and
    5 px on each vertical side. The expansion is symmetric so the box centre
    does not move.

    Args:
        x1, y1, x2, y2: Original box corners.
        padding: Fractional expansion (0.10 = 10 %).
        img_w, img_h: Image dimensions for clamping.

    Returns:
        Padded, clamped, integer box.
    """
    w = x2 - x1
    h = y2 - y1
    dx = w * padding
    dy = h * padding
    return clamp_box(x1 - dx, y1 - dy, x2 + dx, y2 + dy, img_w, img_h)


def crop(image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    """Extract a rectangular region from ``image``.

    Coordinates must already be clamped (use :func:`clamp_box` or
    :func:`pad_box`). Returns an empty array if the box has zero area, so
    callers should check ``crop.size > 0`` rather than its shape.

    Args:
        image: BGR image, shape ``(H, W, C)``.
        x1, y1, x2, y2: Integer, clamped corners.

    Returns:
        A NumPy view of the sub-image, same dtype as ``image``.
    """
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, image.shape[2] if image.ndim == 3 else 1), dtype=image.dtype)
    return image[y1:y2, x1:x2]


def offset_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    offset_x: float,
    offset_y: float,
) -> tuple[float, float, float, float]:
    """Translate a box by ``(offset_x, offset_y)``.

    Use this after running a detector on a cropped sub-image to convert the
    model's output back to the original frame's coordinate space.

    If the plate detector found a box at ``(10, 5, 60, 30)`` inside a vehicle
    crop whose top-left in the frame is at ``(200, 150)``, the plate's frame
    coordinates are ``(210, 155, 260, 180)``.
    """
    return x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y
