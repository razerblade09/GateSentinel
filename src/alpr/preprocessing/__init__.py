"""Preprocessing package — crop extraction and coordinate transforms."""

from alpr.preprocessing.geometry import clamp_box, crop, offset_box, pad_box

__all__ = ["clamp_box", "crop", "offset_box", "pad_box"]
