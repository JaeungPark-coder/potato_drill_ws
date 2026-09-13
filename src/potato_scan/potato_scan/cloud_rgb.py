"""Colour in and out of a PointCloud2, in numpy alone.

PointCloud2 carries colour as a single float32 whose bits spell 0x00RRGGBB.
Packing and unpacking it is two lines, but it lives in its own module for one
reason: eye_detector needs it, and eye_detector must not import
pointcloud_accumulator, which pulls in Open3D. Open3D has no wheel for Python
3.13+, and keeping the detector free of it is what lets the detector be
tested on a machine that cannot install it.
"""
import numpy as np


def unpack_rgb(packed):
    """float32-packed 0x00RRGGBB -> (N, 3) floats in 0..1."""
    as_int = np.ascontiguousarray(np.asarray(packed, dtype=np.float32)).view(np.uint32)
    return np.stack([(as_int >> 16) & 0xFF,
                     (as_int >> 8) & 0xFF,
                     as_int & 0xFF], axis=-1).astype(np.float64) / 255.0


def pack_rgb(colors):
    """(N, 3) floats in 0..1 -> float32-packed 0x00RRGGBB."""
    channels = np.rint(np.clip(np.asarray(colors, dtype=float), 0.0, 1.0) * 255.0)
    as_int = ((channels[:, 0].astype(np.uint32) << 16)
              | (channels[:, 1].astype(np.uint32) << 8)
              | channels[:, 2].astype(np.uint32))
    return np.ascontiguousarray(as_int).view(np.float32)
