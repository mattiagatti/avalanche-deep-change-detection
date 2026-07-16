"""Backwards-compatible shim.

The slope-generation logic now lives in :mod:`utils.slope` so it can be shared
with the inference pipeline. This module re-exports it for any existing callers.
"""

from utils.slope import (  # noqa: F401
    COP30_BUCKET,
    S3,
    _cop30_tile_key,
    _download_cop30_tile,
    ensure_slope,
)

__all__ = ["ensure_slope", "COP30_BUCKET", "S3"]
