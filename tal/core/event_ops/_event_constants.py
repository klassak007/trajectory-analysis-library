from __future__ import annotations

import numpy as np

EDGE_INVALID = np.int8(0)
EDGE_ENTER = np.int8(1)
EDGE_EXIT = np.int8(2)
EDGE_TRIGGER = np.int8(3)
SAMPLE_SENTINEL = np.int64(-1)

__all__ = [
    "EDGE_ENTER",
    "EDGE_EXIT",
    "EDGE_INVALID",
    "EDGE_TRIGGER",
    "SAMPLE_SENTINEL",
]
