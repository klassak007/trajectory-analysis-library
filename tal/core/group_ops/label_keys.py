from __future__ import annotations

import numpy as np
import pandas as pd

_MISSING_LABEL_KEY = object()


def canonical_group_label_key(label: object) -> object:
    """Return a hashable grouping key with NaN-like labels canonicalized.

    Parameters
    ----------
    label : object
        Label/name selection used by this operation.

    Returns
    -------
    object
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if type(label) is tuple:
        return tuple(canonical_group_label_key(value) for value in label)
    try:
        missing = pd.isna(label)
    except (TypeError, ValueError):
        return label
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return _MISSING_LABEL_KEY
    return label


__all__ = ["canonical_group_label_key"]
