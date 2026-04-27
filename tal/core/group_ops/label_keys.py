from __future__ import annotations

import numpy as np

_NAN_LABEL_KEY = object()


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
    if isinstance(label, (float, np.floating)) and bool(np.isnan(label)):
        return _NAN_LABEL_KEY
    return label


__all__ = ["canonical_group_label_key"]
