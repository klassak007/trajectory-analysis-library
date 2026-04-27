from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .types import (
    AlignOptions,
    BatchConcatOptions,
    CoreConcatOptions,
    CoreDecomposeOptions,
    CoreOverlayOptions,
    MergeOptions,
    ParamPrealignOptions,
    SequenceConcatOptions,
)

_JOINS = {"inner", "outer", "exact"}
_SEQ_JOINS = {"inner", "outer", "exact"}
_OVERLAP = {"error", "sort"}
_COMPAT = {"identical", "equals", "broadcast_equals", "no_conflicts", "override"}
_ATTRS = {"drop", "identical", "no_conflicts", "drop_conflicts", "override"}


def coerce_batch_concat_options(opts: object | None, *, owner: str) -> BatchConcatOptions:
    if opts is None:
        out = BatchConcatOptions()
    elif isinstance(opts, BatchConcatOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be BatchConcatOptions or None.")
    validate_batch_concat_options(out, owner=owner)
    return out


def coerce_sequence_concat_options(opts: object | None, *, owner: str) -> SequenceConcatOptions:
    if opts is None:
        out = SequenceConcatOptions()
    elif isinstance(opts, SequenceConcatOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be SequenceConcatOptions or None.")
    validate_sequence_concat_options(out, owner=owner)
    return out


def coerce_merge_options(opts: object | None, *, owner: str) -> MergeOptions:
    if opts is None:
        out = MergeOptions()
    elif isinstance(opts, MergeOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be MergeOptions or None.")
    validate_merge_options(out, owner=owner)
    return out


def coerce_align_options(opts: object | None, *, owner: str) -> AlignOptions:
    if opts is None:
        out = AlignOptions()
    elif isinstance(opts, AlignOptions):
        out = opts
    else:
        raise TypeError(f"{owner}: opts must be AlignOptions or None.")
    validate_align_options(out, owner=owner)
    return out


def coerce_core_concat_options(opts: object | None, *, owner: str) -> CoreConcatOptions:
    if opts is None:
        raise TypeError(f"{owner}: opts must be CoreConcatOptions.")
    if not isinstance(opts, CoreConcatOptions):
        raise TypeError(f"{owner}: opts must be CoreConcatOptions.")
    validate_core_concat_options(opts, owner=owner)
    return opts


def coerce_core_decompose_options(opts: object | None, *, owner: str) -> CoreDecomposeOptions:
    if opts is None:
        raise TypeError(f"{owner}: opts must be CoreDecomposeOptions.")
    if not isinstance(opts, CoreDecomposeOptions):
        raise TypeError(f"{owner}: opts must be CoreDecomposeOptions.")
    validate_core_decompose_options(opts, owner=owner)
    return opts


def coerce_core_overlay_options(opts: object | None, *, owner: str) -> CoreOverlayOptions:
    if opts is None:
        raise TypeError(f"{owner}: opts must be CoreOverlayOptions.")
    if not isinstance(opts, CoreOverlayOptions):
        raise TypeError(f"{owner}: opts must be CoreOverlayOptions.")
    validate_core_overlay_options(opts, owner=owner)
    return opts


def _validate_batch_dim(name: object, *, owner: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{owner}: batch_dim must be a non-empty string.")


def _validate_batch_labels(labels: tuple[object, ...] | None, *, owner: str) -> None:
    if labels is None:
        return
    seen: set[object] = set()
    duplicates: list[str] = []
    for idx, label in enumerate(labels):
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError(f"{owner}: batch_labels entries must be hashable; index {idx} is invalid.") from exc
        if label in seen and repr(label) not in duplicates:
            duplicates.append(repr(label))
        seen.add(label)
    if duplicates:
        raise ValueError(f"{owner}: batch_labels must be unique; duplicates={duplicates!r}.")


def _validate_fill_scalar(value: object, *, owner: str) -> None:
    if value is None:
        return
    if isinstance(value, (bool, np.bool_, str, bytes, bytearray, Mapping)):
        raise ValueError(f"{owner}: outer_fill_value entries must be numeric scalars or None/np.nan.")
    arr = np.asarray(value)
    if arr.ndim != 0 or arr.dtype.kind not in ("i", "u", "f"):
        raise ValueError(f"{owner}: outer_fill_value entries must be numeric scalars or None/np.nan.")


def _validate_fill_mapping(value: object, *, owner: str) -> None:
    if not isinstance(value, Mapping):
        _validate_fill_scalar(value, owner=owner)
        return
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{owner}: outer_fill_value mapping keys must be non-empty strings.")
        _validate_fill_scalar(item, owner=owner)


def validate_batch_concat_options(opts: BatchConcatOptions, *, owner: str) -> None:
    _validate_batch_dim(opts.batch_dim, owner=owner)
    _validate_batch_labels(opts.batch_labels, owner=owner)
    if opts.sequence_join not in _SEQ_JOINS:
        raise ValueError(f"{owner}: opts.sequence_join must be one of {sorted(_SEQ_JOINS)!r}.")


def validate_sequence_concat_options(opts: SequenceConcatOptions, *, owner: str) -> None:
    if opts.batch_join not in _JOINS:
        raise ValueError(f"{owner}: opts.batch_join must be one of {sorted(_JOINS)!r}.")
    if opts.overlap not in _OVERLAP:
        raise ValueError(f"{owner}: opts.overlap must be one of {sorted(_OVERLAP)!r}.")


def _validate_prealign(prealign: ParamPrealignOptions | None, *, owner: str) -> None:
    if prealign is None:
        return
    if not isinstance(prealign, ParamPrealignOptions):
        raise TypeError(f"{owner}: param_prealign must be ParamPrealignOptions or None.")


def validate_merge_options(opts: MergeOptions, *, owner: str) -> None:
    if opts.batch_join not in _JOINS:
        raise ValueError(f"{owner}: opts.batch_join must be one of {sorted(_JOINS)!r}.")
    if opts.sequence_join not in _SEQ_JOINS:
        raise ValueError(f"{owner}: opts.sequence_join must be one of {sorted(_SEQ_JOINS)!r}.")
    if opts.compat not in _COMPAT:
        raise ValueError(f"{owner}: opts.compat must be one of {sorted(_COMPAT)!r}.")
    if opts.combine_attrs not in _ATTRS:
        raise ValueError(f"{owner}: opts.combine_attrs must be one of {sorted(_ATTRS)!r}.")
    _validate_fill_mapping(opts.outer_fill_value, owner=owner)
    _validate_prealign(opts.param_prealign, owner=owner)


def validate_align_options(opts: AlignOptions, *, owner: str) -> None:
    if opts.batch_join not in _JOINS:
        raise ValueError(f"{owner}: opts.batch_join must be one of {sorted(_JOINS)!r}.")
    if opts.sequence_join not in _SEQ_JOINS:
        raise ValueError(f"{owner}: opts.sequence_join must be one of {sorted(_SEQ_JOINS)!r}.")
    if not isinstance(opts.pad_invalid_outer, bool):
        raise ValueError(f"{owner}: opts.pad_invalid_outer must be a bool.")


def _validate_optional_labels(labels: tuple[object, ...] | None, *, owner: str) -> None:
    if labels is None:
        return
    seen: set[object] = set()
    for idx, label in enumerate(labels):
        try:
            hash(label)
        except TypeError as exc:
            raise ValueError(f"{owner}: core_labels entries must be hashable; index {idx} is invalid.") from exc
        if label in seen:
            raise ValueError(f"{owner}: core_labels must be unique.")
        seen.add(label)


def validate_core_concat_options(opts: CoreConcatOptions, *, owner: str) -> None:
    if not isinstance(opts.core_dim, str) or not opts.core_dim:
        raise ValueError(f"{owner}: opts.core_dim must be a non-empty string.")
    if opts.output_var is not None and (not isinstance(opts.output_var, str) or not opts.output_var):
        raise ValueError(f"{owner}: opts.output_var must be a non-empty string when provided.")
    _validate_optional_labels(opts.core_labels, owner=owner)


def _validate_decompose_core_dims(core_dims: tuple[str, ...], *, owner: str) -> None:
    if not isinstance(core_dims, tuple) or not core_dims:
        raise ValueError(f"{owner}: opts.core_dims must be a non-empty tuple of dim names.")
    if any(not isinstance(dim, str) or not dim for dim in core_dims):
        raise ValueError(f"{owner}: opts.core_dims entries must be non-empty strings.")
    if len(set(core_dims)) != len(core_dims):
        raise ValueError(f"{owner}: opts.core_dims must be unique.")


def validate_core_decompose_options(opts: CoreDecomposeOptions, *, owner: str) -> None:
    _validate_decompose_core_dims(opts.core_dims, owner=owner)
    if opts.key_mode not in {"index", "label"}:
        raise ValueError(f"{owner}: opts.key_mode must be one of ['index', 'label'].")
    if opts.output_var is not None and (not isinstance(opts.output_var, str) or not opts.output_var):
        raise ValueError(f"{owner}: opts.output_var must be a non-empty string when provided.")


def validate_core_overlay_options(opts: CoreOverlayOptions, *, owner: str) -> None:
    if not isinstance(opts.core_dim, str) or not opts.core_dim:
        raise ValueError(f"{owner}: opts.core_dim must be a non-empty string.")
    if opts.on_overlap not in {"error", "replace"}:
        raise ValueError(f"{owner}: opts.on_overlap must be one of ['error', 'replace'].")
    if opts.output_var is not None and (not isinstance(opts.output_var, str) or not opts.output_var):
        raise ValueError(f"{owner}: opts.output_var must be a non-empty string when provided.")


def normalize_batch_labels(
    labels: Sequence[object] | None,
    *,
    size: int,
) -> tuple[object, ...]:
    if labels is None:
        return tuple(range(size))
    out = tuple(labels)
    if len(out) != size:
        raise ValueError(f"concat_batch: batch_labels length {len(out)} must match input size {size}.")
    _validate_batch_labels(out, owner="concat_batch")
    return out
