from __future__ import annotations

from collections.abc import Sequence

from .align import align_many as _align_many
from .assemble_core import assemble_core as _assemble_core
from .assemble_core import block_core as _block_core
from .assemble_core import stack_core as _stack_core
from .concat_core import concat_core as _concat_core
from .decompose_core import decompose_core as _decompose_core
from .overlay_core import overlay_core as _overlay_core
from .concat_batch import concat_batch_contexts
from .concat_sequence import concat_sequence_contexts
from .merge import merge_contexts
from .normalize import normalize_inputs, resolve_contexts
from .options import (
    coerce_align_options,
    coerce_batch_concat_options,
    coerce_core_concat_options,
    coerce_core_overlay_options,
    coerce_merge_options,
    coerce_sequence_concat_options,
)
from .types import (
    AlignOptions,
    BatchConcatOptions,
    CombineResolveOptions,
    CoreConcatOptions,
    CoreDecomposeOptions,
    CoreOverlayOptions,
    MergeOptions,
    SequenceConcatOptions,
)


def _with_self(first: "AnalysisObject", others: Sequence[object] | object | None) -> list[object]:
    if others is None:
        return [first]
    if isinstance(others, Sequence) and not isinstance(others, (str, bytes)):
        return [first, *list(others)]
    return [first, others]


def _singleton_nested_layout(value: object, *, depth: int) -> object:
    out: object = value
    for _ in range(depth):
        out = [out]
    return out


def _with_self_core_layout(
    first: "AnalysisObject",
    values: object,
    *,
    depth: int,
) -> object:
    if depth <= 1:
        return _with_self(first, values)
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes, bytearray)):
        return [_singleton_nested_layout(first, depth=depth - 1), *list(values)]
    return [_singleton_nested_layout(first, depth=depth - 1), values]


class CombineAccessor:
    """Combine accessor rooted at ``ao.combine``.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    def __init__(self, ao: "AnalysisObject") -> None:
        self._ao = ao

    def concat_batch(
        self,
        others: Sequence[object] | object,
        *,
        opts: BatchConcatOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Concatenate inputs along a batch axis.

        Parameters
        ----------
        others : Sequence[object] | object
            Additional AO-like operands combined with the receiver.
        opts : BatchConcatOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``BatchConcatOptions`` key fields: ``batch_dim`` (default 'batch'), ``batch_labels`` (default None), ``sequence_join`` (default 'outer'), ``fill_value`` (default nan).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, BatchConcatOptions
        >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> out = left.combine.concat_batch([right], opts=BatchConcatOptions(batch_dim="run", batch_labels=("a", "b")))
        >>> tuple(out.as_dataset().coords["run"].values.tolist())
        ('a', 'b')
        """
        values = _with_self(self._ao, others)
        return concat_batch(values, opts=opts, validate=validate)

    def concat_sequence(
        self,
        others: Sequence[object] | object,
        *,
        opts: SequenceConcatOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Concatenate inputs along sequence semantics.

        Parameters
        ----------
        others : Sequence[object] | object
            Additional AO-like operands combined with the receiver.
        opts : SequenceConcatOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``SequenceConcatOptions`` key fields: ``batch_join`` (default 'outer'), ``overlap`` (default 'error'), ``fill_value`` (default nan).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, SequenceConcatOptions
        >>> first = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> second = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [1]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> out = first.combine.concat_sequence([second], opts=SequenceConcatOptions(overlap="error"))
        >>> out.as_dataset().sizes["sample"]
        2
        """
        values = _with_self(self._ao, others)
        return concat_sequence(values, opts=opts, validate=validate)

    def merge(
        self,
        others: Sequence[object] | object,
        *,
        opts: MergeOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Merge AO inputs with schema-aware reconciliation.

        Parameters
        ----------
        others : Sequence[object] | object
            Additional AO-like operands combined with the receiver.
        opts : MergeOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``MergeOptions`` key fields: ``batch_join`` (default 'inner'), ``sequence_join`` (default 'exact'), ``param_prealign`` (default None), ``compat`` (default 'no_conflicts').
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, MergeOptions
        >>> left = AnalysisObject.from_data(xr.Dataset({"x": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> right = AnalysisObject.from_data(xr.Dataset({"y": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> out = left.combine.merge([right], opts=MergeOptions())
        >>> sorted(out.as_dataset().data_vars)
        ['x', 'y']
        """
        values = _with_self(self._ao, others)
        return merge(values, opts=opts, validate=validate)

    def align(
        self,
        others: Sequence[object] | object,
        *,
        opts: AlignOptions | None = None,
        validate: bool = True,
    ) -> list["AnalysisObject"]:
        """Align many AO inputs under alignment policy options.

        Parameters
        ----------
        others : Sequence[object] | object
            Additional AO-like operands combined with the receiver.
        opts : AlignOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``AlignOptions`` key fields: ``batch_join`` (default 'inner'), ``sequence_join`` (default 'exact'), ``fill_value`` (default nan), ``pad_invalid_outer`` (default True).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        list['AnalysisObject']
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AlignOptions, AnalysisObject
        >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [3.0, 4.0])}, coords={"sample": [1, 2]}), sequence_dim="sample", core_dims=(), validate=True)
        >>> aligned = left.combine.align([right], opts=AlignOptions(sequence_join="outer"))
        >>> [item.as_dataset().sizes["sample"] for item in aligned]
        [3, 3]
        """
        values = _with_self(self._ao, others)
        return align_many(values, opts=opts, validate=validate)

    def assemble_core(
        self,
        values: object,
        *,
        core_dims: Sequence[str],
        core_labels: Sequence[Sequence[object]] | None = None,
        output_var: str | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Assemble nested AO layout into new core dimensions.

        Parameters
        ----------
        values : object
            Nested AO-like layout to assemble together with the receiver. The
            receiver is prepended as the first leaf.
        core_dims : Sequence[str]
            New core dimensions represented by the nesting levels.
        core_labels : Sequence[Sequence[object]] | None, optional
            Optional coordinate labels for each new core dimension.
        output_var : str | None, optional
            Output variable name. Defaults to the first leaf variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            AO whose core dimensions are ``core_dims`` followed by any leaf
            core dimensions.

        Notes
        -----
        All leaves are aligned exactly by batch and sequence dimensions before
        the new core layout is assembled.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> x = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> y = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = x.combine.assemble_core([y], core_dims=("axis",), core_labels=(("x", "y"),))
        >>> out.as_dataset().sizes["axis"]
        2
        """
        merged = _with_self_core_layout(self._ao, values, depth=len(core_dims))
        return assemble_core(
            merged,
            core_dims=core_dims,
            core_labels=core_labels,
            output_var=output_var,
            validate=validate,
        )

    def stack_core(
        self,
        values: object,
        *,
        core_dim: str,
        core_labels: Sequence[object] | None = None,
        output_var: str | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Stack inputs along one existing/declared core dimension.

        Parameters
        ----------
        values : object
            AO-like values to stack with the receiver.
        core_dim : str
            New core dimension to create.
        core_labels : Sequence[object] | None, optional
            Optional coordinate labels for ``core_dim``.
        output_var : str | None, optional
            Output variable name. Defaults to the first leaf variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            AO with a new one-dimensional core axis.

        Notes
        -----
        This is a convenience wrapper around :meth:`assemble_core` for a single
        new core dimension.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> x = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> y = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = x.combine.stack_core([y], core_dim="axis", core_labels=("x", "y"))
        >>> out.as_dataset().sizes["axis"]
        2
        """
        merged = _with_self_core_layout(self._ao, values, depth=1)
        return stack_core(
            merged,
            core_dim=core_dim,
            core_labels=core_labels,
            output_var=output_var,
            validate=validate,
        )

    def block_core(
        self,
        values: object,
        *,
        row_dim: str,
        col_dim: str,
        row_labels: Sequence[object] | None = None,
        col_labels: Sequence[object] | None = None,
        output_var: str | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Assemble 2D nested layout into row/col core dimensions.

        Parameters
        ----------
        values : object
            Nested AO-like layout to assemble with the receiver.
        row_dim : str
            New row core dimension.
        col_dim : str
            New column core dimension.
        row_labels : Sequence[object] | None, optional
            Optional coordinate labels for ``row_dim``.
        col_labels : Sequence[object] | None, optional
            Optional coordinate labels for ``col_dim``.
        output_var : str | None, optional
            Output variable name. Defaults to the first leaf variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            AO with row and column core dimensions.

        Notes
        -----
        This is a two-dimensional convenience wrapper around
        :meth:`assemble_core`.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> x = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> y = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = x.combine.block_core([[y]], row_dim="row", col_dim="col")
        >>> (out.as_dataset().sizes["row"], out.as_dataset().sizes["col"])
        (2, 1)
        """
        merged = _with_self_core_layout(self._ao, values, depth=2)
        return block_core(
            merged,
            row_dim=row_dim,
            col_dim=col_dim,
            row_labels=row_labels,
            col_labels=col_labels,
            output_var=output_var,
            validate=validate,
        )

    def concat_core(
        self,
        others: Sequence[object] | object,
        *,
        opts: CoreConcatOptions | None = None,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Concatenate along an existing core dimension.

        Parameters
        ----------
        others : Sequence[object] | object
            Additional AO-like operands combined with the receiver.
        opts : CoreConcatOptions | None, optional
            When ``None``, operation-specific defaults are resolved by internal option coercion. ``CoreConcatOptions`` key fields: ``core_dim`` (default <required>), ``core_labels`` (default None), ``output_var`` (default None).
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, CoreConcatOptions
        >>> x = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0]])}, coords={"sample": [0], "axis": ["x"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> y = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[2.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> out = x.combine.concat_core([y], opts=CoreConcatOptions(core_dim="axis"))
        >>> tuple(out.as_dataset().coords["axis"].values.tolist())
        ('x', 'y')
        """
        values = _with_self(self._ao, others)
        return concat_core(values, opts=opts, validate=validate)

    def decompose_core(
        self,
        *,
        opts: CoreDecomposeOptions,
        validate: bool = True,
    ) -> dict[tuple[object, ...], "AnalysisObject"]:
        """Decompose selected core dimensions into keyed AO outputs.

        Parameters
        ----------
        opts : CoreDecomposeOptions
            Required decomposition options. Set ``core_dims`` to the dims to split; ``key_mode`` controls whether output keys use indices or labels; ``output_var`` overrides the emitted variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        dict[tuple[object, ...], 'AnalysisObject']
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, CoreDecomposeOptions
        >>> ao = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> parts = ao.combine.decompose_core(opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label"))
        >>> sorted(parts)
        [('x',), ('y',)]
        """
        return decompose_core(self._ao, opts=opts, validate=validate)

    def overlay_core(
        self,
        patches: Sequence[object] | object,
        *,
        opts: CoreOverlayOptions,
        validate: bool = True,
    ) -> "AnalysisObject":
        """Overlay patch labels along a core dimension.

        Parameters
        ----------
        patches : Sequence[object] | object
            Patch payloads applied by overlay operations.
        opts : CoreOverlayOptions
            Required overlay options. Set ``core_dim`` to the overlay axis; ``on_overlap`` chooses error vs replace behavior for duplicate labels; ``output_var`` overrides the emitted variable name.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        AnalysisObject
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject, CoreOverlayOptions
        >>> base = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> patch = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[9.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
        >>> out = base.combine.overlay_core([patch], opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace"))
        >>> out.as_dataset()["v"].sel(axis="y").item()
        9.0
        """
        return overlay_core(self._ao, patches, opts=opts, validate=validate)


def concat_batch(
    aos: Sequence[object],
    *,
    opts: BatchConcatOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Concatenate AO-like inputs along a batch axis.

    Parameters
    ----------
    aos : Sequence[object]
        AO-like inputs consumed by this orchestration boundary.
    opts : BatchConcatOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, BatchConcatOptions, concat_batch
    >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> out = concat_batch([left, right], opts=BatchConcatOptions(batch_dim="run", batch_labels=("a", "b")))
    >>> tuple(out.as_dataset().coords["run"].values.tolist())
    ('a', 'b')
    """
    options = coerce_batch_concat_options(opts, owner="concat_batch")
    objects = normalize_inputs(aos, owner="concat_batch")
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=False, owner="concat_batch"))
    return concat_batch_contexts(contexts, opts=options, validate=validate)


def concat_sequence(
    aos: Sequence[object],
    *,
    opts: SequenceConcatOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Concatenate AO-like inputs along sequence semantics.

    Parameters
    ----------
    aos
        AO-like operands to concatenate.
    opts
        Sequence concatenation options.
    validate
        Whether to validate schema on the output AO.

    Returns
    -------
    AnalysisObject
        Concatenated AO output.

    Notes
    -----
    Uses xarray label-aware alignment and TAL topology owners. Fails closed on
    sequence/topology conflicts.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, SequenceConcatOptions, concat_sequence
    >>> left = AnalysisObject.from_data(
    ...     xr.Dataset(
    ...         {"value": ("sample", [1.0, 2.0])},
    ...         coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])},
    ...     ),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> right = AnalysisObject.from_data(
    ...     xr.Dataset(
    ...         {"value": ("sample", [3.0, 4.0])},
    ...         coords={"sample": [2, 3], "time": ("sample", [2.0, 3.0])},
    ...     ),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     param_coord="time",
    ...     validate=True,
    ... )
    >>> out = concat_sequence([left, right], opts=SequenceConcatOptions(overlap="error"), validate=True)
    >>> out.as_dataset().sizes["sample"]
    4

    See Also
    --------
    tal.core.combine_ops.concat_sequence.concat_sequence_contexts
    """
    options = coerce_sequence_concat_options(opts, owner="concat_sequence")
    objects = normalize_inputs(aos, owner="concat_sequence")
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=True, owner="concat_sequence"))
    return concat_sequence_contexts(contexts, opts=options, validate=validate)


def merge(
    aos: Sequence[object],
    *,
    opts: MergeOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Merge AO-like inputs using schema-aware merge policy.

    Parameters
    ----------
    aos : Sequence[object]
        AO-like inputs consumed by this orchestration boundary.
    opts : MergeOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, MergeOptions, merge
    >>> left = AnalysisObject.from_data(xr.Dataset({"x": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"y": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> sorted(merge([left, right], opts=MergeOptions()).as_dataset().data_vars)
    ['x', 'y']
    """
    options = coerce_merge_options(opts, owner="merge")
    objects = normalize_inputs(aos, owner="merge")
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=False, owner="merge"))
    return merge_contexts(contexts, opts=options, validate=validate)


def align_many(
    aos: Sequence[object],
    *,
    opts: AlignOptions | None = None,
    validate: bool = True,
) -> list["AnalysisObject"]:
    """Align many AO-like operands under a shared alignment policy.

    Parameters
    ----------
    aos : Sequence[object]
        AO-like inputs consumed by this orchestration boundary.
    opts : AlignOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    list['AnalysisObject']
        Ordered collection produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AlignOptions, AnalysisObject, align_many
    >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [3.0, 4.0])}, coords={"sample": [1, 2]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> [item.as_dataset().sizes["sample"] for item in align_many([left, right], opts=AlignOptions(sequence_join="outer"))]
    [3, 3]
    """
    options = coerce_align_options(opts, owner="align_many")
    objects = normalize_inputs(aos, owner="align_many")
    contexts = resolve_contexts(objects, resolve=CombineResolveOptions(require_sequence=False, owner="align_many"))
    return _align_many(contexts, opts=options, validate=validate)


def align_pair(
    left: object,
    right: object,
    *,
    opts: AlignOptions | None = None,
    validate: bool = True,
) -> tuple["AnalysisObject", "AnalysisObject"]:
    """Align two AO-like operands and return ordered pair outputs.

    Parameters
    ----------
    left : object
        Operand value participating in the operation.
    right : object
        Operand value participating in the operation.
    opts : AlignOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    tuple['AnalysisObject', 'AnalysisObject']
        Tuple of output values produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AlignOptions, AnalysisObject, align_pair
    >>> left = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> right = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> a, b = align_pair(left, right, opts=AlignOptions())
    >>> (a.as_dataset().sizes["sample"], b.as_dataset().sizes["sample"])
    (1, 1)
    """
    out = align_many([left, right], opts=opts, validate=validate)
    return out[0], out[1]


def assemble_core(
    values: object,
    *,
    core_dims: Sequence[str],
    core_labels: Sequence[Sequence[object]] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Assemble nested AO-like layout into new core dimensions.

    Parameters
    ----------
    values : object
        Input values consumed by this operation.
    core_dims : Sequence[str], optional
        Core-dimension labels/shape metadata used for structural operations.
    core_labels : Sequence[Sequence[object]] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    output_var : str | None, optional
        Output naming metadata used during finalization.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, assemble_core
    >>> x = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> y = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> out = assemble_core([x, y], core_dims=("axis",), core_labels=(("x", "y"),))
    >>> out.as_dataset().sizes["axis"]
    2
    """
    return _assemble_core(
        values,
        core_dims=core_dims,
        core_labels=core_labels,
        output_var=output_var,
        validate=validate,
    )


def stack_core(
    values: object,
    *,
    core_dim: str,
    core_labels: Sequence[object] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Stack AO-like values along one core dimension.

    Parameters
    ----------
    values : object
        Input values consumed by this operation.
    core_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    core_labels : Sequence[object] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    output_var : str | None, optional
        Output naming metadata used during finalization.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, stack_core
    >>> x = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> y = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> out = stack_core([x, y], core_dim="axis", core_labels=("x", "y"))
    >>> tuple(out.as_dataset().coords["axis"].values.tolist())
    ('x', 'y')
    """
    return _stack_core(
        values,
        core_dim=core_dim,
        core_labels=core_labels,
        output_var=output_var,
        validate=validate,
    )


def block_core(
    values: object,
    *,
    row_dim: str,
    col_dim: str,
    row_labels: Sequence[object] | None = None,
    col_labels: Sequence[object] | None = None,
    output_var: str | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Assemble nested 2D AO-like layout into row/col core dimensions.

    Parameters
    ----------
    values : object
        Input values consumed by this operation.
    row_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    col_dim : str, optional
        Dimension name used to resolve labeled array semantics.
    row_labels : Sequence[object] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    col_labels : Sequence[object] | None, optional
        Core-dimension labels/shape metadata used for structural operations.
    output_var : str | None, optional
        Output naming metadata used during finalization.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, block_core
    >>> x = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> y = AnalysisObject.from_data(xr.Dataset({"value": ("sample", [2.0])}, coords={"sample": [0]}), sequence_dim="sample", core_dims=(), validate=True)
    >>> out = block_core([[x], [y]], row_dim="row", col_dim="col")
    >>> (out.as_dataset().sizes["row"], out.as_dataset().sizes["col"])
    (2, 1)
    """
    return _block_core(
        values,
        row_dim=row_dim,
        col_dim=col_dim,
        row_labels=row_labels,
        col_labels=col_labels,
        output_var=output_var,
        validate=validate,
    )


def concat_core(
    aos: Sequence[object],
    *,
    opts: CoreConcatOptions | None = None,
    validate: bool = True,
) -> "AnalysisObject":
    """Concatenate AO-like operands along an existing core dimension.

    Parameters
    ----------
    aos : Sequence[object]
        AO-like inputs consumed by this orchestration boundary.
    opts : CoreConcatOptions | None, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, CoreConcatOptions, concat_core
    >>> x = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0]])}, coords={"sample": [0], "axis": ["x"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> y = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[2.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> concat_core([x, y], opts=CoreConcatOptions(core_dim="axis")).as_dataset().sizes["axis"]
    2
    """
    options = coerce_core_concat_options(opts, owner="concat_core")
    return _concat_core(
        aos,
        opts=options,
        validate=validate,
    )


def decompose_core(
    ao: object,
    *,
    opts: CoreDecomposeOptions,
    validate: bool = True,
) -> dict[tuple[object, ...], "AnalysisObject"]:
    """Decompose selected core dimensions into keyed AO outputs.

    Parameters
    ----------
    ao : object
        AnalysisObject-like input value.
    opts : CoreDecomposeOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    dict[tuple[object, ...], 'AnalysisObject']
        Mapping-like result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, CoreDecomposeOptions, decompose_core
    >>> ao = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> sorted(decompose_core(ao, opts=CoreDecomposeOptions(core_dims=("axis",), key_mode="label")))
    [('x',), ('y',)]
    """
    return _decompose_core(ao, opts=opts, validate=validate)


def overlay_core(
    base: object,
    patches: Sequence[object] | object,
    *,
    opts: CoreOverlayOptions,
    validate: bool = True,
) -> "AnalysisObject":
    """Overlay patch labels along a selected core dimension.

    Parameters
    ----------
    base : object
        Input dataset/source value processed by this operation.
    patches : Sequence[object] | object
        Operand/component input consumed by this operation.
    opts : CoreOverlayOptions, optional
        Optional options controlling policy and numeric behavior for this operation.
    validate : bool, optional
        When ``True``, validate output schema/layout invariants before returning.

    Returns
    -------
    AnalysisObject
        Result of applying this operation with TAL semantic constraints preserved.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject, CoreOverlayOptions, overlay_core
    >>> base = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[1.0, 2.0]])}, coords={"sample": [0], "axis": ["x", "y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> patch = AnalysisObject.from_data(xr.Dataset({"v": (("sample", "axis"), [[9.0]])}, coords={"sample": [0], "axis": ["y"]}), sequence_dim="sample", core_dims=("axis",), validate=True)
    >>> overlay_core(base, patch, opts=CoreOverlayOptions(core_dim="axis", on_overlap="replace")).as_dataset()["v"].sel(axis="y").item()
    9.0
    """
    options = coerce_core_overlay_options(opts, owner="overlay_core")
    return _overlay_core(base, patches, opts=options, validate=validate)
