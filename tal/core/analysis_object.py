from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Self

import xarray as xr

from . import dataset_ownership as _dataset_ownership
from .dataset_utils import ensure_dataset
from .schema import UNSET, UnsetType
from .schema import _SchemaUpdatePlan
from .schema import _apply_schema_update
from .schema import _is_bootstrap_schema
from .schema import _relocate_dataarray_schema
from .schema import _validate_existing_schema_envelope
from .schema import merge_schema as _merge_schema
from .schema import repair_schema_after_structure as _repair_schema_after_structure
from .schema import set_param_coord as _set_param_coord
from .schema import set_roles as _set_roles
from .schema import set_validity as _set_validity
from .schema import validate_schema as _validate_schema
from .schema_errors import schema_error
from .validity_finalize import reconcile_sequence_validity_after_structure


def _require_sequence_dim_for_schema_fields(
    *,
    sequence_dim: str | None,
    param_coord: str | None,
    sequence_size_coord: str | None,
) -> None:
    required = [
        name
        for name, value in (
            ("param_coord", param_coord),
            ("sequence_size_coord", sequence_size_coord),
        )
        if value is not None
    ]
    if sequence_dim is not None or not required:
        return
    needed = ", ".join(required)
    raise ValueError(
        "from_data requires sequence_dim when schema-bearing arguments are provided. "
        f"Missing sequence_dim with: {needed}."
    )


def _from_data_schema_plan(
    *,
    sequence_dim: str | None,
    batch_dims: Sequence[str],
    core_dims: Sequence[str],
    param_coord: str | None,
    sequence_size_coord: str | None,
    layout: Literal["left_packed"],
) -> _SchemaUpdatePlan:
    roles_declared = sequence_dim is not None or bool(batch_dims) or bool(core_dims)
    return _SchemaUpdatePlan(
        sequence_dim=sequence_dim if sequence_dim is not None else UNSET,
        batch_dims=batch_dims if roles_declared else UNSET,
        core_dims=core_dims if roles_declared else UNSET,
        param_coord=param_coord if param_coord is not None else UNSET,
        sequence_size_coord=sequence_size_coord if sequence_size_coord is not None else UNSET,
        layout=layout,
    )


class AnalysisObject:
    """Dataset-backed core TAL container.

    Notes
    -----
    TAL stores semantic metadata in ``ds.attrs["tal"]`` and preserves xarray's
    label-aware behavior. Alignment is by dimension names and coordinate labels,
    not by positional axis order.

    Operator Families
    -----------------
    Arithmetic operators route through TAL's AO-aware ufunc owners and return
    finalized ``AnalysisObject`` results using xarray labeled alignment.
    Ordering operators (``<``, ``<=``, ``>``, ``>=``) build deferred
    ``Condition`` expressions for event evaluation. ``==`` and ``!=`` retain
    object-identity semantics; use ``tal.ufuncs.equal`` and
    ``tal.ufuncs.not_equal`` for deferred elementwise equality conditions.

    See Also
    --------
    tal.ufuncs
        AO-aware universal function entrypoints used by operator overloads.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> ds = xr.Dataset(
    ...     {"position": (("run", "sample", "axis"), [[[0.0, 1.0, 2.0]]])},
    ...     coords={"run": ["run_0"], "sample": [0], "axis": ["x", "y", "z"]},
    ... )
    >>> ao = AnalysisObject.from_data(
    ...     ds,
    ...     sequence_dim="sample",
    ...     batch_dims=("run",),
    ...     core_dims=("axis",),
    ...     validate=True,
    ... )
    >>> ao.as_dataset().attrs["tal"]["core"]["roles"]["sequence_dim"]
    'sample'
    """

    _WHERE_OTHER_UNSET = object()
    __hash__ = object.__hash__

    @staticmethod
    def _multiindex_dims(ds: xr.Dataset) -> tuple[str, ...]:
        dims: list[str] = []
        for dim in ds.dims:
            try:
                index = ds.get_index(dim)
            except (KeyError, TypeError, ValueError):
                continue
            if int(getattr(index, "nlevels", 1)) > 1:
                dims.append(str(dim))
        return tuple(dims)

    @classmethod
    def _assert_no_multiindex(cls, ds: xr.Dataset, *, owner: str) -> None:
        dims = cls._multiindex_dims(ds)
        if not dims:
            return
        raise ValueError(
            f"{owner}: PandasMultiIndex dimensions are not supported in TAL v3: {list(dims)!r}. "
            "Flatten indexes with reset_index/unstack before using AnalysisObject."
        )

    @staticmethod
    def _input_tal_payload(data: xr.Dataset | xr.DataArray) -> tuple[bool, Any]:
        if "tal" not in data.attrs:
            return False, None
        return True, data.attrs["tal"]

    @classmethod
    def _normalized_ingress_dataset(
        cls,
        data: xr.Dataset | xr.DataArray,
    ) -> xr.Dataset:
        if not isinstance(data, (xr.Dataset, xr.DataArray)):
            ensure_dataset(data)
        input_had_tal, tal_payload = cls._input_tal_payload(data)
        if input_had_tal and not isinstance(tal_payload, Mapping):
            raise schema_error(
                code="schema.not_mapping",
                path="tal",
                expected="mapping",
                actual=type(tal_payload).__name__,
                hint="set ds.attrs['tal'] to a mapping payload",
            )
        ds = ensure_dataset(data)
        if input_had_tal and isinstance(data, xr.DataArray):
            variable_name = next(iter(ds.data_vars))
            ds = _relocate_dataarray_schema(
                ds,
                variable_name=variable_name,
                tal_schema=tal_payload,
            )
        cls._assert_no_multiindex(ds, owner="AnalysisObject")
        return ds

    def __init__(self, data: xr.Dataset | xr.DataArray) -> None:
        candidate = self._prepared_ingress_dataset(data)
        owned = _dataset_ownership.isolate_external_dataset(candidate)
        self._bind_dataset(owned)

    @classmethod
    def _prepared_ingress_dataset(cls, data: xr.Dataset | xr.DataArray) -> xr.Dataset:
        ds = cls._normalized_ingress_dataset(data)
        input_had_tal = "tal" in data.attrs
        if input_had_tal:
            tal_payload = ds.attrs["tal"]
            return (
                _apply_schema_update(
                    _validate_existing_schema_envelope(ds),
                    _SchemaUpdatePlan(),
                    validate=False,
                )
                if _is_bootstrap_schema(tal_payload)
                else _validate_schema(ds)
            )
        return _apply_schema_update(
            ds,
            _SchemaUpdatePlan(),
            validate=False,
        )

    def _bind_dataset(self, ds: xr.Dataset) -> None:
        self._assert_no_multiindex(ds, owner="AnalysisObject")
        self._data = ds
        self._broadcast_intent = None
        self._alignment_intent = None
        self._after_bind_dataset()

    def _after_bind_dataset(self) -> None:
        """Subclass hook invoked after dataset binding."""
        return None

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> AnalysisObject:
        obj = cls.__new__(cls)
        obj._bind_dataset(ensure_dataset(ds))
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray, *, schema_prepared: bool = False) -> AnalysisObject:
        candidate = ensure_dataset(ds)
        if not schema_prepared:
            candidate = _merge_schema(
                candidate,
                {"version": 1, "core": {}},
                validate=False,
            )
        obj = cls.__new__(cls)
        obj._bind_dataset(candidate)
        return obj

    @property
    def param(self) -> "ParamAccessor":
        """Return the param accessor (``ao.param``).

        Returns
        -------
        ParamAccessor
            Resolved property value.
        """
        from .param_ops import ParamAccessor

        return ParamAccessor(self)

    @property
    def combine(self) -> "CombineAccessor":
        """Return the combine accessor (``ao.combine``).

        Returns
        -------
        CombineAccessor
            Resolved property value.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> stacked = ao.combine.stack_core([ao], core_dim="copy", core_labels=("left", "right"))
        >>> stacked.as_dataset().sizes["copy"]
        2
        """
        from .combine_ops import CombineAccessor

        return CombineAccessor(self)

    @property
    def events(self) -> "EventsAccessor":
        """Return the events accessor (``ao.events``).

        Returns
        -------
        EventsAccessor
            Resolved property value.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset(
        ...         {"value": ("sample", [0.0, 2.0])},
        ...         coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])},
        ...     ),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     param_coord="time",
        ...     validate=True,
        ... )
        >>> mask = ao.events.mask(ao > 1.0)
        >>> mask.to_numpy().tolist()
        [False, True]
        """
        from .event_ops import EventsAccessor

        return EventsAccessor(self)

    @property
    def components(self) -> "ComponentsAccessor":
        """Return the components accessor (``ao.components``).

        Returns
        -------
        ComponentsAccessor
            Resolved property value.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.component_ops import ComponentRegistryOptions, ComponentSpec
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"vec": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... )
        >>> tagged = ao.components.define(opts=ComponentRegistryOptions({"xy": ComponentSpec("axis", ("x", "y"))}))
        >>> tagged.components.registry()["xy"].labels
        ('x', 'y')
        """
        from .component_ops import ComponentsAccessor

        return ComponentsAccessor(self)

    @property
    def group(self) -> "GroupAccessor":
        """Return the grouping accessor (``ao.group``).

        Returns
        -------
        GroupAccessor
            Resolved property value.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset(
        ...         {"value": (("run", "sample"), [[1.0, 2.0], [3.0, 4.0]])},
        ...         coords={"run": ["a", "b"], "sample": [0, 1], "kind": ("run", ["sim", "robot"])},
        ...     ),
        ...     sequence_dim="sample",
        ...     batch_dims=("run",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> grouped = ao.group.groupby("kind")
        >>> grouped.mean(dim="sample").as_dataset().sizes["group_key"]
        2
        """
        from .group_ops import GroupAccessor

        return GroupAccessor(self)

    def as_dataset(
        self,
        *,
        copy: Literal["deep", "shallow", "none"] = "deep",
    ) -> xr.Dataset:
        """Return the underlying dataset with the requested ownership policy.

        Parameters
        ----------
        copy
            ``"deep"`` isolates eager buffers and metadata, ``"shallow"``
            shares buffers while isolating wrappers and metadata, and ``"none"``
            returns the exact backing Dataset.

        Returns
        -------
        xr.Dataset
            Dataset exposed under the requested ownership policy.

        Raises
        ------
        ValueError
            If ``copy`` is invalid or the backing Dataset has a MultiIndex.

        Notes
        -----
        Deep and shallow lazy views remain non-owning and require the AO to stay
        open until their lazy work completes. Mutating shared eager buffers from
        a shallow view, or mutating a raw view, can mutate the AO.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> snapshot = ao.as_dataset()
        >>> snapshot is ao.as_dataset(copy="none")
        False
        """
        owner = "AnalysisObject.as_dataset"
        mode = _dataset_ownership.coerce_dataset_copy_mode(copy, owner=owner)
        self._assert_no_multiindex(self._data, owner=owner)
        return _dataset_ownership.dataset_view(self._data, copy=mode, owner=owner)

    def to_dataarray(
        self,
        *,
        name: str | None = None,
        copy: Literal["deep", "shallow", "none"] = "deep",
    ) -> xr.DataArray:
        """Convert a single-variable AO to ``xarray.DataArray``.

        Parameters
        ----------
        name
            Optional output variable name override.
        copy
            Ownership policy applied to the selected variable and coordinates.

        Returns
        -------
        xarray.DataArray
            The only data variable from this AO as a DataArray.

        Raises
        ------
        ValueError
            If ``copy`` is invalid, this AO contains multiple data variables,
            or the backing Dataset has a MultiIndex.

        Notes
        -----
        This helper is a strict single-variable boundary. Multi-variable payloads
        remain datasets to avoid accidental data loss. Dataset-level attrs,
        including Dataset-level TAL schema, are not projected into the selected
        variable's attrs. Every returned DataArray is a non-owning facade.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ds = xr.Dataset({"value": ("sample", [1.0, 2.0])}, coords={"sample": [0, 1]})
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), validate=True)
        >>> ao.to_dataarray().name
        'value'

        See Also
        --------
        tal.core.dataset_utils.dataset_to_dataarray
        """
        owner = "AnalysisObject.to_dataarray"
        mode = _dataset_ownership.coerce_dataset_copy_mode(copy, owner=owner)
        self._assert_no_multiindex(self._data, owner=owner)
        return _dataset_ownership.dataset_to_dataarray_view(
            self._data,
            name=name,
            copy=mode,
            owner=owner,
        )

    def close(self) -> None:
        """Release resources owned by the backing Dataset.

        Notes
        -----
        Calling this method repeatedly is safe. Deep and shallow public views
        are non-owning and must complete lazy work before the AO is closed.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject(xr.Dataset({"value": ("sample", [1.0])}))
        >>> ao.close()
        >>> ao.close()
        """
        self._data.close()

    def _rewrap_dataset(self, ds: xr.Dataset, *, validate: bool) -> AnalysisObject:
        if validate:
            return self.__class__._from_validated(ds)
        return self.__class__._from_unvalidated(ds)

    def _prepare_result_rewrap_context(self, values: tuple[object, ...], *, owner: str) -> object | None:
        """Prepare an opaque domain result context for a multi-input operation."""
        _ = (values, owner)
        return None

    def _apply_result_rewrap_context(self, result: AnalysisObject, *, context: object | None) -> AnalysisObject:
        """Apply an opaque domain result context after ordinary finalization."""
        _ = context
        return result

    def b(self, *, mode: Literal["semantic_broadcast"] = "semantic_broadcast") -> Self:
        """Attach a semantic-broadcast intent to an operand-local AO copy.

        Parameters
        ----------
        mode
            Broadcast intent mode. v3 currently supports
            ``"semantic_broadcast"``.

        Returns
        -------
        Self
            A new AO of the same class carrying broadcast intent.

        Notes
        -----
        Intent is consumed by orchestration topology planning. This method does
        not mutate the source instance.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> hinted = ao.b()
        >>> hinted is ao
        False

        See Also
        --------
        tal.core.orchestration.broadcast_intent.resolve_broadcast_intent
        """
        from .orchestration.broadcast_intent import (
            resolve_broadcast_intent,
            set_broadcast_intent,
        )
        from .orchestration.alignment_intent import read_alignment_intent, set_alignment_intent

        intent = resolve_broadcast_intent(mode=mode, owner="AnalysisObject.b")
        previous_alignment = read_alignment_intent(self, owner="AnalysisObject.b")
        out = self._rewrap_dataset(self._data, validate=False)
        if previous_alignment is not None:
            set_alignment_intent(out, previous_alignment)
        set_broadcast_intent(out, intent)
        return out

    def a(
        self,
        *,
        on: Literal["sequence", "param", "auto"] = "sequence",
        sequence_join: Literal["exact", "inner", "outer", "left", "right"] | None = "exact",
        batch_join: Literal["exact", "inner", "outer", "left", "right"] = "exact",
        core_policy: Literal["strict", "numpy_named"] = "strict",
    ) -> Self:
        """Attach an alignment intent to an operand-local AO copy.

        Parameters
        ----------
        on
            Primary alignment axis family.
        sequence_join
            Join policy for sequence-dimension alignment.
        batch_join
            Join policy for batch-dimension alignment.
        core_policy
            Core-dimension alignment policy.

        Returns
        -------
        Self
            A new AO of the same class carrying alignment intent.

        Notes
        -----
        Alignment intent is policy metadata consumed by orchestration owners.
        It does not execute interpolation or structural alignment eagerly.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> hinted = ao.a(on="sequence", sequence_join="inner")
        >>> hinted is ao
        False

        See Also
        --------
        tal.core.orchestration.alignment_intent.resolve_alignment_intent
        """
        from .orchestration.alignment_intent import (
            read_alignment_intent,
            resolve_alignment_intent,
            set_alignment_intent,
        )
        from .orchestration.broadcast_intent import read_broadcast_intent, set_broadcast_intent

        intent = resolve_alignment_intent(
            on=on,
            sequence_join=sequence_join,
            batch_join=batch_join,
            core_policy=core_policy,
            owner="AnalysisObject.a",
        )
        previous_alignment = read_alignment_intent(self, owner="AnalysisObject.a")
        if previous_alignment is not None and previous_alignment != intent:
            raise ValueError(
                "AnalysisObject.a: conflicting chained alignment intents are not allowed; "
                f"existing={previous_alignment!r}, requested={intent!r}."
            )
        previous_broadcast = read_broadcast_intent(self, owner="AnalysisObject.a")
        out = self._rewrap_dataset(self._data, validate=False)
        set_alignment_intent(out, previous_alignment if previous_alignment is not None else intent)
        if previous_broadcast is not None:
            set_broadcast_intent(out, previous_broadcast)
        return out

    def _finalize_structural(
        self,
        ds: xr.Dataset,
        *,
        validate: bool,
        rename_map: Mapping[str, str] | None = None,
    ) -> AnalysisObject:
        from .component_ops.rewrite import rewrite_component_registry_after_structure

        repaired = _repair_schema_after_structure(ds, validate=False, rename_map=rename_map)
        reconciled = reconcile_sequence_validity_after_structure(
            self._data,
            repaired,
            validate=validate,
            owner=f"{self.__class__.__name__}._finalize_structural",
            rename_map=rename_map,
        )
        rewritten = rewrite_component_registry_after_structure(
            reconciled,
            rename_map=rename_map,
            owner="components.rewrite",
        )
        if validate:
            payload = rewritten.attrs.get("tal")
            if not (isinstance(payload, Mapping) and _is_bootstrap_schema(payload)):
                rewritten = _validate_schema(rewritten)
        return self._rewrap_dataset(rewritten, validate=validate)

    @classmethod
    def from_data(
        cls,
        data: xr.Dataset | xr.DataArray,
        *,
        sequence_dim: str | None = None,
        batch_dims: Sequence[str] = (),
        core_dims: Sequence[str] = (),
        param_coord: str | None = None,
        sequence_size_coord: str | None = None,
        layout: Literal["left_packed"] = "left_packed",
        validate: bool = True,
    ) -> AnalysisObject:
        """Construct an AO while explicitly writing core schema semantics.

        Parameters
        ----------
        data
            Source dataset/dataarray.
        sequence_dim
            Optional sequence dimension name.
        batch_dims
            Optional batch dimension names.
        core_dims
            Optional core dimension names.
        param_coord
            Optional param coordinate name. Requires ``sequence_dim``.
        sequence_size_coord
            Optional validity size coordinate name. Requires ``sequence_dim``.
        layout
            Validity layout policy when ``sequence_size_coord`` is set.
        validate
            Whether to run schema validation before returning.

        Returns
        -------
        AnalysisObject
            AO instance with the requested schema metadata.

        Raises
        ------
        ValueError
            If sequence-dependent fields are requested without ``sequence_dim``.

        Notes
        -----
        ``batch_dims`` are allowed without ``sequence_dim``. With no declared
        roles, runtime consumers treat the AO as core-only where supported.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> ds = xr.Dataset(
        ...     {"value": (("trial", "axis"), [[1.0, 2.0, 3.0]])},
        ...     coords={"trial": ["t0"], "axis": ["x", "y", "z"]},
        ... )
        >>> ao = AnalysisObject.from_data(ds, batch_dims=("trial",), core_dims=("axis",), validate=True)
        >>> read_roles(ao.as_dataset())[1] is None
        True

        See Also
        --------
        tal.core.schema.set_roles
        tal.core.schema.set_param_coord
        tal.core.schema.set_validity
        """
        _require_sequence_dim_for_schema_fields(
            sequence_dim=sequence_dim,
            param_coord=param_coord,
            sequence_size_coord=sequence_size_coord,
        )
        candidate = cls._normalized_ingress_dataset(data)
        if "tal" in data.attrs:
            candidate = _validate_existing_schema_envelope(candidate)
        plan = _from_data_schema_plan(
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=core_dims,
            param_coord=param_coord,
            sequence_size_coord=sequence_size_coord,
            layout=layout,
        )
        candidate = _apply_schema_update(candidate, plan, validate=validate)
        owned = _dataset_ownership.isolate_external_dataset(candidate)
        if validate:
            return cls._from_validated(owned)
        return cls._from_unvalidated(owned, schema_prepared=True)

    def isel(
        self,
        indexers: Mapping[str, Any] | None = None,
        *,
        drop: bool = False,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        validate: bool = True,
        **indexers_kwargs: Any,
    ) -> AnalysisObject:
        """Apply ``xarray.Dataset.isel`` and finalize TAL schema metadata.

        Parameters
        ----------
        indexers : Mapping[str, Any] | None, optional
            Mapping from dimension names to integer, slice, array, or DataArray
            indexers. Follows ``xarray.Dataset.isel`` semantics.
        drop : bool, optional
            Whether scalar-indexed coordinates should be dropped.
        missing_dims : Literal['raise', 'warn', 'ignore'], optional
            How xarray should handle dimensions named in ``indexers`` that are
            absent from the dataset.
        validate : bool, optional
            When ``True``, validate repaired TAL schema before returning.
        **indexers_kwargs : Any
            Additional dimension indexers supplied as keywords.

        Returns
        -------
        AnalysisObject
            AO containing the indexed dataset and repaired TAL metadata.

        Notes
        -----
        TAL delegates the selection itself to xarray, then reconciles role,
        validity, and component metadata with the resulting dimensions.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> ao.isel(sample=slice(0, 2)).as_dataset().sizes["sample"]
        2
        """
        ds = self._data.isel(
            indexers=indexers,
            drop=drop,
            missing_dims=missing_dims,
            **indexers_kwargs,
        )
        return self._finalize_structural(ds, validate=validate)

    def sel(
        self,
        indexers: Mapping[str, Any] | None = None,
        *,
        method: str | None = None,
        tolerance: Any = None,
        drop: bool = False,
        validate: bool = True,
        **indexers_kwargs: Any,
    ) -> AnalysisObject:
        """Apply ``xarray.Dataset.sel`` and finalize TAL schema metadata.

        Parameters
        ----------
        indexers : Mapping[str, Any] | None, optional
            Mapping from dimension or coordinate names to label selectors.
            Follows ``xarray.Dataset.sel`` semantics.
        method : str | None, optional
            Optional xarray inexact-match method such as ``"nearest"``.
        tolerance : Any, optional
            Maximum allowed distance for inexact label matches.
        drop : bool, optional
            Whether scalar-indexed coordinates should be dropped.
        validate : bool, optional
            When ``True``, validate repaired TAL schema before returning.
        **indexers_kwargs : Any
            Additional label selectors supplied as keywords.

        Returns
        -------
        AnalysisObject
            AO containing the selected dataset and repaired TAL metadata.

        Notes
        -----
        Use ``ao.param.sel(...)`` when selection should happen against the
        declared parameter coordinate rather than ordinary xarray labels.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("trial", [1.0, 2.0])}, coords={"trial": ["a", "b"]}),
        ...     batch_dims=("trial",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> ao.sel(trial="b").as_dataset()["value"].item()
        2.0
        """
        ds = self._data.sel(
            indexers=indexers,
            method=method,
            tolerance=tolerance,
            drop=drop,
            **indexers_kwargs,
        )
        return self._finalize_structural(ds, validate=validate)

    def where(
        self,
        cond: Any,
        other: Any = _WHERE_OTHER_UNSET,
        *,
        drop: bool = False,
        validate: bool = True,
    ) -> AnalysisObject:
        """Apply ``xarray.Dataset.where`` and finalize TAL schema metadata.

        Parameters
        ----------
        cond : Any
            Boolean condition accepted by ``xarray.Dataset.where``.
        other : Any, optional
            Fill value used where ``cond`` is false. If omitted, xarray fills
            with missing values.
        drop : bool, optional
            Whether coordinate labels that only correspond to false values
            should be dropped.
        validate : bool, optional
            When ``True``, validate repaired TAL schema before returning.

        Returns
        -------
        AnalysisObject
            AO containing the masked dataset and repaired TAL metadata.

        Notes
        -----
        The mask is applied with xarray alignment rules. TAL then reconciles
        schema and validity metadata with the resulting dataset.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0, 2.0, 3.0])}, coords={"sample": [0, 1, 2]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> condition = ao.as_dataset()["value"] > 1.0
        >>> ao.where(condition, other=0.0).as_dataset()["value"].to_numpy().tolist()
        [0.0, 2.0, 3.0]
        """
        if other is self._WHERE_OTHER_UNSET:
            ds = self._data.where(cond, drop=drop)
        else:
            ds = self._data.where(cond, other=other, drop=drop)
        return self._finalize_structural(ds, validate=validate)

    def drop_vars(
        self,
        names: str | list[str],
        *,
        errors: Literal["raise", "ignore"] = "raise",
        validate: bool = True,
    ) -> AnalysisObject:
        """Drop variables and repair schema/validity metadata.

        Parameters
        ----------
        names : str | list[str]
            Data variable or variables to remove.
        errors : Literal['raise', 'ignore'], optional
            Whether xarray should raise for missing variable names.
        validate : bool, optional
            When ``True``, validate repaired TAL schema before returning.

        Returns
        -------
        AnalysisObject
            AO without the requested variables.

        Notes
        -----
        Component registry entries that reference removed variables are pruned
        during finalization.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0]), "quality": ("sample", [1])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> "quality" in ao.drop_vars("quality").as_dataset()
        False
        """
        ds = self._data.drop_vars(names, errors=errors)
        return self._finalize_structural(ds, validate=validate)

    def rename(
        self,
        name_dict: Mapping[str, str] | None = None,
        *,
        validate: bool = True,
        **names: str,
    ) -> AnalysisObject:
        """Rename dimensions/coords/variables and repair schema references.

        Parameters
        ----------
        name_dict : Mapping[str, str] | None, optional
            Mapping from existing dimension, coordinate, or variable names to
            replacement names.
        validate : bool, optional
            When ``True``, validate repaired TAL schema before returning.
        **names : str
            Additional rename mappings supplied as keywords.

        Returns
        -------
        AnalysisObject
            AO with renamed xarray objects and rewritten TAL schema references.

        Notes
        -----
        Rename mappings must be injective under xarray's own rules. TAL rewrites
        role, parameter-coordinate, validity, and component references that
        point at renamed objects.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> "step" in ao.rename({"sample": "step"}).as_dataset().dims
        True
        """
        ds = self._data.rename(name_dict=name_dict, **names)
        rename_map: dict[str, str] = {}
        if isinstance(name_dict, Mapping):
            rename_map.update({str(old): str(new) for old, new in name_dict.items()})
        if names:
            rename_map.update({str(old): str(new) for old, new in names.items()})
        return self._finalize_structural(ds, validate=validate, rename_map=rename_map)

    def transpose(
        self,
        *dim: str,
        missing_dims: Literal["raise", "warn", "ignore"] = "raise",
        validate: bool = True,
    ) -> AnalysisObject:
        """Transpose dimensions and reconcile schema/validity metadata.

        Parameters
        ----------
        *dim : str
            Target dimension order passed to ``xarray.Dataset.transpose``.
        missing_dims : Literal['raise', 'warn', 'ignore'], optional
            How xarray should handle requested dimensions that are absent.
        validate : bool, optional
            When ``True``, validate repaired TAL schema before returning.

        Returns
        -------
        AnalysisObject
            AO with variables transposed to the requested dimension order.

        Notes
        -----
        Transposition changes physical dimension order only. TAL role metadata
        remains name-based.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": (("trial", "sample"), [[1.0, 2.0]])}, coords={"trial": ["a"], "sample": [0, 1]}),
        ...     sequence_dim="sample",
        ...     batch_dims=("trial",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> ao.transpose("sample", "trial").as_dataset()["value"].dims
        ('sample', 'trial')
        """
        ds = self._data.transpose(
            *dim,
            missing_dims=missing_dims,
        )
        return self._finalize_structural(ds, validate=validate)

    def _apply_unary_ufunc(self, name: str) -> object:
        from tal import ufuncs as tal_ufuncs

        fn = getattr(tal_ufuncs, name)
        return fn(self)

    def _apply_binary_ufunc(self, name: str, other: object) -> object:
        from tal import ufuncs as tal_ufuncs

        fn = getattr(tal_ufuncs, name)
        return fn(self, other)

    def _apply_rbinary_ufunc(self, name: str, other: object) -> object:
        from tal import ufuncs as tal_ufuncs

        fn = getattr(tal_ufuncs, name)
        return fn(other, self)

    def __add__(self, other: object) -> object:
        return self._apply_binary_ufunc("add", other)

    def __radd__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("add", other)

    def __sub__(self, other: object) -> object:
        return self._apply_binary_ufunc("subtract", other)

    def __rsub__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("subtract", other)

    def __mul__(self, other: object) -> object:
        return self._apply_binary_ufunc("multiply", other)

    def __rmul__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("multiply", other)

    def __truediv__(self, other: object) -> object:
        return self._apply_binary_ufunc("true_divide", other)

    def __rtruediv__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("true_divide", other)

    def __floordiv__(self, other: object) -> object:
        return self._apply_binary_ufunc("floor_divide", other)

    def __rfloordiv__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("floor_divide", other)

    def __mod__(self, other: object) -> object:
        return self._apply_binary_ufunc("mod", other)

    def __rmod__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("mod", other)

    def __pow__(self, other: object) -> object:
        return self._apply_binary_ufunc("power", other)

    def __rpow__(self, other: object) -> object:
        return self._apply_rbinary_ufunc("power", other)

    def __neg__(self) -> object:
        return self._apply_unary_ufunc("negative")

    def __pos__(self) -> object:
        return self._apply_unary_ufunc("positive")

    def __abs__(self) -> object:
        return self._apply_unary_ufunc("absolute")

    def __lt__(self, other: object) -> object:
        return self._apply_binary_ufunc("less", other)

    def __le__(self, other: object) -> object:
        return self._apply_binary_ufunc("less_equal", other)

    def __gt__(self, other: object) -> object:
        return self._apply_binary_ufunc("greater", other)

    def __ge__(self, other: object) -> object:
        return self._apply_binary_ufunc("greater_equal", other)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __ne__(self, other: object) -> bool:
        return self is not other

    def set_roles(
        self,
        *,
        sequence_dim: str | UnsetType = UNSET,
        batch_dims: Sequence[str] | UnsetType = UNSET,
        core_dims: Sequence[str] | UnsetType = UNSET,
        validate: bool = True,
    ) -> AnalysisObject:
        """Write role metadata on this AO and return a new AO.

        Parameters
        ----------
        sequence_dim
            Sequence dimension override or ``UNSET`` to preserve.
        batch_dims
            Batch dimensions override or ``UNSET`` to preserve.
        core_dims
            Core dimensions override or ``UNSET`` to preserve.
        validate
            Whether to validate schema before returning.

        Returns
        -------
        AnalysisObject
            AO with updated role metadata.

        Notes
        -----
        Role metadata drives orchestrators in core/linalg/spatial owners.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_roles
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset(
        ...         {"value": (("trial", "sample"), [[1.0, 2.0]])},
        ...         coords={"trial": ["t0"], "sample": [0, 1]},
        ...     ),
        ...     sequence_dim="sample",
        ...     batch_dims=("trial",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = ao.set_roles(sequence_dim=None, batch_dims=("trial",), core_dims=(), validate=True)
        >>> read_roles(out.as_dataset())[1] is None
        True

        See Also
        --------
        tal.core.schema.set_roles
        """
        ds = _set_roles(
            self._data,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            core_dims=core_dims,
            validate=validate,
        )
        return self._rewrap_dataset(ds, validate=validate)

    def set_param_coord(
        self,
        *,
        name: str | None,
        validate: bool = True,
    ) -> AnalysisObject:
        """Write ``param_coord`` metadata and return a new AO.

        Parameters
        ----------
        name
            Coordinate name or ``None`` to clear.
        validate
            Whether to validate schema before returning.

        Returns
        -------
        AnalysisObject
            AO with updated param metadata.

        Notes
        -----
        Param-aware selection/evaluation owners require valid sequence semantics.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_param_coord_name
        >>> ds = xr.Dataset(
        ...     {"value": ("sample", [0.0, 1.0])},
        ...     coords={"sample": [0, 1], "time": ("sample", [0.0, 1.0])},
        ... )
        >>> ao = AnalysisObject.from_data(ds, sequence_dim="sample", core_dims=(), validate=True)
        >>> out = ao.set_param_coord(name="time", validate=True)
        >>> read_param_coord_name(out.as_dataset())
        'time'

        See Also
        --------
        tal.core.schema.set_param_coord
        """
        ds = _set_param_coord(self._data, name=name, validate=validate)
        return self._rewrap_dataset(ds, validate=validate)

    def set_validity(
        self,
        *,
        sequence_size_coord: str | None,
        layout: Literal["left_packed"] = "left_packed",
        validate: bool = True,
    ) -> AnalysisObject:
        """Write sequence validity metadata and return a new AO.

        Parameters
        ----------
        sequence_size_coord
            Name of the sequence-size coordinate, or ``None`` to clear validity.
        layout
            Validity layout policy.
        validate
            Whether to validate schema before returning.

        Returns
        -------
        AnalysisObject
            AO with updated validity metadata.

        Notes
        -----
        Validity is sequence-scoped. If sequence semantics are removed, validity
        owners prune stale sequence validity blocks.

        Examples
        --------
        >>> import numpy as np
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.schema_read import read_sequence_size_coord_name
        >>> ds = xr.Dataset(
        ...     {"value": (("trial", "sample"), [[1.0, 2.0, 0.0]])},
        ...     coords={
        ...         "trial": ["t0"],
        ...         "sample": [0, 1, 2],
        ...         "group_size": ("trial", np.array([2], dtype=np.int64)),
        ...     },
        ... )
        >>> ao = AnalysisObject.from_data(
        ...     ds,
        ...     sequence_dim="sample",
        ...     batch_dims=("trial",),
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> out = ao.set_validity(sequence_size_coord="group_size", validate=True)
        >>> read_sequence_size_coord_name(out.as_dataset())
        'group_size'

        See Also
        --------
        tal.core.schema.set_validity
        """
        ds = _set_validity(
            self._data,
            sequence_size_coord=sequence_size_coord,
            layout=layout,
            validate=validate,
        )
        return self._rewrap_dataset(ds, validate=validate)

    def merge_schema(
        self,
        patch: Mapping[str, Any],
        *,
        validate: bool = True,
    ) -> AnalysisObject:
        """Merge a schema patch into ``ds.attrs['tal']`` and return a new AO.

        Parameters
        ----------
        patch
            Schema patch rooted at TAL payload keys.
        validate
            Whether to validate schema before returning.

        Returns
        -------
        AnalysisObject
            AO with merged schema payload.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ds = xr.Dataset({"value": ("axis", [1.0, 2.0, 3.0])}, coords={"axis": ["x", "y", "z"]})
        >>> ao = AnalysisObject.from_data(ds, core_dims=("axis",), validate=True)
        >>> out = ao.merge_schema({"core": {"roles": {"core_dims": ["axis"]}}}, validate=True)
        >>> out.as_dataset().attrs["tal"]["core"]["roles"]["core_dims"]
        ['axis']

        See Also
        --------
        tal.core.schema.merge_schema
        """
        ds = _merge_schema(self._data, patch=patch, validate=validate)
        return self._rewrap_dataset(ds, validate=validate)

    def validate_schema(self) -> AnalysisObject:
        """Validate schema and return a validated AO instance.

        Returns
        -------
        AnalysisObject
            New AO instance wrapping a schema-validated dataset.

        Notes
        -----
        Validation checks that declared role dimensions, parameter coordinates,
        validity metadata, and extension metadata reference objects that still
        exist in the dataset.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"value": ("sample", [1.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> isinstance(ao.validate_schema(), AnalysisObject)
        True
        """
        return self._rewrap_dataset(_validate_schema(self._data), validate=True)


from .reducer_ops.surface import install_analysis_object_reducers as _install_analysis_object_reducers

_install_analysis_object_reducers(AnalysisObject)
