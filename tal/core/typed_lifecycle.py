from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

import xarray as xr

from .analysis_object import AnalysisObject
from .dataset_ownership import (
    analysis_object_dataset,
    couple_dataset_resource,
    metadata_isolated_dataset,
)

LifecyclePhase = Literal["init", "from_validated", "from_unvalidated"]


@dataclass(frozen=True)
class TypedLifecycleContext:
    """Runtime context passed to typed AO lifecycle hooks.

    Parameters
    ----------
    owner : str
        Owner string used by hooks when raising deterministic diagnostics.
    phase : str
        Lifecycle path currently invoking the hook: ``"init"``,
        ``"from_validated"``, or ``"from_unvalidated"``.
    options : object | None, optional
        Constructor-specific options supplied by a subclass.

    Notes
    -----
    Hooks should use ``owner`` in error messages and should treat ``phase`` as
    the validation policy context. In particular, a normalization hook can avoid
    schema validation during ``"from_unvalidated"`` rewraps while still
    enforcing cheap subtype invariants.

    Examples
    --------
    >>> from tal.core.typed_lifecycle import TypedLifecycleContext
    >>> ctx = TypedLifecycleContext(owner="thermal.Temperature.__init__", phase="init")
    >>> (ctx.owner, ctx.phase, ctx.options)
    ('thermal.Temperature.__init__', 'init', None)
    """

    owner: str
    phase: LifecyclePhase
    options: object | None = None


SourceCoercer = Callable[[object, TypedLifecycleContext], AnalysisObject]
DatasetHook = Callable[[xr.Dataset, TypedLifecycleContext], xr.Dataset]
EnforceHook = Callable[[xr.Dataset, TypedLifecycleContext], None]


def default_coerce_source(value: object, ctx: TypedLifecycleContext) -> AnalysisObject:
    """Coerce a typed lifecycle source with the canonical AO input boundary.

    Parameters
    ----------
    value : object
        Source object supplied to a ``TypedAnalysisObject`` constructor.
    ctx : TypedLifecycleContext
        Lifecycle context providing the owner string for diagnostics.

    Returns
    -------
    AnalysisObject
        Coerced analysis object.

    Raises
    ------
    TypeError
        If ``value`` is not an ``AnalysisObject``, ``xarray.Dataset``, or
        ``xarray.DataArray``.

    See Also
    --------
    tal.core.orchestration.inputs.coerce_analysis_object_input

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.typed_lifecycle import TypedLifecycleContext, default_coerce_source
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> ctx = TypedLifecycleContext(owner="thermal.Temperature.__init__", phase="init")
    >>> isinstance(default_coerce_source(ds, ctx), AnalysisObject)
    True
    """
    from .orchestration.inputs import coerce_analysis_object_input

    return coerce_analysis_object_input(value, owner=ctx.owner)


def _prepare_typed_promotion(source: AnalysisObject, *, owner: str) -> xr.Dataset:
    candidate = AnalysisObject._prepared_ingress_dataset(
        analysis_object_dataset(source)
    )
    return metadata_isolated_dataset(candidate, owner=owner)


def _finish_typed_promotion(source: AnalysisObject, target: xr.Dataset) -> None:
    couple_dataset_resource(analysis_object_dataset(source), target)


def identity_init_options(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    """Return the input dataset unchanged for option-free lifecycle specs.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset currently bound to the typed object.
    ctx : TypedLifecycleContext
        Lifecycle context for the hook invocation.

    Returns
    -------
    xr.Dataset
        The same dataset object, unchanged.

    Notes
    -----
    Use this default when a subtype does not need constructor options to rewrite
    schema or payload metadata.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.typed_lifecycle import TypedLifecycleContext, identity_init_options
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> ctx = TypedLifecycleContext(owner="thermal.Temperature.__init__", phase="init")
    >>> identity_init_options(ds, ctx) is ds
    True
    """
    _ = ctx
    return ds


def identity_normalize(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    """Return the input dataset unchanged for metadata-neutral subtypes.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset currently bound to the typed object.
    ctx : TypedLifecycleContext
        Lifecycle context for the hook invocation.

    Returns
    -------
    xr.Dataset
        The same dataset object, unchanged.

    Notes
    -----
    Use this default when a subtype does not need to add or normalize metadata
    under ``ds.attrs["tal"]``.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.typed_lifecycle import TypedLifecycleContext, identity_normalize
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> ctx = TypedLifecycleContext(owner="thermal.Temperature.__init__", phase="init")
    >>> identity_normalize(ds, ctx) is ds
    True
    """
    _ = ctx
    return ds


def no_op_enforce(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    """Skip subtype-specific invariant checks.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset currently bound to the typed object.
    ctx : TypedLifecycleContext
        Lifecycle context for the hook invocation.

    Returns
    -------
    None
        This hook performs no validation.

    Notes
    -----
    Use this default only when the base ``AnalysisObject`` schema contract is
    sufficient for the subtype. Subtypes with domain-specific shape, role, or
    metadata requirements should provide an ``enforce`` hook.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core.typed_lifecycle import TypedLifecycleContext, no_op_enforce
    >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
    >>> ctx = TypedLifecycleContext(owner="thermal.Temperature.__init__", phase="init")
    >>> no_op_enforce(ds, ctx) is None
    True
    """
    _ = (ds, ctx)
    return None


def _require_non_empty_string(value: object, *, field: str) -> None:
    if isinstance(value, str) and value.strip():
        return
    raise ValueError(f"TypedLifecycleSpec: {field} must be a non-empty string.")


def _require_callable(value: object, *, field: str) -> None:
    if callable(value):
        return
    raise TypeError(f"TypedLifecycleSpec: {field} must be callable.")


@dataclass(frozen=True)
class TypedLifecycleSpec:
    """Declare the lifecycle hooks for a typed ``AnalysisObject`` subclass.

    Parameters
    ----------
    type_name : str
        Human-readable subtype name used by diagnostics and documentation.
    owner_prefix : str
        Public owner prefix used to construct default lifecycle owner strings.
    coerce_source : SourceCoercer, optional
        Hook that converts constructor input into an ``AnalysisObject``.
    apply_init_options : DatasetHook, optional
        Hook that applies constructor-specific options before normalization.
    normalize : DatasetHook, optional
        Hook that adds or normalizes subtype metadata.
    enforce : EnforceHook, optional
        Hook that checks subtype invariants.

    Raises
    ------
    TypeError
        If a lifecycle hook is not callable.
    ValueError
        If ``type_name`` or ``owner_prefix`` is empty.

    Notes
    -----
    Hooks run in a fixed order during construction: source coercion,
    initialization options, metadata normalization, and invariant enforcement.
    The normalize and enforce hooks also run for typed ``_from_validated`` and
    ``_from_unvalidated`` rewrap paths.

    Examples
    --------
    >>> from tal.core.typed_lifecycle import TypedLifecycleSpec
    >>> spec = TypedLifecycleSpec(type_name="Temperature", owner_prefix="thermal.Temperature")
    >>> (spec.type_name, spec.normalize.__name__, spec.enforce.__name__)
    ('Temperature', 'identity_normalize', 'no_op_enforce')
    """

    type_name: str
    owner_prefix: str
    coerce_source: SourceCoercer = default_coerce_source
    apply_init_options: DatasetHook = identity_init_options
    normalize: DatasetHook = identity_normalize
    enforce: EnforceHook = no_op_enforce

    def __post_init__(self) -> None:
        _require_non_empty_string(self.type_name, field="type_name")
        _require_non_empty_string(self.owner_prefix, field="owner_prefix")
        _require_callable(self.coerce_source, field="coerce_source")
        _require_callable(self.apply_init_options, field="apply_init_options")
        _require_callable(self.normalize, field="normalize")
        _require_callable(self.enforce, field="enforce")


class TypedAnalysisObject(AnalysisObject):
    """Base class for domain-specific typed analysis objects.

    Subclasses define a class-level ``LIFECYCLE`` with ``TypedLifecycleSpec``.
    The lifecycle centralizes source coercion, constructor options, metadata
    normalization, invariant enforcement, and self-typed rewrap behavior.

    Parameters
    ----------
    data : object
        Source accepted by the subclass lifecycle. The default source coercer
        accepts ``AnalysisObject``, ``xarray.Dataset``, and ``xarray.DataArray``.

    Notes
    -----
    ``TypedAnalysisObject`` remains an ``AnalysisObject``: data is stored as an
    xarray dataset and TAL metadata lives in ``ds.attrs["tal"]``. Lifecycle
    invariant hooks should be inexpensive metadata checks so Dask-backed payload
    arrays remain lazy.

    Examples
    --------
    >>> import xarray as xr
    >>> from tal.core import AnalysisObject
    >>> from tal.core.schema import merge_schema
    >>> from tal.core.schema_read import read_roles
    >>> from tal.core.typed_lifecycle import (
    ...     TypedAnalysisObject,
    ...     TypedLifecycleContext,
    ...     TypedLifecycleSpec,
    ... )
    >>> def normalize_thermal(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    ...     patch = {"ext": {"thermal": {"kind": "temperature"}}}
    ...     return merge_schema(ds, patch, validate=ctx.phase != "from_unvalidated")
    >>> def enforce_temperature(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    ...     declared, sequence_dim, _, _ = read_roles(ds)
    ...     if not declared or sequence_dim != "sample":
    ...         raise ValueError(f"{ctx.owner}: expected sample sequence roles.")
    >>> class Temperature(TypedAnalysisObject):
    ...     LIFECYCLE = TypedLifecycleSpec(
    ...         type_name="Temperature",
    ...         owner_prefix="thermal.Temperature",
    ...         normalize=normalize_thermal,
    ...         enforce=enforce_temperature,
    ...     )
    >>> base = AnalysisObject.from_data(
    ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
    ...     sequence_dim="sample",
    ...     core_dims=(),
    ...     validate=True,
    ... )
    >>> temp = Temperature(base)
    >>> temp.as_dataset().attrs["tal"]["ext"]["thermal"]["kind"]
    'temperature'
    """

    LIFECYCLE: ClassVar[TypedLifecycleSpec]

    def __init__(self, data: object) -> None:
        """Construct a typed AO by running the subclass lifecycle.

        Parameters
        ----------
        data : object
            Source accepted by the subclass lifecycle.

        Raises
        ------
        TypeError
            If ``LIFECYCLE`` is missing or source coercion returns a non-AO.
        ValueError
            If lifecycle hooks reject subtype metadata or invariants.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleSpec
        >>> class Temperature(TypedAnalysisObject):
        ...     LIFECYCLE = TypedLifecycleSpec(
        ...         type_name="Temperature",
        ...         owner_prefix="thermal.Temperature",
        ...     )
        >>> temp = Temperature(xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}))
        >>> temp.as_dataset().attrs["tal"]["version"]
        1
        """
        self._init_typed(data, options=None)

    @classmethod
    def _lifecycle_spec(cls) -> TypedLifecycleSpec:
        spec = getattr(cls, "LIFECYCLE", None)
        if isinstance(spec, TypedLifecycleSpec):
            return spec
        raise TypeError(f"{cls.__name__}: LIFECYCLE must be TypedLifecycleSpec.")

    @classmethod
    def _lifecycle_owner(cls, *, phase: LifecyclePhase) -> str:
        spec = cls._lifecycle_spec()
        if phase == "init":
            return f"{spec.owner_prefix}.__init__"
        return f"{cls.__name__}._{phase}"

    @classmethod
    def _lifecycle_context(
        cls,
        *,
        phase: LifecyclePhase,
        options: object | None = None,
    ) -> TypedLifecycleContext:
        return TypedLifecycleContext(
            owner=cls._lifecycle_owner(phase=phase),
            phase=phase,
            options=options,
        )

    @staticmethod
    def _require_dataset(value: object, *, owner: str, hook: str) -> xr.Dataset:
        if isinstance(value, xr.Dataset):
            return value
        raise TypeError(f"{owner}: lifecycle hook {hook} returned {type(value).__name__}; expected xr.Dataset.")

    def _bind_from_hook(self, value: object, *, ctx: TypedLifecycleContext, hook: str) -> None:
        self._bind_dataset(self._require_dataset(value, owner=ctx.owner, hook=hook))

    def _init_typed(self, data: object, *, options: object | None = None) -> None:
        spec = self.__class__._lifecycle_spec()
        ctx = self.__class__._lifecycle_context(phase="init", options=options)
        source = spec.coerce_source(data, ctx)
        if not isinstance(source, AnalysisObject):
            raise TypeError(f"{ctx.owner}: lifecycle source coercer returned {type(source).__name__}; expected AnalysisObject.")
        self._bind_dataset(_prepare_typed_promotion(source, owner=ctx.owner))
        initialized = spec.apply_init_options(analysis_object_dataset(self), ctx)
        self._bind_from_hook(initialized, ctx=ctx, hook="apply_init_options")
        self._run_typed_lifecycle(ctx)
        _finish_typed_promotion(source, analysis_object_dataset(self))

    def _run_typed_lifecycle(self, ctx: TypedLifecycleContext) -> None:
        spec = self.__class__._lifecycle_spec()
        source = analysis_object_dataset(self)
        self._bind_from_hook(spec.normalize(source, ctx), ctx=ctx, hook="normalize")
        spec.enforce(analysis_object_dataset(self), ctx)

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> Self:
        """Bind a validated dataset and run typed lifecycle hooks.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            Dataset or data array already validated by the caller.

        Returns
        -------
        Self
            Instance of the concrete typed subclass.

        Notes
        -----
        This is a typed rewrap boundary for operation owners that have already
        selected ``validate=True``. The subtype ``normalize`` and ``enforce``
        hooks still run after binding.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleSpec
        >>> class Temperature(TypedAnalysisObject):
        ...     LIFECYCLE = TypedLifecycleSpec(
        ...         type_name="Temperature",
        ...         owner_prefix="thermal.Temperature",
        ...     )
        >>> base = AnalysisObject.from_data(
        ...     xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]}),
        ...     sequence_dim="sample",
        ...     core_dims=(),
        ...     validate=True,
        ... )
        >>> isinstance(Temperature._from_validated(base.as_dataset()), Temperature)
        True
        """
        obj = super()._from_validated(ds)
        obj._run_typed_lifecycle(cls._lifecycle_context(phase="from_validated"))
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray, *, schema_prepared: bool = False) -> Self:
        """Bind a dataset without core schema validation and run lifecycle hooks.

        Parameters
        ----------
        ds : xr.Dataset | xr.DataArray
            Dataset or data array produced by an owner that intentionally chose
            the unvalidated rewrap path.
        schema_prepared : bool, optional
            Whether the trusted caller already completed schema preparation.

        Returns
        -------
        Self
            Instance of the concrete typed subclass.

        Notes
        -----
        This path skips core schema validation but does not bypass the typed
        lifecycle. Subtype ``normalize`` and ``enforce`` hooks still run.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core.typed_lifecycle import TypedAnalysisObject, TypedLifecycleSpec
        >>> class Temperature(TypedAnalysisObject):
        ...     LIFECYCLE = TypedLifecycleSpec(
        ...         type_name="Temperature",
        ...         owner_prefix="thermal.Temperature",
        ...     )
        >>> ds = xr.Dataset({"celsius": ("sample", [20.0])}, coords={"sample": [0]})
        >>> isinstance(Temperature._from_unvalidated(ds), Temperature)
        True
        """
        obj = super()._from_unvalidated(ds, schema_prepared=schema_prepared)
        obj._run_typed_lifecycle(cls._lifecycle_context(phase="from_unvalidated"))
        return obj

    def _normalize_metadata(self, *, owner: str) -> None:
        ctx = TypedLifecycleContext(owner=owner, phase="init", options=None)
        spec = self.__class__._lifecycle_spec()
        self._bind_from_hook(spec.normalize(analysis_object_dataset(self), ctx), ctx=ctx, hook="normalize")

    def _enforce_invariants(self, *, owner: str) -> None:
        ctx = TypedLifecycleContext(owner=owner, phase="init", options=None)
        self.__class__._lifecycle_spec().enforce(analysis_object_dataset(self), ctx)


__all__ = [
    "DatasetHook",
    "EnforceHook",
    "LifecyclePhase",
    "SourceCoercer",
    "TypedAnalysisObject",
    "TypedLifecycleContext",
    "TypedLifecycleSpec",
    "default_coerce_source",
    "identity_init_options",
    "identity_normalize",
    "no_op_enforce",
]
