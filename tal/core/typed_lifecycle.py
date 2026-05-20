from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar, Literal, Self

import xarray as xr

from .analysis_object import AnalysisObject

LifecyclePhase = Literal["init", "from_validated", "from_unvalidated"]


@dataclass(frozen=True)
class TypedLifecycleContext:
    owner: str
    phase: LifecyclePhase
    options: object | None = None


SourceCoercer = Callable[[object, TypedLifecycleContext], AnalysisObject]
DatasetHook = Callable[[xr.Dataset, TypedLifecycleContext], xr.Dataset]
EnforceHook = Callable[[xr.Dataset, TypedLifecycleContext], None]


def default_coerce_source(value: object, ctx: TypedLifecycleContext) -> AnalysisObject:
    from .orchestration.inputs import coerce_analysis_object_input

    return coerce_analysis_object_input(value, owner=ctx.owner)


def identity_init_options(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    _ = ctx
    return ds


def identity_normalize(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    _ = ctx
    return ds


def no_op_enforce(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
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
    LIFECYCLE: ClassVar[TypedLifecycleSpec]

    def __init__(self, data: object) -> None:
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
        AnalysisObject.__init__(self, source.unsafe_data)
        self._bind_from_hook(spec.apply_init_options(self.unsafe_data, ctx), ctx=ctx, hook="apply_init_options")
        self._run_typed_lifecycle(ctx)

    def _run_typed_lifecycle(self, ctx: TypedLifecycleContext) -> None:
        spec = self.__class__._lifecycle_spec()
        self._bind_from_hook(spec.normalize(self.unsafe_data, ctx), ctx=ctx, hook="normalize")
        spec.enforce(self.unsafe_data, ctx)

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> Self:
        obj = super()._from_validated(ds)
        obj._run_typed_lifecycle(cls._lifecycle_context(phase="from_validated"))
        return obj

    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray) -> Self:
        obj = super()._from_unvalidated(ds)
        obj._run_typed_lifecycle(cls._lifecycle_context(phase="from_unvalidated"))
        return obj

    def _normalize_metadata(self, *, owner: str) -> None:
        ctx = TypedLifecycleContext(owner=owner, phase="init", options=None)
        spec = self.__class__._lifecycle_spec()
        self._bind_from_hook(spec.normalize(self.unsafe_data, ctx), ctx=ctx, hook="normalize")

    def _enforce_invariants(self, *, owner: str) -> None:
        ctx = TypedLifecycleContext(owner=owner, phase="init", options=None)
        self.__class__._lifecycle_spec().enforce(self.unsafe_data, ctx)


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
