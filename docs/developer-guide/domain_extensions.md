(developer-guide-domain-extensions)=
# Domain Extensions

A TAL domain extension is a small package layer that defines typed
`AnalysisObject` subclasses, claims metadata under `tal.ext.<namespace>`, and
delegates shared trajectory semantics to TAL core helpers. The domain package
owns its own meanings; TAL core preserves extension metadata but does not
interpret it.

Use this recipe when adding a domain such as thermal sensors, dynamics,
biomechanics, or other typed analysis surfaces.

## Dataset Anatomy

TAL stores trajectory semantics in `ds.attrs["tal"]`.

- `tal.core` stores shared roles such as `sequence_dim`, `batch_dims`,
  `core_dims`, parameter coordinates, and validity metadata.
- `tal.ext.<namespace>` stores domain-owned metadata. Choose one stable
  namespace for your package, for example `tal.ext.thermal`.
- Unknown extension namespaces must be preserved unless a user explicitly asks
  to remove them.

Declare core roles as data enters the domain. Add domain metadata with
`AnalysisObject.merge_schema(...)` or `tal.core.schema.merge_schema(...)`.

## Typed AO Anatomy

A typed AO is a thin `TypedAnalysisObject` subclass over an xarray dataset. Keep
the shape predictable:

- Storage comes from `AnalysisObject`: the typed class still exposes
  `unsafe_data` and keeps TAL schema in `ds.attrs["tal"]`.
- Ingress constructors such as `from_celsius(...)` should declare shared TAL
  roles with `AnalysisObject.from_data(...)`.
- `TypedLifecycleSpec.normalize` should add or normalize domain metadata under
  the domain namespace.
- `TypedLifecycleSpec.enforce` should check the domain invariants that make the
  subtype meaningful.
- Public operations should coerce inputs, resolve dataset context, run a pure
  xarray kernel, finalize schema, and return the typed subclass.

`TypedAnalysisObject` supplies the lifecycle and self-typed rewrap behavior, so
most domain classes only need a small `LIFECYCLE` spec plus domain-specific
constructors and methods.

### Type Annotations And Self

When defining custom constructors, subclass methods, or classmethod factories on
a `TypedAnalysisObject` subclass, use `typing.Self` as the return type. This
helps static type checkers and IDEs preserve the exact subclass type during
method chaining instead of widening the result to `TypedAnalysisObject` or
`AnalysisObject`.

Module-level operations should name their concrete public return type. Subclass
methods and classmethod factories should prefer `Self`; the examples below use
`from typing import Self` and annotate `from_celsius(...) -> Self`.

### Lifecycle Defaults

If your subclass does not require custom initialization arguments or metadata
normalization, omit those hooks from the `LIFECYCLE` spec. `TypedAnalysisObject`
uses core defaults automatically:

- `coerce_source` defaults to `default_coerce_source`, which runs standard
  `coerce_analysis_object_input(...)`.
- `apply_init_options` defaults to `identity_init_options`, which returns the
  dataset unchanged.
- `normalize` defaults to `identity_normalize`, which returns the dataset
  unchanged.
- `enforce` defaults to `no_op_enforce`, which skips subtype-specific invariant
  checks.

<!-- example-id: DOC_EXAMPLE_001_domain_extension_minimal_type_recipe -->
```python
from typing import Self

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.schema import merge_schema
from tal.core.schema_read import read_roles
from tal.core.typed_lifecycle import (
    TypedAnalysisObject,
    TypedLifecycleContext,
    TypedLifecycleSpec,
)


_THERMAL_SCHEMA = {"ext": {"thermal": {"quantity": "temperature", "unit": "degC"}}}


def normalize_temperature_metadata(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    return merge_schema(ds, _THERMAL_SCHEMA, validate=ctx.phase != "from_unvalidated")


def enforce_temperature_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{ctx.owner}: Temperature requires declared TAL roles.")
    if core_dims != ():
        raise ValueError(f"{ctx.owner}: Temperature expects scalar samples with no core dims.")
    if "temperature_c" not in ds.data_vars:
        raise ValueError(f"{ctx.owner}: expected variable 'temperature_c'.")


class Temperature(TypedAnalysisObject):
    LIFECYCLE = TypedLifecycleSpec(
        type_name="Temperature",
        owner_prefix="thermal.temperature",
        normalize=normalize_temperature_metadata,
        enforce=enforce_temperature_invariants,
    )

    @classmethod
    def from_celsius(cls, ds: xr.Dataset) -> Self:
        ao = AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("sensor",),
            core_dims=(),
            param_coord="time_s",
            validate=True,
        )
        return cls(ao)


sample = np.arange(3)
base = xr.Dataset(
    {"temperature_c": (("sensor", "sample"), [[20.0, 21.5, 22.0]])},
    coords={
        "sensor": ["probe_0"],
        "sample": sample,
        "time_s": (("sensor", "sample"), [[0.0, 0.5, 1.0]]),
    },
    attrs={"tal": {"version": 1, "core": {}, "ext": {"instrument": {"serial": "T-01"}}}},
)

temperature = Temperature.from_celsius(base)
schema = temperature.unsafe_data.attrs["tal"]
declared, sequence_dim, batch_dims, core_dims = read_roles(temperature.unsafe_data)

assert isinstance(temperature, Temperature)
assert declared is True
assert sequence_dim == "sample"
assert batch_dims == ("sensor",)
assert core_dims == ()
assert schema["ext"]["thermal"] == {"quantity": "temperature", "unit": "degC"}
assert schema["ext"]["instrument"] == {"serial": "T-01"}
```

## Operation Shape

Keep public operation entrypoints thin and predictable:

1. Coerce user input at the boundary with `coerce_operand(...)` when the
   argument has operand semantics.
2. Resolve runtime information with `resolve_dataset_context(...)`.
3. Run the kernel as pure array math on already-resolved xarray objects.
4. Finalize with shared owners such as `finalize_like(...)`.
5. Return the typed AO with `_from_validated(...)` or `_from_unvalidated(...)`
   according to the operation's validation policy.

The owner string passed through the operation should appear in diagnostics, so
users can identify which boundary rejected invalid input.

### The Owner Naming Convention

The owner string passed to core orchestration helpers should match the fully qualified public import path of the entrypoint calling the helper. This keeps
diagnostic messages aligned with user-facing code.

- Module-level operations: `<package>.<function_name>`, for example
  `thermal.bias_temperature`.
- Class methods: `<package>.<class_name>.<method_name>`, for example
  `thermal.Temperature.from_celsius`.
- Accessor properties: `<package>.<class_name>.<property_name>`, for example
  `thermal.Temperature.celsius_values`.

<!-- example-id: DOC_EXAMPLE_002_domain_extension_operation_recipe -->
```python
from typing import Self

import numpy as np
import xarray as xr

from tal.core import AnalysisObject
from tal.core.orchestration.context import DatasetContextOptions, resolve_dataset_context
from tal.core.orchestration.finalize import finalize_like
from tal.core.orchestration.inputs import coerce_operand
from tal.core.schema import merge_schema
from tal.core.schema_read import read_roles
from tal.core.typed_lifecycle import (
    TypedAnalysisObject,
    TypedLifecycleContext,
    TypedLifecycleSpec,
)


_THERMAL_SCHEMA = {"ext": {"thermal": {"quantity": "temperature", "unit": "degC"}}}


def normalize_temperature_metadata(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
    return merge_schema(ds, _THERMAL_SCHEMA, validate=ctx.phase != "from_unvalidated")


def enforce_temperature_invariants(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{ctx.owner}: Temperature requires declared TAL roles.")
    if core_dims != ():
        raise ValueError(f"{ctx.owner}: Temperature expects scalar samples with no core dims.")
    if "temperature_c" not in ds.data_vars:
        raise ValueError(f"{ctx.owner}: expected variable 'temperature_c'.")


class Temperature(TypedAnalysisObject):
    LIFECYCLE = TypedLifecycleSpec(
        type_name="Temperature",
        owner_prefix="thermal.temperature",
        normalize=normalize_temperature_metadata,
        enforce=enforce_temperature_invariants,
    )

    @classmethod
    def from_celsius(cls, ds: xr.Dataset) -> Self:
        ao = AnalysisObject.from_data(
            ds,
            sequence_dim="sample",
            batch_dims=("sensor",),
            core_dims=(),
            param_coord="time_s",
            validate=True,
        )
        return cls(ao)


def bias_temperature(value: object, offset_c: float, *, validate: bool = True) -> Temperature:
    owner = "thermal.bias_temperature"
    coerced = coerce_operand(value, owner=owner, label="temperature")
    if not isinstance(coerced, AnalysisObject):
        raise TypeError(f"{owner}: expected AnalysisObject-like temperature input.")
    context = resolve_dataset_context(
        coerced,
        owner=owner,
        options=DatasetContextOptions(
            require_roles=True,
            require_sequence_dim=True,
            select_numeric_var=True,
            require_single_numeric_var=True,
            allowed_core_arity=(0,),
            require_semantic_dims_in_var=True,
        ),
    )
    if context.var_name != "temperature_c" or context.data is None:
        raise ValueError(f"{owner}: expected a single variable named 'temperature_c'.")

    out = context.ds.copy(deep=False)
    out["temperature_c"] = context.data + float(offset_c)
    finalized = finalize_like(context.ao, out, validate=validate, owner=owner)
    if validate:
        return Temperature._from_validated(finalized.unsafe_data)
    return Temperature._from_unvalidated(finalized.unsafe_data)


sample = np.arange(3)
source = Temperature.from_celsius(
    xr.Dataset(
        {"temperature_c": (("sensor", "sample"), [[20.0, 21.5, 22.0]])},
        coords={
            "sensor": ["probe_0"],
            "sample": sample,
            "time_s": (("sensor", "sample"), [[0.0, 0.5, 1.0]]),
        },
    )
)

biased = bias_temperature(source, 2.0)
expected = xr.DataArray(
    [[22.0, 23.5, 24.0]],
    dims=("sensor", "sample"),
    coords={"sensor": ["probe_0"], "sample": sample, "time_s": (("sensor", "sample"), [[0.0, 0.5, 1.0]])},
    name="temperature_c",
)

xr.testing.assert_allclose(biased.unsafe_data["temperature_c"], expected)
assert isinstance(biased, Temperature)
assert read_roles(biased.unsafe_data)[1:] == ("sample", ("sensor",), ())
assert biased.unsafe_data.attrs["tal"]["ext"]["thermal"]["unit"] == "degC"

try:
    bias_temperature(object(), 1.0)
except TypeError as exc:
    assert "thermal.bias_temperature:" in str(exc)
else:
    raise AssertionError("bad operands must fail with owner-prefixed diagnostics")
```

## Core Helpers

Use core helpers for cross-domain trajectory mechanics:

- `coerce_operand(...)` for user-facing operands.
- `coerce_analysis_object_input(...)` for AO-like internal inputs.
- `resolve_dataset_context(...)` for roles, parameter coordinates, validity
  coordinates, and numeric variable selection.
- `finalize_like(...)` or `finalize_with_schema(...)` when returning an
  `AnalysisObject`.

Keep domain meanings in the domain package. A helper belongs in `tal.core` only
when it is generic trajectory machinery and has more than one active domain
consumer. `tal.core` must not import domain packages; domain packages may
import `tal.core`.

## Dask And Laziness

Preserve Dask-backed arrays in orchestration and operation paths. Avoid eager materialization in shared logic, including `.values`, `.item()`,
`np.asarray(...)`, and `.compute()`.

If an operation truly needs eager data, isolate that behavior in one small
boundary helper, document the reason in user-facing terms, and keep the rest of
the operation xarray-native.

### Invariant Performance And Fast-Paths

Subclass invariant hooks run on every typed object instantiation path by
default, including `_from_validated(...)` and `_from_unvalidated(...)`. Keep
`enforce` computationally inexpensive.

- Do check dataset shape, declared dimension roles, coordinate label existence,
  and variable names. These are cheap metadata checks.
- Do not check actual data values or numeric ranges when doing so would force
  eager evaluation of Dask arrays.
- When chaining multiple operations internally, `validate=False` can skip core
  TAL schema validation for intermediate steps, but the subclass lifecycle still
  enforces its own invariants.
- If profiling shows invariant checks on a hot path are a bottleneck, do not
  bypass them globally. Use a dedicated private rewrap helper with local
  documentation and performance regression coverage.

## Public Documentation And Coverage

Public documentation should explain the API without relying on project-private
notes. Any runnable example should include its imports, construct its own data,
and assert the behavior it is demonstrating.

For each documented example, add executable coverage that checks the same
roles, metadata, values, and failure behavior shown in the documentation.
Examples should be deterministic, short, and independent from each other.
