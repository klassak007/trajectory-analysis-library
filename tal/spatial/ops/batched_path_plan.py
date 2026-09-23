from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.orchestration.runtime_checks import select_single_numeric_var
from tal.core.param_engine.blocking import (
    LogicalRowBlockPlan,
    prepare_logical_row_blocks,
)
from tal.core.param_engine.physical_partition import (
    PhysicalRowPartitionPlan,
    prepare_physical_row_partitions,
)
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
)
from tal.frames import FramePath
from tal.utils.frame_schema import get_frames

from ..metadata import get_pose_rep
from ..metadata.common import _read_roles_block
from ..pose import Pose
from ..temporal.options import resolve_rotation_method
from .path_query_plan import PathOutputIntent, PreparedPathQuery, PreparedProviderQuery
from .pose_component_ops import resolve_pose_component_specs

PathStorageKind = Literal["eager", "dask", "unsupported"]
PayloadStorageKind = Literal["eager", "dask"]


class BatchedPathReason(Enum):
    """Stable internal classifications for the prospective batched route."""

    ELIGIBLE = "eligible"
    UNSUPPORTED_OUTPUT = "unsupported-output"
    NON_DYNAMIC = "non-dynamic"
    PARAMETER_KIND = "parameter-kind"
    TEMPORAL_POLICY = "temporal-policy"
    REPRESENTATION = "representation"
    PROVIDER_METADATA = "provider-metadata"
    PAYLOAD_LAYOUT = "payload-layout"
    STORAGE = "storage"


@dataclass(frozen=True)
class BatchedPathFinalizationIntent:
    """Output declarations needed by the later single-commit executor."""

    output: PathOutputIntent
    batch_dims: tuple[str, ...]
    query_dim: str
    sequence_dim: str
    param_name: str
    parent_frame: str
    child_frame: str | None
    expressed_in: str
    result_context: object | None


@dataclass(frozen=True)
class BatchedPathMetadataClassification:
    """Metadata result established before parameter-map backend selection."""

    reason: BatchedPathReason
    storage: PathStorageKind
    payload_storage: tuple[PayloadStorageKind, ...]
    map_is_lazy: bool


@dataclass(frozen=True)
class PreparedBatchedPathExecution:
    """Immutable metadata-only candidate for batched path execution."""

    path: FramePath
    query: PreparedPathQuery
    finalization: BatchedPathFinalizationIntent
    reason: BatchedPathReason
    storage: PathStorageKind
    payload_storage: tuple[PayloadStorageKind, ...]
    logical_rows: LogicalRowBlockPlan
    physical_rows: PhysicalRowPartitionPlan

    @property
    def eligible(self) -> bool:
        return self.reason is BatchedPathReason.ELIGIBLE

    @property
    def mixed_storage(self) -> bool:
        return len(self.payload_storage) > 1


def _array_storage(value: xr.DataArray) -> PathStorageKind:
    data = value.data
    if isinstance(data, np.ndarray):
        return "eager"
    chunks = getattr(data, "chunks", None)
    meta = getattr(data, "_meta", None)
    if chunks is not None and isinstance(meta, np.ndarray):
        return "dask"
    return "unsupported"


def _combined_storage(values: tuple[xr.DataArray, ...]) -> PathStorageKind:
    kinds = tuple(_array_storage(value) for value in values)
    if "unsupported" in kinds:
        return "unsupported"
    return "dask" if "dask" in kinds else "eager"


def _payload_storage(values: tuple[xr.DataArray, ...]) -> tuple[PayloadStorageKind, ...]:
    kinds = tuple(_array_storage(value) for value in values)
    return tuple(dict.fromkeys(kind for kind in kinds if kind != "unsupported"))


def _is_supported_payload_dtype(dtype: object) -> bool:
    try:
        resolved = np.dtype(dtype)
    except TypeError:
        return False
    return resolved in {np.dtype(np.float32), np.dtype(np.float64)}


def _tal_extensions(ds: xr.Dataset) -> Mapping[str, object]:
    tal = ds.attrs.get("tal")
    if not isinstance(tal, Mapping):
        return {}
    ext = tal.get("ext", {})
    return ext if isinstance(ext, Mapping) else {}


def _provider_metadata_is_canonical(
    ds: xr.Dataset,
    *,
    component_vars: tuple[str, str],
    structural_names: frozenset[str],
) -> bool:
    if set(ds.attrs) != {"tal"} or ds.encoding:
        return False
    if set(ds.data_vars) != set(component_vars):
        return False
    if any(value.attrs or value.encoding for value in ds.variables.values()):
        return False
    if set(_tal_extensions(ds)) - {"components", "frames", "spatial"}:
        return False
    if _read_roles_block(ds, owner="spatial.path_solve.pose"):
        return False
    return all(str(name) in structural_names for name in ds.coords)


def _provider_arrays(item: PreparedProviderQuery) -> tuple[xr.DataArray, xr.DataArray] | None:
    if item.projection is None:
        return None
    value = item.projection.value
    if not isinstance(value, Pose):
        return None
    ds = analysis_object_dataset(value)
    if get_pose_rep(ds, owner="spatial.path_solve.pose") != "components":
        return None
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    specs = resolve_pose_component_specs(
        ds,
        owner="spatial.path_solve.pose",
        core_dims=core_dims,
    )
    (position_dim, position_var), (rotation_dim, rotation_var) = specs
    expected = (
        frozenset((*batch_dims, sequence_dim, position_dim)),
        frozenset((*batch_dims, sequence_dim, rotation_dim)),
    )
    arrays = (ds[position_var], ds[rotation_var])
    if any(
        len(value.dims) != len(dims) or frozenset(value.dims) != dims
        for value, dims in zip(arrays, expected, strict=True)
    ):
        return None
    return arrays


def _provider_reason(item: PreparedProviderQuery) -> BatchedPathReason | None:
    if item.required.source_representation not in {None, "components"}:
        return BatchedPathReason.REPRESENTATION
    arrays = _provider_arrays(item)
    if arrays is None:
        return BatchedPathReason.PAYLOAD_LAYOUT
    ds = analysis_object_dataset(item.projection.value)
    _, sequence_dim, batch_dims, core_dims = read_roles(ds)
    (_, position_var), (_, rotation_var) = resolve_pose_component_specs(
        ds, owner="spatial.path_solve.pose", core_dims=core_dims,
    )
    structural = {sequence_dim, *batch_dims, *core_dims}
    structural.update(filter(None, (read_param_coord_name(ds), read_sequence_size_coord_name(ds))))
    metadata_ok = _provider_metadata_is_canonical(
        ds,
        component_vars=(position_var, rotation_var),
        structural_names=frozenset(structural),
    )
    if not metadata_ok:
        return BatchedPathReason.PROVIDER_METADATA
    if any(not _is_supported_payload_dtype(value.dtype) for value in arrays):
        return BatchedPathReason.PAYLOAD_LAYOUT
    return None


def _caller_payload(plan: PreparedPathQuery, intent: PathOutputIntent) -> xr.DataArray | None:
    topology = plan.topology
    if intent != "position" or topology is None or topology.caller is None:
        return None
    ds = topology.caller.ds
    name = select_single_numeric_var(ds, owner="spatial.path_solve.pose", what="Position caller")
    value = ds[name]
    if topology.sequence_dim == topology.query_dim:
        return value
    return value.rename({topology.sequence_dim: topology.query_dim})


def _caller_array(plan: PreparedPathQuery, intent: PathOutputIntent) -> xr.DataArray | None:
    value = _caller_payload(plan, intent)
    return value if value is not None and _is_supported_payload_dtype(value.dtype) else None


def _prepared_arrays(plan: PreparedPathQuery, intent: PathOutputIntent) -> tuple[xr.DataArray, ...]:
    if plan.topology is None:
        return ()
    values: list[xr.DataArray] = [plan.topology.query]
    caller = _caller_payload(plan, intent)
    if caller is not None:
        values.append(caller)
    for item in plan.items:
        arrays = _provider_arrays(item)
        if arrays is not None:
            values.extend(arrays)
        for evaluation in item.evaluations:
            mapping = evaluation.param_map
            values.extend((mapping.i0, mapping.i1, mapping.alpha, mapping.valid))
    return tuple(values)


def _metadata_arrays(plan: PreparedPathQuery, intent: PathOutputIntent) -> tuple[xr.DataArray, ...]:
    if plan.topology is None:
        return ()
    values: list[xr.DataArray] = [plan.topology.query]
    caller = _caller_payload(plan, intent)
    if caller is not None:
        values.append(caller)
    for item in plan.items:
        arrays = _provider_arrays(item)
        if arrays is not None:
            values.extend(arrays)
        if item.context is not None:
            values.extend((item.context.spec.coord, item.context.valid_mask))
    return tuple(values)


def _payload_arrays(plan: PreparedPathQuery, intent: PathOutputIntent) -> tuple[xr.DataArray, ...]:
    values: list[xr.DataArray] = []
    caller = _caller_payload(plan, intent)
    if caller is not None:
        values.append(caller)
    for item in plan.items:
        arrays = _provider_arrays(item)
        if arrays is not None:
            values.extend(arrays)
    return tuple(values)


def _temporal_is_supported(item: PreparedProviderQuery) -> bool:
    return (
        item.temporal.position_opts.method == "linear"
        and resolve_rotation_method(item.temporal.rotation_opts) == "slerp"
    )


def _eligibility(
    plan: PreparedPathQuery,
    intent: PathOutputIntent,
    arrays: tuple[xr.DataArray, ...],
) -> tuple[BatchedPathReason, PathStorageKind, tuple[PayloadStorageKind, ...]]:
    payloads = _payload_arrays(plan, intent)
    payload_storage = _payload_storage(payloads)
    storage = _combined_storage(payloads or arrays)
    if intent not in {"pose", "position"}:
        return BatchedPathReason.UNSUPPORTED_OUTPUT, storage, payload_storage
    if not plan.items or any(item.required.topology != "dynamic" for item in plan.items):
        return BatchedPathReason.NON_DYNAMIC, storage, payload_storage
    if plan.topology is None or plan.topology.param_kind != "numeric":
        return BatchedPathReason.PARAMETER_KIND, storage, payload_storage
    if any(not _temporal_is_supported(item) for item in plan.items):
        return BatchedPathReason.TEMPORAL_POLICY, storage, payload_storage
    for item in plan.items:
        provider_reason = _provider_reason(item)
        if provider_reason is not None:
            return provider_reason, storage, payload_storage
    if intent == "position" and _caller_array(plan, intent) is None:
        return BatchedPathReason.PAYLOAD_LAYOUT, storage, payload_storage
    if _combined_storage(arrays) == "unsupported":
        return BatchedPathReason.STORAGE, "unsupported", payload_storage
    return BatchedPathReason.ELIGIBLE, storage, payload_storage


def classify_batched_path_metadata(
    plan: PreparedPathQuery,
) -> BatchedPathMetadataClassification:
    """Classify a batched request before parameter-map backend selection."""
    arrays = _metadata_arrays(plan, plan.output_intent)
    reason, storage, payload_storage = _eligibility(
        plan,
        plan.output_intent,
        arrays,
    )
    return BatchedPathMetadataClassification(
        reason,
        storage,
        payload_storage,
        _combined_storage(arrays) == "dask",
    )


def prepare_batched_path_execution(
    path: FramePath,
    query: PreparedPathQuery,
) -> PreparedBatchedPathExecution | None:
    """Prepare a batched fused candidate while public dispatch stays generic."""
    topology = query.topology
    if topology is None or not topology.batch_dims:
        return None
    output = query.output_intent
    arrays = _prepared_arrays(query, output)
    logical = prepare_logical_row_blocks(
        *arrays,
        excluded_dims=frozenset(),
        included_dims=(*topology.batch_dims, topology.query_dim),
        fastest_dim=topology.query_dim,
    )
    physical = prepare_physical_row_partitions(logical, *arrays)
    classification = query.batched_classification or classify_batched_path_metadata(query)
    parent_frame = path.nodes[-1].id
    child_frame = path.nodes[0].id
    if output == "position" and topology.caller is not None:
        _, child_frame = get_frames(topology.caller.ds)
    finalization = BatchedPathFinalizationIntent(
        output,
        topology.batch_dims,
        topology.query_dim,
        topology.sequence_dim,
        topology.param_name,
        parent_frame,
        child_frame,
        parent_frame,
        query.result_context,
    )
    return PreparedBatchedPathExecution(
        path,
        query,
        finalization,
        classification.reason,
        classification.storage,
        classification.payload_storage,
        logical,
        physical,
    )


__all__ = [
    "BatchedPathMetadataClassification",
    "BatchedPathReason",
    "PreparedBatchedPathExecution",
    "classify_batched_path_metadata",
    "prepare_batched_path_execution",
]
