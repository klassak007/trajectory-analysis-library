from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tal.spatial.ops.batched_path_execution import (
    _execute_arrays,
    execute_eager_batched_path,
)
from tal.spatial.ops.batched_path_finalize import commit_batched_position
from tal.spatial.ops.batched_path_inputs import (
    PackedBatchedPathInputs,
    pack_batched_path_inputs,
)
from tal.spatial.ops.batched_path_plan import PreparedBatchedPathExecution
from tal.utils.numba_support import _numba_available


@dataclass(frozen=True)
class ProductionBatchedRoute:
    plan: PreparedBatchedPathExecution
    packed: PackedBatchedPathInputs
    arrays: tuple[np.ndarray, np.ndarray | None]
    backend: str


def prepare_production_route(plan: PreparedBatchedPathExecution) -> ProductionBatchedRoute:
    """Prepare immutable production inputs outside route measurement."""
    backend = "numba" if _numba_available() else "scipy"
    packed = pack_batched_path_inputs(plan)
    arrays = _execute_arrays(plan, packed, backend=backend)
    return ProductionBatchedRoute(plan, packed, arrays, backend)


def production_pack(plan: PreparedBatchedPathExecution) -> PackedBatchedPathInputs:
    return pack_batched_path_inputs(plan)


def production_execute(route: ProductionBatchedRoute) -> tuple[np.ndarray, np.ndarray | None]:
    return _execute_arrays(route.plan, route.packed, backend=route.backend)


def production_finalize(route: ProductionBatchedRoute):
    return commit_batched_position(route.plan, route.arrays[0])


def production_dispatch(route: ProductionBatchedRoute):
    return execute_eager_batched_path(
        route.plan,
        backend=route.backend,
        owner="benchmarks.batched_path_execution",
    )


__all__ = [
    "ProductionBatchedRoute",
    "prepare_production_route",
    "production_dispatch",
    "production_execute",
    "production_finalize",
    "production_pack",
]
