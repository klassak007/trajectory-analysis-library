"""Manual benchmark for parameter-map backends and prepared evaluation."""

from __future__ import annotations

import argparse
from time import perf_counter

import numpy as np
import xarray as xr
from _numba_bench import cold_subprocess, warm_median

from tal.core import AnalysisObject
from tal.core.param_engine import ParamMapOptions
from tal.core.param_engine.backends import (
    PARAM_BOUNDS_BACKEND_NUMBA,
    PARAM_BOUNDS_BACKEND_NUMPY_BLOCK,
    PARAM_MAP_BACKEND_NUMBA,
    PARAM_MAP_BACKEND_NUMPY_BLOCK,
    bounds_block_backend,
    map_block_backend,
)
from tal.core.param_engine.prepared import prepare_param_evaluation


def _case_data(name: str) -> tuple[np.ndarray, ...]:
    if name == "many-short":
        rows, sequence, query_size = 4096, 16, 8
    elif name == "fewer-long":
        rows, sequence, query_size = 128, 1024, 128
    else:
        raise ValueError(f"unknown benchmark case {name!r}")
    base = np.linspace(0.0, 1.0, sequence)
    param = np.broadcast_to(base, (rows, sequence)).copy()
    valid = np.ones_like(param, dtype=bool)
    query = np.broadcast_to(np.linspace(0.05, 0.95, query_size), (rows, query_size)).copy()
    start = np.full(rows, 0.2)
    stop = np.full(rows, 0.8)
    return param, valid, query, start, stop


def _map_backend(param: np.ndarray, valid: np.ndarray, query: np.ndarray, backend: str) -> object:
    return map_block_backend(
        param,
        valid,
        query,
        method="linear",
        dup_code=0,
        backend=backend,
    )


def _bounds_backend(
    param: np.ndarray,
    valid: np.ndarray,
    start: np.ndarray,
    stop: np.ndarray,
    backend: str,
) -> object:
    return bounds_block_backend(param, valid, start, stop, backend=backend)


def _prepared_evaluation(param: np.ndarray, valid: np.ndarray, query: np.ndarray) -> object:
    rows = np.arange(param.shape[0])
    return prepare_param_evaluation(
        param=xr.DataArray(param, dims=("row", "sample"), coords={"row": rows}),
        query=xr.DataArray(query, dims=("row", "query"), coords={"row": rows}),
        sequence_dim="sample",
        batch_dims=("row",),
        batch_coords=xr.Coordinates({"row": rows}),
        valid_mask=xr.DataArray(valid, dims=("row", "sample"), coords={"row": rows}),
        options=ParamMapOptions(),
        param_kind="numeric",
        query_dim="query",
    )


def _public_evaluation(param: np.ndarray, query: np.ndarray) -> object:
    rows = np.arange(param.shape[0])
    source = AnalysisObject.from_data(
        xr.Dataset(
            {"value": (("row", "sample"), param)},
            coords={
                "row": rows,
                "sample": np.arange(param.shape[1]),
                "time": (("row", "sample"), param),
            },
        ),
        sequence_dim="sample",
        batch_dims=("row",),
        param_coord="time",
    )
    target = xr.DataArray(query, dims=("row", "query"), coords={"row": rows})
    return source.param.at(target, validate=False)


def _elapsed(operation, *args) -> float:
    started = perf_counter()
    operation(*args)
    return perf_counter() - started


def _cold_elapsed(case_name: str, operation: str) -> float:
    return cold_subprocess(
        __file__,
        ("--cold-case", case_name, "--cold-operation", operation),
        cache_prefix="tal-param-benchmark-",
    )


def _run_cold(case_name: str, operation: str) -> None:
    param, valid, query, start, stop = _case_data(case_name)
    if operation == "map-numba":
        elapsed = _elapsed(_map_backend, param, valid, query, PARAM_MAP_BACKEND_NUMBA)
    elif operation == "bounds-numba":
        elapsed = _elapsed(_bounds_backend, param, valid, start, stop, PARAM_BOUNDS_BACKEND_NUMBA)
    elif operation == "prepared":
        elapsed = _elapsed(_prepared_evaluation, param, valid, query)
    else:
        raise ValueError(f"unknown benchmark operation {operation!r}")
    print(f"{elapsed:.9f}")


def _report_case(case_name: str) -> None:
    param, valid, query, start, stop = _case_data(case_name)
    _map_backend(param, valid, query, PARAM_MAP_BACKEND_NUMBA)
    _bounds_backend(param, valid, start, stop, PARAM_BOUNDS_BACKEND_NUMBA)
    routes = (
        ("map-numpy", _map_backend, (param, valid, query, PARAM_MAP_BACKEND_NUMPY_BLOCK)),
        ("map-numba", _map_backend, (param, valid, query, PARAM_MAP_BACKEND_NUMBA)),
        ("bounds-numpy", _bounds_backend, (param, valid, start, stop, PARAM_BOUNDS_BACKEND_NUMPY_BLOCK)),
        ("bounds-numba", _bounds_backend, (param, valid, start, stop, PARAM_BOUNDS_BACKEND_NUMBA)),
        ("prepared", _prepared_evaluation, (param, valid, query)),
        ("public", _public_evaluation, (param, query)),
    )
    print(f"\ncase={case_name}; rows={param.shape[0]}; sample={param.shape[1]}; query={query.shape[1]}")
    for name, operation, args in routes:
        print(f"  route={name}; warm_median={warm_median(operation, *args):.6f}s")
    for operation in ("map-numba", "bounds-numba", "prepared"):
        print(f"  route={operation}; cold={_cold_elapsed(case_name, operation):.6f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cold-case")
    parser.add_argument("--cold-operation")
    args = parser.parse_args()
    if args.cold_case or args.cold_operation:
        if not args.cold_case or not args.cold_operation:
            raise SystemExit("--cold-case and --cold-operation must be provided together")
        _run_cold(args.cold_case, args.cold_operation)
        return
    _report_case("many-short")
    _report_case("fewer-long")


if __name__ == "__main__":
    main()
