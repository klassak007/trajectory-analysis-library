import numpy as np
import pytest
import xarray as xr
from pathlib import Path

from tal.core import AnalysisObject, ParamEvalOptions
from tal.core.param_engine import ParamMapOptions, build_param_map
import tal.core.param_ops.evaluate as eval_mod


def _ao_two_vars() -> AnalysisObject:
    ds = xr.Dataset(
        data_vars={
            "v1": (("sample",), [0.0, 10.0, 20.0, 30.0]),
            "v2": (("sample",), [1.0, 11.0, 21.0, 31.0]),
        },
        coords={"sample": [0, 1, 2, 3], "tau": ("sample", [0.0, 0.5, 1.0, 1.5])},
    )
    return AnalysisObject.from_data(ds, sequence_dim="sample", batch_dims=(), core_dims=(), param_coord="tau")


def test_param_perf_015_map_reuse_across_multiple_vars(monkeypatch) -> None:
    """ID: PARAM_PERF_015_map_reuse_across_multiple_vars."""
    ao = _ao_two_vars()
    calls = {"n": 0}
    original = eval_mod.build_param_map

    def _count(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(eval_mod, "build_param_map", _count)
    out = ao.param.at([0.25, 1.25], opts=ParamEvalOptions(method="linear"))
    assert calls["n"] == 1
    np.testing.assert_allclose(out.data["v1"].values, [5.0, 25.0])
    np.testing.assert_allclose(out.data["v2"].values, [6.0, 26.0])


def test_param_perf_016_integral_param_mapping_preserves_dask_laziness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: PARAM_PERF_016_integral_param_mapping_preserves_dask_laziness."""
    da = pytest.importorskip("dask.array")
    base = 2**53
    param = xr.DataArray(
        da.from_array(np.asarray([base, base + 1], dtype="int64"), chunks=1),
        dims=("sample",),
    )
    query = xr.DataArray(
        da.from_array(np.asarray([base + 1], dtype="int64"), chunks=1),
        dims=("query",),
    )

    with monkeypatch.context() as guarded:
        guarded.setattr(da.Array, "compute", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("eager")))
        pmap = build_param_map(
            param=param,
            query=query,
            sequence_dim="sample",
            query_dim="query",
            options=ParamMapOptions(method="nearest"),
        )
        assert hasattr(pmap.i0.data, "chunks")
        assert hasattr(pmap.valid.data, "chunks")

    assert int(pmap.i0.compute().item()) == 1
    assert bool(pmap.valid.compute().item()) is True


def test_param_ops_063_sequence_size_finalize_policy_single_owner() -> None:
    """ID: PARAM_OPS_063_sequence_size_finalize_policy_single_owner."""
    param_finalize = Path("tal/core/param_ops/finalize.py").read_text(encoding="utf-8")
    validity_finalize = Path("tal/core/validity_finalize.py").read_text(encoding="utf-8")
    assert "assign_sequence_size_from_valid_mask(" in param_finalize
    assert "def assign_sequence_size_if_left_packed(" not in param_finalize
    assert "def assign_sequence_size_from_valid_mask(" in validity_finalize
