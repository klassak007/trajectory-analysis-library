"""Contract 111: cold concurrent execution retains ordinary numerical semantics."""

import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import dask
import numpy as np
import pytest
import xarray as xr
from dask.callbacks import Callback
from scipy.spatial.transform import Rotation as SciRotation
from scipy.spatial.transform import Slerp

from tal.core import AnalysisLayoutSpec
from tal.spatial import Pose, Position, Rotation, solve_pose_path_transform


def _public_fixed():
    """First execution is a mixed threaded graph, with no serial warmup."""
    q = SciRotation.from_euler("z", .3).as_quat()
    coords = {"sample": np.arange(32), "quat": list("xyzw"), "axis": list("xyz")}
    ds = xr.Dataset({"q": (("sample", "quat"), np.tile(q, (32, 1))),
                     "p": (("sample", "axis"), np.tile([1., 0., 0.], (32, 1)))}, coords=coords)
    ds = ds.chunk({"sample": 1})
    rotation = Rotation(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("quat",)).wrap(ds[["q"]]))
    position = Position(AnalysisLayoutSpec(sequence_dim="sample", core_dims=("axis",)).wrap(ds[["p"]]))
    pose = Pose.from_components(rotation, position)
    before = [value.as_dataset() for value in (rotation, position, pose)]
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        outputs = [rotation.compose(rotation).as_matrix(), rotation.inverse().as_matrix(),
                   rotation.apply(position), pose.as_matrix(), pose.compose(pose).as_matrix()]
    assert not tasks and "numba" not in sys.modules
    actual = dask.compute(*(value.to_dataarray() for value in outputs), scheduler="threads", num_workers=8)
    r = SciRotation.from_quat(q)
    h = np.eye(4)
    h[:3, :3], h[:3, 3] = r.as_matrix(), [1., 0., 0.]
    expected = [(r * r).as_matrix(), r.inv().as_matrix(), r.apply([1., 0., 0.]), h, h @ h]
    for output, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(output, np.broadcast_to(reference, output.shape), atol=1e-12)
        np.testing.assert_array_equal(output["sample"], np.arange(32))
    for value, original in zip((rotation, position, pose), before, strict=True):
        xr.testing.assert_identical(value.as_dataset(), original)


def _concurrent_reference_cases(family):
    # Reuse the maintained independent numerical parity cases, now under cold
    # concurrent entry. No private factory names or call sequences are asserted.
    from tests.core import test_event_numba as event
    from tests.core import test_linalg_numba as linalg
    from tests.core import test_param_engine_numba as param
    from tests.core import test_spatial_numba as spatial

    cases = {
        "spatial": [spatial.test_spatial_numba_001_slerp_backend_parity,
                    spatial.test_spatial_numba_062_quaternion_squad_backend_parity_if_retained,
                    spatial.test_spatial_numba_063_pose_cubic_squad_backend_parity_if_retained,
                    spatial.test_spatial_numba_011_moving_average_backend_parity,
                    spatial.test_spatial_numba_012_gaussian_smoothing_backend_parity,
                    spatial.test_spatial_numba_013_trapezoid_backend_parity,
                    spatial.test_spatial_numba_017_simpson_backend_parity_if_retained,
                    spatial.test_spatial_numba_041_rotation_mean_backend_parity_if_implemented],
        "core": [param.test_param_numba_001_map_linear_backend_parity,
                 param.test_param_numba_005_bounds_backend_parity,
                 event.test_event_numba_001_boundary_bounded_transition_parity,
                 event.test_event_numba_005_intervals_bounded_pairing_parity,
                 linalg.test_linalg_numba_002_lstsq_numba_backend_parity_if_implemented,
                 spatial.test_spatial_topo_numba_001_chain_pose_compose_backend_parity],
    }[family]
    barrier = Barrier(len(cases))

    def run(case):
        barrier.wait(timeout=60)
        case()

    with ThreadPoolExecutor(max_workers=len(cases)) as workers:
        list(workers.map(run, cases * 2))


def _public_paths():
    from tests.core.test_spatial_path_execution import _registered_path

    eager_graph, providers = _registered_path(1)
    lazy_graph, lazy_providers = _registered_path(1, lazy=True)
    before = lazy_providers[0].as_dataset()
    queries = np.linspace(.1, .9, 8)
    query = xr.DataArray(np.tile(queries, (2, 1)), dims=("trial", "when"))
    tasks = []
    with Callback(pretask=lambda *args: tasks.append(args[0])):
        lazy = [solve_pose_path_transform("f1", "f0", graph=lazy_graph, query=query) for _ in range(4)]
    assert not tasks and "numba" not in sys.modules
    source = providers[0].as_dataset()
    expected_q = Slerp(source.time.data, SciRotation.from_quat(source.rotation.data))(queries).as_matrix()
    expected_t = np.stack([np.interp(queries, source.time.data, source.position.data[:, i]) for i in range(3)], axis=-1)
    barrier = Barrier(8)

    def execute(i):
        barrier.wait(timeout=60)
        if i < 4:
            result = solve_pose_path_transform("f1", "f0", graph=eager_graph, query=queries)
            ds = result.as_dataset()
            assert result.graph is eager_graph
        else:
            result = lazy[i - 4]
            ds = result.as_dataset().compute(scheduler="synchronous")
            assert result.graph is lazy_graph
        actual_t = ds.position.data
        actual_q = SciRotation.from_quat(ds.rotation.data.reshape(-1, 4)).as_matrix()
        np.testing.assert_allclose(actual_t, np.broadcast_to(expected_t, actual_t.shape), atol=1e-12)
        np.testing.assert_allclose(actual_q, np.tile(expected_q, (1 if i < 4 else 2, 1, 1)), atol=1e-12)

    with ThreadPoolExecutor(max_workers=8) as workers:
        list(workers.map(execute, range(8)))
    xr.testing.assert_identical(lazy_providers[0].as_dataset(), before)


@pytest.mark.parametrize("family", ["public_fixed", "public_paths", "spatial", "core"])
def test_numba_concurrent_cold_execution(family, tmp_path):
    """ID: NUMBA_CONCURRENT_001; empty cache and fresh process per family."""
    if importlib.util.find_spec("numba") is None:
        pytest.skip("Numba is physically absent")
    code = "from tests.core.test_backend_concurrent_initialization import _public_fixed, _public_paths, _concurrent_reference_cases; "
    code += f"_{family}()" if family.startswith("public_") else f"_concurrent_reference_cases({family!r})"
    env = dict(os.environ, NUMBA_CACHE_DIR=str(tmp_path / "numba"), NUMBA_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                            env=env, capture_output=True, text=True, timeout=300, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
