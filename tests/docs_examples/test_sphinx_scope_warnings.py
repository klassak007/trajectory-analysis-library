from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


_SCOPED_WARNING_TOKENS = (
    "tal.spatial.position.to_frame",
    "tal.spatial.rotation.to_rep",
    "tal.spatial.pose.from_components",
    "tal.spatial.path_solve.solve_pose_path_transform",
    "tal.io.csv_logs.read_csv_logs",
    "tal.frames.topology.find_path",
    "tal.frames.snapshot.snapshot_from_seeds",
    "tal.viz.surface.line",
    "tal.spatial.pose.Pose.register",
    "tal.utils.frame_schema.get_frames",
    "tal.utils.topology_operation_families.operation_intent_support_for_operation_family",
    "tal.utils.xarray_namespace.rename_dims_collision_safe",
)


def _collect_scoped_warnings(output: str) -> list[str]:
    scoped: list[str] = []
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if "WARNING" not in line:
            continue
        lowered = line.lower()
        if any(token in lowered for token in _SCOPED_WARNING_TOKENS):
            scoped.append(line)
    return scoped


def test_sphinx_scope_warnings_for_touched_domains() -> None:
    sphinx_build = shutil.which("sphinx-build")
    if sphinx_build is None:
        pytest.skip("sphinx-build is not available in this environment")

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = repo_root / "docs" / "_build" / "html_docs_examples_scope"
    command = [
        sphinx_build,
        "-n",
        "-b",
        "html",
        "docs",
        str(out_dir),
    ]
    proc = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    output = f"{proc.stdout}\n{proc.stderr}"
    assert proc.returncode == 0, output
    scoped = _collect_scoped_warnings(output)
    assert not scoped, "Scoped Sphinx warnings found:\n" + "\n".join(scoped)
