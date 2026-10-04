"""Compare TAL's CSV header boundary with direct pandas parsing.

This is a manual, non-CI benchmark. By default it generates a representative
multi-megabyte numeric log; pass ``--path`` to use an existing UTF-8 CSV file.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from functools import partial
import gc
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter
import warnings

import numpy as np
import pandas as pd

from tal.io.csv_logs import _read_csv_frame
from tal.io.csv_validation import require_valid_csv_header


def _read_pandas(path: Path) -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.ParserWarning)
        return pd.read_csv(path, index_col=False, on_bad_lines="error")


def _read_tal_boundary(path: Path, *, time_col: str) -> pd.DataFrame:
    header = require_valid_csv_header(str(path), owner="benchmark")
    if time_col not in header:
        raise ValueError(f"benchmark: timestamp column {time_col!r} is not in the CSV header.")
    return _read_csv_frame(
        str(path),
        time_col=time_col,
        owner="benchmark",
    )


def _time_read(reader: Callable[[Path], pd.DataFrame], path: Path) -> float:
    gc.collect()
    started = perf_counter()
    frame = reader(path)
    elapsed = perf_counter() - started
    if frame.empty:
        raise RuntimeError("benchmark input must contain data rows")
    return elapsed


def _generate_log(path: Path, *, rows: int) -> None:
    sample = np.arange(rows, dtype=np.int64)
    pd.DataFrame(
        {
            "time": sample.astype(np.float64) * 0.01,
            "position": np.sin(sample * 0.001),
            "velocity": np.cos(sample * 0.001),
            "quality": sample % 7,
        }
    ).to_csv(path, index=False)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--rows", type=int, default=400_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1.25)
    parser.add_argument("--time-col", default="time")
    return parser.parse_args()


def _run(path: Path, *, repeats: int, threshold: float, time_col: str) -> int:
    tal_reader = partial(_read_tal_boundary, time_col=time_col)
    _read_pandas(path)
    tal_reader(path)
    direct: list[float] = []
    boundary: list[float] = []
    for index in range(repeats):
        order = (
            ((_read_pandas, direct), (tal_reader, boundary))
            if index % 2 == 0
            else ((tal_reader, boundary), (_read_pandas, direct))
        )
        for reader, samples in order:
            samples.append(_time_read(reader, path))
    direct_median = median(direct)
    boundary_median = median(boundary)
    ratio = boundary_median / direct_median
    size_mib = path.stat().st_size / (1024 * 1024)
    print(f"input: {path} ({size_mib:.1f} MiB)")
    print(f"TAL timestamp column: {time_col!r}")
    print(f"direct pandas median: {direct_median:.4f} s")
    print(f"header + pandas median: {boundary_median:.4f} s")
    print(f"ratio: {ratio:.3f}x (target <= {threshold:.2f}x)")
    return int(ratio > threshold)


def main() -> int:
    args = _parse_args()
    if args.path is not None:
        return _run(
            args.path,
            repeats=args.repeats,
            threshold=args.threshold,
            time_col=args.time_col,
        )
    with tempfile.TemporaryDirectory(prefix="tal-csv-benchmark-") as raw_dir:
        path = Path(raw_dir) / "representative.csv"
        _generate_log(path, rows=args.rows)
        return _run(
            path,
            repeats=args.repeats,
            threshold=args.threshold,
            time_col=args.time_col,
        )


if __name__ == "__main__":
    raise SystemExit(main())
