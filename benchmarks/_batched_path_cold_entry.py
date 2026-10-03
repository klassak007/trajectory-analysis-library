"""Minimal fresh-process entrypoint for batched-path cold timing."""

from __future__ import annotations

import json
import sys
from time import perf_counter

_STARTED = perf_counter()

from benchmarks._batched_path_process_protocol import (
    _COLD_MARKER,
    CapstoneConfig,
    cold_child,
)


def main() -> None:
    """Report imports separately from setup, first call, and validation."""
    import_seconds = perf_counter() - _STARTED
    if len(sys.argv) != 2:
        raise SystemExit("expected one JSON benchmark configuration")
    config = CapstoneConfig(**json.loads(sys.argv[1]))
    payload = {"import_seconds": import_seconds, **cold_child(config)}
    print(f"{_COLD_MARKER}{json.dumps(payload)}")


if __name__ == "__main__":
    main()
