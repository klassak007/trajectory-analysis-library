from __future__ import annotations

import pytest

import tal.io.adapter_temp as adapter_temp_module
from tal.io.adapter_temp import owned_temporary_directory


def test_io_hard_p10b_063_active_temp_cleanup_interrupt_preserves_primary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: IO_HARD_P10B_063_active_temp_cleanup_interrupt_preserves_primary."""

    class InterruptingTemporaryDirectory:
        name = "/unused/tal-cleanup-probe"

        def __init__(self, *args: object, **kwargs: object) -> None:
            _ = args, kwargs

        def cleanup(self) -> None:
            raise KeyboardInterrupt("cleanup interrupted")

    monkeypatch.setattr(
        adapter_temp_module,
        "TemporaryDirectory",
        InterruptingTemporaryDirectory,
    )
    primary = RuntimeError("primary serialization failure")

    with pytest.raises(RuntimeError, match="primary serialization failure") as error:
        with owned_temporary_directory(
            prefix="probe-",
            owner="tal.io.write_csv_logs",
            purpose="CSV export spool directory",
        ):
            raise primary

    assert error.value is primary
