from __future__ import annotations

import doctest
import inspect

import pytest

from tal.io import CsvExportOptions
from tests.docs_examples._examples import EXECUTABLE_EXAMPLES
from tests.docs_examples._manifest import required_example_ids


IO_EXAMPLE_IDS = tuple(sorted(example_id for example_id in required_example_ids() if example_id.startswith("IO-")))


def test_io_zarr_roundtrip_example_is_execution_required() -> None:
    assert "IO-ROUNDTRIP-SURFACE" in IO_EXAMPLE_IDS


def test_csv_export_options_public_doctest_executes() -> None:
    tests = doctest.DocTestFinder().find(CsvExportOptions)
    runner = doctest.DocTestRunner()
    for example in tests:
        runner.run(example)
    assert runner.summarize().failed == 0


def test_io_doc_p10b_005_csv_export_option_describes_validity_selection_not_column() -> None:
    """ID: IO_DOC_P10B_005_csv_export_option_describes_validity_selection_not_column."""
    doc = inspect.getdoc(CsvExportOptions)
    assert doc is not None
    assert "determine the exported row count" in doc
    assert "validity coordinate itself is not exported" in doc


@pytest.mark.parametrize("example_id", IO_EXAMPLE_IDS)
def test_io_doc_example_executes(example_id: str) -> None:
    EXECUTABLE_EXAMPLES[example_id]()
