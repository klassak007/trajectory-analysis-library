"""Extension-owned Contract 101 documentation coverage after extraction."""

import doctest
import re

import pytest

from tal_extensions.geo import accessor
from tal_extension_tests.docs_examples._examples import EXECUTABLE_EXAMPLES
from tal_extension_tests.docs_examples._manifest import (
    CURATED_SCOPE_COUNTS,
    DOCSTRING_SECTION_REQUIREMENTS,
    EXAMPLE_REQUIRED_SYMBOLS,
    curated_scope_counts,
    iter_curated_public_symbols,
    required_example_ids,
)


def test_curated_extension_sections_and_example_coverage():
    records = iter_curated_public_symbols()
    assert curated_scope_counts() == CURATED_SCOPE_COUNTS
    assert sum(CURATED_SCOPE_COUNTS.values()) == len(records)
    assert set(required_example_ids()) <= set(EXECUTABLE_EXAMPLES)
    for record in records:
        doc = getattr(record.obj, "__doc__", "") or ""
        for section in DOCSTRING_SECTION_REQUIREMENTS.get(record.symbol, ()):
            assert re.search(rf"(?m)^\s*{section}\s*\n\s*-{{3,}}", doc), (
                record.symbol,
                section,
            )
        if record.symbol in EXAMPLE_REQUIRED_SYMBOLS and record.kind != "type_alias":
            assert ">>>" in doc, record.symbol


@pytest.mark.parametrize("method", ("to_lla", "to_enu", "to_ecef"))
def test_accessor_docstrings_register_explicitly_and_execute(method):
    owner = getattr(accessor.PositionGeoAccessor, method)
    test = doctest.DocTestParser().get_doctest(
        owner.__doc__, vars(accessor), method, None, 0
    )
    result = doctest.DocTestRunner().run(test)
    assert result.failed == 0 and result.attempted > 0
