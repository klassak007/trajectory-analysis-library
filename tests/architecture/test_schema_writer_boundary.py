import ast
from pathlib import Path

import pytest
import tal.core

from tests.architecture._schema_write import (
    tal_attr_write_lines,
    tal_write_lines_from_source,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOT = REPO_ROOT / "tal"
APPROVED_WRITERS = {
    (SCAN_ROOT / "core" / "schema_validate" / "finalize.py").resolve(),
}
IGNORED_PATH_PARTS = {"tests"}


def _iter_python_files() -> list[Path]:
    files: list[Path] = []
    for path in SCAN_ROOT.rglob("*.py"):
        if any(part in IGNORED_PATH_PARTS for part in path.parts):
            continue
        files.append(path)
    return sorted(files)


def test_arch_schemawrite_001_only_approved_modules_write_tal_attrs() -> None:
    """ID: ARCH_SCHEMAWRITE_001_only_approved_modules_write_tal_attrs."""
    violations: list[str] = []
    for path in _iter_python_files():
        lines = tal_attr_write_lines(path)
        if not lines or path.resolve() in APPROVED_WRITERS:
            continue
        rel = path.relative_to(REPO_ROOT)
        locations = ",".join(str(line) for line in sorted(lines))
        violations.append(f"{rel}:{locations}")
    assert not violations, (
        "Direct attrs['tal'] writes are restricted to canonical writer modules. "
        f"Violations: {violations}"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("attrs['tal'] = {}", [1]),
        ("attrs['tal']['x'] = 1", [1]),
        ("attrs_alias = attrs\nattrs_alias['tal'] = {}", [2]),
        ("attrs.setdefault('tal', {})", [1]),
        ("attrs.update({'tal': {}})", [1]),
        ("attrs_alias = attrs\nattrs_alias.update({'tal': {}})", [2]),
        ("attrs.update(tal={})", [1]),
        ("tal_schema = attrs['tal']\ntal_schema['x'] = 1", [2]),
        ("attrs['tal'].update({'x': 1})", [1]),
        ('new_attrs = {**ds.attrs, "tal": schema}\nds.attrs = new_attrs', [2]),
        ("new_attrs = dict(ds.attrs, tal=schema)\nds.attrs = new_attrs", [2]),
        (
            "new_attrs = dict(ds.attrs)\n"
            "new_attrs['tal'] = schema\n"
            "ds.attrs = new_attrs",
            [3],
        ),
        (
            "new_attrs = {}\nnew_attrs.update([('tal', schema)])\n"
            "ds.attrs = new_attrs",
            [3],
        ),
        (
            "new_attrs = dict(ds.attrs)\n"
            "if condition:\n"
            "    new_attrs['tal'] = schema\n"
            "else:\n"
            "    new_attrs = {}\n"
            "ds.attrs = new_attrs",
            [6],
        ),
        (
            "new_attrs = dict(ds.attrs)\n"
            "new_attrs['tal'].update({'roles': roles})",
            [2],
        ),
        (
            "new_attrs = dict(ds.attrs)\n"
            "schema = new_attrs['tal']\n"
            "schema['roles'] = roles",
            [3],
        ),
        (
            "new_attrs = ds.attrs | {}\n"
            "new_attrs['tal'].update({'roles': roles})",
            [2],
        ),
        (
            "new_attrs = dict(ds.attrs)\n"
            "newer_attrs = new_attrs | {}\n"
            "newer_attrs['tal'].update({'roles': roles})",
            [3],
        ),
        (
            "schema = ds.attrs.get('tal')\n"
            "schema['roles'] = roles",
            [2],
        ),
        (
            "schema = dict(ds.attrs)['tal']\n"
            "schema['roles'] = roles",
            [2],
        ),
        (
            "if condition:\n"
            "    new_attrs = ds.attrs | {}\n"
            "else:\n"
            "    new_attrs = {}\n"
            "new_attrs['tal'].update({'roles': roles})",
            [5],
        ),
        (
            "attrs = ds.attrs\n"
            "def helper():\n"
            "    global attrs\n"
            "    attrs['tal'] = value",
            [4],
        ),
        (
            "new_attrs = dict(ds.attrs)\n"
            "for item in items:\n"
            "    new_attrs['tal'] = schema\n"
            "    if item:\n"
            "        break\n"
            "else:\n"
            "    new_attrs = {}\n"
            "ds.attrs = new_attrs",
            [8],
        ),
        (
            "attrs_alias = attrs\n"
            "attrs_alias = attrs_alias.update({'tal': schema})",
            [2],
        ),
        (
            "schema = attrs['tal']\n"
            "schema = schema.update({'roles': roles})",
            [2],
        ),
        ("attrs_alias = attrs\nattrs_alias |= {'tal': schema}", [2]),
        ("schema = attrs['tal']\nschema |= {'roles': roles}", [2]),
        (
            "attrs_alias, other = (attrs, value)\n"
            "attrs_alias['tal'] = schema",
            [2],
        ),
        (
            "attrs_alias, *rest = (attrs, value)\n"
            "attrs_alias['tal'] = schema",
            [2],
        ),
        (
            "if (attrs_alias := attrs):\n"
            "    attrs_alias['tal'] = schema",
            [2],
        ),
        (
            "attrs_alias = ds.attrs if condition else {}\n"
            "attrs_alias['tal'] = schema",
            [2],
        ),
        (
            "schema = ds.attrs['tal'].copy()\n"
            "schema['roles']['group'] = 'trial'",
            [2],
        ),
        (
            "schema = ds.attrs['tal']\n"
            "roles = schema.get('roles')\n"
            "roles['group'] = 'trial'",
            [3],
        ),
        (
            "attrs = ds.attrs\n"
            "[attrs.update({'tal': schema}) for value in values]",
            [2],
        ),
        (
            "[(attrs_alias := attrs) for value in values]\n"
            "attrs_alias['tal'] = schema",
            [2],
        ),
        (
            "mapping = {}\n"
            "[mapping.update({'tal': schema}) for value in values]\n"
            "ds.attrs = mapping",
            [3],
        ),
        ("copied = ds.attrs.copy()\nout.attrs = copied", [2]),
        ("copied = {**ds.attrs}\nout.attrs = copied", [2]),
        (
            "copied = {key: value for key, value in ds.attrs.items()}\n"
            "out.attrs = copied",
            [2],
        ),
        (
            "copied = dict((key, value) for key, value in ds.attrs.items())\n"
            "out.attrs = copied",
            [2],
        ),
        (
            "[alias.__setitem__('tal', schema) for alias in (ds.attrs,)]",
            [1],
        ),
        (
            "for alias in (ds.attrs,):\n    alias['tal'] = schema",
            [2],
        ),
        ("out.attrs = ds.attrs", [1]),
        ("attrs = ds.attrs\nattrs.__ior__({'tal': schema})", [2]),
        ("attrs = ds.attrs\nattrs.__delitem__('tal')", [2]),
        ("schema = ds.attrs['tal']\nschema.__setitem__('roles', roles)", [2]),
        ("schema = ds.attrs['tal']\nschema.__delitem__('roles')", [2]),
        ("schema = ds.attrs['tal']\nschema.__ior__({'roles': roles})", [2]),
        (
            "(alias := ds.attrs) if condition else (alias := {})\n"
            "alias['tal'] = schema",
            [2],
        ),
        ("alias = ds.attrs or {}\nalias['tal'] = schema", [2]),
        (
            "alias = ds.attrs\ncondition or (alias := {})\n"
            "alias['tal'] = schema",
            [3],
        ),
        (
            "mapping = {}\nfor item in items:\n    if mapping:\n"
            "        ds.attrs = mapping\n    mapping['tal'] = schema",
            [4],
        ),
        (
            "mapping = {}\nwhile condition:\n    if mapping:\n"
            "        ds.attrs = mapping\n    mapping['tal'] = schema",
            [4],
        ),
        (
            "mapping = {}\nfor item in items:\n"
            "    mapping = ds.attrs\n    break\n    mapping = {}\n"
            "out.attrs = mapping",
            [6],
        ),
        (
            "mapping = {}\nfor item in items:\n"
            "    mapping = ds.attrs\n    continue\n    mapping = {}\n"
            "out.attrs = mapping",
            [6],
        ),
        (
            "alias = {}\ntry:\n    alias = ds.attrs\n    operation()\n"
            "    alias = {}\nexcept Exception:\n    out.attrs = alias",
            [7],
        ),
        (
            "alias = {}\ntry:\n    operation(alias := ds.attrs)\n"
            "except Exception:\n    out.attrs = alias",
            [5],
        ),
        (
            "def helper():\n    alias['tal'] = schema\n"
            "alias = ds.attrs\nhelper()",
            [2],
        ),
        (
            "def outer():\n    def helper():\n        alias['tal'] = schema\n"
            "    alias = ds.attrs\n    helper()",
            [3],
        ),
        (
            "def outer():\n    alias = {}\n    def helper():\n"
            "        nonlocal alias\n        alias['tal'] = schema\n"
            "    alias = ds.attrs\n    helper()",
            [5],
        ),
        ("out = xr.Dataset(attrs=dict(ds.attrs))", [1]),
        ("out = xr.Dataset(attrs=ds.attrs)", [1]),
        ("out = xr.Dataset(data_vars, coords, ds.attrs)", [1]),
        ("out = xr.DataArray(values, attrs=ds.attrs.copy())", [1]),
        ("out = xr.DataArray(values, coords, dims, name, ds.attrs)", [1]),
        ("factory = xr.Dataset\nout = factory(data_vars, coords, ds.attrs)", [2]),
        (
            "from xarray import Dataset as factory\n"
            "out = factory(data_vars, coords, ds.attrs)",
            [2],
        ),
        (
            "def build(factory=xr.Dataset):\n"
            "    return factory(data_vars, coords, ds.attrs)",
            [2],
        ),
        (
            "def build(alias=ds.attrs):\n    out.attrs = alias",
            [2],
        ),
        ("out = target.assign_attrs(**ds.attrs)", [1]),
        ("target.attrs.update(**ds.attrs)", [1]),
        ("copied = dict(**ds.attrs)\nout.attrs = copied", [2]),
    ],
)
def test_arch_schemawrite_002_detector_flags_indirect_patterns(
    source: str, expected: list[int]
) -> None:
    """ID: ARCH_SCHEMAWRITE_002_detector_flags_indirect_patterns."""
    assert tal_write_lines_from_source(source) == expected


def test_arch_schemawrite_003_detector_ignores_read_only_alias() -> None:
    """ID: ARCH_SCHEMAWRITE_003_detector_ignores_read_only_alias."""
    source_tal = "tal_schema = attrs['tal']\nvalue = tal_schema.get('x')\n"
    source_attrs = "attrs_alias = attrs\nvalue = attrs_alias.get('tal')\n"
    source_shadow = "attrs = ds.attrs\ndef helper(attrs):\n    attrs['tal'] = value\n"
    source_local = (
        "attrs = ds.attrs\ndef helper():\n"
        "    attrs['tal'] = value\n    attrs = {}\n"
    )
    source_with = (
        "attrs = ds.attrs\nwith manager() as attrs:\n    attrs['tal'] = value\n"
    )
    source_with_local = (
        "attrs = ds.attrs\ndef helper():\n    attrs['tal'] = value\n"
        "    with manager() as attrs:\n        pass\n"
    )
    source_except = (
        "attrs = ds.attrs\ntry:\n    operation()\n"
        "except Exception as attrs:\n    attrs['tal'] = value\n"
    )
    source_except_local = (
        "attrs = ds.attrs\ndef helper():\n    attrs['tal'] = value\n"
        "    try:\n        operation()\n    except Exception as attrs:\n        pass\n"
    )
    source_import_local = (
        "attrs = ds.attrs\ndef helper():\n"
        "    attrs['tal'] = value\n    import attrs\n"
    )
    source_function_name = (
        "attrs = ds.attrs\ndef attrs():\n    attrs['tal'] = value\n"
    )
    source_match_local = (
        "attrs = ds.attrs\ndef helper(subject):\n    attrs['tal'] = value\n"
        "    match subject:\n        case {'value': attrs}:\n            pass\n"
    )
    source_destructured_shadow = (
        "attrs = ds.attrs\nattrs, other = ({}, value)\nattrs['tal'] = schema\n"
    )
    source_starred_shadow = (
        "attrs = ds.attrs\nattrs, *rest = ({}, value)\nattrs['tal'] = schema\n"
    )
    source_named_shadow = (
        "attrs = ds.attrs\nif (attrs := {'value': 1}):\n"
        "    attrs['tal'] = schema\n"
    )
    source_import_shadow = (
        "attrs = ds.attrs\nimport attrs\nattrs['tal'] = schema\n"
    )
    source_class_import_shadow = (
        "attrs = ds.attrs\nclass Holder:\n"
        "    import attrs\n    attrs['tal'] = schema\n"
    )
    source_shallow_schema_copy = (
        "schema = ds.attrs['tal'].copy()\nschema['roles'] = roles\n"
    )
    source_comprehension_shadow = (
        "attrs = ds.attrs\n"
        "[attrs.update({'tal': schema}) for attrs in values]\n"
    )
    source_constructor = "out = xr.Dataset(attrs={'note': value})\n"
    source_positional_constructor = "out = xr.Dataset(data_vars, coords, {'note': value})\n"
    source_unpack = "out = target.assign_attrs(**{'note': value})\n"
    source_unrelated_attrs_keyword = "inspect_metadata(attrs=ds.attrs)\n"
    source_shadowed_xarray = (
        "xr = custom\nout = xr.Dataset(data_vars, coords, ds.attrs)\n"
    )
    assert tal_write_lines_from_source(source_tal) == []
    assert tal_write_lines_from_source(source_attrs) == []
    assert tal_write_lines_from_source(source_shadow) == []
    assert tal_write_lines_from_source(source_local) == []
    assert tal_write_lines_from_source(source_with) == []
    assert tal_write_lines_from_source(source_with_local) == []
    assert tal_write_lines_from_source(source_except) == []
    assert tal_write_lines_from_source(source_except_local) == []
    assert tal_write_lines_from_source(source_import_local) == []
    assert tal_write_lines_from_source(source_function_name) == []
    assert tal_write_lines_from_source(source_match_local) == []
    assert tal_write_lines_from_source(source_destructured_shadow) == []
    assert tal_write_lines_from_source(source_starred_shadow) == []
    assert tal_write_lines_from_source(source_named_shadow) == []
    assert tal_write_lines_from_source(source_import_shadow) == []
    assert tal_write_lines_from_source(source_class_import_shadow) == []
    assert tal_write_lines_from_source(source_shallow_schema_copy) == []
    assert tal_write_lines_from_source(source_comprehension_shadow) == []
    assert tal_write_lines_from_source(source_constructor) == []
    assert tal_write_lines_from_source(source_positional_constructor) == []
    assert tal_write_lines_from_source(source_unpack) == []
    assert tal_write_lines_from_source(source_unrelated_attrs_keyword) == []
    assert tal_write_lines_from_source(source_shadowed_xarray) == []


def test_arch_schemawrite_004_detector_flags_attrs_alias_writes() -> None:
    """ID: ARCH_SCHEMAWRITE_004_detector_flags_attrs_alias_writes."""
    source = "attrs_alias = attrs\nattrs_alias['tal'] = {}\n"
    assert tal_write_lines_from_source(source) == [2]


def test_arch_schemawrite_005_new_cleanup_owner_modules_remain_schema_write_free() -> None:
    """ID: ARCH_SCHEMAWRITE_005_new_cleanup_owner_modules_remain_schema_write_free."""
    paths = [
        REPO_ROOT / "tal" / "core" / "orchestration" / "context.py",
        REPO_ROOT / "tal" / "core" / "orchestration" / "schema_finalize.py",
        REPO_ROOT / "tal" / "core" / "validity_finalize.py",
        REPO_ROOT / "tal" / "core" / "event_ops" / "finalize.py",
        REPO_ROOT / "tal" / "linalg" / "component_context.py",
    ]
    for path in paths:
        assert path.exists(), path
        assert tal_attr_write_lines(path) == [], path


def test_arch_schemawrite_006_dataset_attr_transfer_is_internal_and_centralized() -> None:
    """ID: ARCH_SCHEMAWRITE_006_dataset_attr_transfer_is_internal_and_centralized."""
    assert not hasattr(tal.core, "copy_dataset_attrs")
    definitions = []
    for path in _iter_python_files():
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            if getattr(node, "name", None) == "transfer_dataset_attrs":
                definitions.append(path.relative_to(REPO_ROOT).as_posix())
    assert definitions == ["tal/core/orchestration/finalize.py"]


def test_arch_schemawrite_007_private_schema_assignment_has_one_owner() -> None:
    """ID: ARCH_SCHEMAWRITE_007_private_schema_assignment_has_one_owner."""
    schema_path = SCAN_ROOT / "core" / "schema.py"
    finalize_path = SCAN_ROOT / "core" / "schema_validate" / "finalize.py"
    assert tal_attr_write_lines(schema_path) == []

    lines = tal_attr_write_lines(finalize_path)
    module = ast.parse(finalize_path.read_text(encoding="utf-8"))
    assert lines
    owner_names: list[str] = []
    for line in lines:
        owners = [
            node.name
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.lineno <= line <= node.end_lineno
        ]
        assert len(owners) == 1
        owner_names.extend(owners)
    assert len(set(owner_names)) == 1
    assert owner_names[0].startswith("_")


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "value = ds.attrs\n"
            "copied = {key: value for key, value in records}\n"
            "out.attrs = copied",
            [],
        ),
        (
            "copied = {key: value for attrs_source in (ds.attrs,) "
            "for key, value in attrs_source.items()}\n"
            "out.attrs = copied",
            [2],
        ),
        (
            "attrs_source = ds.attrs\n"
            "copied = {key: value for attrs_source in (attrs_source,) "
            "for key, value in attrs_source.items()}\n"
            "out.attrs = copied",
            [3],
        ),
        (
            "copied = {'tal': schema for _ in records}\n"
            "out.attrs = copied",
            [2],
        ),
        (
            "copied = dict([('tal', schema) for _ in records])\n"
            "out.attrs = copied",
            [2],
        ),
        (
            "copied = dict({('tal', schema) for _ in records})\n"
            "out.attrs = copied",
            [2],
        ),
        (
            "copied = dict(('tal', schema) for _ in records)\n"
            "out.attrs = copied",
            [2],
        ),
    ],
)
def test_arch_schemawrite_008_comprehension_origins_use_local_scope(
    source: str, expected: list[int]
) -> None:
    """ID: ARCH_SCHEMAWRITE_008_comprehension_origins_use_local_scope."""
    assert tal_write_lines_from_source(source) == expected


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "import copy\n"
            "copied = copy.copy(ds.attrs)\n"
            "out.attrs = copied",
            [3],
        ),
        (
            "import copy as copier\n"
            "copied = copier.deepcopy(ds.attrs)\n"
            "out.attrs = copied",
            [3],
        ),
        (
            "from copy import copy as clone\n"
            "copied = clone(ds.attrs)\n"
            "out.attrs = copied",
            [3],
        ),
        (
            "from copy import deepcopy\n"
            "copied = deepcopy(x=ds.attrs)\n"
            "out.attrs = copied",
            [3],
        ),
        (
            "import copy\n"
            "copy = custom\n"
            "copied = copy.copy(ds.attrs)\n"
            "out.attrs = copied",
            [],
        ),
        (
            "from copy import copy\n"
            "def build(copy):\n"
            "    copied = copy(ds.attrs)\n"
            "    out.attrs = copied",
            [],
        ),
    ],
)
def test_arch_schemawrite_009_stdlib_copy_tracks_aliases_and_shadowing(
    source: str, expected: list[int]
) -> None:
    """ID: ARCH_SCHEMAWRITE_009_stdlib_copy_tracks_aliases_and_shadowing."""
    assert tal_write_lines_from_source(source) == expected
