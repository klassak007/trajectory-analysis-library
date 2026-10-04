from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
import xarray as xr

from tal.core import AnalysisObject
from tal.core.event_ops import AroundOptions, ConditionEvalOptions


def _ao(*, clock_name: str = "time", chunked: bool = False) -> AnalysisObject:
    values: object = np.asarray([0.0, 1.0, 1.0, 0.0], dtype="float64")
    if chunked:
        da = pytest.importorskip("dask.array")
        values = da.from_array(values, chunks=2)
    ds = xr.Dataset(
        {"value": ("sample", values)},
        coords={
            "sample": np.arange(4, dtype="int64"),
            clock_name: ("sample", np.arange(4, dtype="float64")),
        },
    )
    return AnalysisObject.from_data(
        ds,
        sequence_dim="sample",
        core_dims=(),
        param_coord=clock_name,
    )


def _assert_same(left: AnalysisObject, right: AnalysisObject) -> None:
    xr.testing.assert_identical(
        left.as_dataset(copy="none"),
        right.as_dataset(copy="none"),
    )


@pytest.mark.parametrize(
    ("overrides", "opts"),
    [
        ({"dt": 1.0}, AroundOptions(dt=1.0)),
        ({"edge": "exit", "dt": 1.0}, AroundOptions(edge="exit", dt=1.0)),
        ({"pre": 0.0, "dt": 1.0}, AroundOptions(pre=0.0, dt=1.0)),
        ({"post": 0.0, "dt": 1.0}, AroundOptions(post=0.0, dt=1.0)),
        (
            {"layout": "stacked", "dt": 1.0},
            AroundOptions(layout="stacked", dt=1.0),
        ),
        (
            {
                "edge": "exit",
                "pre": 0.0,
                "post": 0.0,
                "dt": 1.0,
                "layout": "segments",
            },
            AroundOptions(
                edge="exit",
                pre=0.0,
                post=0.0,
                dt=1.0,
                layout="segments",
            ),
        ),
    ],
)
def test_event_around_sugar_001_keywords_match_options(
    overrides: dict[str, object],
    opts: AroundOptions,
) -> None:
    """ID: EVENT_AROUND_SUGAR_001_keywords_match_options."""
    ao = _ao()
    condition = ao > 0.5
    _assert_same(
        ao.events.around(condition, **overrides),
        ao.events.around(condition, opts=opts),
    )


def test_event_around_sugar_002_explicit_anchors_match_options() -> None:
    """ID: EVENT_AROUND_SUGAR_002_explicit_anchors_match_options."""
    ao = _ao()
    anchors = xr.DataArray([1.0, 3.0], dims="anchor")
    _assert_same(
        ao.events.around(anchors, pre=0.0, post=0.0, dt=1.0),
        ao.events.around(
            anchors,
            opts=AroundOptions(pre=0.0, post=0.0, dt=1.0),
        ),
    )


def test_event_around_sugar_003_advanced_options_remain_supported() -> None:
    """ID: EVENT_AROUND_SUGAR_003_advanced_options_remain_supported."""
    ao = _ao(clock_name="clock")
    condition = ao > 0.5
    out = ao.events.around(
        condition,
        opts=AroundOptions(
            eval=ConditionEvalOptions(coord_name="clock"),
            grid=np.asarray([-1.0, 0.0, 1.0]),
        ),
    )
    ds = out.as_dataset(copy="none")
    sequence_dim = str(ds.attrs["tal"]["core"]["roles"]["sequence_dim"])
    assert ds.sizes[sequence_dim] == 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"edge": "enter"},
        {"pre": 0.0},
        {"post": 0.0},
        {"dt": 1.0},
        {"layout": "segments"},
    ],
)
def test_event_around_sugar_004_mixed_forms_fail_before_input_use(
    overrides: dict[str, object],
) -> None:
    """ID: EVENT_AROUND_SUGAR_004_mixed_forms_fail_before_input_use."""
    ao = _ao()
    with pytest.raises(TypeError, match="events.around: opts cannot be combined"):
        ao.events.around(object(), opts=AroundOptions(dt=1.0), **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["eval", "grid"])
def test_event_around_sugar_005_advanced_fields_are_not_keywords(field: str) -> None:
    """ID: EVENT_AROUND_SUGAR_005_advanced_fields_are_not_keywords."""
    ao = _ao()
    condition = ao > 0.5
    value: object = ConditionEvalOptions() if field == "eval" else np.asarray([0.0])
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{field}'"):
        ao.events.around(condition, **{field: value})


def test_event_around_sugar_006_explicit_none_is_not_omitted() -> None:
    """ID: EVENT_AROUND_SUGAR_006_explicit_none_is_not_omitted."""
    ao = _ao()
    condition = ao > 0.5
    with pytest.raises(ValueError, match="events.around: opts.dt is required"):
        ao.events.around(condition, dt=None)
    with pytest.raises(TypeError, match="events.around: opts cannot be combined.*dt"):
        ao.events.around(condition, opts=AroundOptions(dt=1.0), dt=None)


def test_event_around_sugar_007_no_overrides_keep_default_validation() -> None:
    """ID: EVENT_AROUND_SUGAR_007_no_overrides_keep_default_validation."""
    ao = _ao()
    with pytest.raises(ValueError, match="events.around: opts.dt is required"):
        ao.events.around(ao > 0.5)


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (lambda ao: ao.events.around(ao > 0.5, edge="bad", dt=1.0), "opts.edge"),
        (lambda ao: ao.events.around(ao > 0.5, pre=-1.0, dt=1.0), "opts.pre"),
        (lambda ao: ao.events.around(ao > 0.5, layout="bad", dt=1.0), "opts.layout"),
    ],
)
def test_event_around_sugar_008_invalid_keywords_keep_owner(
    call: Callable[[AnalysisObject], object],
    message: str,
) -> None:
    """ID: EVENT_AROUND_SUGAR_008_invalid_keywords_keep_owner."""
    with pytest.raises(ValueError, match=rf"events\.around: {message}"):
        call(_ao())


def test_event_around_sugar_009_keyword_path_preserves_laziness() -> None:
    """ID: EVENT_AROUND_SUGAR_009_keyword_path_preserves_laziness."""
    dask = pytest.importorskip("dask")
    ao = _ao(chunked=True)
    anchors = xr.DataArray([1.0], dims="anchor")
    executed: list[object] = []
    with dask.callbacks.Callback(pretask=lambda key, *_: executed.append(key)):
        out = ao.events.around(
            anchors,
            pre=0.0,
            post=0.0,
            dt=1.0,
            layout="stacked",
        )
    assert not executed
    assert out.as_dataset(copy="none")["value"].chunks is not None
