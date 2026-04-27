from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any

import pytest

from tal.frames import FrameGraph, FrameGraphDrawOptions, draw_frame_graph, find_path


def _build_graph() -> tuple[FrameGraph, Any, Any, Any, Any]:
    graph = FrameGraph()
    with graph:
        world = graph.get_or_create_frame("world")
        robot = graph.get_or_create_frame("robot", parent=world)
        camera = graph.get_or_create_frame("camera", parent=robot)
        base = graph.get_or_create_frame("base", parent=world)
        lidar = graph.get_or_create_frame("lidar", parent=base)
        alpha = graph.get_or_create_frame("alpha")
        graph.get_or_create_frame("zeta", parent=alpha)
    return graph, world, camera, lidar, alpha


@dataclass
class _FakeAxes:
    edge_calls: list[dict[str, Any]]
    node_calls: list[dict[str, Any]]
    label_calls: list[dict[str, Any]]
    plot_calls: list[dict[str, Any]]
    legend_calls: list[dict[str, Any]]
    axis_off: bool = False

    def set_axis_off(self) -> None:
        self.axis_off = True

    def plot(self, *args: Any, **kwargs: Any) -> None:
        self.plot_calls.append({"args": args, "kwargs": kwargs})

    def legend(self, *args: Any, **kwargs: Any) -> None:
        self.legend_calls.append({"args": args, "kwargs": kwargs})


class _FakeDiGraph:
    def __init__(self) -> None:
        self._nodes: list[str] = []
        self._edges: list[tuple[str, str]] = []
        self.graph: dict[str, Any] = {}

    def add_node(self, node_id: str) -> None:
        if node_id not in self._nodes:
            self._nodes.append(node_id)

    def add_edge(self, src: str, dst: str) -> None:
        edge = (src, dst)
        if edge not in self._edges:
            self._edges.append(edge)

    def edges(self) -> list[tuple[str, str]]:
        return list(self._edges)


class _FakeNetworkx:
    DiGraph = _FakeDiGraph

    @staticmethod
    def _layout(graph: _FakeDiGraph) -> dict[str, tuple[float, float]]:
        return {node: (float(index), 0.0) for index, node in enumerate(graph._nodes)}

    @staticmethod
    def spring_layout(graph: _FakeDiGraph) -> dict[str, tuple[float, float]]:
        return _FakeNetworkx._layout(graph)

    @staticmethod
    def kamada_kawai_layout(graph: _FakeDiGraph) -> dict[str, tuple[float, float]]:
        return _FakeNetworkx._layout(graph)

    @staticmethod
    def circular_layout(graph: _FakeDiGraph) -> dict[str, tuple[float, float]]:
        return _FakeNetworkx._layout(graph)

    @staticmethod
    def shell_layout(graph: _FakeDiGraph) -> dict[str, tuple[float, float]]:
        return _FakeNetworkx._layout(graph)

    @staticmethod
    def planar_layout(graph: _FakeDiGraph) -> dict[str, tuple[float, float]]:
        return _FakeNetworkx._layout(graph)

    @staticmethod
    def draw_networkx_edges(
        graph: _FakeDiGraph,
        pos: dict[str, tuple[float, float]],
        *,
        edgelist: list[tuple[str, str]] | None = None,
        edge_color: str | None = None,
        width: float | None = None,
        style: str = "solid",
        ax: _FakeAxes | None = None,
        **kwargs: Any,
    ) -> None:
        assert pos
        assert kwargs
        assert ax is not None
        ax.edge_calls.append(
            {
                "edgelist": tuple(edgelist or graph.edges()),
                "edge_color": edge_color,
                "width": width,
                "style": style,
            }
        )

    @staticmethod
    def draw_networkx_nodes(
        graph: _FakeDiGraph,
        pos: dict[str, tuple[float, float]],
        *,
        nodelist: list[str] | None = None,
        ax: _FakeAxes | None = None,
        **kwargs: Any,
    ) -> None:
        assert pos
        assert kwargs
        assert ax is not None
        ax.node_calls.append({"nodelist": tuple(nodelist or graph._nodes)})

    @staticmethod
    def draw_networkx_labels(
        graph: _FakeDiGraph,
        pos: dict[str, tuple[float, float]],
        *,
        labels: dict[str, str] | None = None,
        ax: _FakeAxes | None = None,
        **kwargs: Any,
    ) -> None:
        assert pos
        assert kwargs
        assert ax is not None
        ax.label_calls.append({"labels": dict(labels or {})})


class _FakePyplot:
    def subplots(self) -> tuple[None, _FakeAxes]:
        axis = _FakeAxes(edge_calls=[], node_calls=[], label_calls=[], plot_calls=[], legend_calls=[])
        return None, axis


def _install_fake_viz_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "networkx":
            return _FakeNetworkx
        if name == "matplotlib.pyplot":
            return _FakePyplot()
        return original(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)


def _highlight_edges(ax: _FakeAxes, opts: FrameGraphDrawOptions, *, color: str) -> set[tuple[str, str]]:
    for call in ax.edge_calls:
        if call["edge_color"] == color:
            return set(call["edgelist"])
    return set()


def test_frame_core_019_draw_frame_graph_directed_edges_and_axes_return(monkeypatch: pytest.MonkeyPatch) -> None:
    """ID: FRAME_CORE_019_draw_frame_graph_directed_edges_and_axes_return."""
    _install_fake_viz_modules(monkeypatch)
    graph, _, _, _, _ = _build_graph()
    opts = FrameGraphDrawOptions(include_legend=False)
    axis = draw_frame_graph(graph=graph, opts=opts)
    assert isinstance(axis, _FakeAxes)
    assert axis.axis_off is True
    assert len(axis.edge_calls) == 1
    assert axis.edge_calls[0]["edge_color"] == opts.edge_color
    assert set(axis.edge_calls[0]["edgelist"]) == {
        ("world", "robot"),
        ("robot", "camera"),
        ("world", "base"),
        ("base", "lidar"),
        ("alpha", "zeta"),
    }


def test_frame_core_020_draw_frame_graph_src_dst_highlight_direction_roles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_CORE_020_draw_frame_graph_src_dst_highlight_direction_roles."""
    _install_fake_viz_modules(monkeypatch)
    graph, world, camera, lidar, _ = _build_graph()
    opts = FrameGraphDrawOptions()
    axis = draw_frame_graph(graph=graph, src=camera, dst=lidar, opts=opts)
    assert isinstance(axis, _FakeAxes)
    assert _highlight_edges(axis, opts, color=opts.path_forward_color) == {("world", "base"), ("base", "lidar")}
    assert _highlight_edges(axis, opts, color=opts.path_reverse_color) == {("world", "robot"), ("robot", "camera")}
    reverse_call = [call for call in axis.edge_calls if call["edge_color"] == opts.path_reverse_color][0]
    assert reverse_call["style"] == "dashed"
    assert len(axis.plot_calls) == 2
    assert len(axis.legend_calls) == 1


def test_frame_core_021_draw_frame_graph_path_and_src_dst_highlight_equivalence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_CORE_021_draw_frame_graph_path_and_src_dst_highlight_equivalence."""
    _install_fake_viz_modules(monkeypatch)
    graph, _, camera, lidar, _ = _build_graph()
    path = find_path(camera, lidar)
    opts = FrameGraphDrawOptions(include_legend=False)
    axis_from_path = draw_frame_graph(graph=graph, path=path, opts=opts)
    axis_from_endpoints = draw_frame_graph(graph=graph, src=camera, dst=lidar, opts=opts)
    assert _highlight_edges(axis_from_path, opts, color=opts.path_forward_color) == _highlight_edges(
        axis_from_endpoints,
        opts,
        color=opts.path_forward_color,
    )
    assert _highlight_edges(axis_from_path, opts, color=opts.path_reverse_color) == _highlight_edges(
        axis_from_endpoints,
        opts,
        color=opts.path_reverse_color,
    )


def test_frame_core_022_draw_frame_graph_string_endpoint_resolution_and_unknown_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_CORE_022_draw_frame_graph_string_endpoint_resolution_and_unknown_fail_closed."""
    _install_fake_viz_modules(monkeypatch)
    graph, _, _, _, _ = _build_graph()
    opts = FrameGraphDrawOptions(include_legend=False)
    axis = draw_frame_graph(graph=graph, src="camera", dst="lidar", opts=opts)
    assert isinstance(axis, _FakeAxes)
    with pytest.raises(ValueError, match="src frame 'missing' not found"):
        draw_frame_graph(graph=graph, src="missing", dst="lidar", opts=opts)


def test_frame_hard_028_draw_frame_graph_path_outside_snapshot_domain_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_HARD_028_draw_frame_graph_path_outside_snapshot_domain_fail_closed."""
    _install_fake_viz_modules(monkeypatch)
    graph, _, camera, lidar, alpha = _build_graph()
    opts = FrameGraphDrawOptions(include_legend=False)
    with pytest.raises(ValueError, match="outside rendered snapshot domain"):
        draw_frame_graph(
            graph=graph,
            seeds=(alpha,),
            include_unreachable=False,
            src=camera,
            dst=lidar,
            opts=opts,
        )


def test_frame_hard_029_draw_frame_graph_missing_networkx_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_HARD_029_draw_frame_graph_missing_networkx_fail_closed."""
    graph, _, _, _, _ = _build_graph()
    original = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "networkx":
            raise ImportError("missing networkx")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="draw_frame_graph: networkx is required"):
        draw_frame_graph(graph=graph)


def test_frame_hard_030_draw_frame_graph_missing_matplotlib_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_HARD_030_draw_frame_graph_missing_matplotlib_fail_closed."""
    graph, _, _, _, _ = _build_graph()
    original = importlib.import_module

    def _fake_import(name: str) -> Any:
        if name == "networkx":
            return _FakeNetworkx
        if name == "matplotlib.pyplot":
            raise ImportError("missing matplotlib")
        return original(name)

    monkeypatch.setattr(importlib, "import_module", _fake_import)
    with pytest.raises(ImportError, match="draw_frame_graph: matplotlib is required"):
        draw_frame_graph(graph=graph, opts=FrameGraphDrawOptions(include_legend=False))


def test_frame_hard_031_draw_frame_graph_invalid_path_endpoint_argument_combinations_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ID: FRAME_HARD_031_draw_frame_graph_invalid_path_endpoint_argument_combinations_fail_closed."""
    _install_fake_viz_modules(monkeypatch)
    graph, _, camera, lidar, _ = _build_graph()
    path = find_path(camera, lidar)
    with pytest.raises(ValueError, match="mutually exclusive"):
        draw_frame_graph(graph=graph, path=path, src=camera, dst=lidar)
    with pytest.raises(ValueError, match="src and dst must be provided together"):
        draw_frame_graph(graph=graph, src=camera)
    with pytest.raises(TypeError, match="path must be FramePath"):
        draw_frame_graph(graph=graph, path="bad")  # type: ignore[arg-type]


def test_frame_hard_032_draw_frame_graph_cross_graph_path_resolution_fail_closed() -> None:
    """ID: FRAME_HARD_032_draw_frame_graph_cross_graph_path_resolution_fail_closed."""
    g1 = FrameGraph()
    with g1:
        left = g1.get_or_create_frame("left")
    g2 = FrameGraph()
    with g2:
        right = g2.get_or_create_frame("right")
    with pytest.raises(ValueError, match="different FrameGraph"):
        draw_frame_graph(graph=g1, src=left, dst=right)

