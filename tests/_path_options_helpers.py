"""Runtime inputs and dependency probes for the shared path-options contract."""

from __future__ import annotations

import pytest

from tal.spatial import PathSolveOptions


class FalseyPathSolveOptions(PathSolveOptions):
    def __bool__(self):
        return False


class UninspectableOptions:
    def __bool__(self):
        raise AssertionError("option truthiness must not be evaluated")

    @property
    def graph(self):
        raise AssertionError("wrong-type option fields must not be read")

    strict = True
    kinematics_support = None


BAD_PATH_OPTIONS = [
    pytest.param({}, id="empty-dict"),
    pytest.param(0, id="zero"),
    pytest.param(False, id="false"),
    pytest.param("", id="empty-string"),
    pytest.param([], id="empty-list"),
    pytest.param({"strict": True}, id="nonempty-dict"),
    pytest.param(object(), id="object"),
    pytest.param(PathSolveOptions, id="options-class"),
    pytest.param(UninspectableOptions(), id="uninspectable-options"),
]

DELEGATOR_BAD_PATH_OPTIONS = [
    pytest.param({}, id="falsey"),
    pytest.param(UninspectableOptions(), id="truthy-uninspectable"),
]


class ResolutionProbe:
    def __init__(self):
        self.events = []

    def __call__(self, *args, **kwargs):
        self.events.append("resolver-call")
        raise AssertionError("resolver must not be invoked")

    @property
    def __signature__(self):
        self.events.append("resolver-signature")
        raise AssertionError("resolver must not be inspected")

    def graph_lookup(self, *args, **kwargs):
        self.events.append("graph-lookup")
        raise AssertionError("graph must not be resolved")


def forbid_path_resolution(monkeypatch, graph):
    """Observe access through caller-supplied graph and resolver collaborators."""
    probe = ResolutionProbe()
    monkeypatch.setattr(graph, "get_frame", probe.graph_lookup)
    return probe
