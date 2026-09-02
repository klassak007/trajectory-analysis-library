"""Sphinx configuration for TAL docs.

This repo uses MyST Markdown as the primary authoring format (xarray-style),
and Sphinx `autodoc`/`autosummary` for API tables and docstring extraction.
"""

from __future__ import annotations

import os
import sys

# Ensure `import tal` works when building docs from a source checkout.
sys.path.insert(0, os.path.abspath(".."))


project = "TAL"
author = "TAL Contributors"

try:  # pragma: no cover (docs build only)
    import tal  # type: ignore

    release = getattr(tal, "__version__", "0.0.0")
except Exception:  # pragma: no cover (docs build only)
    release = "0.0.0"


extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
]


root_doc = "index"
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

exclude_patterns = [
    "_build",
    "api/_generated/frames/tal.utils.frame_schema.get_frames.rst",
    "api/_generated/frames/tal.utils.frame_schema.set_frames.rst",
    "**/.ipynb_checkpoints",
    ".DS_Store",
]

templates_path = ["_templates"]


# ----------------------------- MyST config ----------------------------- #

myst_enable_extensions = [
    # Allow ::: fenced directives (optional, but convenient).
    "colon_fence",
    # Definition lists are common in reference docs.
    "deflist",
]

# Generate heading anchors so deep links behave like xarray docs.
myst_heading_anchors = 3


# --------------------------- Autodoc config ---------------------------- #

autosummary_generate = True

# Keep type hints readable in the rendered docs.
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Optional deps should never break API docs import.
autodoc_mock_imports = [
    "rosbags",
    "networkx",
    "matplotlib",
    "holoviews",
    "hvplot",
    "pygraphviz",
    "pydot",
]


# -------------------------- Intersphinx config ------------------------- #

intersphinx_mapping = {
    # Sphinx expects the inventory spec to be None or a non-empty string/path.
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}

nitpick_ignore_regex = [
    ("py:class", r".*Accessor"),
    ("py:class", r".*Options"),
    ("py:class", r"'AnalysisObject"),
    ("py:class", r"'AnalysisObject'"),
    ("py:class", r"'Array'"),
    ("py:class", r"'Frame'"),
    ("py:class", r"\{\"quat\""),
    ("py:class", r"\"matrix\"\}"),
    ("py:class", r"callable"),
    ("py:class", r"default=True"),
    ("py:class", r"DimLike"),
    ("py:class", r"iterable"),
    ("py:class", r"np\..*"),
    ("py:class", r"optional"),
    ("py:class", r"tal\.core\..*types\..*"),
    ("py:class", r"tal\.core\.schema\.UnsetType"),
    ("py:class", r"tal\.io\.options\..*"),
    ("py:class", r"tal\.spatial\.path_solve\..*"),
    ("py:class", r"UnsetType"),
    ("py:class", r"VizKind"),
    ("py:class", r"WeightInput"),
    ("py:class", r"xr\..*"),
    ("py:data", r"typing\.Union"),
    ("py:meth", r"Array\.set_core_dims"),
    ("py:meth", r"Array\.set_matrix_axes"),
    ("py:meth", r"Array\.set_vector_axis"),
    ("py:meth", r"set_core_dims"),
    ("py:meth", r"set_matrix_axes"),
    ("py:meth", r"set_vector_axis"),
    ("py:obj", r"tal\.core\.combine_ops\.concat_sequence\.concat_sequence_contexts"),
    ("py:obj", r"tal\.core\.dataset_utils\.dataset_to_dataarray"),
    ("py:obj", r"tal\.core\.orchestration\.alignment_intent\.resolve_alignment_intent"),
    ("py:obj", r"tal\.core\.orchestration\.broadcast_intent\.resolve_broadcast_intent"),
    ("py:obj", r"tal\.core\.schema\.merge_schema"),
    ("py:obj", r"tal\.core\.schema\.set_param_coord"),
    ("py:obj", r"tal\.core\.schema\.set_roles"),
    ("py:obj", r"tal\.core\.schema\.set_validity"),
    ("py:obj", r"tal\.ufuncs\..*"),
]

suppress_warnings = [
    "autodoc.duplicate_object",
    "sphinx_autodoc_typehints.forward_reference",
    "toc.not_included",
]


# ----------------------------- HTML theme ------------------------------ #

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    # Keep the navigation reasonably shallow; deeper pages still show local TOCs.
    "navigation_depth": 3,
    "show_toc_level": 2,
}

html_static_path = ["_static"]
