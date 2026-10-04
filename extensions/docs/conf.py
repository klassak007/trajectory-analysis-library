"""Independent Sphinx build for the experimental extension distribution."""

from pathlib import Path
import sys
import tomllib

EXTENSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXTENSION_ROOT / "src"))
sys.path.insert(0, str(EXTENSION_ROOT.parent))

project = "TAL Extensions"
author = "TAL Contributors"
release = tomllib.loads((EXTENSION_ROOT / "pyproject.toml").read_text())["project"][
    "version"
]
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
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
exclude_patterns = ["_build", "**/.ipynb_checkpoints", ".DS_Store"]
templates_path = ["_templates"]
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3
autosummary_generate = True
autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_mock_imports = ["rosbags", "networkx", "matplotlib", "holoviews", "hvplot"]
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "tal": ("https://klassak007.github.io/trajectory-analysis-library/", None),
}
nitpick_ignore_regex = [
    ("py:class", r".*Options"),
    ("py:class", r".*Accessor"),
    ("py:class", r"tal\.core\..*"),
    ("py:class", r"tal\.spatial\..*"),
    ("py:class", r"AnalysisObject|Position|Vector3|DimLike|Self|optional"),
    ("py:class", r"xr\..*|np\..*"),
]
html_theme = "pydata_sphinx_theme"
html_theme_options = {"navigation_depth": 3, "show_toc_level": 2}
suppress_warnings = [
    "sphinx_autodoc_typehints.forward_reference",
    "autodoc.duplicate_object",
]
