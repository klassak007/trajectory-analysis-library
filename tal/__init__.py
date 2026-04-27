from . import frames
from . import io
from . import spatial
from . import catalog
from . import viz
from . import ufuncs
from .core import AnalysisObject, SchemaError
from .core import merge_schema, set_param_coord, set_roles, set_validity, validate_schema
from .io import install_analysis_object_io_surface
from .utils.frame_ops import install_analysis_object_frames_accessor
from .viz import install_analysis_object_viz_surface
from .utils.frame_schema import get_frames, set_frames

install_analysis_object_frames_accessor()
install_analysis_object_io_surface()
install_analysis_object_viz_surface()

__all__ = [
    "AnalysisObject",
    "SchemaError",
    "frames",
    "io",
    "viz",
    "catalog",
    "set_roles",
    "get_frames",
    "set_frames",
    "set_param_coord",
    "set_validity",
    "merge_schema",
    "validate_schema",
    "spatial",
    "ufuncs",
]
