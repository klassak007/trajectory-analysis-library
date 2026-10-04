from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from types import FunctionType, UnionType
from typing import Literal, Union, get_origin


@dataclass(frozen=True)
class SymbolRecord:
    symbol: str
    kind: str
    obj: object
    owner_class: str | None = None


CURATED_SYMBOLS_BY_SUBSYSTEM = {
    "geo": (
        "tal_extensions.geo.register_position_accessor",
        "tal_extensions.geo.options.ENUOptions",
        "tal_extensions.geo.options.GeodesicOptions",
        "tal_extensions.geo.options.GeodeticInterpolationOptions",
        "tal_extensions.geo.options.GeodeticOptions",
        "tal_extensions.geo.options.LocalOrigin",
        "tal_extensions.geo.accessor.PositionGeoAccessor.to_lla",
        "tal_extensions.geo.accessor.PositionGeoAccessor.to_enu",
        "tal_extensions.geo.accessor.PositionGeoAccessor.to_ecef",
        "tal_extensions.geo.temporal.GeodeticParamAccessor.at",
        "tal_extensions.geo.temporal.GeodeticParamAccessor.resample_to",
        "tal_extensions.geo.temporal.GeodeticParamAccessor.interp_like",
        "tal_extensions.geo.geodetic.GeodeticPosition",
        "tal_extensions.geo.geodetic.GeodeticPosition.from_lla",
        "tal_extensions.geo.geodetic.GeodeticPosition.to_ecef",
        "tal_extensions.geo.geodetic.GeodeticPosition.from_ecef",
        "tal_extensions.geo.geodetic.GeodeticPosition.to_enu",
        "tal_extensions.geo.geodetic.GeodeticPosition.to_crs",
        "tal_extensions.geo.geodetic.GeodeticPosition.distance_to",
        "tal_extensions.geo.geodetic.GeodeticPosition.initial_bearing_to",
        "tal_extensions.geo.geodetic.GeodeticPosition.final_bearing_to",
        "tal_extensions.geo.projected.ProjectedPosition",
        "tal_extensions.geo.projected.ProjectedPosition.from_projected",
        "tal_extensions.geo.projected.ProjectedPosition.to_crs",
        "tal_extensions.geo.from_ecef",
        "tal_extensions.geo.transform_crs",
    ),
    "astro": (
        "tal_extensions.astro.options.AstroBackend",
        "tal_extensions.astro.options.AstroIERSOptions",
        "tal_extensions.astro.options.AstroOptions",
        "tal_extensions.astro.options.AstroTimeOptions",
        "tal_extensions.astro.direction.TopocentricDirection",
        "tal_extensions.astro.direction.TopocentricDirection.to_vector3",
        "tal_extensions.astro.sun.SpiceSunOptions",
        "tal_extensions.astro.sun.SunDirectionOptions",
        "tal_extensions.astro.sun.direction_to_sun",
    ),
}

CURATED_SCOPE_COUNTS = {"geo": 26, "astro": 9}

SUPPORT_OWNER_SYMBOLS = ()

DOCSTRING_SECTION_REQUIREMENTS = {
    "tal_extensions.geo.register_position_accessor": (
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.from_lla": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_ecef": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.from_ecef": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_enu": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_crs": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.distance_to": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.initial_bearing_to": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.final_bearing_to": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.at": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.resample_to": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.interp_like": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_lla": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_enu": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_ecef": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.projected.ProjectedPosition.from_projected": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.projected.ProjectedPosition.to_crs": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.from_ecef": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.geo.transform_crs": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.astro.sun.SpiceSunOptions": ("Notes", "Examples"),
    "tal_extensions.astro.sun.SunDirectionOptions": ("Parameters", "Notes", "Examples"),
    "tal_extensions.astro.sun.direction_to_sun": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
    "tal_extensions.astro.direction.TopocentricDirection.to_vector3": (
        "Parameters",
        "Returns",
        "Raises",
        "Notes",
        "Examples",
    ),
}

EXAMPLE_REQUIRED_SYMBOLS = {
    "tal_extensions.geo.register_position_accessor": ("GEO-REGISTRATION",),
    "tal_extensions.geo.options.ENUOptions": ("GEO-ENU-OPTIONS",),
    "tal_extensions.geo.options.GeodesicOptions": ("GEO-GEODESIC-OPTIONS",),
    "tal_extensions.geo.options.GeodeticInterpolationOptions": (
        "GEO-INTERPOLATION-OPTIONS",
    ),
    "tal_extensions.geo.options.GeodeticOptions": ("GEO-GEODETIC-OPTIONS",),
    "tal_extensions.geo.options.LocalOrigin": ("GEO-ENU-OPTIONS",),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_lla": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_enu": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_ecef": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.at": ("GEO-INTERPOLATION",),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.resample_to": (
        "GEO-INTERPOLATION",
    ),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.interp_like": (
        "GEO-INTERPOLATION",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.from_lla": ("GEO-GEODETIC-FROM-LLA",),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_ecef": (
        "GEO-GEODETIC-CONVERSION",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.from_ecef": (
        "GEO-GEODETIC-CONVERSION",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_enu": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_crs": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.geo.geodetic.GeodeticPosition.distance_to": (
        "GEO-DISTANCE-BEARING",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.initial_bearing_to": (
        "GEO-DISTANCE-BEARING",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.final_bearing_to": (
        "GEO-DISTANCE-BEARING",
    ),
    "tal_extensions.geo.projected.ProjectedPosition.from_projected": (
        "GEO-CRS-TRANSFORM",
    ),
    "tal_extensions.geo.projected.ProjectedPosition.to_crs": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.geo.from_ecef": ("GEO-GEODETIC-CONVERSION",),
    "tal_extensions.geo.transform_crs": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.astro.options.AstroBackend": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.options.AstroIERSOptions": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.options.AstroOptions": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.options.AstroTimeOptions": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.direction.TopocentricDirection": (
        "ASTRO-TOPOCENTRIC-DIRECTION",
    ),
    "tal_extensions.astro.direction.TopocentricDirection.to_vector3": (
        "ASTRO-DIRECTION-TO-VECTOR3",
    ),
    "tal_extensions.astro.sun.SpiceSunOptions": ("ASTRO-SUN-OPTIONS",),
    "tal_extensions.astro.sun.SunDirectionOptions": ("ASTRO-SUN-OPTIONS",),
    "tal_extensions.astro.sun.direction_to_sun": ("ASTRO-SUN-DIRECTION",),
}

INVENTORY_EXAMPLE_REQUIRED_SYMBOLS = {
    "tal_extensions.geo.register_position_accessor": ("GEO-REGISTRATION",),
    "tal_extensions.geo.options.ENUOptions": ("GEO-ENU-OPTIONS",),
    "tal_extensions.geo.options.GeodesicOptions": ("GEO-GEODESIC-OPTIONS",),
    "tal_extensions.geo.options.GeodeticInterpolationOptions": (
        "GEO-INTERPOLATION-OPTIONS",
    ),
    "tal_extensions.geo.options.GeodeticOptions": ("GEO-GEODETIC-OPTIONS",),
    "tal_extensions.geo.options.LocalOrigin": ("GEO-ENU-OPTIONS",),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_lla": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_enu": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.accessor.PositionGeoAccessor.to_ecef": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.at": ("GEO-INTERPOLATION",),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.resample_to": (
        "GEO-INTERPOLATION",
    ),
    "tal_extensions.geo.temporal.GeodeticParamAccessor.interp_like": (
        "GEO-INTERPOLATION",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition": ("GEO-GEODETIC-FROM-LLA",),
    "tal_extensions.geo.geodetic.GeodeticPosition.from_lla": ("GEO-GEODETIC-FROM-LLA",),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_ecef": (
        "GEO-GEODETIC-CONVERSION",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.from_ecef": (
        "GEO-GEODETIC-CONVERSION",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_enu": ("GEO-ENU-CONVERSION",),
    "tal_extensions.geo.geodetic.GeodeticPosition.to_crs": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.geo.geodetic.GeodeticPosition.distance_to": (
        "GEO-DISTANCE-BEARING",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.initial_bearing_to": (
        "GEO-DISTANCE-BEARING",
    ),
    "tal_extensions.geo.geodetic.GeodeticPosition.final_bearing_to": (
        "GEO-DISTANCE-BEARING",
    ),
    "tal_extensions.geo.projected.ProjectedPosition": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.geo.projected.ProjectedPosition.from_projected": (
        "GEO-CRS-TRANSFORM",
    ),
    "tal_extensions.geo.projected.ProjectedPosition.to_crs": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.geo.from_ecef": ("GEO-GEODETIC-CONVERSION",),
    "tal_extensions.geo.transform_crs": ("GEO-CRS-TRANSFORM",),
    "tal_extensions.astro.options.AstroIERSOptions": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.options.AstroOptions": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.options.AstroTimeOptions": ("ASTRO-OPTIONS",),
    "tal_extensions.astro.direction.TopocentricDirection": (
        "ASTRO-TOPOCENTRIC-DIRECTION",
    ),
}

DUUNDER_FAMILY_DOC_OWNER = {}


_MISSING_SYMBOL = object()


def _import_candidate(module_name: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ImportError:
        return _MISSING_SYMBOL


def _resolve_candidate_attrs(obj: object, parts: list[str]) -> object:
    try:
        for part in parts:
            obj = getattr(obj, part)
    except AttributeError:
        return _MISSING_SYMBOL
    return obj


def _resolve_symbol(symbol: str) -> object:
    parts = symbol.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        obj = _import_candidate(module_name)
        if obj is _MISSING_SYMBOL:
            continue
        obj = _resolve_candidate_attrs(obj, parts[index:])
        if obj is _MISSING_SYMBOL:
            continue
        return obj
    raise ValueError(f"Unable to resolve symbol: {symbol}")


def _owner_class_symbol(symbol: str) -> str | None:
    parts = symbol.split(".")
    if len(parts) < 3:
        return None
    owner = ".".join(parts[:-1])
    try:
        obj = _resolve_symbol(owner)
    except ValueError:
        return None
    if inspect.isclass(obj):
        return owner
    return None


def _symbol_kind(obj: object, owner_class: str | None) -> str:
    if get_origin(obj) is Literal:
        return "type_alias"
    if get_origin(obj) in {Union, UnionType} or isinstance(obj, UnionType):
        return "type_alias"
    if inspect.isclass(obj):
        return "class"
    if isinstance(obj, FunctionType):
        return "method" if owner_class is not None else "function"
    if isinstance(obj, property):
        return "property"
    return "symbol"


def iter_curated_public_symbols() -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    for subsystem in CURATED_SCOPE_COUNTS:
        for symbol in CURATED_SYMBOLS_BY_SUBSYSTEM[subsystem]:
            obj = _resolve_symbol(symbol)
            owner_class = _owner_class_symbol(symbol)
            records.append(
                SymbolRecord(
                    symbol=symbol,
                    kind=_symbol_kind(obj, owner_class),
                    obj=obj,
                    owner_class=owner_class,
                )
            )
    return records


def iter_scoped_public_symbols() -> list[SymbolRecord]:
    records: dict[str, SymbolRecord] = {
        record.symbol: record for record in iter_curated_public_symbols()
    }
    for symbol in SUPPORT_OWNER_SYMBOLS:
        obj = _resolve_symbol(symbol)
        records[symbol] = SymbolRecord(
            symbol=symbol, kind="class", obj=obj, owner_class=None
        )
    return [records[key] for key in sorted(records)]


def iter_inventory_example_symbols() -> list[SymbolRecord]:
    records: list[SymbolRecord] = []
    for symbol in sorted(INVENTORY_EXAMPLE_REQUIRED_SYMBOLS):
        obj = _resolve_symbol(symbol)
        owner_class = _owner_class_symbol(symbol)
        records.append(
            SymbolRecord(
                symbol=symbol,
                kind=_symbol_kind(obj, owner_class),
                obj=obj,
                owner_class=owner_class,
            )
        )
    return records


def curated_scope_counts() -> dict[str, int]:
    return {key: len(CURATED_SYMBOLS_BY_SUBSYSTEM[key]) for key in CURATED_SCOPE_COUNTS}


def required_example_ids() -> set[str]:
    out: set[str] = set()
    for ids in EXAMPLE_REQUIRED_SYMBOLS.values():
        out.update(ids)
    return out


def inventory_required_example_ids() -> set[str]:
    out: set[str] = set()
    for ids in INVENTORY_EXAMPLE_REQUIRED_SYMBOLS.values():
        out.update(ids)
    return out
