from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CatalogBackendOption = Literal["auto", "dataset", "datatree"]
UnknownFieldPolicy = Literal["error", "ignore"]
MetadataQueryEagerPolicy = Literal["forbid", "allow"]
MetadataScalarTarget = Literal["batch_coord", "attr", "none"]
MetadataNonscalarTarget = Literal["none", "attr"]


@dataclass(frozen=True)
class CatalogInitOptions:
    backend: CatalogBackendOption = "auto"
    batch_dim: str | None = None


@dataclass(frozen=True)
class CatalogQueryOptions:
    unknown_field_policy: UnknownFieldPolicy = "error"
    metadata_eager_policy: MetadataQueryEagerPolicy = "forbid"


@dataclass(frozen=True)
class CatalogMetadataPromotionOptions:
    scalar_target: MetadataScalarTarget = "batch_coord"
    nonscalar_target: MetadataNonscalarTarget = "none"


@dataclass(frozen=True)
class CatalogExtractOptions:
    ignore_missing_vars: bool = False
    require_sequence_size_coord: bool = False
    metadata_promotion: CatalogMetadataPromotionOptions = CatalogMetadataPromotionOptions()
    validate: bool = True


def coerce_catalog_init_options(
    *,
    backend: CatalogBackendOption = "auto",
    batch_dim: str | None = None,
    owner: str,
) -> CatalogInitOptions:
    if backend not in {"auto", "dataset", "datatree"}:
        raise ValueError(
            f"{owner}: backend must be one of 'auto', 'dataset', or 'datatree'; got {backend!r}."
        )
    if batch_dim is not None and (not isinstance(batch_dim, str) or not batch_dim):
        raise ValueError(f"{owner}: batch_dim must be a non-empty string when provided.")
    return CatalogInitOptions(backend=backend, batch_dim=batch_dim)


def coerce_catalog_query_options(
    opts: CatalogQueryOptions | None,
    *,
    owner: str,
) -> CatalogQueryOptions:
    if opts is None:
        return CatalogQueryOptions()
    if not isinstance(opts, CatalogQueryOptions):
        raise TypeError(f"{owner}: opts must be CatalogQueryOptions or None.")
    if opts.unknown_field_policy not in {"error", "ignore"}:
        raise ValueError(
            f"{owner}: unknown_field_policy must be 'error' or 'ignore'; got {opts.unknown_field_policy!r}."
        )
    if opts.metadata_eager_policy not in {"forbid", "allow"}:
        raise ValueError(
            f"{owner}: metadata_eager_policy must be 'forbid' or 'allow'; got {opts.metadata_eager_policy!r}."
        )
    return opts


def coerce_catalog_metadata_promotion_options(
    opts: CatalogMetadataPromotionOptions | None,
    *,
    owner: str,
) -> CatalogMetadataPromotionOptions:
    if opts is None:
        return CatalogMetadataPromotionOptions()
    if not isinstance(opts, CatalogMetadataPromotionOptions):
        raise TypeError(f"{owner}: metadata_promotion must be CatalogMetadataPromotionOptions or None.")
    if opts.scalar_target not in {"batch_coord", "attr", "none"}:
        raise ValueError(
            f"{owner}: scalar_target must be one of 'batch_coord', 'attr', 'none'; got {opts.scalar_target!r}."
        )
    if opts.nonscalar_target not in {"none", "attr"}:
        raise ValueError(
            f"{owner}: nonscalar_target must be one of 'none' or 'attr'; got {opts.nonscalar_target!r}."
        )
    return opts


def coerce_catalog_extract_options(
    opts: CatalogExtractOptions | None,
    *,
    owner: str,
) -> CatalogExtractOptions:
    if opts is None:
        return CatalogExtractOptions()
    if not isinstance(opts, CatalogExtractOptions):
        raise TypeError(f"{owner}: opts must be CatalogExtractOptions or None.")
    promotion = coerce_catalog_metadata_promotion_options(
        opts.metadata_promotion,
        owner=owner,
    )
    return CatalogExtractOptions(
        ignore_missing_vars=bool(opts.ignore_missing_vars),
        require_sequence_size_coord=bool(opts.require_sequence_size_coord),
        metadata_promotion=promotion,
        validate=bool(opts.validate),
    )


def coerce_extract_variables(
    variables: object,
    *,
    owner: str,
) -> tuple[str, ...] | None:
    if variables is None:
        return None
    if isinstance(variables, str):
        if not variables:
            raise ValueError(f"{owner}: variable names must be non-empty strings.")
        return (variables,)
    if not isinstance(variables, (tuple, list)):
        raise TypeError(f"{owner}: variables must be None, a string, or a tuple/list of strings.")
    names: list[str] = []
    seen: set[str] = set()
    for idx, name in enumerate(variables):
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{owner}: variables[{idx}] must be a non-empty string; got {name!r}."
            )
        if name in seen:
            raise ValueError(f"{owner}: duplicate variable {name!r} in extract request.")
        seen.add(name)
        names.append(name)
    return tuple(names)


__all__ = [
    "CatalogBackendOption",
    "CatalogExtractOptions",
    "CatalogInitOptions",
    "CatalogMetadataPromotionOptions",
    "CatalogQueryOptions",
    "coerce_catalog_extract_options",
    "coerce_catalog_init_options",
    "coerce_catalog_metadata_promotion_options",
    "coerce_catalog_query_options",
    "coerce_extract_variables",
]
