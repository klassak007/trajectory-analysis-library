from __future__ import annotations

from collections.abc import Mapping

from .extract import ensure_grouping_context_for_query_extract
from .metadata_domain import discover_metadata_fields, project_metadata
from .options import CatalogQueryOptions
from .query_eval import apply_query_predicate
from .query_plan import normalize_catalog_query_plan
from .types import CatalogState


def run_catalog_query(
    state: CatalogState,
    *,
    where: object,
    filters: Mapping[str, object],
    options: CatalogQueryOptions,
    owner: str,
) -> CatalogState:
    ensure_grouping_context_for_query_extract(state, owner=owner)
    fields = discover_metadata_fields(state, owner=owner)
    plan = normalize_catalog_query_plan(
        where=where,
        filters=filters,
        field_index=fields,
        options=options,
        owner=owner,
    )
    projection = project_metadata(
        state,
        fields=plan.projections,
        options=options,
        owner=owner,
    )
    return apply_query_predicate(
        state,
        predicate=plan.predicate,
        labels=projection.labels,
        columns=projection.columns,
        owner=owner,
    )


__all__ = ["run_catalog_query"]
