from __future__ import annotations

from typing import Literal

import xarray as xr

from tal.core.typed_lifecycle import TypedLifecycleContext, TypedLifecycleSpec

from ..construction import SpatialConstructionPlan, apply_spatial_construction
from .family import (
    KinematicsFamilyConfig,
    enforce_angular_invariants,
    enforce_family_invariants,
    enforce_linear_invariants,
    normalize_typed_metadata,
)

KinematicsLifecycleRole = Literal["linear", "angular", "family"]


def _normalize_role(role: str) -> KinematicsLifecycleRole:
    if role in {"linear", "angular", "family"}:
        return role
    raise ValueError(f"kinematics lifecycle role must be 'linear', 'angular', or 'family'; got {role!r}.")


def _normalize_kinematics_metadata(
    ds: xr.Dataset,
    ctx: TypedLifecycleContext,
    *,
    cfg: KinematicsFamilyConfig,
    role: KinematicsLifecycleRole,
) -> xr.Dataset:
    if role == "linear":
        out = normalize_typed_metadata(
            ds,
            rep_getter=cfg.get_linear_rep,
            rep_setter=cfg.set_linear_rep,
            expected_kind=cfg.linear_kind,
            cfg=cfg,
            owner=ctx.owner,
        )
    elif role == "angular":
        out = normalize_typed_metadata(
            ds,
            rep_getter=cfg.get_angular_rep,
            rep_setter=cfg.set_angular_rep,
            expected_kind=cfg.angular_kind,
            cfg=cfg,
            owner=ctx.owner,
        )
    else:
        out = normalize_typed_metadata(
            ds,
            rep_getter=cfg.get_family_rep,
            rep_setter=cfg.set_family_rep,
            expected_kind=cfg.family_kind,
            cfg=cfg,
            owner=ctx.owner,
        )
    if ctx.options is None:
        return out
    if not isinstance(ctx.options, SpatialConstructionPlan):
        raise TypeError(
            f"{ctx.owner}: typed spatial construction options must be SpatialConstructionPlan."
        )
    return apply_spatial_construction(out, plan=ctx.options, owner=ctx.owner)


def _enforce_kinematics_invariants(
    ds: xr.Dataset,
    ctx: TypedLifecycleContext,
    *,
    cfg: KinematicsFamilyConfig,
    role: KinematicsLifecycleRole,
) -> None:
    if role == "linear":
        enforce_linear_invariants(ds, owner=ctx.owner, cfg=cfg)
        return
    if role == "angular":
        enforce_angular_invariants(ds, owner=ctx.owner, cfg=cfg)
        return
    enforce_family_invariants(ds, cfg=cfg, owner=ctx.owner)


def make_kinematics_lifecycle_spec(
    *,
    type_name: str,
    owner_prefix: str,
    cfg: KinematicsFamilyConfig,
    role: KinematicsLifecycleRole,
) -> TypedLifecycleSpec:
    normalized_role = _normalize_role(role)

    def normalize(ds: xr.Dataset, ctx: TypedLifecycleContext) -> xr.Dataset:
        return _normalize_kinematics_metadata(ds, ctx, cfg=cfg, role=normalized_role)

    def enforce(ds: xr.Dataset, ctx: TypedLifecycleContext) -> None:
        _enforce_kinematics_invariants(ds, ctx, cfg=cfg, role=normalized_role)

    return TypedLifecycleSpec(
        type_name=type_name,
        owner_prefix=owner_prefix,
        normalize=normalize,
        enforce=enforce,
    )


__all__ = [
    "KinematicsLifecycleRole",
    "make_kinematics_lifecycle_spec",
]
