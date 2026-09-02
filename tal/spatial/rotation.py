from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.analysis_object import AnalysisObject
from tal.core.orchestration.alignment import align_exact_for_plan
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.context import resolve_semantic_topology_from_dataset
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.topology import (
    SEMANTIC_NON_CORE_POLICY,
    STRICT_NON_CORE_POLICY,
    TopologyPolicy,
    TopologyOperand,
    resolve_binary_topology,
)
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    resolve_single_numeric_var_single_core_dim,
    select_single_numeric_var,
)
from tal.core.orchestration.finalize import transfer_dataset_attrs
from tal.core.schema_errors import SchemaError
from tal.core.schema_read import read_param_coord_name, validate_schema_if_needed
from tal.utils.frame_schema import get_frames, set_frames
from tal.utils.topology_operation_families import operation_intent_support_for_operation_family

from .conversion.finalize import (
    allocate_dim_pair,
    allocate_free_dim_name,
    conversion_dataset_from_array,
    dataset_dim_names,
    finalize_conversion_dataset,
)
from .policies.frame import resolve_compose_output_frames
from .ops.frame_api_ops import rotation_class_solve_path_transform
from .ops.quat_role_dim_ops import resolve_quat_dim_with_role_fallback, resolve_rotation_component_dims_for_reduce
from .metadata import (
    get_rotation_rep,
    normalize_configuration_relation_semantics,
    set_rotation_rep,
    validate_spatial_roles,
)
from .ops.rotation_apply_ops import rotation_apply
from .kernels.rotation_compose_kernels import compose_quat_kernel, inverse_quat_kernel
from .kernels.rotation_kernels import matrix_to_quat_kernel, quat_to_matrix_kernel
from .policies.wrap import wrap_as

if TYPE_CHECKING:
    from tal.frames import Frame
    from tal.core.param_ops.types import ParamEvalOptions
    from .path_solve import PathSolveOptions
    from .temporal.options import RotationTemporalOptions

_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_MATRIX_LABELS: tuple[str, str, str] = ("x", "y", "z")
_ALLOWED_TARGET_REPS = {"quat", "matrix"}


def _coerce_rotation_source(value: object, *, owner: str) -> AnalysisObject:
    return coerce_analysis_object_input(value, owner=owner)


def _coerce_rotation_operand(value: object, *, owner: str) -> "Rotation":
    if isinstance(value, Rotation):
        return value
    try:
        return Rotation(value)
    except SchemaError:
        raise
    except TypeError as exc:
        raise TypeError(
            f"{owner}: other must be Rotation, AnalysisObject, xr.Dataset, or xr.DataArray."
        ) from exc
    except ValueError as exc:
        raise ValueError(f"{owner}: other operand is not a valid Rotation: {exc}") from exc


def _require_quat_labels(ds: xr.Dataset, *, axis: str, owner: str) -> None:
    labels = require_explicit_unique_dim_labels(ds, dim=axis, owner=owner, what="Rotation")
    require_exact_labels(labels, expected=_QUAT_LABELS, owner=owner, what="Rotation core")


def _require_quat_var_and_dim(ds: xr.Dataset, *, owner: str) -> tuple[str, str]:
    var_name = select_single_numeric_var(ds, owner=owner, what="Rotation")
    quat_dim = resolve_quat_dim_with_role_fallback(ds, var_name=var_name, owner=owner, what="Rotation")
    require_var_contains_dims(ds, var_name=var_name, required_dims=(quat_dim,), owner=owner, what="Rotation")
    _require_quat_labels(ds, axis=quat_dim, owner=owner)
    return var_name, quat_dim


def _require_matrix_core_dims(ds: xr.Dataset, *, owner: str) -> tuple[str, str]:
    from tal.core.schema_read import read_roles

    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: Rotation requires declared roles.")
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: Rotation matrix layout requires exactly two core dims; got {core_dims!r}.")
    row_dim, col_dim = core_dims
    if row_dim == col_dim:
        raise ValueError(f"{owner}: Rotation matrix layout core dims must be distinct; got {core_dims!r}.")
    if int(ds.sizes.get(row_dim, -1)) != 3 or int(ds.sizes.get(col_dim, -1)) != 3:
        raise ValueError(f"{owner}: Rotation matrix core dims must both have length 3.")
    return row_dim, col_dim


def _require_matrix_labels(ds: xr.Dataset, *, row_dim: str, col_dim: str, owner: str) -> None:
    row_labels = require_explicit_unique_dim_labels(ds, dim=row_dim, owner=owner, what="Rotation matrix")
    col_labels = require_explicit_unique_dim_labels(ds, dim=col_dim, owner=owner, what="Rotation matrix")
    require_exact_labels(row_labels, expected=_MATRIX_LABELS, owner=owner, what=f"Rotation matrix row dim {row_dim!r}")
    require_exact_labels(col_labels, expected=_MATRIX_LABELS, owner=owner, what=f"Rotation matrix col dim {col_dim!r}")


def _enforce_quat_layout_invariants(ds: xr.Dataset, *, owner: str) -> None:
    _require_quat_var_and_dim(ds, owner=owner)


def _enforce_matrix_layout_invariants(ds: xr.Dataset, *, owner: str) -> None:
    var_name = select_single_numeric_var(ds, owner=owner, what="Rotation matrix layout")
    row_dim, col_dim = _require_matrix_core_dims(ds, owner=owner)
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(row_dim, col_dim),
        owner=owner,
        what="Rotation matrix layout",
    )
    _require_matrix_labels(ds, row_dim=row_dim, col_dim=col_dim, owner=owner)


def _enforce_rotation_dataset_invariants(ds: xr.Dataset, *, owner: str) -> None:
    candidate = validate_schema_if_needed(ds)
    validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    rep = get_rotation_rep(candidate, owner=owner)
    if rep == "quat":
        _enforce_quat_layout_invariants(candidate, owner=owner)
        return
    if rep == "matrix":
        _enforce_matrix_layout_invariants(candidate, owner=owner)
        return
    raise ValueError(f"{owner}: unsupported rotation representation {rep!r}.")


def _normalize_rotation_metadata(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    rep = get_rotation_rep(ds, owner=owner)
    out = set_rotation_rep(ds, rep=rep, validate=False, owner=owner)
    return normalize_configuration_relation_semantics(
        out,
        validate=False,
        owner=owner,
    )


def _finalize_rotation_conversion(
    ds: xr.Dataset,
    *,
    core_dims: tuple[str, ...],
    rep: Literal["quat", "matrix"],
    owner: str,
) -> xr.Dataset:
    return finalize_conversion_dataset(
        ds,
        core_dims=core_dims,
        rep_value=rep,
        set_rep=set_rotation_rep,
        owner=owner,
    )


def _convert_quat_to_matrix(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    candidate = validate_schema_if_needed(ds)
    var_name, quat_dim = _require_quat_var_and_dim(candidate, owner=owner)
    row_dim, col_dim = allocate_dim_pair(
        existing_dims=dataset_dim_names(candidate),
        first_candidates=("row", "rot_row", "matrix_row"),
        first_base="row",
        first_what="rotation matrix row dim",
        second_candidates=("col", "rot_col", "matrix_col"),
        second_base="col",
        second_what="rotation matrix col dim",
        owner=owner,
    )
    matrix = xr.apply_ufunc(
        quat_to_matrix_kernel,
        candidate[var_name],
        input_core_dims=[[quat_dim]],
        output_core_dims=[[row_dim, col_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {row_dim: 3, col_dim: 3}},
    )
    matrix = matrix.assign_coords({row_dim: list(_MATRIX_LABELS), col_dim: list(_MATRIX_LABELS)})
    out = conversion_dataset_from_array(matrix, var_name=var_name, source_ds=candidate)
    return _finalize_rotation_conversion(out, core_dims=(row_dim, col_dim), rep="matrix", owner=owner)


def _convert_matrix_to_quat(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    candidate = validate_schema_if_needed(ds)
    var_name = select_single_numeric_var(candidate, owner=owner, what="Rotation matrix layout")
    row_dim, col_dim = _require_matrix_core_dims(candidate, owner=owner)
    require_var_contains_dims(
        candidate,
        var_name=var_name,
        required_dims=(row_dim, col_dim),
        owner=owner,
        what="Rotation matrix layout",
    )
    _require_matrix_labels(candidate, row_dim=row_dim, col_dim=col_dim, owner=owner)
    quat_dim = allocate_free_dim_name(
        existing_dims=dataset_dim_names(candidate),
        candidates=("quat", "quat_component", "rotation_component"),
        base="quat",
        owner=owner,
        what="rotation quaternion dim",
    )
    quat = xr.apply_ufunc(
        matrix_to_quat_kernel,
        candidate[var_name],
        input_core_dims=[[row_dim, col_dim]],
        output_core_dims=[[quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}},
    )
    quat = quat.assign_coords({quat_dim: list(_QUAT_LABELS)})
    out = conversion_dataset_from_array(quat, var_name=var_name, source_ds=candidate)
    return _finalize_rotation_conversion(out, core_dims=(quat_dim,), rep="quat", owner=owner)


def _normalize_target_rep(rep: object, *, owner: str) -> Literal["quat", "matrix"]:
    if not isinstance(rep, str):
        raise TypeError(f"{owner}: rep must be a string; got {type(rep).__name__}.")
    cleaned = rep.strip()
    if not cleaned:
        raise ValueError(f"{owner}: rep must be a non-empty string.")
    if cleaned == "quat":
        return "quat"
    if cleaned == "matrix":
        return "matrix"
    raise ValueError(f"{owner}: unsupported target rotation representation {cleaned!r}; allowed={sorted(_ALLOWED_TARGET_REPS)!r}.")
def _wrap_rotation_output(ds: xr.Dataset, *, validate: bool) -> "Rotation":
    return wrap_as(Rotation, ds, validate=validate)


def _prepare_compose_quat_inputs(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    policy: TopologyPolicy,
) -> tuple[xr.Dataset, str, str, str, xr.DataArray, xr.DataArray]:
    left_candidate = validate_schema_if_needed(left_ds)
    right_candidate = validate_schema_if_needed(right_ds)
    left_var, left_quat_dim = resolve_single_numeric_var_single_core_dim(left_candidate, owner=owner, what="left rotation")
    right_var, right_quat_dim = resolve_single_numeric_var_single_core_dim(right_candidate, owner=owner, what="right rotation")
    _require_quat_labels(left_candidate, axis=left_quat_dim, owner=owner)
    _require_quat_labels(right_candidate, axis=right_quat_dim, owner=owner)
    left_da = left_candidate[left_var]
    right_da = right_candidate[right_var]
    plan = resolve_binary_topology(
        TopologyOperand(
            index=0,
            data=left_da,
            semantic=resolve_semantic_topology_from_dataset(
                left_candidate,
                var_name=left_var,
                core_dims=(left_quat_dim,),
                owner=owner,
                what="left rotation",
                allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
                allow_missing_batch_dims=policy.mode == "semantic_broadcast",
            ),
            param_coord=read_param_coord_name(left_candidate),
        ),
        TopologyOperand(
            index=1,
            data=right_da,
            semantic=resolve_semantic_topology_from_dataset(
                right_candidate,
                var_name=right_var,
                core_dims=(right_quat_dim,),
                owner=owner,
                what="right rotation",
                allow_missing_sequence_dim=policy.mode == "semantic_broadcast",
                allow_missing_batch_dims=policy.mode == "semantic_broadcast",
            ),
            param_coord=read_param_coord_name(right_candidate),
        ),
        owner=owner,
        what="compose",
        policy=policy,
    )
    aligned_left, aligned_right = align_exact_for_plan(plan, owner=owner, what="compose")
    return left_candidate, left_var, left_quat_dim, right_quat_dim, aligned_left, aligned_right


def _compose_quat_datasets(
    left_ds: xr.Dataset,
    right_ds: xr.Dataset,
    *,
    owner: str,
    policy: TopologyPolicy,
) -> tuple[xr.Dataset, str]:
    left_candidate, left_var, left_quat_dim, right_quat_dim, aligned_left, aligned_right = _prepare_compose_quat_inputs(
        left_ds,
        right_ds,
        owner=owner,
        policy=policy,
    )
    out = xr.apply_ufunc(
        partial(_wrap_compose_quat_kernel, owner=owner),
        aligned_left,
        aligned_right,
        input_core_dims=[[left_quat_dim], [right_quat_dim]],
        output_core_dims=[[left_quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {left_quat_dim: 4}},
    )
    out = out.assign_coords({left_quat_dim: list(_QUAT_LABELS)})
    out_ds = out.to_dataset(name=left_var)
    out_ds = transfer_dataset_attrs(left_candidate, out_ds, validate=False)
    return _finalize_rotation_conversion(out_ds, core_dims=(left_quat_dim,), rep="quat", owner=owner), left_quat_dim


def _inverse_quat_dataset(ds: xr.Dataset, *, owner: str) -> tuple[xr.Dataset, str]:
    candidate = validate_schema_if_needed(ds)
    var_name, quat_dim = _require_quat_var_and_dim(candidate, owner=owner)
    quat = xr.apply_ufunc(
        partial(_wrap_inverse_quat_kernel, owner=owner),
        candidate[var_name],
        input_core_dims=[[quat_dim]],
        output_core_dims=[[quat_dim]],
        vectorize=False,
        dask="parallelized",
        output_dtypes=[np.float64],
        dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}},
    )
    quat = quat.assign_coords({quat_dim: list(_QUAT_LABELS)})
    out_ds = quat.to_dataset(name=var_name)
    out_ds = transfer_dataset_attrs(candidate, out_ds, validate=False)
    return _finalize_rotation_conversion(out_ds, core_dims=(quat_dim,), rep="quat", owner=owner), quat_dim


def _wrap_compose_quat_kernel(left: np.ndarray, right: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return compose_quat_kernel(left, right)
    except ValueError as exc:
        if owner.startswith("spatial.path_solve."):
            raise ValueError(f"{owner}: compose kernel failed after alignment.") from exc
        raise ValueError(f"{owner}: compose kernel failed after alignment: {exc}") from exc


def _wrap_inverse_quat_kernel(values: np.ndarray, *, owner: str) -> np.ndarray:
    try:
        return inverse_quat_kernel(values)
    except ValueError as exc:
        if owner.startswith("spatial.path_solve."):
            raise ValueError(f"{owner}: inverse kernel failed.") from exc
        raise ValueError(f"{owner}: inverse kernel failed: {exc}") from exc


def _convert_quat_result_to_rep(
    ds: xr.Dataset,
    *,
    quat_dim: str,
    target_rep: str,
    owner: str,
) -> xr.Dataset:
    if target_rep == "matrix":
        return _convert_quat_to_matrix(ds, owner=owner)
    return _finalize_rotation_conversion(ds, core_dims=(quat_dim,), rep="quat", owner=owner)


def _rotation_compose_with_owner(
    left: "Rotation",
    right: "Rotation",
    *,
    validate: bool,
    owner: str,
) -> "Rotation":
    left._enforce_invariants(owner=owner)
    right._enforce_invariants(owner=owner)
    left_rep = get_rotation_rep(left_ds := analysis_object_dataset(left), owner=owner)
    parent, child = resolve_compose_output_frames(left_ds, analysis_object_dataset(right), owner=owner)
    left_quat = left.as_quat(validate=False)
    right_quat = right.as_quat(validate=False)
    selection = select_topology_policy_with_intents(
        (left, right),
        owner=owner,
        operation_family="spatial.rotation.compose",
        support=operation_intent_support_for_operation_family(
            "spatial.rotation.compose",
            owner=owner,
        ),
        strict_policy=STRICT_NON_CORE_POLICY,
        semantic_policy=SEMANTIC_NON_CORE_POLICY,
    )
    policy = selection.policy
    composed_quat, quat_dim = _compose_quat_datasets(
        analysis_object_dataset(left_quat),
        analysis_object_dataset(right_quat),
        owner=owner,
        policy=policy,
    )
    composed_quat = set_frames(composed_quat, parent=parent, child=child, validate=False)
    result = _convert_quat_result_to_rep(composed_quat, quat_dim=quat_dim, target_rep=left_rep, owner=owner)
    return _wrap_rotation_output(result, validate=validate)


def _rotation_inverse_with_owner(
    rotation: "Rotation",
    *,
    validate: bool,
    owner: str,
) -> "Rotation":
    rotation._enforce_invariants(owner=owner)
    source_rep = get_rotation_rep(source := analysis_object_dataset(rotation), owner=owner)
    source_quat = rotation.as_quat(validate=False)
    inverse_quat, quat_dim = _inverse_quat_dataset(analysis_object_dataset(source_quat), owner=owner)
    parent, child = get_frames(source)
    inverse_quat = set_frames(inverse_quat, parent=child, child=parent, validate=False)
    result = _convert_quat_result_to_rep(inverse_quat, quat_dim=quat_dim, target_rep=source_rep, owner=owner)
    return _wrap_rotation_output(result, validate=validate)


class Rotation(AnalysisObject):
    """3D orientation type with quaternion/matrix representations.

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    CANONICAL_REP: str = "quat"
    QUAT_LABELS: tuple[str, str, str, str] = _QUAT_LABELS
    MATRIX_LABELS: tuple[str, str, str] = _MATRIX_LABELS

    def __init__(self, data: "AnalysisObject | xr.Dataset | xr.DataArray") -> None:
        source = _coerce_rotation_source(data, owner="spatial.rotation.__init__")
        super().__init__(analysis_object_dataset(source))
        self._normalize_metadata(owner="spatial.rotation.__init__")
        self._enforce_invariants(owner="spatial.rotation.__init__")

    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> "Rotation":
        obj = super()._from_validated(ds)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_validated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_validated")
        return obj
    
    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray, *, schema_prepared: bool = False) -> "Rotation":
        obj = super()._from_unvalidated(ds, schema_prepared=schema_prepared)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_unvalidated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_unvalidated")
        return obj
    
    def _normalize_metadata(self, *, owner: str) -> None:
        normalized = _normalize_rotation_metadata(analysis_object_dataset(self), owner=owner)
        self._bind_dataset(normalized)

    def _enforce_invariants(self, *, owner: str) -> None:
        _enforce_rotation_dataset_invariants(analysis_object_dataset(self), owner=owner)

    def _required_component_dims_for_reduce(self) -> tuple[str, ...]:
        return resolve_rotation_component_dims_for_reduce(
            analysis_object_dataset(self),
            owner="spatial.rotation._required_component_dims_for_reduce",
        )
    
    def to_rep(self, rep: Literal["quat", "matrix"], *, validate: bool = True) -> "Rotation":
        """Convert this rotation between quaternion and matrix representation.

        Parameters
        ----------
        rep : {"quat", "matrix"}
            Target storage representation.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Rotation
            Rotation with the requested representation and preserved non-core
            topology.

        Notes
        -----
        Quaternion layout uses one four-label core dimension ``x, y, z, w``.
        Matrix layout uses two 3-label core dimensions.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Rotation
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... )
        >>> Rotation(ao).to_rep("matrix").as_dataset()["rotation"].shape[-2:]
        (3, 3)
        """
        owner = "spatial.rotation.to_rep"
        target = _normalize_target_rep(rep, owner=owner)
        self._enforce_invariants(owner=owner)
        source = analysis_object_dataset(self)
        current = get_rotation_rep(source, owner=owner)
        if target == current:
            return _wrap_rotation_output(source, validate=validate)
        if target == "matrix":
            converted = _convert_quat_to_matrix(source, owner=owner)
        else:
            converted = _convert_matrix_to_quat(source, owner=owner)
        return _wrap_rotation_output(converted, validate=validate)
    
    def as_quat(self, *, validate: bool = True) -> "Rotation":
        """Return this rotation in quaternion representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Rotation
            Rotation stored as quaternion components.

        Notes
        -----
        This is equivalent to ``to_rep("quat")``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Rotation
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... )
        >>> Rotation(ao).as_quat().as_dataset()["rotation"].shape[-1]
        4
        """
        return self.to_rep("quat", validate=validate)
    
    def as_matrix(self, *, validate: bool = True) -> "Rotation":
        """Return this rotation in 3x3 matrix representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Rotation
            Rotation stored as 3x3 matrices.

        Notes
        -----
        This is equivalent to ``to_rep("matrix")``.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Rotation
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... )
        >>> Rotation(ao).as_matrix().as_dataset()["rotation"].shape[-2:]
        (3, 3)
        """
        return self.to_rep("matrix", validate=validate)
    
    @property
    def preferred_interpolator(self) -> str:
        """Return the default interpolation method name for rotations.

        Returns
        -------
        str
            Resolved property value.
        """
        return "slerp"
    
    @property
    def param(self) -> "RotationParamAccessor":
        """Return the parameter-domain accessor for temporal rotation operations.

        Returns
        -------
        RotationParamAccessor
            Resolved property value.
        """
        from .temporal.accessor import RotationParamAccessor

        return RotationParamAccessor(self)
    
    def compose(
        self,
        other: "Rotation | AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        validate: bool = True,
    ) -> "Rotation":
        """Compose this rotation with ``other`` (apply ``other`` after ``self``).

        Parameters
        ----------
        other : Rotation | AnalysisObject | xr.Dataset | xr.DataArray
            Rotation to compose with this one.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Rotation
            Composed rotation with aligned non-core topology.

        Notes
        -----
        Inputs are aligned by xarray/TAL topology rules before quaternion
        composition. Frame metadata is resolved fail-closed.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Rotation
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... )
        >>> isinstance(Rotation(ao).compose(Rotation(ao)), Rotation)
        True
        """
        owner = "spatial.rotation.compose"
        right = _coerce_rotation_operand(other, owner=owner)
        return _rotation_compose_with_owner(self, right, validate=validate, owner=owner)
    
    def inverse(self, *, validate: bool = True) -> "Rotation":
        """Return the inverse rotation (same magnitude, opposite orientation).

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Rotation
            Inverse rotation.

        Notes
        -----
        The inverse preserves sequence, batch, parameter, validity, and frame
        metadata.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Rotation
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... )
        >>> Rotation(ao).inverse().as_dataset()["rotation"].shape
        (1, 4)
        """
        owner = "spatial.rotation.inverse"
        return _rotation_inverse_with_owner(self, validate=validate, owner=owner)
    
    def slerp(
        self,
        query: "xr.DataArray | np.ndarray | Sequence[float] | float",
        *,
        on: str | None = None,
        opts: "RotationTemporalOptions | ParamEvalOptions | None" = None,
        validate: bool = True,
        sequence_dim: str | None = None,
        batch_dims: "Sequence[str] | None" = None,
        sequence_size_coord: str | None = None,
    ) -> "Rotation":
        """Interpolate rotation samples with spherical linear interpolation.

        Parameters
        ----------
        query : xr.DataArray | np.ndarray | Sequence[float] | float
            Query coordinate/grid for param-aware selection/evaluation.
        on : str | None, optional
            Coordinate/dimension name used as the operation domain.
        opts : RotationTemporalOptions | ParamEvalOptions | None, optional
            When ``None``, defaults use rotation-aware interpolation. ``ParamEvalOptions`` allows nearest/linear interpolation policies. ``RotationTemporalOptions`` controls rotation method (including ``'slerp'``), duplicate handling, and query dim naming.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.
        sequence_dim : str | None, optional
            Optional override for the sequence dimension used by temporal semantics.
        batch_dims : Sequence[str] | None, optional
            Optional override for batch dimensions used by temporal semantics.
        sequence_size_coord : str | None, optional
            Optional sequence-size coordinate used for ragged validity handling.

        Returns
        -------
        Rotation
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> opts = ParamEvalOptions(method='nearest')
        >>> isinstance(opts, ParamEvalOptions)
        True
        """
        from .ops.rotation_temporal_ops import rotation_param_at
        from .temporal.options import as_rotation_method, coerce_rotation_temporal_options

        normalized = coerce_rotation_temporal_options(opts, owner="spatial.rotation.slerp")
        slerp_opts = as_rotation_method(normalized, method="slerp")
        return rotation_param_at(
            self,
            query=query,
            on=on,
            opts=slerp_opts,
            validate=validate,
            sequence_dim=sequence_dim,
            batch_dims=batch_dims,
            sequence_size_coord=sequence_size_coord,
            owner="spatial.rotation.slerp",
        )
    
    def apply(
        self,
        target: "Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration",
        *,
        validate: bool = True,
    ) -> "Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration":
        """Rotate a compatible spatial target by this rotation.

        Parameters
        ----------
        target : Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration
            Spatial value whose vector components should be rotated.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration
            Target type with rotated vector components.

        Notes
        -----
        The target must be a supported spatial vector-like type. The return type
        matches the input target type.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Position, Rotation
        >>> rot = Rotation(AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... ))
        >>> pos = Position(AnalysisObject.from_data(
        ...     xr.Dataset({"position": (("sample", "axis"), [[1.0, 0.0, 0.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> isinstance(rot.apply(pos), Position)
        True
        """
        return rotation_apply(self, target, validate=validate)
    
    @classmethod
    def solve_path_transform(
        cls,
        src: "Frame | str",
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Rotation":
        """Solve and compose edge rotations from ``src`` to ``dst`` frames.

        Parameters
        ----------
        src : Frame | str
            Source frame identifier or ``Frame`` object.
        dst : Frame | str
            Destination frame identifier or ``Frame`` object.
        edge_rotation_fn : object, optional
            Callable resolving edge rotations during frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph``
            (override frame graph), ``strict`` (strict path checks), and
            ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Rotation
            Result of applying this operation with TAL semantic constraints preserved.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        return rotation_class_solve_path_transform(
            cls,
            src,
            dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
        )
    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_rotation_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Rotation":
        """Express this rotation in another frame using graph-resolved edge rotations.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_rotation_fn : object
            Callable resolving rotation edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Rotation
            Operation result preserving TAL semantic/topology guarantees.

        Raises
        ------
        TypeError
            If option payload types are invalid for this API.
        ValueError
            If option values violate fail-closed semantic/layout constraints.

        Notes
        -----
        Uses xarray label-aware alignment and TAL fail-closed schema/runtime guards.

        Examples
        --------
        >>> opts = PathSolveOptions(strict=False)
        >>> isinstance(opts, PathSolveOptions)
        True
        """
        from .ops.frame_expression_ops import rotation_express_in

        return rotation_express_in(
            self,
            dst,
            edge_rotation_fn=edge_rotation_fn,
            opts=opts,
            validate=validate,
        )

from .ops.magnitude_ops import install_rotation_magnitude_methods as _install_rotation_magnitude_methods
from .ops.rotation_reduce_ops import install_rotation_reducer_methods as _install_rotation_reducer_methods
_install_rotation_magnitude_methods(Rotation)
_install_rotation_reducer_methods(Rotation)


__all__ = ["Rotation"]
