from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import xarray as xr

from tal.core.dataset_ownership import analysis_object_dataset
from tal.core.analysis_object import AnalysisObject
from tal.core.component_ops import (
    ComponentExtractOptions,
    extract_components,
)
from tal.core.component_ops.runtime_checks import require_component_numeric_var
from tal.core.orchestration.alignment_intent import select_topology_policy_with_intents
from tal.core.orchestration.inputs import coerce_analysis_object_input
from tal.core.orchestration.topology import SEMANTIC_NON_CORE_POLICY, STRICT_NON_CORE_POLICY, TopologyPolicy
from tal.core.orchestration.runtime_checks import (
    require_exact_labels,
    require_explicit_unique_dim_labels,
    require_var_contains_dims,
    select_single_numeric_var,
)
from tal.core.schema_read import (
    read_param_coord_name,
    read_roles,
    read_sequence_size_coord_name,
    validate_schema_if_needed,
)
from tal.core.typed_lifecycle import _finish_typed_promotion, _prepare_typed_promotion
from tal.utils.frame_schema import get_frames, set_frames
from tal.utils.topology_operation_families import operation_intent_support_for_operation_family

from .metadata import (
    get_pose_rep,
    normalize_configuration_relation_semantics,
    set_pose_rep,
    set_position_rep,
    set_rotation_rep,
    validate_spatial_roles,
)
from .policies.frame import resolve_components_shared_frames
from .ops.frame_api_ops import pose_class_solve_path_transform
from .kinematics.paired_components import (
    PairAssemblyOptions,
    align_paired_component_payloads,
    build_paired_components_dataset,
    clear_component_registry,
    component_var_names,
    resolve_component_spec,
    resolve_pair_registry,
    resolve_paired_roles,
)
from .kernels.pose_kernels import _matrix3_to_quat_prevalidated_kernel
from .ops.pose_apply_ops import pose_apply
from .ops.pose_matrix_validation import prepare_pose_matrix_for_conversion, validate_pose_matrix_dataset
from .ops.pose_ops import pose_as_components, pose_as_matrix, pose_compose, pose_inverse, pose_to_rep
from .position import Position
from .rotation import Rotation
from .policies.wrap import wrap_as

if TYPE_CHECKING:
    from tal.frames import Frame
    from tal.core.param_ops.types import ParamEvalOptions
    from .path_solve import PathSolveOptions
    from .temporal.options import PoseTemporalOptions

_COMPONENT_NAMES = {"position", "rotation"}
_XYZ_LABELS: tuple[str, str, str] = ("x", "y", "z")
_QUAT_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_MATRIX_LABELS: tuple[str, str, str, str] = ("x", "y", "z", "w")
_POSE_PAIR_OPTS = PairAssemblyOptions(
    left_what="Position",
    right_what="Rotation",
    pair_what="pose",
    left_component_name="position",
    right_component_name="rotation",
    left_expected_labels=_XYZ_LABELS,
    right_expected_labels=_QUAT_LABELS,
)

def _coerce_pose_source(value: object, *, owner: str) -> AnalysisObject:
    return coerce_analysis_object_input(value, owner=owner)


def _coerce_rotation_operand(value: object, *, owner: str) -> Rotation:
    if isinstance(value, Rotation):
        return value
    try:
        return Rotation(value)
    except TypeError as exc:
        raise TypeError(f"{owner}: rotation operand must be Rotation, AnalysisObject, xr.Dataset, or xr.DataArray.") from exc
    except ValueError as exc:
        raise ValueError(f"{owner}: rotation operand is not a valid Rotation: {exc}") from exc


def _coerce_position_operand(value: object, *, owner: str) -> Position:
    if isinstance(value, Position):
        return value
    try:
        return Position(value)
    except TypeError as exc:
        raise TypeError(f"{owner}: position operand must be Position, AnalysisObject, xr.Dataset, or xr.DataArray.") from exc
    except ValueError as exc:
        raise ValueError(f"{owner}: position operand is not a valid Position: {exc}") from exc


def _required_non_core_dims(sequence_dim: str | None, batch_dims: tuple[str, ...]) -> tuple[str, ...]:
    if sequence_dim is None:
        required_non_core_dims = tuple(batch_dims)
    else:
        required_non_core_dims = (sequence_dim, *batch_dims)
    return required_non_core_dims


def _resolve_pose_component_specs(
    ds: xr.Dataset,
    *,
    owner: str,
    core_dims: tuple[str, ...],
) -> tuple[tuple[str, str], tuple[str, str]]:
    registry = resolve_pair_registry(
        ds,
        owner=owner,
        pair_what="Pose",
        left_component_name=_POSE_PAIR_OPTS.left_component_name,
        right_component_name=_POSE_PAIR_OPTS.right_component_name,
    )
    pos_spec = resolve_component_spec(
        registry[_POSE_PAIR_OPTS.left_component_name],
        component_name=_POSE_PAIR_OPTS.left_component_name,
        core_dims=core_dims,
        expected_labels=_POSE_PAIR_OPTS.left_expected_labels,
        owner=owner,
        pair_what="Pose",
    )
    rot_spec = resolve_component_spec(
        registry[_POSE_PAIR_OPTS.right_component_name],
        component_name=_POSE_PAIR_OPTS.right_component_name,
        core_dims=core_dims,
        expected_labels=_POSE_PAIR_OPTS.right_expected_labels,
        owner=owner,
        pair_what="Pose",
    )
    return pos_spec, rot_spec


def _enforce_components_layout_invariants(ds: xr.Dataset, *, owner: str) -> None:
    declared, sequence_dim, batch_dims, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: Pose requires declared roles.")
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: Pose components layout requires exactly two core dims; got {core_dims!r}.")
    required_non_core_dims = _required_non_core_dims(sequence_dim, batch_dims)
    (pos_dim, pos_var), (rot_dim, rot_var) = _resolve_pose_component_specs(ds, owner=owner, core_dims=core_dims)
    if pos_dim == rot_dim:
        raise ValueError(f"{owner}: Pose position/rotation components must use distinct core dims.")
    require_component_numeric_var(
        ds,
        component_name="position",
        var_name=pos_var,
        required_dims=required_non_core_dims + (pos_dim,),
        owner=owner,
        operand="Pose",
    )
    require_component_numeric_var(
        ds,
        component_name="rotation",
        var_name=rot_var,
        required_dims=required_non_core_dims + (rot_dim,),
        owner=owner,
        operand="Pose",
    )
    pos_labels = require_explicit_unique_dim_labels(ds, dim=pos_dim, owner=owner, what="Pose position")
    rot_labels = require_explicit_unique_dim_labels(ds, dim=rot_dim, owner=owner, what="Pose rotation")
    require_exact_labels(pos_labels, expected=_XYZ_LABELS, owner=owner, what=f"Pose position core dim {pos_dim!r}")
    require_exact_labels(rot_labels, expected=_QUAT_LABELS, owner=owner, what=f"Pose rotation core dim {rot_dim!r}")


def _enforce_matrix_layout_invariants(ds: xr.Dataset, *, owner: str) -> None:
    declared, _, _, core_dims = read_roles(ds)
    if not declared:
        raise ValueError(f"{owner}: Pose requires declared roles.")
    if len(core_dims) != 2:
        raise ValueError(f"{owner}: Pose matrix layout requires exactly two core dims; got {core_dims!r}.")
    row_dim, col_dim = core_dims
    if row_dim == col_dim:
        raise ValueError(f"{owner}: Pose matrix core dims must be distinct; got {core_dims!r}.")
    var_name = select_single_numeric_var(ds, owner=owner, what="Pose matrix layout")
    require_var_contains_dims(
        ds,
        var_name=var_name,
        required_dims=(row_dim, col_dim),
        owner=owner,
        what="Pose matrix layout",
    )
    if int(ds.sizes.get(row_dim, -1)) != 4 or int(ds.sizes.get(col_dim, -1)) != 4:
        raise ValueError(f"{owner}: Pose matrix core dims must both have length 4.")
    row_labels = require_explicit_unique_dim_labels(ds, dim=row_dim, owner=owner, what="Pose matrix")
    col_labels = require_explicit_unique_dim_labels(ds, dim=col_dim, owner=owner, what="Pose matrix")
    require_exact_labels(row_labels, expected=_MATRIX_LABELS, owner=owner, what=f"Pose matrix row dim {row_dim!r}")
    require_exact_labels(col_labels, expected=_MATRIX_LABELS, owner=owner, what=f"Pose matrix col dim {col_dim!r}")


def _enforce_pose_dataset_invariants(ds: xr.Dataset, *, owner: str) -> None:
    candidate = validate_schema_if_needed(ds)
    validate_spatial_roles(candidate, owner=owner)
    _ = get_frames(candidate)
    rep = get_pose_rep(candidate, owner=owner)
    if rep == "components":
        _enforce_components_layout_invariants(candidate, owner=owner)
        return
    if rep == "matrix":
        _enforce_matrix_layout_invariants(candidate, owner=owner)
        return
    raise ValueError(f"{owner}: unsupported pose representation {rep!r}.")


def _normalize_pose_metadata(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    rep = get_pose_rep(ds, owner=owner)
    out = set_pose_rep(ds, rep=rep, validate=False, owner=owner)
    return normalize_configuration_relation_semantics(
        out,
        validate=False,
        owner=owner,
    )


def _validate_pose_matrix_if_needed(ds: xr.Dataset, *, owner: str) -> xr.Dataset:
    if get_pose_rep(ds, owner=owner) != "matrix":
        return ds
    return validate_pose_matrix_dataset(ds, owner=owner)


def _build_components_pose_dataset(
    rotation_ds: xr.Dataset,
    position_ds: xr.Dataset,
    *,
    owner: str,
    validate: bool,
    policy: TopologyPolicy,
) -> xr.Dataset:
    pos_dim, rot_dim = resolve_paired_roles(
        position_ds,
        rotation_ds,
        owner=owner,
        left_what="Position",
        right_what="Rotation",
    )
    pos_var, rot_var = component_var_names(
        position_ds,
        rotation_ds,
        owner=owner,
        opts=_POSE_PAIR_OPTS,
    )
    aligned_position_ds, aligned_rotation_ds, sequence_dim, batch_dims = align_paired_component_payloads(
        position_ds,
        rotation_ds,
        left_var=pos_var,
        right_var=rot_var,
        left_dim=pos_dim,
        right_dim=rot_dim,
        owner=owner,
        opts=_POSE_PAIR_OPTS,
        policy=policy,
    )
    return build_paired_components_dataset(
        left_ds=aligned_position_ds,
        right_ds=aligned_rotation_ds,
        sequence_dim=sequence_dim,
        batch_dims=batch_dims,
        left_dim=pos_dim,
        right_dim=rot_dim,
        left_var=pos_var,
        right_var=rot_var,
        owner=owner,
        validate=validate,
        opts=_POSE_PAIR_OPTS,
        policy=policy,
    )


def _clear_component_registry_for_matrix_layout(
    source: AnalysisObject,
    *,
    owner: str,
) -> xr.Dataset:
    return clear_component_registry(analysis_object_dataset(source), owner=owner)


def _wrap_pose_output(ds: xr.Dataset, *, validate: bool) -> "Pose":
    return wrap_as(Pose, ds, validate=validate)


def _wrap_position_output(ds: xr.Dataset, *, validate: bool) -> Position:
    return wrap_as(Position, ds, validate=validate)


def _wrap_rotation_output(ds: xr.Dataset, *, validate: bool) -> Rotation:
    return wrap_as(Rotation, ds, validate=validate)


def _resolve_quat_dim_name(ds: xr.Dataset) -> str:
    for candidate in ("quat", "quat_component", "rotation_component"):
        if candidate not in ds.dims:
            return candidate
    raise ValueError("spatial.pose.decompose: unable to allocate quaternion output dim name.")


def _matrix3_to_quat_decompose_kernel(values: np.ndarray) -> np.ndarray:
    owner = "spatial.pose.decompose"
    try:
        return _matrix3_to_quat_prevalidated_kernel(values)
    except ValueError as exc:
        raise ValueError(f"{owner}: matrix decomposition quaternion kernel failed.") from exc


def _build_matrix_component_outputs(
    *,
    translation: xr.DataArray,
    quat: xr.DataArray,
    position_core_dim: str,
    rotation_core_dim: str,
    sequence_dim: str | None,
    batch_dims: tuple[str, ...],
    param_coord: str | None,
    sequence_size_coord: str | None,
    owner: str,
    validate: bool,
) -> tuple[AnalysisObject, AnalysisObject]:
    if position_core_dim not in translation.dims:
        raise ValueError(
            f"{owner}: matrix decomposition translation output is missing expected core dim {position_core_dim!r}."
        )
    if rotation_core_dim not in quat.dims:
        raise ValueError(
            f"{owner}: matrix decomposition quaternion output is missing expected core dim {rotation_core_dim!r}."
        )
    param_name = param_coord if sequence_dim is not None else None
    size_name = sequence_size_coord if sequence_dim is not None else None
    base_kwargs: dict[str, object] = {
        "batch_dims": batch_dims,
        "param_coord": param_name,
        "sequence_size_coord": size_name,
        "validate": validate,
    }
    if sequence_dim is not None:
        base_kwargs["sequence_dim"] = sequence_dim
    pos_ao = AnalysisObject.from_data(
        translation.to_dataset(name="position"),
        core_dims=(position_core_dim,),
        **base_kwargs,
    )
    rot_ao = AnalysisObject.from_data(
        quat.to_dataset(name="rotation"),
        core_dims=(rotation_core_dim,),
        **base_kwargs,
    )
    return pos_ao, rot_ao
def _finalize_components_pose_output(
    ds: xr.Dataset,
    *,
    parent: str | None,
    child: str | None,
    validate: bool,
    owner: str,
) -> "Pose":
    ds = set_pose_rep(ds, rep="components", validate=False, owner=owner)
    ds = set_frames(ds, parent=parent, child=child, validate=False)
    return _wrap_pose_output(ds, validate=validate)

class Pose(AnalysisObject):
    """Rigid-body pose type (rotation + translation).

    Notes
    -----
    Public TAL class surface. See class methods/properties for operational semantics.
    """

    CANONICAL_POSITION_REP: str = "cart"
    CANONICAL_ROTATION_REP: str = "quat"
    def __init__(self, data: "AnalysisObject | xr.Dataset | xr.DataArray") -> None:
        owner = "spatial.pose.__init__"
        source = _coerce_pose_source(data, owner=owner)
        self._bind_dataset(_prepare_typed_promotion(source, owner=owner))
        self._normalize_metadata(owner=owner)
        self._enforce_invariants(owner=owner)
        self._bind_dataset(
            _validate_pose_matrix_if_needed(analysis_object_dataset(self), owner=owner)
        )
        _finish_typed_promotion(source, analysis_object_dataset(self))
    @classmethod
    def _from_validated(cls, ds: xr.Dataset | xr.DataArray) -> "Pose":
        owner = f"{cls.__name__}._from_validated"
        obj = cls._from_rigid_validated(ds, owner=owner)
        obj._bind_dataset(_validate_pose_matrix_if_needed(analysis_object_dataset(obj), owner=owner))
        return obj
    @classmethod
    def _from_rigid_validated(
        cls,
        ds: xr.Dataset | xr.DataArray,
        *,
        owner: str,
    ) -> "Pose":
        """Wrap data whose matrix payload has already passed rigid validation."""
        obj = super()._from_validated(ds)
        obj._normalize_metadata(owner=owner)
        obj._enforce_invariants(owner=owner)
        return obj
    @classmethod
    def _from_unvalidated(cls, ds: xr.Dataset | xr.DataArray, *, schema_prepared: bool = False) -> "Pose":
        obj = super()._from_unvalidated(ds, schema_prepared=schema_prepared)
        obj._normalize_metadata(owner=f"{cls.__name__}._from_unvalidated")
        obj._enforce_invariants(owner=f"{cls.__name__}._from_unvalidated")
        return obj
    def _normalize_metadata(self, *, owner: str) -> None:
        normalized = _normalize_pose_metadata(analysis_object_dataset(self), owner=owner)
        self._bind_dataset(normalized)
    def _enforce_invariants(self, *, owner: str) -> None:
        _enforce_pose_dataset_invariants(analysis_object_dataset(self), owner=owner)
    @property
    def preferred_interpolator(self) -> str:
        """Return the default temporal interpolation strategy for poses.

        Returns
        -------
        str
            Resolved property value.
        """
        return "slerp_split"
    @property
    def param(self) -> "PoseParamAccessor":
        """Return the parameter-domain accessor for temporal pose operations.

        Returns
        -------
        PoseParamAccessor
            Resolved property value.
        """
        from .temporal.accessor import PoseParamAccessor

        return PoseParamAccessor(self)
    @classmethod
    def from_components(
        cls,
        rotation: object,
        position: object,
        *,
        validate: bool = True,
    ) -> "Pose":
        """Build a pose from compatible rotation and position components.

        Parameters
        ----------
        rotation : object
            Rotation-like input coercible to :class:`Rotation`.
        position : object
            Position-like input coercible to :class:`Position`.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Pose
            Pose stored in component representation.

        Notes
        -----
        Rotation and position components are aligned by TAL topology policy.
        Parent/child frame metadata must be compatible.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Pose, Position, Rotation
        >>> rot = Rotation(AnalysisObject.from_data(
        ...     xr.Dataset({"rotation": (("sample", "quat"), [[0.0, 0.0, 0.0, 1.0]])}, coords={"sample": [0], "quat": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("quat",),
        ...     validate=True,
        ... ))
        >>> pos = Position(AnalysisObject.from_data(
        ...     xr.Dataset({"position": (("sample", "axis"), [[1.0, 2.0, 3.0]])}, coords={"sample": [0], "axis": ["x", "y", "z"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("axis",),
        ...     validate=True,
        ... ))
        >>> sorted(Pose.from_components(rot, pos).as_dataset().data_vars)
        ['position', 'rotation']
        """
        owner = "spatial.pose.from_components"
        rot = _coerce_rotation_operand(rotation, owner=owner)
        pos = _coerce_position_operand(position, owner=owner)
        rot_ds = analysis_object_dataset(rot)
        pos_ds = analysis_object_dataset(pos)
        parent, child = resolve_components_shared_frames(rot_ds, pos_ds, owner=owner, left_name="rotation", right_name="position")
        selection = select_topology_policy_with_intents(
            (rot, pos),
            owner=owner,
            operation_family="spatial.pose.components",
            support=operation_intent_support_for_operation_family("spatial.pose.components", owner=owner),
            strict_policy=STRICT_NON_CORE_POLICY,
            semantic_policy=SEMANTIC_NON_CORE_POLICY,
        )
        ds = _build_components_pose_dataset(
            rot_ds,
            pos_ds,
            owner=owner,
            validate=validate,
            policy=selection.policy,
        )
        return _finalize_components_pose_output(ds, parent=parent, child=child, validate=validate, owner=owner)
    @classmethod
    def from_matrix(
        cls,
        matrix: object,
        *,
        validate: bool = True,
    ) -> "Pose":
        """Construct a pose from a homogeneous matrix layout dataset.

        Parameters
        ----------
        matrix : object
            Operand/component input consumed by this operation.
        validate : bool, optional
            When ``True``, validate output schema, layout, and rigid-transform
            invariants before returning.

        Returns
        -------
        Pose
            Result of applying this operation with TAL semantic constraints preserved.

        Raises
        ------
        ValueError
            If a structurally valid matrix is non-finite, not a proper rigid
            transform, or has a malformed homogeneous bottom row.

        Notes
        -----
        The input must already declare two 4-element core dimensions containing
        homogeneous transform matrices. Validation is immediate for eager
        arrays and remains lazy for Dask-backed payloads.

        Examples
        --------
        >>> import xarray as xr
        >>> import numpy as np
        >>> from tal.core import AnalysisObject
        >>> from tal.spatial import Pose
        >>> matrix = np.eye(4)[None, :, :]
        >>> ao = AnalysisObject.from_data(
        ...     xr.Dataset({"pose_matrix": (("sample", "row", "col"), matrix)}, coords={"sample": [0], "row": ["x", "y", "z", "w"], "col": ["x", "y", "z", "w"]}),
        ...     sequence_dim="sample",
        ...     core_dims=("row", "col"),
        ...     validate=True,
        ... )
        >>> Pose.from_matrix(ao).as_dataset()["pose_matrix"].shape[-2:]
        (4, 4)
        """
        owner = "spatial.pose.from_matrix"
        source = _coerce_pose_source(matrix, owner=owner)
        ds = _clear_component_registry_for_matrix_layout(source, owner=owner)
        ds = set_pose_rep(ds, rep="matrix", validate=False, owner=owner)
        if validate:
            _enforce_matrix_layout_invariants(ds, owner=owner)
            ds = validate_pose_matrix_dataset(ds, owner=owner)
            return cls._from_rigid_validated(ds, owner=owner)
        return cls._from_unvalidated(ds)
    def decompose(self, *, validate: bool = True) -> tuple[Position, Rotation]:
        """Decompose this pose into ``(Position, Rotation)`` components.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        tuple[Position, Rotation]
            Position and rotation components in that order.

        Notes
        -----
        Matrix poses are decomposed into cartesian translation and quaternion
        rotation components. Component poses return their existing parts.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> position, rotation = pose.decompose()  # doctest: +SKIP
        """
        rep = get_pose_rep(analysis_object_dataset(self), owner="spatial.pose.decompose")
        if rep == "components":
            return self._decompose_components(validate=validate)
        return self._decompose_matrix(validate=validate)
    def to_rep(self, rep: Literal["components", "matrix"], *, validate: bool = True) -> "Pose":
        """Convert between ``components`` and ``matrix`` pose representations.

        Parameters
        ----------
        rep : Literal['components', 'matrix']
            Target representation, either ``"components"`` or ``"matrix"``.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Pose
            Pose stored in the requested representation.

        Notes
        -----
        Component representation stores separate ``position`` and ``rotation``
        variables. Matrix representation stores homogeneous 4x4 transforms.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> pose.to_rep("matrix").as_dataset()["pose_matrix"].shape[-2:]  # doctest: +SKIP
        (4, 4)
        """
        return pose_to_rep(self, rep=rep, validate=validate)
    def as_components(self, *, validate: bool = True) -> "Pose":
        """Return this pose in ``components`` representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Pose
            Pose with separate position and rotation variables.

        Notes
        -----
        This is equivalent to ``to_rep("components")``.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> sorted(pose.as_components().as_dataset().data_vars)  # doctest: +SKIP
        ['position', 'rotation']
        """
        return pose_as_components(self, validate=validate)
    def as_matrix(self, *, validate: bool = True) -> "Pose":
        """Return this pose in homogeneous matrix representation.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Pose
            Pose with homogeneous 4x4 matrix payloads.

        Notes
        -----
        This is equivalent to ``to_rep("matrix")``.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> pose.as_matrix().as_dataset()["pose_matrix"].shape[-2:]  # doctest: +SKIP
        (4, 4)
        """
        return pose_as_matrix(self, validate=validate)
    def compose(
        self,
        other: "Pose | AnalysisObject | xr.Dataset | xr.DataArray",
        *,
        validate: bool = True,
    ) -> "Pose":
        """Compose this pose with another pose.

        Parameters
        ----------
        other : Pose | AnalysisObject | xr.Dataset | xr.DataArray
            Pose-like transform to compose with this pose.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Pose
            Composed rigid transform.

        Notes
        -----
        Composition preserves TAL topology and validates frame compatibility
        before returning.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> isinstance(pose.compose(pose), Pose)  # doctest: +SKIP
        True
        """
        return pose_compose(self, other, validate=validate)
    def inverse(self, *, validate: bool = True) -> "Pose":
        """Return the inverse rigid transform.

        Parameters
        ----------
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Pose
            Inverse rigid transform.

        Notes
        -----
        The inverse preserves non-core topology and swaps transform direction
        according to pose algebra.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> isinstance(pose.inverse(), Pose)  # doctest: +SKIP
        True
        """
        return pose_inverse(self, validate=validate)
    def apply(
        self,
        target: "Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration",
        *,
        validate: bool = True,
    ) -> "Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration":
        """Apply this pose transform to a compatible spatial target.

        Parameters
        ----------
        target : Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration
            Spatial value to transform.
        validate : bool, optional
            When ``True``, validate output spatial/schema invariants before
            returning.

        Returns
        -------
        Position | LinearVelocity | AngularVelocity | LinearAcceleration | AngularAcceleration | Velocity | Acceleration
            Transformed target value. The return type matches the target type.

        Notes
        -----
        Positions are transformed with rotation and translation; vector-like
        velocity and acceleration values use the appropriate spatial rules.

        Examples
        --------
        >>> pose = Pose.from_components(rot, pos)  # doctest: +SKIP
        >>> isinstance(pose.apply(pos), Position)  # doctest: +SKIP
        True
        """
        return pose_apply(self, target, validate=validate)
    @classmethod
    def solve_path_transform(
        cls,
        src: "Frame | str",
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Pose":
        """Solve and compose edge poses from ``src`` to ``dst`` frames.

        Parameters
        ----------
        src : Frame | str
            Source frame identifier or ``Frame`` object.
        dst : Frame | str
            Destination frame identifier or ``Frame`` object.
        edge_pose_fn : object, optional
            Callable resolving edge poses during frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph``
            (override frame graph), ``strict`` (strict path checks), and
            ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Pose
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
        return pose_class_solve_path_transform(
            cls,
            src,
            dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
        )
    def express_in(
        self,
        dst: "Frame | str",
        *,
        edge_pose_fn,
        opts: "PathSolveOptions | None" = None,
        validate: bool = True,
    ) -> "Pose":
        """Express this pose in another frame through pose-path composition.

        Parameters
        ----------
        dst : Frame | str
            Destination frame id/object.
        edge_pose_fn : object
            Callable resolving pose edges for frame-path traversal.
        opts : PathSolveOptions | None, optional
            When ``None``, defaults are used. Key fields are ``graph`` (override graph source), ``strict`` (strict path checks), and ``kinematics_support`` for velocity/acceleration transport metadata.
        validate : bool, optional
            When ``True``, validate output schema/layout invariants before returning.

        Returns
        -------
        Pose
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
        from .ops.frame_expression_ops import pose_express_in

        return pose_express_in(
            self,
            dst,
            edge_pose_fn=edge_pose_fn,
            opts=opts,
            validate=validate,
        )
    def _decompose_components(self, *, validate: bool) -> tuple[Position, Rotation]:
        owner = "spatial.pose.decompose"
        source_ds = analysis_object_dataset(self)
        source = AnalysisObject._from_validated(source_ds) if validate else AnalysisObject._from_unvalidated(source_ds)
        extracted = extract_components(source, opts=ComponentExtractOptions(names=("position", "rotation")), validate=validate)
        parent, child = get_frames(source_ds)
        pos_ds = set_position_rep(analysis_object_dataset(extracted["position"]), rep=self.CANONICAL_POSITION_REP, validate=False, owner=owner)
        rot_ds = set_rotation_rep(analysis_object_dataset(extracted["rotation"]), rep=self.CANONICAL_ROTATION_REP, validate=False, owner=owner)
        pos_ds = set_frames(pos_ds, parent=parent, child=child, validate=False)
        rot_ds = set_frames(rot_ds, parent=parent, child=child, validate=False)
        return _wrap_position_output(pos_ds, validate=validate), _wrap_rotation_output(rot_ds, validate=validate)
    def _decompose_matrix(self, *, validate: bool) -> tuple[Position, Rotation]:
        owner = "spatial.pose.decompose"
        candidate = validate_schema_if_needed(analysis_object_dataset(self))
        declared, seq_dim, batch_dims, core_dims = read_roles(candidate)
        if not declared:
            raise ValueError(f"{owner}: Pose requires declared roles.")
        row_dim, col_dim = core_dims
        var_name = select_single_numeric_var(candidate, owner=owner, what="Pose matrix layout")
        require_var_contains_dims(candidate, var_name=var_name, required_dims=(row_dim, col_dim), owner=owner, what="Pose matrix layout")
        matrix = prepare_pose_matrix_for_conversion(candidate, owner=owner)
        rot_matrix = matrix.sel({row_dim: list(_XYZ_LABELS), col_dim: list(_XYZ_LABELS)})
        translation = matrix.sel({row_dim: list(_XYZ_LABELS), col_dim: "w"})

        quat_dim = _resolve_quat_dim_name(candidate)
        quat = xr.apply_ufunc(
            _matrix3_to_quat_decompose_kernel,
            rot_matrix,
            input_core_dims=[[row_dim, col_dim]],
            output_core_dims=[[quat_dim]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[np.float64],
            dask_gufunc_kwargs={"output_sizes": {quat_dim: 4}},
        )
        quat = quat.assign_coords({quat_dim: list(_QUAT_LABELS)})

        param_name = read_param_coord_name(candidate)
        size_name = read_sequence_size_coord_name(candidate)
        pos_ao, rot_ao = _build_matrix_component_outputs(
            translation=translation,
            quat=quat,
            position_core_dim=row_dim,
            rotation_core_dim=quat_dim,
            sequence_dim=seq_dim,
            batch_dims=batch_dims,
            param_coord=param_name,
            sequence_size_coord=size_name,
            owner=owner,
            validate=validate,
        )

        parent, child = get_frames(candidate)
        pos_ds = set_position_rep(analysis_object_dataset(pos_ao), rep=self.CANONICAL_POSITION_REP, validate=False, owner=owner)
        rot_ds = set_rotation_rep(analysis_object_dataset(rot_ao), rep=self.CANONICAL_ROTATION_REP, validate=False, owner=owner)
        pos_ds = set_frames(pos_ds, parent=parent, child=child, validate=False)
        rot_ds = set_frames(rot_ds, parent=parent, child=child, validate=False)
        return _wrap_position_output(pos_ds, validate=validate), _wrap_rotation_output(rot_ds, validate=validate)

__all__ = ["Pose"]
