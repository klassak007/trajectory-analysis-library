"""Public immutable recipes and thin typed field-factory surfaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Generic, Literal, Self, TypeVar, cast

from tal.core import AnalysisLayoutSpec
from tal.core.schema import UNSET, UnsetType
from tal.frames import FrameGraph

from .ops.field_selectors import FieldSelectorDeclaration, prepare_field_selector

T = TypeVar("T")
FieldTarget = Literal["position", "rotation", "pose"]

_POSITION_LABELS = ("x", "y", "z")
_ROTATION_LABELS = ("x", "y", "z", "w")


@dataclass(frozen=True, init=False)
class SpatialFieldRecipe(Generic[T]):
    """Reusable immutable declaration for typed construction from scalar fields.

    Notes
    -----
    Recipes retain only their target and selectors. Source layout, prefixes,
    frame declarations, graph association, and validation belong to each build.
    Create recipes with ``Position.fields``, ``Rotation.fields``, or
    ``Pose.fields`` rather than by calling this class directly.

    Examples
    --------
    >>> from tal.spatial import Position
    >>> recipe = Position.fields("position.{x,y,z}")
    >>> type(recipe).__name__
    'SpatialFieldRecipe'
    """

    _target: FieldTarget
    _selectors: tuple[FieldSelectorDeclaration, ...]

    def __new__(cls):
        raise TypeError(
            "SpatialFieldRecipe objects must be created by Position.fields, "
            "Rotation.fields, or Pose.fields."
        )

    @classmethod
    def _create(
        cls,
        target: FieldTarget,
        selectors: tuple[FieldSelectorDeclaration, ...],
    ) -> SpatialFieldRecipe[object]:
        instance = object.__new__(cls)
        object.__setattr__(instance, "_target", target)
        object.__setattr__(instance, "_selectors", selectors)
        instance.__post_init__()
        return instance

    def __post_init__(self) -> None:
        expected = {
            "position": ("position",),
            "rotation": ("rotation",),
            "pose": ("position", "rotation"),
        }
        slots = tuple(selector.slot for selector in self._selectors)
        if self._target not in expected or slots != expected[self._target]:
            raise TypeError(
                "SpatialFieldRecipe objects must be created by Position.fields, "
                "Rotation.fields, or Pose.fields."
            )

    def build(
        self,
        source: object,
        *,
        prefix: str = "",
        source_layout: AnalysisLayoutSpec | None = None,
        parent: str | None | UnsetType = UNSET,
        child: str | None | UnsetType = UNSET,
        expressed_in: str | None | UnsetType = UNSET,
        graph: FrameGraph | None | UnsetType = UNSET,
    ) -> T:
        """Build the declared spatial type from scalar source fields.

        Parameters
        ----------
        source
            Source ``AnalysisObject`` or ``xarray.Dataset``.
        prefix
            Literal prefix added to every expanded source field name.
        source_layout
            Complete source layout for an untagged Dataset. Omit for an AO or
            schema-bearing Dataset.
        parent, child, expressed_in
            Frame declarations to inherit, confirm, add, or explicitly clear.
        graph
            Passive graph association; construction does not mutate topology.

        Returns
        -------
        T
            Fully validated typed spatial result declared by the recipe.

        Raises
        ------
        TypeError
            If an option or source has an unsupported type.
        ValueError
            If fields, topology, or spatial declarations are incompatible.
        tal.core.SchemaError
            If the complete source schema is invalid.

        Notes
        -----
        Parameters are resolved for this build only; neither recipe nor source
        is mutated. Construction always validates the source and typed result.
        Newly assembled component variables do not inherit field attributes.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisLayoutSpec
        >>> from tal.spatial import Position
        >>> data = xr.Dataset({"camera.x": ("sample", [1.0]),
        ...                    "camera.y": ("sample", [2.0]),
        ...                    "camera.z": ("sample", [3.0])})
        >>> recipe = Position.fields("{x,y,z}")
        >>> out = recipe.build(data, prefix="camera.",
        ...                    source_layout=AnalysisLayoutSpec(sequence_dim="sample"))
        >>> out.as_dataset()["position"].shape
        (1, 3)
        """
        return self._execute(
            source,
            prefix=prefix,
            source_layout=source_layout,
            parent=parent,
            child=child,
            expressed_in=expressed_in,
            graph=graph,
            owner="SpatialFieldRecipe.build",
        )

    def _execute(
        self,
        source: object,
        *,
        prefix: object,
        source_layout: object,
        parent: object,
        child: object,
        expressed_in: object,
        graph: object,
        owner: str,
    ) -> T:
        from .ops.field_construction import execute_field_recipe

        return execute_field_recipe(
            self,
            source,
            prefix=prefix,
            source_layout=source_layout,
            parent=parent,
            child=child,
            expressed_in=expressed_in,
            graph=graph,
            owner=owner,
        )


def _single_recipe(
    target: FieldTarget,
    fields: object,
    *,
    owner: str,
) -> SpatialFieldRecipe[object]:
    expected = _POSITION_LABELS if target == "position" else _ROTATION_LABELS
    selector = prepare_field_selector(
        fields,
        expected=expected,
        slot=target,
        owner=owner,
    )
    return SpatialFieldRecipe._create(target, (selector,))


def _pose_recipe(
    *,
    position: object,
    rotation: object,
    owner: str,
) -> SpatialFieldRecipe[object]:
    position_selector = prepare_field_selector(
        position,
        expected=_POSITION_LABELS,
        slot="position",
        owner=owner,
    )
    rotation_selector = prepare_field_selector(
        rotation,
        expected=_ROTATION_LABELS,
        slot="rotation",
        owner=owner,
    )
    return SpatialFieldRecipe._create(
        "pose",
        (position_selector, rotation_selector),
    )


class SingleSpatialFieldFactoryMixin:
    """Thin inherited Position/Rotation field-construction surface."""

    SPATIAL_FIELD_TARGET: ClassVar[Literal["position", "rotation"]]

    @classmethod
    def fields(cls, fields: object) -> SpatialFieldRecipe[Self]:
        """Declare a reusable immutable scalar-field recipe.

        Parameters
        ----------
        fields
            One brace selector or target-label-to-source-name mapping.

        Returns
        -------
        SpatialFieldRecipe
            Immutable recipe for this Position or Rotation type.

        Raises
        ------
        TypeError
            If the selector container or mapping items have invalid types.
        ValueError
            If selector grammar or target component labels are invalid.

        Notes
        -----
        A recipe stores no source, prefix, layout, frame, or graph. Those are
        supplied independently to each ``build`` call.

        Examples
        --------
        >>> from tal.spatial import Position
        >>> recipe = Position.fields("position.{x,y,z}")
        >>> type(recipe).__name__
        'SpatialFieldRecipe'
        """
        return cast(
            SpatialFieldRecipe[Self],
            _single_recipe(
                cls.SPATIAL_FIELD_TARGET,
                fields,
                owner=f"{cls.__name__}.fields",
            ),
        )

    @classmethod
    def from_fields(
        cls,
        source: object,
        fields: object,
        *,
        prefix: str = "",
        source_layout: AnalysisLayoutSpec | None = None,
        parent: str | None | UnsetType = UNSET,
        child: str | None | UnsetType = UNSET,
        expressed_in: str | None | UnsetType = UNSET,
        graph: FrameGraph | None | UnsetType = UNSET,
    ) -> Self:
        """Construct a Position or Rotation from explicit scalar fields.

        Parameters
        ----------
        source
            Source ``AnalysisObject`` or ``xarray.Dataset``.
        fields
            One brace selector or target-label-to-source-name mapping.
        prefix
            Literal prefix added to expanded source field names.
        source_layout
            Complete source layout for an untagged Dataset.
        parent, child, expressed_in
            Frame declarations to inherit, confirm, add, or explicitly clear.
        graph
            Passive graph association.

        Returns
        -------
        Self
            Fully validated typed result.

        Raises
        ------
        TypeError
            If selector, option, or source types are unsupported.
        ValueError
            If fields, topology, or spatial declarations are incompatible.
        tal.core.SchemaError
            If the complete source schema is invalid.

        Notes
        -----
        This is the one-shot equivalent of ``fields(...).build(...)``. Newly
        assembled component variables start with empty attributes and encoding.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisLayoutSpec
        >>> from tal.spatial import Position
        >>> data = xr.Dataset({"p.x": ("sample", [1.0]),
        ...                    "p.y": ("sample", [2.0]),
        ...                    "p.z": ("sample", [3.0])})
        >>> out = Position.from_fields(data, "p.{x,y,z}",
        ...     source_layout=AnalysisLayoutSpec(sequence_dim="sample"))
        >>> out.as_dataset()["position"].shape
        (1, 3)
        """
        recipe = _single_recipe(
            cls.SPATIAL_FIELD_TARGET,
            fields,
            owner=f"{cls.__name__}.from_fields",
        )
        return cast(
            Self,
            recipe._execute(
                source,
                prefix=prefix,
                source_layout=source_layout,
                parent=parent,
                child=child,
                expressed_in=expressed_in,
                graph=graph,
                owner=f"{cls.__name__}.from_fields",
            ),
        )


class PoseSpatialFieldFactoryMixin:
    """Thin inherited Pose field-construction surface."""

    @classmethod
    def fields(
        cls,
        *,
        position: object,
        rotation: object,
    ) -> SpatialFieldRecipe[Self]:
        """Declare a reusable immutable Pose scalar-field recipe.

        Parameters
        ----------
        position
            Position brace selector or target-label mapping.
        rotation
            Quaternion brace selector or target-label mapping.

        Returns
        -------
        SpatialFieldRecipe
            Immutable component-Pose recipe.

        Raises
        ------
        TypeError
            If a selector container or mapping item has an invalid type.
        ValueError
            If selector grammar or target labels are invalid.

        Notes
        -----
        The recipe snapshots both selectors but stores no source or build
        options. Quaternion source fields are reordered by explicit labels.

        Examples
        --------
        >>> from tal.spatial import Pose
        >>> recipe = Pose.fields(position="position.{x,y,z}",
        ...                      rotation="rotation.{x,y,z,w}")
        >>> type(recipe).__name__
        'SpatialFieldRecipe'
        """
        return cast(
            SpatialFieldRecipe[Self],
            _pose_recipe(
                position=position,
                rotation=rotation,
                owner=f"{cls.__name__}.fields",
            ),
        )

    @classmethod
    def from_fields(
        cls,
        source: object,
        *,
        position: object,
        rotation: object,
        prefix: str = "",
        source_layout: AnalysisLayoutSpec | None = None,
        parent: str | None | UnsetType = UNSET,
        child: str | None | UnsetType = UNSET,
        expressed_in: str | None | UnsetType = UNSET,
        graph: FrameGraph | None | UnsetType = UNSET,
    ) -> Self:
        """Construct a component Pose from explicit scalar fields.

        Parameters
        ----------
        source
            Source ``AnalysisObject`` or ``xarray.Dataset``.
        position
            Position brace selector or target-label mapping.
        rotation
            Quaternion brace selector or target-label mapping.
        prefix
            Literal prefix added to expanded source field names.
        source_layout
            Complete source layout for an untagged Dataset.
        parent, child, expressed_in
            Frame declarations to inherit, confirm, add, or explicitly clear.
        graph
            Passive graph association.

        Returns
        -------
        Self
            Fully validated component-representation Pose.

        Raises
        ------
        TypeError
            If selector, option, or source types are unsupported.
        ValueError
            If fields, topology, or spatial declarations are incompatible.
        tal.core.SchemaError
            If the complete source schema is invalid.

        Notes
        -----
        The source union is prepared once. Position and rotation variables are
        assembled with empty attributes and storage encodings.

        Examples
        --------
        >>> import xarray as xr
        >>> from tal.core import AnalysisLayoutSpec
        >>> from tal.spatial import Pose
        >>> data = xr.Dataset({"p.x": 1.0, "p.y": 2.0, "p.z": 3.0,
        ...                    "q.x": 0.0, "q.y": 0.0, "q.z": 0.0, "q.w": 1.0})
        >>> pose = Pose.from_fields(data, position="p.{x,y,z}",
        ...     rotation="q.{x,y,z,w}", source_layout=AnalysisLayoutSpec())
        >>> list(pose.as_dataset().data_vars)
        ['position', 'rotation']
        """
        recipe = _pose_recipe(
            position=position,
            rotation=rotation,
            owner=f"{cls.__name__}.from_fields",
        )
        return cast(
            Self,
            recipe._execute(
                source,
                prefix=prefix,
                source_layout=source_layout,
                parent=parent,
                child=child,
                expressed_in=expressed_in,
                graph=graph,
                owner=f"{cls.__name__}.from_fields",
            ),
        )


__all__ = ["SpatialFieldRecipe"]
