from __future__ import annotations

from tal.core.orchestration.alignment_intent import OperationIntentSupport

_SEMANTIC_DEFAULT_OPERATION_FAMILIES = frozenset(
    {
        "linalg.elementwise",
        "ufunc.arithmetic",
        "spatial.rotation.apply",
        "spatial.rotation.compose",
        "spatial.pose.apply",
        "spatial.pose.compose",
        "spatial.pose.components",
        "spatial.velocity.components",
        "spatial.acceleration.components",
        "spatial.velocity.vector6",
        "spatial.acceleration.vector6",
        "spatial.kinematics.path_coupling",
    }
)

_ALIGNMENT_SUPPORTED_OPERATION_FAMILIES = frozenset(_SEMANTIC_DEFAULT_OPERATION_FAMILIES)


def default_topology_mode_for_operation_family(
    operation_family: str,
    *,
    owner: str,
) -> str:
    """Return the default topology policy mode for an operation family token.

    Parameters
    ----------
    operation_family : str
        Canonical operation-family token used for policy lookup.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    str
        String result produced by this operation.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not operation_family:
        raise ValueError(f"{owner}: operation_family must be a non-empty string.")
    if operation_family in _SEMANTIC_DEFAULT_OPERATION_FAMILIES:
        return "semantic_broadcast"
    return "strict"


def alignment_intent_supported_for_operation_family(
    operation_family: str,
    *,
    owner: str,
) -> bool:
    """Return whether explicit alignment intents are supported for a family.

    Parameters
    ----------
    operation_family : str
        Canonical operation-family token used for policy lookup.
    owner : str, optional
        Owner prefix used to build deterministic fail-closed error messages.

    Returns
    -------
    bool
        Boolean result indicating whether the requested condition is satisfied.

    Notes
    -----
    Raises deterministic fail-closed errors when semantic/layout assumptions are not met.
    """
    if not operation_family:
        raise ValueError(f"{owner}: operation_family must be a non-empty string.")
    return operation_family in _ALIGNMENT_SUPPORTED_OPERATION_FAMILIES


def operation_intent_support_for_operation_family(
    operation_family: str,
    *,
    owner: str,
) -> OperationIntentSupport:
    """Return topology-intent support flags for an operation family.

    Parameters
    ----------
    operation_family : str
        Canonical operation-family token (for example ``"spatial.pose.compose"``).
    owner : str
        Error owner prefix used in fail-closed validation messages.

    Returns
    -------
    OperationIntentSupport
        Support flags containing:
        - ``semantic_default``: whether semantic-broadcast is the default mode,
        - ``alignment_intent_supported``: whether explicit alignment intents are supported.

    Notes
    -----
    This helper centralizes policy lookup so orchestrators do not duplicate
    operation-family tables.

    Examples
    --------
    >>> from tal.utils.topology_operation_families import operation_intent_support_for_operation_family
    >>> support = operation_intent_support_for_operation_family(
    ...     "spatial.pose.compose",
    ...     owner="example",
    ... )
    >>> support.semantic_default
    True
    """
    return OperationIntentSupport(
        semantic_default=(
            default_topology_mode_for_operation_family(operation_family, owner=owner)
            == "semantic_broadcast"
        ),
        alignment_intent_supported=alignment_intent_supported_for_operation_family(
            operation_family,
            owner=owner,
        ),
    )


__all__ = [
    "alignment_intent_supported_for_operation_family",
    "default_topology_mode_for_operation_family",
    "operation_intent_support_for_operation_family",
]
