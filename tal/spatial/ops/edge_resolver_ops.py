from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import NoReturn

from tal.frames import Frame


class _PathCallbackSignatureError(TypeError):
    """Internal marker for deterministic path-callback invocation misuse."""


def _public_signature_cause(exc: _PathCallbackSignatureError) -> TypeError:
    """Return a public cause without leaking TAL's internal marker type."""
    cause = exc.__cause__
    if type(cause) is TypeError:
        return cause
    return TypeError(str(exc))


def _raise_public_signature_error(
    exc: _PathCallbackSignatureError,
    *,
    owner: str | None = None,
) -> NoReturn:
    """Translate the internal signature marker at a public callback boundary."""
    text = str(exc)
    if owner is not None and not text.startswith(f"{owner}:"):
        text = f"{owner}: {text}"
    raise TypeError(text) from _public_signature_cause(exc)


@dataclass(frozen=True)
class _TrustedEdgeResolverAdapter:
    callback: Callable[..., object]


@dataclass(frozen=True)
class _PreparedPathCallback:
    callback: Callable[..., object]
    arg: str
    signature_checked: bool
    call_shape: str
    propagate_signature_error: bool = False


@dataclass(frozen=True)
class PreparedEdgeResolver:
    """Validated resolver input retained across one path request."""

    resolver: Callable[[Frame, Frame], object] | None
    arg: str
    signature_checked: bool = False
    propagate_signature_error: bool = False


def _require_callable(value: object, *, owner: str, arg: str, call_shape: str) -> Callable[..., object]:
    if callable(value):
        return value  # type: ignore[return-value]
    raise _PathCallbackSignatureError(f"{owner}: {arg} must be callable{call_shape}.")


def _require_callback_signature(
    callback: Callable[..., object],
    sample_args: tuple[object, ...],
    *,
    owner: str,
    arg: str,
    call_shape: str,
) -> bool:
    try:
        signature = inspect.signature(callback)
    except (TypeError, ValueError):
        return False
    try:
        signature.bind(*sample_args)
    except TypeError as exc:
        raise _PathCallbackSignatureError(f"{owner}: {arg} must be callable{call_shape}.") from exc
    return True


def _is_invocation_signature_typeerror(exc: TypeError) -> bool:
    traceback_obj = exc.__traceback__
    if traceback_obj is None:
        return True
    return traceback_obj.tb_next is None


def _prepare_callback(
    value: object,
    *,
    owner: str,
    arg: str,
    sample_args: tuple[object, ...],
    call_shape: str,
) -> _PreparedPathCallback:
    callback = _require_callable(value, owner=owner, arg=arg, call_shape=call_shape)
    checked = _require_callback_signature(
        callback,
        sample_args,
        owner=owner,
        arg=arg,
        call_shape=call_shape,
    )
    return _PreparedPathCallback(callback, arg, checked, call_shape)


def _call_prepared_callback(
    prepared: _PreparedPathCallback,
    args: tuple[object, ...],
    *,
    owner: str,
    failure_context: str,
) -> object:
    try:
        return prepared.callback(*args)
    except _PathCallbackSignatureError as exc:
        if prepared.propagate_signature_error:
            raise
        raise ValueError(
            f"{owner}: {prepared.arg} failed for {failure_context}."
        ) from _public_signature_cause(exc)
    except TypeError as exc:
        if not prepared.signature_checked and _is_invocation_signature_typeerror(exc):
            raise _PathCallbackSignatureError(
                f"{owner}: {prepared.arg} must be callable{prepared.call_shape}."
            ) from exc
        raise ValueError(f"{owner}: {prepared.arg} failed for {failure_context}.") from exc
    except Exception as exc:
        raise ValueError(f"{owner}: {prepared.arg} failed for {failure_context}.") from exc


def _prepare_required_edge_callback(value: object, *, owner: str, arg: str) -> _PreparedPathCallback:
    return _prepare_callback(
        value,
        owner=owner,
        arg=arg,
        sample_args=(object(), object()),
        call_shape="(child, parent)",
    )


def _prepare_required_frame_callback(value: object, *, owner: str, arg: str) -> _PreparedPathCallback:
    return _prepare_callback(
        value,
        owner=owner,
        arg=arg,
        sample_args=(object(),),
        call_shape="(frame)",
    )


def _call_prepared_edge_callback(
    prepared: _PreparedPathCallback,
    child: Frame,
    parent: Frame,
    *,
    owner: str,
) -> object:
    context = f"edge (child={child.id!r}, parent={parent.id!r})"
    return _call_prepared_callback(prepared, (child, parent), owner=owner, failure_context=context)


def _call_prepared_frame_callback(
    prepared: _PreparedPathCallback,
    frame: Frame,
    *,
    owner: str,
) -> object:
    return _call_prepared_callback(prepared, (frame,), owner=owner, failure_context=f"frame {frame.id!r}")


def prepare_edge_resolver(value: object, *, owner: str, arg: str) -> PreparedEdgeResolver:
    """Validate one optional resolver without invoking it."""
    if value is None:
        return PreparedEdgeResolver(None, arg)
    propagate_signature_error = isinstance(value, _TrustedEdgeResolverAdapter)
    if propagate_signature_error:
        value = value.callback
    prepared = _prepare_required_edge_callback(value, owner=owner, arg=arg)
    return PreparedEdgeResolver(
        prepared.callback,
        prepared.arg,
        prepared.signature_checked,
        propagate_signature_error,
    )


def _trusted_edge_resolver_adapter(callback: Callable[..., object]) -> object:
    """Mark one TAL-owned adapter as eligible to propagate signature misuse."""
    return _TrustedEdgeResolverAdapter(callback)


def call_prepared_edge_resolver(
    prepared: PreparedEdgeResolver,
    child: Frame,
    parent: Frame,
    *,
    owner: str,
) -> object:
    """Invoke one prepared resolver through the shared error envelope."""
    if prepared.resolver is None:
        raise RuntimeError("bound-provider selection must precede resolver invocation")
    callback = _PreparedPathCallback(
        prepared.resolver,
        prepared.arg,
        prepared.signature_checked,
        "(child, parent)",
        prepared.propagate_signature_error,
    )
    return _call_prepared_edge_callback(
        callback,
        child,
        parent,
        owner=owner,
    )


__all__ = [
    "PreparedEdgeResolver",
    "call_prepared_edge_resolver",
    "prepare_edge_resolver",
]
