from __future__ import annotations

from .registry import UfuncCallable


def call_unary_ufunc_kernel(ufunc: UfuncCallable, operand: object, *, owner: str) -> object:
    _ = owner
    return ufunc(operand)


def call_binary_ufunc_kernel(ufunc: UfuncCallable, left: object, right: object, *, owner: str) -> object:
    _ = owner
    return ufunc(left, right)


__all__ = [
    "call_binary_ufunc_kernel",
    "call_unary_ufunc_kernel",
]
