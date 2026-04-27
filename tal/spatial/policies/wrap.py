from __future__ import annotations


def wrap_as(cls: type, ds, *, validate: bool):
    if validate:
        return cls._from_validated(ds)
    return cls._from_unvalidated(ds)


def wrap_like(instance, ds, *, validate: bool):
    return wrap_as(instance.__class__, ds, validate=validate)


__all__ = ["wrap_as", "wrap_like"]
