from .types import ParamEvalOptions, ParamSelectOptions, ParamSyncOptions

__all__ = [
    "ParamAccessor",
    "ParamEvalOptions",
    "ParamSelectOptions",
    "ParamSyncOptions",
    "synchronize",
    "synchronize_param",
]


def __getattr__(name: str):
    if name == "ParamAccessor":
        from .accessor import ParamAccessor

        return ParamAccessor
    if name in {"synchronize", "synchronize_param"}:
        from .sync import synchronize, synchronize_param

        return {"synchronize": synchronize, "synchronize_param": synchronize_param}[name]
    raise AttributeError(name)
