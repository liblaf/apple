import inspect
from typing import Any, overload


# TODO: this method does not tell pyright whether a property has getter, setter, or deleter
@overload
def delegate(prop: str, fn: property) -> Any: ...
@overload
def delegate[T](prop: str, fn: T) -> T: ...
@overload
def delegate[T](prop: str, fn: Any, typ: T) -> T: ...
def delegate(prop: str, fn: Any, typ: Any = ...) -> Any:  # noqa: ARG001
    if inspect.isfunction(fn):
        name: str | None = getattr(fn, "__name__", None)
        assert name is not None
        return lambda self, *args, **kwargs: getattr(getattr(self, prop), name)(
            *args, **kwargs
        )
    if inspect.isdatadescriptor(fn):
        name: str | None = getattr(fn, "__name__", None)
        if name is None and isinstance(fn, property):
            name = getattr(fn.fget, "__name__", None)
        assert name is not None
        return property(
            fget=lambda self: getattr(getattr(self, prop), name),
            fset=lambda self, value: setattr(getattr(self, prop), name, value),
            fdel=lambda self: delattr(getattr(self, prop), name),
        )
    raise NotImplementedError
