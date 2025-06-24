import collections
from collections.abc import Callable, Sequence
from typing import Any, Self

import attrs
import equinox as eqx
import wadler_lindig as wl
from wadler_lindig._definitions import _WithRepr

from ._field_specifiers import static
from ._pytree import pytree

type Node = Any


class PyTreeMixin:
    def __pdoc__(self, **kwargs) -> wl.AbstractDoc:
        """...

        References:
            1. [`wadler_lindig._definitions._pformat_dataclass()`](https://github.com/patrick-kidger/wadler_lindig/blob/3d4c81d5099bc96da6e5a5f058430cf1a80bcc60/wadler_lindig/_definitions.py#L302-L320)
        """
        type_name: str = "_" + type(self).__name__
        uninitialized = _WithRepr("<uninitialized>")
        selfs = []
        for field in attrs.fields(type(self)):
            field: attrs.Attribute
            if field.repr:
                value: str | _WithRepr = getattr(self, field.name, uninitialized)
                if not (kwargs["hide_defaults"] and value is field.default):
                    selfs.append((field.name.removeprefix(type_name), value))
        selfs = wl.named_objs(selfs, **kwargs)
        name_kwargs: dict[str, Any] = kwargs.copy()
        name_kwargs["show_type_module"] = kwargs["show_dataclass_module"]
        return wl.bracketed(
            begin=wl.pdoc(type(self), **name_kwargs) + wl.TextDoc("("),
            docs=selfs,
            sep=wl.comma,
            end=wl.TextDoc(")"),
            indent=kwargs["indent"],
        )

    def __repr__(self) -> str:
        return eqx.tree_pformat(self)

    def evolve(self, **changes) -> Self:
        return attrs.evolve(self, **changes)

    def tree_at(
        self,
        where: Callable[[Self], Node | Sequence[Node]],
        replace: Any | Sequence[Any] = ...,
        replace_fn: Callable[[Node], Any] = ...,  # pyright: ignore[reportArgumentType]
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Self:
        kwargs: dict[str, Any] = {}
        if replace is not ...:
            kwargs["replace"] = replace
        if replace_fn is not ...:
            kwargs["replace_fn"] = replace_fn
        if is_leaf is not None:
            kwargs["is_leaf"] = is_leaf
        return eqx.tree_at(where, self, **kwargs)


_counter: collections.Counter[str] = collections.Counter()


def uniq_id(self: Any) -> str:
    prefix: str = type(self).__qualname__
    id_: str = f"{prefix}-{_counter[prefix]:03d}"
    _counter[prefix] += 1
    return id_


@pytree
class PyTreeNode(PyTreeMixin):
    id: str = static(default=attrs.Factory(uniq_id, takes_self=True), kw_only=True)
