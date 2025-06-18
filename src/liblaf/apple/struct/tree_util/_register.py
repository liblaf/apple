from collections.abc import Generator, Sequence

import attrs
import jax


def register_attrs[T: type](cls: T) -> T:
    data_fields: Sequence[str] = tuple(_collect_fields(cls, static=False))
    meta_fields: Sequence[str] = tuple(_collect_fields(cls, static=True))
    return jax.tree_util.register_dataclass(
        cls, data_fields=data_fields, meta_fields=meta_fields
    )


def _collect_fields(cls: type, *, static: bool) -> Generator[str]:
    for field in attrs.fields(cls):
        field: attrs.Attribute
        if not field.init:
            continue
        if field.metadata.get("static", False) == static:
            yield field.name
