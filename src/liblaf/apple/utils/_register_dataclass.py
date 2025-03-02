from collections.abc import Sequence
from typing import Protocol

import attrs
import jax


class RegisterDataclassDecorater(Protocol):
    def __call__[T: type](self, cls: T) -> T: ...


def register_dataclass(
    data_fields: Sequence[str] | None = None,
    meta_fields: Sequence[str] | None = None,
    drop_fields: Sequence[str] = (),
) -> RegisterDataclassDecorater:
    def decorater[T: type](cls: T) -> T:
        nonlocal data_fields, meta_fields
        if jax.__version_info__ >= (0, 4, 36):
            return jax.tree_util.register_dataclass(
                cls,
                data_fields=data_fields,  # pyright: ignore[reportArgumentType]
                meta_fields=meta_fields,  # pyright: ignore[reportArgumentType]
                drop_fields=drop_fields,
            )
        if data_fields is None:
            data_fields = extract_fields(cls, static=False)
        if meta_fields is None:
            meta_fields = extract_fields(cls, static=True)
        return jax.tree_util.register_dataclass(
            cls,
            data_fields=data_fields,
            meta_fields=meta_fields,
            drop_fields=drop_fields,
        )

    return decorater


def extract_fields(cls: type, *, static: bool) -> list[str]:
    fields: tuple[attrs.Attribute] = attrs.fields(cls)
    return [
        field.name for field in fields if field.metadata.get("static", False) == static
    ]
