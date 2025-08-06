from collections.abc import Sequence
from typing import Self

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Bool, Integer, Shaped

from liblaf.apple.struct import tree


class Index(tree.PyTree):
    index: Integer[Array, " N"] = tree.array(factory=lambda: jnp.empty((0,), dtype=int))
    shape: Sequence[int] = tree.field(default=None)

    @classmethod
    def from_index(
        cls, index: Integer[ArrayLike, " N"], shape: Sequence[int] | None = None
    ) -> Self:
        index = jnp.asarray(index)
        if shape is None:
            shape = index.shape
        return cls(index=index.ravel(), shape=shape)

    @classmethod
    def from_mask(cls, mask: Bool[ArrayLike, "*shape"]) -> Self:
        index: Integer[Array, " N"]
        (index,) = jnp.nonzero(jnp.asarray(mask).ravel())
        return cls.from_index(index)

    @classmethod
    def union1d(
        cls,
        *indices: "ArrayLike | Index",
        size: int | None = None,
        fill_value: ArrayLike | None = None,
    ) -> Self:
        if not indices:
            return cls()
        index: Integer[Array, " N"] = jnp.union1d(
            *(jnp.asarray(i).ravel() for i in indices), size=size, fill_value=fill_value
        )
        return cls.from_index(index)

    def __post_init__(self) -> None:
        if self.shape is None:
            self.shape = self.index.shape  # pyright: ignore[reportAttributeAccessIssue]

    def __getitem__(self, local_index: "ArrayLike | Index") -> Self:
        local_index: Integer[Array, " ..."] = jnp.asarray(local_index)
        local_index_flat: Integer[Array, " N"] = local_index.ravel()
        return self.replace(index=self.index[local_index_flat], shape=local_index.shape)

    def __jax_array__(self) -> Integer[Array, "*shape"]:
        return self.index.reshape(self.shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self.index.dtype

    @property
    def size(self) -> int:
        return self.index.size

    def remap(self, global_index: "ArrayLike | Index", /) -> "Index":
        """`global[self]`."""
        global_index: Integer[Array, " N"] = jnp.asarray(global_index).ravel()
        return self.replace(index=global_index[self.index])

    # region Index Update

    def add(
        self, x: Shaped[ArrayLike, "*dim"], y: Shaped[ArrayLike, "..."], /
    ) -> Shaped[Array, "*dim"]:
        """`x[index] += y`."""
        x: Shaped[ArrayLike, "*dim"] = jnp.asarray(x)
        x_flat: Shaped[Array, " N"] = x.ravel()
        y_flat: Shaped[Array, " N"] = jnp.asarray(y).ravel()
        x_flat = x_flat.at[self.index].add(y_flat)
        return x_flat.reshape(x.shape)

    def get(self, x: Shaped[ArrayLike, "*dim"], /) -> Shaped[Array, "..."]:
        """`x[index]`."""
        x_flat: Shaped[Array, " N"] = jnp.asarray(x).ravel()
        return x_flat[self.index].reshape(self.shape)

    def set(
        self, x: Shaped[ArrayLike, "*dim"], y: Shaped[ArrayLike, "..."], /
    ) -> Shaped[Array, "*dim"]:
        """`x[index] = y`."""
        x: Shaped[Array, "*dim"] = jnp.asarray(x)
        x_flat: Shaped[Array, " N"] = x.ravel()
        y_flat: Shaped[Array, " N"] = jnp.asarray(y).ravel()
        x_flat = x_flat.at[self.index].set(y_flat)
        return x_flat.reshape(x.shape)

    # endregion Index Update
