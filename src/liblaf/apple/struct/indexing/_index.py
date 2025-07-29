from collections.abc import Sequence

import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, Integer, Shaped

from liblaf.apple.struct import tree


class Index(tree.PyTree):
    index: Integer[Array, " N"] = tree.array(factory=lambda: jnp.empty((0,), dtype=int))
    shape: Sequence[int] = tree.field(default=None)

    def __post_init__(self) -> None:
        if self.shape is None:
            self.shape = self.index.shape  # pyright: ignore[reportAttributeAccessIssue]

    def __getitem__(self, index: "ArrayLike | Index") -> "Index":
        index: Integer[Array, " N"] = jnp.asarray(index)
        index_flat: Integer[Array, " N"] = jnp.asarray(index).ravel()
        return self.replace(index=self.index[index_flat], shape=index.shape)

    def __jax_array__(self) -> Integer[Array, " N"]:
        return self.index.reshape(self.shape)

    @property
    def dtype(self) -> jnp.dtype:
        return self.index.dtype

    @property
    def size(self) -> int:
        return self.index.size

    # region Index Update

    def add(
        self, x: Shaped[ArrayLike, "*dim"], y: Shaped[ArrayLike, "..."], /
    ) -> Shaped[Array, "*dim"]:
        x: Shaped[ArrayLike, "*dim"] = jnp.asarray(x)
        x_flat: Shaped[Array, " N"] = x.ravel()
        y_flat: Shaped[Array, " N"] = jnp.asarray(y).ravel()
        x_flat = x_flat.at[self.index].add(y_flat)
        return x_flat.reshape(x.shape)

    def get(self, x: "ArrayLike | Index", /) -> Shaped[Array, "..."]:
        x_flat: Shaped[Array, " N"] = jnp.asarray(x).ravel()
        return x_flat[self.index].reshape(self.shape)

    def set(
        self, x: Shaped[ArrayLike, "*dim"], y: Shaped[ArrayLike, "..."], /
    ) -> Shaped[Array, "*dim"]:
        x: Shaped[Array, "*dim"] = jnp.asarray(x)
        x_flat: Shaped[Array, " N"] = x.ravel()
        y_flat: Shaped[Array, " N"] = jnp.asarray(y).ravel()
        x_flat = x_flat.at[self.index].set(y_flat)
        return x_flat.reshape(x.shape)

    # endregion Index Update
