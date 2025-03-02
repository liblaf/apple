import attrs
import jax
import numpy as np
from jaxtyping import Bool, Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True)
class Fixed(apple.AbstractPhysicsProblem):
    fixed_mask: Bool[np.ndarray, " D"] = attrs.field(metadata={"static": True})
    fixed_values: Float[jax.Array, " D"]
    problem: apple.AbstractPhysicsProblem

    @property
    def free_mask(self) -> Bool[np.ndarray, " D"]:
        return ~self.fixed_mask

    @property
    def n_dof(self) -> int:
        return self.problem.n_dof - self.n_fixed

    @property
    def n_fixed(self) -> int:
        return np.count_nonzero(self.fixed_mask)

    def fill(self, u: PyTree) -> PyTree:
        u_flat: Float[jax.Array, " DoF"] = self.ravel_u(u)
        u_flat: Float[jax.Array, " D"] = self.fill_flat(u_flat)
        u: PyTree = self.problem.unravel_u(u_flat)
        return u

    def fill_flat(self, u_flat: Float[jax.Array, " D"]) -> Float[jax.Array, " D"]:
        u_new: Float[jax.Array, " D"] = self.fixed_values.copy()
        u_new: Float[jax.Array, " D"] = u_new.at[self.free_mask].set(u_flat)
        return u_new

    @apple.jit()
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        return self.fun_flat(self.ravel_u(u), self.ravel_q(q))

    @apple.jit()
    def fun_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[jax.Array, ""]:
        u_flat: Float[jax.Array, " D"] = self.fill_flat(u_flat)
        return self.problem.fun_flat(u_flat, q_flat)

    def prepare(self, params: PyTree | None = None) -> None:
        self.problem.prepare(params)

    def ravel_params(self, params: PyTree) -> jax.Array:
        return self.problem.ravel_params(params)

    def ravel_q(self, q: PyTree | None) -> Float[jax.Array, " Q"] | None:  # pyright: ignore[reportIncompatibleMethodOverride]
        return self.problem.ravel_q(q)

    def ravel_u(self, u: PyTree) -> Float[jax.Array, " DoF"]:
        return u

    def unravel_params(self, params_flat: Float[jax.Array, " P"]) -> PyTree:
        return self.problem.ravel_params(params_flat)

    def unravel_q(self, q: Float[jax.Array, " Q"] | None) -> PyTree | None:
        return self.problem.unravel_q(q)

    def unravel_u(self, u: Float[jax.Array, " DoF"]) -> PyTree:
        return u
