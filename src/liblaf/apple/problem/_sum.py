from typing import override

import attrs
import jax
import jax.numpy as jnp
from jaxtyping import Float, PyTree

from liblaf import apple


@apple.register_dataclass()
@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class Sum(apple.AbstractPhysicsProblem):
    name: str = attrs.field(default="sum", metadata={"static": True})
    problems: list[apple.AbstractPhysicsProblem]

    @property
    def n_dof(self) -> int:
        return self.problems[0].n_dof

    @apple.jit()
    def fun(self, u: PyTree, q: PyTree | None = None) -> Float[jax.Array, ""]:
        return jnp.sum(jnp.asarray([problem.fun(u, q) for problem in self.problems]))

    @apple.jit()
    def fun_flat(
        self,
        u_flat: Float[jax.Array, " DoF"],
        q_flat: Float[jax.Array, " Q"] | None = None,
    ) -> Float[jax.Array, ""]:
        return jnp.sum(
            jnp.asarray([problem.fun_flat(u_flat, q_flat) for problem in self.problems])
        )

    @override
    def ravel_u(self, u: PyTree) -> Float[jax.Array, " DoF"]:
        return self.problems[0].ravel_u(u)

    @override
    def unravel_u(self, u_flat: Float[jax.Array, " DoF"]) -> PyTree:
        return self.problems[0].unravel_u(u_flat)


@attrs.define(kw_only=True, on_setattr=attrs.setters.convert)
class SumBuilder(apple.AbstractPhysicsProblemBuilder):
    name: str = attrs.field(default="sum", metadata={"static": True})
    problems: list[apple.AbstractPhysicsProblemBuilder]

    def build(self, q: PyTree | None = None) -> Sum:
        super().build(q)
        return Sum(
            name=self.name,
            problems=[problem.build(q) for problem in self.problems],
            _q_unravel=self._q_unravel,
        )
