from collections.abc import Mapping

import equinox as eqx
import jarp
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from liblaf.apple import utils

from ._state import JaxEnergyState

type EnergyMaterials = Mapping[str, Array]
type Index = Integer[Array, " points"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, "points dim"]


@jarp.define
class JaxEnergy:
    name: str = jarp.static(default=utils.name_factory, kw_only=True)
    requires_grad: frozenset[str] = jarp.static(default=frozenset(), kw_only=True)

    def init_state(self, u: Vector) -> JaxEnergyState:  # noqa: ARG002
        return JaxEnergyState()

    def update(self, state: JaxEnergyState, u: Vector) -> JaxEnergyState:  # noqa: ARG002
        return state

    def update_materials(self, materials: EnergyMaterials) -> None:
        pass

    def fun(self, state: JaxEnergyState, u: Vector) -> Scalar:
        raise NotImplementedError

    @jarp.jit(inline=True)
    def grad(self, state: JaxEnergyState, u: Vector) -> Updates:
        values: Vector = eqx.filter_grad(jarp.partial(self.fun, state))(u)
        return values, jnp.arange(u.shape[0])

    def hess_diag(self, state: JaxEnergyState, u: Vector) -> Updates:
        raise NotImplementedError

    @jarp.jit(inline=True)
    def hess_prod(self, state: JaxEnergyState, u: Vector, p: Vector) -> Updates:
        values: Vector
        _, values = jax.jvp(jax.grad(jarp.partial(self.fun, state)), (u,), (p,))
        return values, jnp.arange(u.shape[0])

    @jarp.jit(inline=True)
    def hess_quad(self, state: JaxEnergyState, u: Vector, p: Vector) -> Scalar:
        values: Vector
        index: Index
        values, index = self.hess_prod(state, u, p)
        return jnp.vdot(p[index], values)

    @jarp.jit(inline=True)
    def value_and_grad(
        self, state: JaxEnergyState, u: Vector
    ) -> tuple[Scalar, Updates]:
        value: Scalar
        grad: Vector
        value, grad = jax.value_and_grad(jarp.partial(self.fun, state))(u)
        return value, (grad, jnp.arange(u.shape[0]))

    @jarp.jit(inline=True)
    def grad_and_hess_diag(
        self, state: JaxEnergyState, u: Vector
    ) -> tuple[Updates, Updates]:
        return self.grad(state, u), self.hess_diag(state, u)

    @jarp.jit(inline=True)
    def mixed_derivative_prod(
        self, state: JaxEnergyState, u: Vector, p: Vector
    ) -> dict[str, Array]:
        outputs: dict[str, Array] = {}
        for name in self.requires_grad:
            outputs[name] = getattr(self, f"mixed_derivative_prod_{name}")(state, u, p)
        return outputs
