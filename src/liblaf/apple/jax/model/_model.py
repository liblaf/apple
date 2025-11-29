import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer
from liblaf.peach import tree

from ._energy import JaxEnergy

type Index = Integer[Array, " N"]
type Scalar = Float[Array, ""]
type Updates = tuple[Vector, Index]
type Vector = Float[Array, " N"]


@tree.define
class JaxModel:
    energies: dict[str, JaxEnergy] = tree.field(factory=dict, kw_only=True)

    def update(self, x: Vector) -> None:
        for energy in self.energies.values():
            energy.update(x)

    @eqx.filter_jit
    def fun(self, x: Vector) -> Scalar:
        output: Scalar = jnp.zeros(())
        for energy in self.energies.values():
            output += energy.fun(x)
        return output

    @eqx.filter_jit
    def grad(self, x: Vector) -> Vector:
        output: Vector = jnp.zeros_like(x)
        for energy in self.energies.values():
            grad: Vector
            index: Index
            grad, index = energy.grad(x)
            output = output.at[index].add(grad)
        return output

    @eqx.filter_jit
    def hess_diag(self, x: Vector) -> Vector:
        output: Vector = jnp.zeros_like(x)
        for energy in self.energies.values():
            diag: Vector
            index: Index
            diag, index = energy.hess_diag(x)
            output = output.at[index].add(diag)
        return output

    @eqx.filter_jit
    def hess_prod(self, x: Vector, p: Vector) -> Vector:
        output: Vector = jnp.zeros_like(x)
        for energy in self.energies.values():
            prod: Vector
            index: Index
            prod, index = energy.hess_prod(x, p)
            output = output.at[index].add(prod)
        return output

    @eqx.filter_jit
    def hess_quad(self, x: Vector, p: Vector) -> Scalar:
        output: Scalar = jnp.zeros(())
        for energy in self.energies.values():
            output += energy.hess_quad(x, p)
        return output

    @eqx.filter_jit
    def mixed_derivative_prod(
        self, x: Vector, p: Vector
    ) -> dict[str, dict[str, Array]]:
        return {
            name: energy.mixed_derivative_prod(x, p)
            for name, energy in self.energies.items()
        }

    @eqx.filter_jit
    def value_and_grad(self, x: Vector) -> tuple[Scalar, Vector]:
        value: Scalar = jnp.zeros(())
        grad: Vector = jnp.zeros_like(x)
        for energy in self.energies.values():
            value_i: Scalar
            grad_i: Vector
            value_i, (grad_i, index) = energy.value_and_grad(x)
            value += value_i
            grad = grad.at[index].add(grad_i)
        return value, grad

    @eqx.filter_jit
    def grad_and_hess_diag(self, x: Vector) -> tuple[Vector, Vector]:
        grad: Vector = jnp.zeros_like(x)
        hess_diag: Vector = jnp.zeros_like(x)
        for energy in self.energies.values():
            grad_i: Vector
            index_g: Index
            hess_diag_i: Vector
            index_h: Index
            (grad_i, index_g), (hess_diag_i, index_h) = energy.grad_and_hess_diag(x)
            grad = grad.at[index_g].add(grad_i)
            hess_diag = hess_diag.at[index_h].add(hess_diag_i)
        return grad, hess_diag
