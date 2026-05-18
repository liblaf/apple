import jax.numpy as jnp
from jaxtyping import Array, Float

from liblaf import jarp
from liblaf.apple.collision import Collision
from liblaf.apple.warp.model import WarpModelAdapter

from .dof_map import DofMap

type Full = Float[Array, "points dim"]
type Scalar = Float[Array, ""]


@jarp.define
class Model:
    from ._state import ModelState as State

    dof_map: DofMap
    warp_model: WarpModelAdapter = jarp.static()
    collision: Collision | None = jarp.static(default=None)

    @property
    def dim(self) -> int:
        return self.dof_map.dim

    @property
    def n_fixed(self) -> int:
        return self.dof_map.n_fixed

    @property
    def n_free(self) -> int:
        return self.dof_map.n_free

    @property
    def n_full(self) -> int:
        return self.dof_map.n_full

    @property
    def n_points(self) -> int:
        return self.dof_map.n_points

    def init(self) -> State:
        u_full: Full = self.dof_map.to_full(jnp.zeros((self.n_free,)))
        return self.State(u=u_full)

    def fun(self, u: Full) -> Scalar:
        output: Scalar = self.warp_model.fun(u)
        if self.collision is not None:
            output += self.collision.fun(u)
        return output

    def grad(self, u: Full) -> Full:
        output: Full = self.warp_model.grad(u)
        if self.collision is not None:
            output += self.collision.grad(u)
        return output

    def hess_diag(self, u: Full) -> Full:
        output: Full = self.warp_model.hess_diag(u)
        if self.collision is not None:
            output += self.collision.hess_diag(u)
        return output

    def hess_prod(self, u: Full, p: Full) -> Full:
        output: Full = self.warp_model.hess_prod(u, p)
        if self.collision is not None:
            output += self.collision.hess_prod(u, p)
        return output

    def hess_quad(self, u: Full, p: Full) -> Scalar:
        output: Scalar = self.warp_model.hess_quad(u, p)
        if self.collision is not None:
            output += self.collision.hess_quad(u, p)
        return output
