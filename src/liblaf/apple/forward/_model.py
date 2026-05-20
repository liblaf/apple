from collections.abc import Mapping

import attrs
import torch
import warp as wp
from jaxtyping import Float
from torch import Tensor

from liblaf.apple.collision import Collision
from liblaf.apple.torch.utils import method_with_device
from liblaf.apple.warp.model import WarpModelAdapter

from .dof_map import DofMap

type Full = Float[Tensor, "points dim"]
type Scalar = Float[Tensor, ""]


@attrs.define
class Model:
    @attrs.define
    class State:
        u: Full
        collision: Collision.State | None = None

    dof_map: DofMap
    warp_model: WarpModelAdapter
    collision: Collision | None = None
    device: torch.device = attrs.field(factory=torch.get_default_device)

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

    def get_materials(self) -> dict[str, dict[str, Tensor]]:
        return self.warp_model.get_materials()

    def set_materials(
        self, materials: Mapping[str, Mapping[str, wp.array | Tensor]]
    ) -> None:
        self.warp_model.set_materials(materials)

    def require_grad(self, materials: Mapping[str, Mapping[str, bool]]) -> None:
        self.warp_model.require_grad(materials)

    @method_with_device
    def init(self) -> State:
        u_full: Full = self.dof_map.to_full(torch.zeros((self.n_free,)))
        state: Model.State = self.State(u=u_full)
        if self.collision is not None:
            state.collision = self.collision.init()
        return state

    @method_with_device
    def max_step_size(self, state: State, p: Full) -> Scalar:
        if self.collision is None:
            return torch.ones((), dtype=state.u.dtype)
        assert state.collision is not None
        return self.collision.max_step_size(state.collision, state.u, p)

    @method_with_device
    def update(self, state: State, u: Full) -> None:
        state.u.copy_(u)
        if self.collision is not None:
            assert state.collision is not None
            self.collision.update(state.collision, u)

    @method_with_device
    def fun(self, state: State) -> Scalar:
        output: Scalar = self.warp_model.fun(state.u)
        if self.collision is not None:
            assert state.collision is not None
            output += self.collision.fun(state.collision, state.u)
        return output

    @method_with_device
    def grad(self, state: State) -> Full:
        output: Full = torch.zeros_like(state.u)
        self.warp_model.grad(state.u, output)
        if self.collision is not None:
            assert state.collision is not None
            self.collision.grad(state.collision, state.u, output)
        return output

    @method_with_device
    def hess_diag(self, state: State) -> Full:
        output: Full = torch.zeros_like(state.u)
        self.warp_model.hess_diag(state.u, output)
        if self.collision is not None:
            assert state.collision is not None
            self.collision.hess_diag(state.collision, state.u, output)
        return output

    @method_with_device
    def hess_prod(self, state: State, p: Full) -> Full:
        output: Full = torch.zeros_like(state.u)
        self.warp_model.hess_prod(state.u, p, output)
        if self.collision is not None:
            assert state.collision is not None
            self.collision.hess_prod(state.collision, state.u, p, output)
        return output

    @method_with_device
    def hess_quad(self, state: State, p: Full) -> Scalar:
        output: Scalar = self.warp_model.hess_quad(state.u, p)
        if self.collision is not None:
            assert state.collision is not None
            output += self.collision.hess_quad(state.collision, state.u, p)
        return output

    def mixed_derivative_prod(self, state: State, p: Full) -> Full:
        output: Full = torch.zeros_like(state.u)
        self.warp_model.mixed_derivative_prod(state.u, p)
        return output
