import attrs
import torch
from jaxtyping import Float
from torch import Tensor

type Scalar = Float[Tensor, ""]
type Vector = Float[Tensor, " N"]


@attrs.define
class HessianDampingState:
    """Adaptive diagonal Hessian-damping state."""

    factor: float = attrs.field()
    hess_diag_mean: Scalar = attrs.field(default=None)


@attrs.define
class HessianDamping:
    """Adaptive Levenberg-style diagonal Hessian damping."""

    def _default_factor_max(self) -> float:
        return 1e3 * self.initial

    def _default_factor_min(self) -> float:
        return 1e-3 * self.initial

    initial: float = 0.0
    factor_max: float = attrs.field(
        default=attrs.Factory(_default_factor_max, takes_self=True)
    )
    factor_min: float = attrs.field(
        default=attrs.Factory(_default_factor_min, takes_self=True)
    )

    def init(self) -> HessianDampingState:
        """Create damping state with the initial factor."""
        return HessianDampingState(factor=self.initial)

    def hess_diag(self, state: HessianDampingState, H_diag: Vector) -> Vector:
        """Return `abs(H_diag) + factor * mean(abs(H_diag))`."""
        H_diag: Vector = torch.abs(H_diag)
        H_diag_mean: Scalar = torch.mean(H_diag)
        state.hess_diag_mean = H_diag_mean
        return H_diag + state.factor * H_diag_mean

    def hess_quad(self, state: HessianDampingState, p: Vector, pHp: Scalar) -> Scalar:
        """Add the damping contribution to a Hessian quadratic form."""
        return pHp + state.factor * state.hess_diag_mean * torch.dot(p, p)

    def update(
        self,
        state: HessianDampingState,
        *,
        actual_decrease: Scalar,
        line_search_step: int,
        predicted_decrease: Scalar,
    ) -> None:
        """Adapt the damping factor from line-search behavior."""
        factor: float = state.factor
        if line_search_step == 0 and actual_decrease > predicted_decrease:
            factor *= 0.5
        else:
            factor *= (line_search_step + 1) ** 2
        state.factor = min(max(factor, self.factor_min), self.factor_max)
