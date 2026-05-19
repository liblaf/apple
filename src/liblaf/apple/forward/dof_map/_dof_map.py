import attrs
import torch
from jaxtyping import Float, Integer
from torch import Tensor

type Free = Float[Tensor, " free"]
type Full = Float[Tensor, "points dim"]


@attrs.define
class DofMap:
    n_points: int
    fixed_indices: Integer[Tensor, " fixed"]
    fixed_values: Float[Tensor, " fixed"]
    free_indices: Integer[Tensor, " free"]
    dim: int = attrs.field(default=3)

    @property
    def n_fixed(self) -> int:
        return self.fixed_indices.numel()

    @property
    def n_free(self) -> int:
        return self.free_indices.numel()

    @property
    def n_full(self) -> int:
        return self.n_points * self.dim

    def to_free(self, full: Full) -> Free:
        return full.flatten()[self.free_indices]

    def to_free_grad(self, full: Full) -> Free:
        return full.flatten()[self.free_indices]

    def to_free_hess_diag(self, full: Full) -> Free:
        return full.flatten()[self.free_indices]

    def to_full(self, free: Free) -> Full:
        result: Full = torch.empty((self.n_full,), dtype=free.dtype)
        result[self.fixed_indices] = self.fixed_values
        result[self.free_indices] = free
        return result.reshape((self.n_points, self.dim))

    def to_full_grad(self, grad_free: Free) -> Full:
        result: Full = torch.empty((self.n_full,), dtype=grad_free.dtype)
        result[self.fixed_indices] = 0.0
        result[self.free_indices] = grad_free
        return result.reshape((self.n_points, self.dim))
