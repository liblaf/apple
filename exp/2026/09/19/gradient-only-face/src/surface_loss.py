# ruff: noqa: PT018
"""Fixed-reference, piecewise-linear surface-gradient matching."""

from __future__ import annotations

import numpy as np
import torch


class SurfaceGradientLoss:
    """Area-normalized squared gradient error on a triangular reference surface.

    Gradients of the barycentric basis functions and triangle areas are computed
    once from ``reference_points``.  Therefore, only ``positions`` participates
    in autograd; the operator is a fixed correspondence-based surface operator.
    """

    def __init__(
        self,
        reference_points: np.ndarray,
        triangles: np.ndarray,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        points = np.asarray(reference_points)
        connectivity = np.asarray(triangles)
        assert points.ndim == 2 and points.shape[1] == 3
        assert connectivity.ndim == 2 and connectivity.shape[1] == 3
        assert np.issubdtype(connectivity.dtype, np.integer)
        assert len(points) > 0 and len(connectivity) > 0
        assert connectivity.min() >= 0 and connectivity.max() < len(points)

        reference = torch.as_tensor(points, device=device, dtype=dtype)
        self.triangles = torch.as_tensor(connectivity, device=device, dtype=torch.long)
        vertices = reference[self.triangles]
        normal_twice_area = torch.linalg.cross(
            vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
        )
        squared_twice_area = (normal_twice_area**2).sum(dim=-1)
        assert bool(torch.all(squared_twice_area > 0))
        self.areas = 0.5 * torch.sqrt(squared_twice_area)
        self.total_area = self.areas.sum()

        # grad(phi_i) = cross(n, opposite edge_i) / ||n||^2.
        self.barycentric_gradients = (
            torch.stack(
                (
                    torch.linalg.cross(
                        normal_twice_area, vertices[:, 2] - vertices[:, 1]
                    ),
                    torch.linalg.cross(
                        normal_twice_area, vertices[:, 0] - vertices[:, 2]
                    ),
                    torch.linalg.cross(
                        normal_twice_area, vertices[:, 1] - vertices[:, 0]
                    ),
                ),
                dim=1,
            )
            / squared_twice_area[:, None, None]
        )

    def __call__(self, positions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return mean area-weighted Frobenius-squared gradient residual."""
        assert positions.ndim == target.ndim == 2
        assert positions.shape == target.shape
        assert positions.shape[1] == 3
        assert positions.shape[0] > int(self.triangles.max())
        assert positions.device == target.device == self.areas.device
        assert positions.dtype == target.dtype == self.areas.dtype

        residual = (positions - target)[self.triangles]
        gradient = torch.einsum("tvi,tvj->tij", residual, self.barycentric_gradients)
        energy = (gradient**2).sum(dim=(-2, -1))
        return (self.areas * energy).sum() / self.total_area
