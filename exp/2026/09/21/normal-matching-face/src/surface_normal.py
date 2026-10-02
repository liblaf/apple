"""Reference-area-weighted oriented triangle-normal matching for face skins."""

from __future__ import annotations

from typing import Any

import torch


class SurfaceNormalLoss:
    """Match corresponding oriented unit triangle normals with fixed area weights.

    ``triangles`` is assumed to share winding between the reference and target
    skin.  The scalar is the reference-area average squared chord distance of
    unit normals.  It is zero only when each corresponding oriented normal
    agrees; a reversed normal contributes four.
    """

    def __init__(
        self,
        reference_points: Any,
        triangles: Any,
        target_displacement: Any,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
    ) -> None:
        self.reference = torch.as_tensor(reference_points, device=device, dtype=dtype)
        self.triangles = torch.as_tensor(triangles, device=device, dtype=torch.long)
        self.target = torch.as_tensor(target_displacement, device=device, dtype=dtype)
        assert self.reference.ndim == 2
        assert self.reference.shape[1] == 3
        assert self.target.shape == self.reference.shape
        assert self.triangles.ndim == 2
        assert self.triangles.shape[1] == 3
        assert self.triangles.numel() > 0
        assert bool(torch.all(self.triangles >= 0))
        assert bool(torch.all(self.triangles < len(self.reference)))
        assert torch.isfinite(self.reference).all()
        assert torch.isfinite(self.target).all()
        self.reference_normals, self.reference_double_areas = self._normals(
            self.reference
        )
        self.target_normals, _ = self._normals(self.reference + self.target)
        self.reference_areas = self.reference_double_areas / 2
        self.area_sum = self.reference_areas.sum()
        assert bool(torch.isfinite(self.area_sum))
        assert bool(self.area_sum > 0)

    def _normals(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        assert points.shape == self.reference.shape
        triangles = points[self.triangles]
        cross = torch.linalg.cross(
            triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        )
        double_area = torch.linalg.vector_norm(cross, dim=1)
        assert bool(torch.all(torch.isfinite(double_area)))
        assert bool(torch.all(double_area > 1e-14)), "collapsed skin triangle"
        return cross / double_area[:, None], double_area

    def __call__(self, skin_displacement: torch.Tensor) -> torch.Tensor:
        """Return fixed-reference-area mean squared chord normal mismatch."""
        assert skin_displacement.shape == self.reference.shape
        assert skin_displacement.device == self.reference.device
        assert skin_displacement.dtype == self.reference.dtype
        normal, _ = self._normals(self.reference + skin_displacement)
        chord2 = (normal - self.target_normals).square().sum(dim=1)
        value = (self.reference_areas * chord2).sum() / self.area_sum
        assert torch.isfinite(value)
        return value

    def metrics(self, skin_displacement: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return target-relative angle and current-area diagnostics."""
        normal, double_area = self._normals(self.reference + skin_displacement)
        dot = (normal * self.target_normals).sum(dim=1).clamp(-1, 1)
        angle = torch.arccos(dot)
        return {
            "normal_angle_rms_deg": torch.rad2deg(
                torch.sqrt(
                    (self.reference_areas * angle.square()).sum() / self.area_sum
                )
            ),
            "normal_chord_rms": torch.sqrt(
                (
                    self.reference_areas
                    * (normal - self.target_normals).square().sum(dim=1)
                ).sum()
                / self.area_sum
            ),
            "minimum_triangle_area_ratio": (
                double_area / self.reference_double_areas
            ).min(),
        }
