"""Independent algebraic and autograd checks for ``SurfaceGradientLoss``."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from liblaf import cherries

sys.path.insert(0, str(Path(__file__).parent))
from surface_loss import SurfaceGradientLoss


class Config(cherries.BaseConfig):
    output: Path = cherries.output("05-surface-loss/checks.json", mkdir=True)


def unit_square(resolution: int) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(0.0, 1.0, resolution + 1)
    x, y = np.meshgrid(grid, grid, indexing="xy")
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
    triangles: list[tuple[int, int, int]] = []
    for row in range(resolution):
        for column in range(resolution):
            lower_left = row * (resolution + 1) + column
            lower_right = lower_left + 1
            upper_left = lower_left + resolution + 1
            upper_right = upper_left + 1
            triangles.extend(
                (
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                )
            )
    return points, np.asarray(triangles, dtype=np.int64)


def value_for_affine_field(resolution: int, matrix: torch.Tensor) -> float:
    reference, triangles = unit_square(resolution)
    loss = SurfaceGradientLoss(reference, triangles)
    point_tensor = torch.as_tensor(reference, dtype=torch.float64)
    positions = point_tensor + point_tensor @ matrix.T
    return float(loss(positions, point_tensor))


def main(cfg: Config) -> None:
    torch.manual_seed(17)
    reference, triangles = unit_square(3)
    loss = SurfaceGradientLoss(reference, triangles)
    target = torch.as_tensor(reference, dtype=torch.float64)

    zero = float(loss(target, target))
    translation = torch.tensor([0.31, -0.22, 0.47], dtype=torch.float64)
    translation_nullspace = float(loss(target + translation, target))

    # Corresponding vertices are matched: a rotation is expected to have energy.
    angle = torch.tensor(0.31, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle), torch.zeros_like(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle), torch.zeros_like(angle))),
            torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64),
        )
    )
    rotation_energy = float(loss(target @ rotation.T, target))

    affine_matrix = torch.tensor(
        [[0.25, -0.10, 0.0], [0.30, 0.15, 0.0], [-0.20, 0.40, 0.0]],
        dtype=torch.float64,
    )
    affine_energy = value_for_affine_field(3, affine_matrix)
    affine_expected = float((affine_matrix[:, :2] ** 2).sum())

    positions = (target + 0.1 * torch.randn_like(target)).requires_grad_()
    autograd_value = loss(positions, target)
    autograd_gradient = torch.autograd.grad(autograd_value, positions)[0]
    direction = torch.randn_like(positions)
    direction /= torch.linalg.vector_norm(direction)
    epsilon = 1.0e-6
    with torch.no_grad():
        finite_difference = float(
            (
                loss(positions + epsilon * direction, target)
                - loss(positions - epsilon * direction, target)
            )
            / (2 * epsilon)
        )
    directional_derivative = float((autograd_gradient * direction).sum())
    finite_difference_relative_error = abs(
        finite_difference - directional_derivative
    ) / max(1.0, abs(finite_difference), abs(directional_derivative))

    resolution_one = value_for_affine_field(1, affine_matrix)
    resolution_eight = value_for_affine_field(8, affine_matrix)
    resolution_relative_error = abs(resolution_one - resolution_eight) / affine_expected

    checks = {
        "zero_residual": zero,
        "constant_translation_nullspace": translation_nullspace,
        "rotation_energy": rotation_energy,
        "affine_energy": affine_energy,
        "affine_expected": affine_expected,
        "affine_relative_error": abs(affine_energy - affine_expected) / affine_expected,
        "finite_difference": finite_difference,
        "directional_derivative": directional_derivative,
        "finite_difference_relative_error": finite_difference_relative_error,
        "resolution_one": resolution_one,
        "resolution_eight": resolution_eight,
        "resolution_relative_error": resolution_relative_error,
    }
    passed = (
        zero < 1.0e-28
        and translation_nullspace < 1.0e-28
        and rotation_energy > 1.0e-6
        and checks["affine_relative_error"] < 1.0e-12
        and finite_difference_relative_error < 1.0e-9
        and resolution_relative_error < 1.0e-12
    )
    checks["passed"] = passed
    assert passed, checks
    cfg.output.write_text(json.dumps(checks, indent=2) + "\n")
    cherries.log_metrics(
        {
            f"surface_loss/{key}": value
            for key, value in checks.items()
            if isinstance(value, (int, float))
        }
    )


if __name__ == "__main__":
    cherries.main(main)
