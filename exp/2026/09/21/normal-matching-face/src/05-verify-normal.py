"""CPU algebra and autograd checks for the face target-normal loss."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from experiment import Profile
from study import Study
from surface_normal import SurfaceNormalLoss

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("05-normal-verification")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def fixture() -> tuple[torch.Tensor, torch.Tensor]:
    points = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.0, 1.0],
            [0.0, 0.5, 1.0],
        ],
        dtype=torch.float64,
    )
    triangles = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.long)
    return points, triangles


def main(cfg: Config) -> None:  # noqa: PLR0915
    points, triangles = fixture()
    target = torch.zeros_like(points)
    loss = SurfaceNormalLoss(
        points, triangles, target, device="cpu", dtype=torch.float64
    )
    zero = torch.zeros_like(points, requires_grad=True)
    value = loss(zero)
    value.backward()
    assert float(value.detach()) == 0.0
    assert float(zero.grad.abs().max()) == 0.0
    assert torch.allclose(
        loss.reference_normals[0], torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )
    reversed_loss = SurfaceNormalLoss(
        points, triangles[:, [0, 2, 1]], target, device="cpu", dtype=torch.float64
    )
    assert torch.allclose(
        reversed_loss.reference_normals[0],
        torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64),
    )
    assert float(reversed_loss(torch.zeros_like(points))) == 0.0

    target = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.2],
            [0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.03],
            [0.0, 0.0, -0.02],
        ],
        dtype=torch.float64,
    )
    loss = SurfaceNormalLoss(
        points, triangles, target, device="cpu", dtype=torch.float64
    )
    assert float(loss(target)) <= 1e-29
    translation = torch.tensor([0.7, -0.4, 1.2], dtype=torch.float64)
    assert float(loss(target + translation)) <= 1e-29

    trial = target.clone()
    trial[0, 2] += 0.3
    measured = loss(trial)
    trial_triangles = (points + trial)[triangles]
    cross = torch.linalg.cross(
        trial_triangles[:, 1] - trial_triangles[:, 0],
        trial_triangles[:, 2] - trial_triangles[:, 0],
    )
    normal = cross / torch.linalg.vector_norm(cross, dim=1)[:, None]
    manual = (
        loss.reference_areas * (normal - loss.target_normals).square().sum(dim=1)
    ).sum() / loss.area_sum
    assert torch.allclose(measured, manual, rtol=0, atol=1e-15)
    assert torch.allclose(
        loss.reference_areas / loss.area_sum,
        torch.tensor([16 / 17, 1 / 17], dtype=torch.float64),
    )

    trial = (
        target
        + torch.tensor(
            np.random.default_rng(20260921).normal(scale=0.05, size=points.shape),
            dtype=torch.float64,
        )
    ).requires_grad_()
    analytic = torch.autograd.grad(loss(trial), trial)[0]
    direction = torch.tensor(
        np.random.default_rng(20260922).normal(size=points.shape), dtype=torch.float64
    )
    direction /= torch.linalg.vector_norm(direction)
    epsilon = 1e-6
    numeric = (
        loss(trial.detach() + epsilon * direction)
        - loss(trial.detach() - epsilon * direction)
    ) / (2 * epsilon)
    exact = (analytic * direction).sum()
    relative = float(
        abs(exact - numeric)
        / torch.maximum(torch.maximum(abs(exact), abs(numeric)), torch.tensor(1e-12))
    )
    assert relative < 2e-8, (exact, numeric, relative)
    metrics = loss.metrics(trial.detach())
    assert float(metrics["minimum_triangle_area_ratio"]) > 0

    collapsed = torch.zeros_like(points)
    collapsed[2] = points[1] - points[2]
    collapsed_fails = False
    try:
        loss(collapsed)
    except AssertionError as exc:
        collapsed_fails = "collapsed skin triangle" in str(exc)
    assert collapsed_fails

    fake = SimpleNamespace(
        edge_i=torch.tensor([0, 1]),
        edge_j=torch.tensor([1, 2]),
        conductance=torch.tensor([2.0, 0.5], dtype=torch.float64),
        regularizer_factor=0.3,
    )
    q = torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.4, -0.5, 0.6],
            [-0.3, 0.2, 0.1, -0.2, 0.4, -0.1],
            [0.2, 0.0, -0.1, 0.3, -0.2, 0.5],
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    regularizer = Study.regularizer(fake, q)
    difference = q[[0, 1]] - q[[1, 2]]
    manual_regularizer = 0.3 * torch.sum(
        torch.tensor([2.0, 0.5], dtype=torch.float64)
        * (difference[:, :3].square().sum(1) + 2 * difference[:, 3:].square().sum(1))
    )
    assert torch.allclose(regularizer, manual_regularizer, rtol=0, atol=1e-15)
    regularizer_gradient = torch.autograd.grad(regularizer, q)[0]
    regularizer_direction = torch.tensor(
        np.random.default_rng(20260923).normal(size=q.shape), dtype=torch.float64
    )
    regularizer_direction /= torch.linalg.vector_norm(regularizer_direction)
    regularizer_numeric = (
        Study.regularizer(fake, q.detach() + epsilon * regularizer_direction)
        - Study.regularizer(fake, q.detach() - epsilon * regularizer_direction)
    ) / (2 * epsilon)
    regularizer_exact = torch.sum(regularizer_gradient * regularizer_direction)
    regularizer_relative = float(
        abs(regularizer_exact - regularizer_numeric)
        / torch.maximum(
            torch.maximum(abs(regularizer_exact), abs(regularizer_numeric)),
            torch.tensor(1e-12),
        )
    )
    assert regularizer_relative < 2e-9

    report: dict[str, Any] = {
        "passed": True,
        "target_zero": float(loss(target)),
        "translation_invariance_loss": float(loss(target + translation)),
        "normal_derivative_relative_error": relative,
        "reference_area_weights": [
            float(value) for value in (loss.reference_areas / loss.area_sum)
        ],
        "normal_angle_rms_deg": float(metrics["normal_angle_rms_deg"]),
        "minimum_triangle_area_ratio": float(metrics["minimum_triangle_area_ratio"]),
        "collapsed_triangle_fails_visibly": True,
        "raw6_regularizer": float(regularizer.detach()),
        "raw6_regularizer_derivative_relative_error": regularizer_relative,
        "device": "cpu",
        "dtype": "float64",
    }
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "checks.json", report)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
