"""Validate the metric-projected 19D BFGS quadratic subproblem."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pydantic_settings as ps
import torch
from joint_common import ProfileJoint, archive_sources, write_json
from joint_fields import research_informed_material_config, symmetric_coordinates
from joint_outer import metric_projected_direction, project_shared_free

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("outer-validation", mkdir=True)


def box(lower: float, upper: float):
    def project(value: torch.Tensor) -> torch.Tensor:
        return value.clamp(min=lower, max=upper)

    return project


def assert_rejected(inverse_hessian: torch.Tensor) -> None:
    try:
        metric_projected_direction(
            torch.zeros(2),
            torch.ones(2),
            inverse_hessian,
            box(-1.0, 1.0),
        )
    except AssertionError:
        return
    message = "invalid inverse Hessian was accepted"
    raise AssertionError(message)


def main(cfg: Config) -> None:  # noqa: PLR0915
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    tolerance = 1e-10

    diagonal_x = torch.zeros(2)
    diagonal_gradient = torch.tensor([-2.0, 4.0])
    diagonal_inverse = torch.diag(torch.tensor([2.0, 0.5]))
    diagonal_direction, diagonal_receipt = metric_projected_direction(
        diagonal_x,
        diagonal_gradient,
        diagonal_inverse,
        box(-1.0, 1.0),
        residual_tolerance=tolerance,
    )
    diagonal_expected = torch.tensor([1.0, -1.0])
    torch.testing.assert_close(
        diagonal_x + diagonal_direction,
        diagonal_expected,
        atol=2e-10,
        rtol=0,
    )

    coupled_metric = torch.tensor([[2.0, 0.5], [0.5, 1.0]])
    coupled_inverse = torch.linalg.inv(coupled_metric)
    coupled_expected = torch.tensor([1.0, 0.2])
    coupled_terminal_gradient = torch.tensor([-0.3, 0.0])
    coupled_gradient = coupled_terminal_gradient - coupled_metric @ coupled_expected
    coupled_direction, coupled_receipt = metric_projected_direction(
        torch.zeros(2),
        coupled_gradient,
        coupled_inverse,
        box(-1.0, 1.0),
        residual_tolerance=tolerance,
    )
    torch.testing.assert_close(coupled_direction, coupled_expected, atol=2e-9, rtol=0)
    assert coupled_receipt["method"] == "accelerated_metric_projection"
    coupled_quadratic = float(
        torch.dot(coupled_gradient, coupled_direction)
        + 0.5 * torch.dot(coupled_direction, coupled_metric @ coupled_direction)
    )
    assert coupled_quadratic <= 0

    constraints = research_informed_material_config()["constraints"]
    shared_x = torch.zeros(19)
    shared_unconstrained = torch.zeros(19)
    angle = 0.37
    rotation = torch.tensor(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    unbounded_eigenvalues = torch.tensor([12.0, -2.0, 0.4])
    unbounded_matrix = (
        rotation @ torch.diag(unbounded_eigenvalues) @ rotation.transpose(-1, -2)
    )
    shared_unconstrained[:6] = symmetric_coordinates(unbounded_matrix)
    shared_unconstrained[18] = 2.0
    shared_inverse = 0.5 * torch.eye(19)
    shared_gradient = -torch.linalg.solve(shared_inverse, shared_unconstrained)
    shared_direction, shared_receipt = metric_projected_direction(
        shared_x,
        shared_gradient,
        shared_inverse,
        lambda value: project_shared_free(value, constraints),
        residual_tolerance=tolerance,
    )
    shared_expected = shared_unconstrained.clone()
    expected_matrix = (
        rotation
        @ torch.diag(torch.tensor([10.0, -0.9, 0.4]))
        @ rotation.transpose(-1, -2)
    )
    shared_expected[:6] = symmetric_coordinates(expected_matrix)
    shared_expected[18] = math.log(3.0)
    torch.testing.assert_close(shared_direction, shared_expected, atol=2e-9, rtol=0)

    direct_gradient = torch.linspace(-0.1, 0.1, 19)
    direct_inverse = 0.01 * torch.eye(19)
    direct_direction, direct_receipt = metric_projected_direction(
        shared_x,
        direct_gradient,
        direct_inverse,
        lambda value: project_shared_free(value, constraints),
        residual_tolerance=tolerance,
    )
    torch.testing.assert_close(
        direct_direction,
        -direct_inverse @ direct_gradient,
        atol=2e-12,
        rtol=0,
    )
    assert direct_receipt["method"] == "direct_unconstrained_feasible"

    interior_stationary, interior_receipt = metric_projected_direction(
        torch.zeros(2),
        torch.zeros(2),
        torch.eye(2),
        box(-1.0, 1.0),
        residual_tolerance=tolerance,
    )
    torch.testing.assert_close(interior_stationary, torch.zeros(2))
    assert interior_receipt["stationary"] is True

    active_stationary, active_receipt = metric_projected_direction(
        torch.ones(1),
        torch.tensor([-1.0]),
        torch.eye(1),
        box(-1.0, 1.0),
        residual_tolerance=tolerance,
    )
    torch.testing.assert_close(active_stationary, torch.zeros(1))
    assert active_receipt["stationary"] is True

    assert_rejected(torch.tensor([[1.0, 0.1], [0.0, 1.0]]))
    assert_rejected(torch.diag(torch.tensor([1.0, -1.0])))

    receipts = {
        "diagonal_box": diagonal_receipt,
        "coupled_box": coupled_receipt,
        "shared_spectral_and_log_bounds": shared_receipt,
        "direct_feasible": direct_receipt,
        "interior_stationary": interior_receipt,
        "active_bound_stationary": active_receipt,
    }
    assert max(item["kkt_residual_inf"] for item in receipts.values()) <= tolerance
    summary = {
        "schema": "joint-outer-metric-projection-validation-v1",
        "success": True,
        "scope": (
            "CPU analytic active-bound quadratics, the exact shared spectral/log "
            "projection, and the direct feasible BFGS path"
        ),
        "residual_tolerance": tolerance,
        "receipts": receipts,
    }
    archive_sources(cfg.output_dir)
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "outer/max_kkt_residual": max(
                item["kkt_residual_inf"] for item in receipts.values()
            ),
            "outer/max_iterations": max(
                item["iterations"] for item in receipts.values()
            ),
        }
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
    os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
    cherries.main(main, profile=ProfileJoint)
