"""Validate the scalar hinge map and its derivatives without a GPU solve."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_equilibrium import rigid_displacement

from liblaf import cherries


class Config(cherries.BaseConfig):
    inputs_dir: Path = GROUP / "data/expression-inputs-002"
    output_dir: Path = GROUP / "data/mandible-hinge-validation-001"


def main(cfg: Config) -> None:
    source = GROUP / "src/93-fit-expressions.py"
    module = runpy.run_path(str(source))
    hinge_pose = module["hinge_pose"]
    scale = module["HINGE_SCALE_RAD"]
    torch.set_default_dtype(torch.float64)
    with np.load(cfg.inputs_dir / "state.npz") as arrays:
        axis = torch.tensor(arrays["mandible_frame_world"][:, 0])
        pivot = torch.tensor(arrays["mandible_pivot_m"])
        points = torch.tensor(arrays["neutral_points_m"][arrays["mandible_node_ids"]])
    torch.testing.assert_close(torch.linalg.vector_norm(axis), torch.tensor(1.0))
    # Positive angle lowers the anterior-most support point in world +Y-up.
    anterior = points[points[:, 2].argmax()]
    assert torch.linalg.cross(axis, anterior - pivot)[1] < 0
    selected = torch.stack((points[0], anterior, points[-1]))
    on_axis = torch.stack((pivot - 0.05 * axis, pivot, pivot + 0.05 * axis))
    rows = []
    for angle_deg in (0.0, 5.0, 20.0, 40.0):
        jaw = torch.tensor([angle_deg / 10], requires_grad=True)
        pose = hinge_pose(jaw, axis)
        assert pose.shape == (6,)
        assert torch.count_nonzero(pose[3:]) == 0
        torch.testing.assert_close(pose[:3], jaw[0] * scale * axis)
        torch.testing.assert_close(
            rigid_displacement(on_axis, pivot, pose),
            torch.zeros_like(on_axis),
            atol=1e-14,
            rtol=0,
        )
        moved = selected + rigid_displacement(selected, pivot, pose)
        torch.testing.assert_close(
            torch.pdist(moved), torch.pdist(selected), atol=1e-14, rtol=1e-12
        )
        derivative = torch.autograd.functional.jacobian(
            lambda value: rigid_displacement(selected, pivot, hinge_pose(value, axis)),
            jaw,
        )[..., 0]
        epsilon = 1e-5
        plus = rigid_displacement(
            selected, pivot, hinge_pose(jaw.detach() + epsilon, axis)
        )
        minus = rigid_displacement(
            selected, pivot, hinge_pose(jaw.detach() - epsilon, axis)
        )
        numerical = (plus - minus) / (2 * epsilon)
        torch.testing.assert_close(derivative, numerical, atol=5e-11, rtol=1e-7)
        # The old prior restricted to the hinge equals the new scalar prior.
        old_normalized = pose / torch.tensor([scale] * 3 + [0.005] * 3)
        torch.testing.assert_close(
            old_normalized.square().mean(), jaw.square().sum() / 6
        )
        rows.append(
            {
                "angle_deg": angle_deg,
                "derivative_max_abs_error": float((derivative - numerical).abs().max()),
            }
        )
    try:
        hinge_pose(torch.zeros(6), axis)
    except AssertionError:
        pass
    else:
        message = "Legacy six-coordinate jaw must be rejected"
        raise AssertionError(message)
    # At the opening-only lower bound a gradient towards closing is stationary.
    stationarity = module["stationarity"]
    args = (torch.zeros((1, 6)), torch.zeros(1), torch.zeros((1, 6)))
    assert stationarity(*args, torch.ones(1), torch.ones(1))["jaw_mapping_inf"] == 0
    assert stationarity(*args, -torch.ones(1), torch.ones(1))["jaw_mapping_inf"] == 1
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    receipt = {
        "success": True,
        "scope": "scalar rigid map, finite-angle geometry, autograd finite differences, prior equivalence, projected-gradient bounds; no new full-equilibrium derivative claim",
        "runner_sha256": sha256(source),
        "input_manifest_sha256": sha256(cfg.inputs_dir / "manifest.json"),
        "axis_world": axis.tolist(),
        "pivot_m": pivot.tolist(),
        "angle_bounds_deg": [0, 40],
        "translation_m": [0, 0, 0],
        "positive_angle_lowers_anterior_support": True,
        "legacy_six_coordinate_input_rejected": True,
        "checks": rows,
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir / "summary.json")


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
