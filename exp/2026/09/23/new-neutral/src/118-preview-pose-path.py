"""Audit a chin rigid estimate and save bounded pose schedules without physics."""

# ruff: noqa: E402
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json
from joint_equilibrium import rigid_displacement
from mouthopen_pose_path import pose_waypoints


class Config(cherries.BaseConfig):
    estimate_path: Path = GROUP / "data/chin-rigid-pose-001/estimate.json"
    output: Path = GROUP / "data/chin-rigid-pose-001/proposed-continuation.json"


def main(cfg: Config):
    estimate = json.loads(cfg.estimate_path.read_text())
    protocol = json.loads(
        Path(estimate["sources"]["source_protocol"]["path"]).read_text()
    )
    pose = np.asarray(estimate["pose_rad_m"])
    pivot = np.asarray(estimate["mandible_pivot_m"])
    with np.load(estimate["sources"]["blendshapes"]["path"]) as data:
        neutral = data["new_neutral_points_m"]
        target = data["target_points_m"][
            list(data["expression_names"]).index("MouthOpen")
        ]
        triangles = data["skin_triangles"]
    dtype = torch.float64
    x = torch.as_tensor(neutral, dtype=dtype, device="cpu")
    native = (
        x
        + rigid_displacement(
            x, torch.as_tensor(pivot, dtype=dtype), torch.as_tensor(pose, dtype=dtype)
        )
    ).numpy()
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    numpy = (neutral - pivot) @ rotation.T + pivot + pose[3:]
    max_error = float(np.max(np.abs(native - numpy)))
    assert max_error < 1e-12
    patch = np.asarray(estimate["patch_local_ids"])
    tri = neutral[triangles]
    area = (
        np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        / 2
    )
    weight = np.zeros(len(neutral))
    np.add.at(weight, triangles.ravel(), np.repeat(area / 3, 3))
    rms = float(
        np.sqrt(
            np.average(
                np.sum((native[patch] - target[patch]) ** 2, axis=1),
                weights=weight[patch],
            )
        )
    )
    assert abs(rms - estimate["weighted_rms_after_m"]) < 1e-12
    current_path = GROUP / "data/inverse-mouthopen-003/initialization.pt"
    current = torch.load(current_path, map_location="cpu", weights_only=False)
    axis = np.asarray(protocol["parameterization"]["jaw_axis"])
    old_pose = np.r_[float(current["jaw"][0]) * np.pi / 18 * axis, np.zeros(3)]
    paths = {}
    for label, source in [
        ("current_4deg_checkpoint", old_pose),
        ("neutral", np.zeros(6)),
    ]:
        poses, receipt = pose_waypoints(source, pose)
        paths[label] = {
            **receipt,
            "pose_rad_m": poses.tolist(),
            "steps_count": len(poses) - 1,
        }
    write_json(
        cfg.output,
        {
            "schema": "chin-rigid-pose-preview-plan-v1",
            "status": "awaiting_user_pose_review",
            "physics_run": False,
            "estimate": {
                "path": str(cfg.estimate_path.resolve()),
                "sha256": sha256(cfg.estimate_path),
            },
            "current_checkpoint": {
                "path": str(current_path.resolve()),
                "sha256": sha256(current_path),
            },
            "independent_native_transform_max_error_m": max_error,
            "native_chin_rms_m": rms,
            "paths": paths,
            "limits_meaning": "Relative rotation angle <=1 degree; world displacement of shared pivot <=1 mm per increment. Limits do not bound every mandible vertex displacement.",
        },
    )
    cherries.log_output(cfg.output)
    cherries.log_metrics(
        {
            "native_transform_max_error_m": max_error,
            "native_chin_rms_mm": 1000 * rms,
            "planned_steps_from_checkpoint": paths["current_4deg_checkpoint"][
                "steps_count"
            ],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
