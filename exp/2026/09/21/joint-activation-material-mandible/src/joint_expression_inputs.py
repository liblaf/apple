# ruff: noqa: EM101, EM102, TRY003
"""Hash-bound 36-expression inputs on the eye-inclusive loaded neutral."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from joint_common import sha256
from joint_data import array_sha256
from joint_expression_equilibrium import install_expression_runtime
from joint_frozen_neutral import FrozenNeutral
from joint_rigid_eye_contact import RigidEyeJointPhysics, build_eye_collision_physics


@dataclass(frozen=True)
class EyeExpressionInputs:
    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]
    directory: Path

    @classmethod
    def load(cls, directory: Path) -> EyeExpressionInputs:  # noqa: C901
        directory = directory.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        if (
            manifest["schema"] != "joint-eye-expression-inputs-v1"
            or not manifest["success"]
        ):
            raise ValueError("expression inputs are not admitted")
        for item in manifest["sources"].values():
            path = Path(item["path"])
            if sha256(path) != item["sha256"]:
                raise ValueError(f"changed source: {path}")
        artifact = directory / "state.npz"
        if sha256(artifact) != manifest["artifacts"]["state.npz"]["sha256"]:
            raise ValueError("expression state hash changed")
        with np.load(artifact, allow_pickle=False) as values:
            arrays = {key: values[key] for key in values.files}
        if set(arrays) != set(manifest["arrays"]):
            raise ValueError("array manifest differs from state")
        for key, value in arrays.items():
            item = manifest["arrays"][key]
            if item["shape"] != list(value.shape) or item["dtype"] != value.dtype.str:
                raise ValueError(f"array layout changed: {key}")
            if array_sha256(value) != item["sha256"]:
                raise ValueError(f"array hash changed: {key}")
        names = tuple(manifest["expression_names"])
        if arrays["target_displacement_m"].shape[0] != len(names):
            raise ValueError("target names and target array differ")
        if not np.array_equal(
            arrays["expression_displacement_m"], arrays["target_displacement_m"]
        ):
            raise ValueError(
                "expression displacement alias differs from objective target"
            )
        return cls(arrays, manifest, directory)

    @property
    def expression_names(self) -> tuple[str, ...]:
        return tuple(self.manifest["expression_names"])

    @property
    def neutral_displacement_m(self) -> np.ndarray:
        return self.arrays["neutral_displacement_m"]

    def total_displacement(self, increment: torch.Tensor) -> torch.Tensor:
        return increment + torch.as_tensor(
            self.neutral_displacement_m, device=increment.device, dtype=increment.dtype
        )

    def expression_residual(
        self, total: torch.Tensor, target_index: int
    ) -> torch.Tensor:
        ids = torch.as_tensor(self.arrays["observation_node_ids"], device=total.device)
        target = torch.as_tensor(
            self.arrays["target_total_displacement_m"][target_index],
            device=total.device,
            dtype=total.dtype,
        )
        return total[ids] - target

    def build_physics(
        self,
    ) -> tuple[RigidEyeJointPhysics, dict[str, dict[str, torch.Tensor]]]:
        """Build fixed parent materials/contact, then install this target bundle."""
        parent = FrozenNeutral.load(
            Path(self.manifest["parent_frozen_neutral"]["directory"])
        )
        eyes = Path(self.manifest["sources"]["eyes_manifest"]["path"]).parent
        physics, baseline = build_eye_collision_physics(parent, eyes)
        install_expression_runtime(physics)
        base = physics.base
        base.arrays = {key: value.copy() for key, value in self.arrays.items()}
        base.observation_t = torch.as_tensor(
            base.arrays["observation_node_ids"], dtype=torch.long
        )
        base.weights_t = torch.as_tensor(base.arrays["observation_weight_normalized"])
        # Runtime metrics operate on total displacement from the constitutive
        # reference. The objective keeps the original expression increment
        # separately in expression_displacement_m for its scale.
        base.arrays["target_displacement_m"] = base.arrays[
            "target_total_displacement_m"
        ].copy()
        base.targets_t = torch.as_tensor(base.arrays["target_displacement_m"])
        mass = base.arrays["active_effective_volume_m3"]
        base.muscle_mass = mass / mass.sum()
        base.muscle_mass_t = torch.as_tensor(base.muscle_mass)
        return physics, baseline
