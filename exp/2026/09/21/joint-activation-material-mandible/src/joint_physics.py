"""Full anatomical mesh assembly for the joint stress/material/jaw experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pyvista as pv
import torch
from joint_equilibrium import Equilibrium, rigid_displacement
from joint_materials import StableNeoHookeanMembrane, StableNeoHookeanStress

from liblaf.apple.common import FRACTION, LAMBDA, MU
from liblaf.apple.forward import Forward, ModelBuilder

BULK_NAMES = ("fat", "aponeurosis", "muscle")


def moduli(young: float, nu: float) -> tuple[float, float]:
    mu = young / (2 * (1 + nu))
    return mu, young * nu / ((1 + nu) * (1 - 2 * nu)) + mu


class JointPhysics:
    def __init__(  # noqa: PLR0915
        self,
        volume_path: Path,
        skin_path: Path,
        arrays: dict[str, np.ndarray],
        *,
        bulk_young_mpa: dict[str, float],
        bulk_nu: dict[str, float],
        skin_young_mpa: float = 0.2,
        skin_nu: float = 0.46,
        thickness_m: float = 0.001,
        rtol: float = 5e-4,
        atol: float = 1e-10,
        adjoint_rtol: float = 5e-4,
        max_steps: int = 5000,
        contact_config: dict | None = None,
        forward_method: Literal["pncg", "newton_cg"] = "pncg",
        newton_linear_rtol: float = 1e-3,
        newton_max_steps: int = 12,
    ) -> None:
        self.arrays = arrays
        self.mesh = pv.read(volume_path)
        self.skin = pv.read(skin_path)
        self.points = np.asarray(self.mesh.points).copy()
        self.tets = np.asarray(self.mesh.cells).reshape(-1, 5)[:, 1:].copy()
        self.ids = arrays["active_cell_ids"]
        self.skin_ids = np.asarray(
            self.skin.point_data["GlobalPointId"], dtype=np.int64
        ).copy()
        assert np.allclose(
            self.skin.points, self.points[self.skin_ids], atol=1e-12, rtol=0
        )
        self.skin_tri = self.skin_ids[np.asarray(self.skin.faces).reshape(-1, 4)[:, 1:]]
        xyz = self.points[self.skin_tri]
        self.skin_area = (
            np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            / 2
        )
        assert np.all(self.skin_area > 0)
        dm = np.transpose(
            self.points[self.tets[:, 1:]] - self.points[self.tets[:, :1]], (0, 2, 1)
        )
        self.dm_inv = np.linalg.inv(dm)
        self.volumes = np.linalg.det(dm) / 6
        assert np.all(self.volumes > 0)
        self.muscle_mass = arrays["active_effective_volume_m3"]
        self.muscle_mass = self.muscle_mass / self.muscle_mass.sum()
        # Anatomy labels select the motion of prescribed nodes; only IsFixed
        # grants a node a prescribed degree of freedom.
        is_fixed = np.asarray(self.mesh.point_data["IsFixed"])
        assert is_fixed.shape == (len(self.points),)
        assert np.isin(is_fixed, (0, 1)).all()
        fixed = np.flatnonzero(is_fixed).astype(np.int64)
        assert np.array_equal(fixed, np.sort(arrays["historical_fixed_node_ids"])), (
            "Historical fixed IDs must match the source IsFixed label"
        )
        assert not np.intersect1d(
            arrays["cranium_node_ids"], arrays["mandible_node_ids"]
        ).size
        mask = np.zeros((len(self.points), 3), dtype=bool)
        mask[fixed] = True
        self.mesh.point_data["FixedMask"] = mask
        self.mesh.point_data["FixedValue"] = np.zeros_like(self.points)
        builder = ModelBuilder()
        builder.add_vertices(self.mesh)
        builder.add_fixed(self.mesh)
        self.bulk_mu = {}
        self.bulk_lambda = {}
        for name in BULK_NAMES:
            mu, la = moduli(bulk_young_mpa[name], bulk_nu[name])
            self.bulk_mu[name], self.bulk_lambda[name] = mu, la
            self.mesh.cell_data[MU.vtk] = np.full(self.mesh.n_cells, mu)
            self.mesh.cell_data[LAMBDA.vtk] = np.full(self.mesh.n_cells, la)
            self.mesh.cell_data[FRACTION.vtk] = np.asarray(
                self.mesh.cell_data[name.title() + "Fraction"]
            ).copy()
            builder.add_potential(
                StableNeoHookeanStress.from_pyvista(self.mesh, name=name)
            )
        skin_mu, skin_la = moduli(skin_young_mpa, skin_nu)
        self.skin.cell_data[MU.vtk] = np.full(self.skin.n_cells, skin_mu)
        self.skin.cell_data[LAMBDA.vtk] = np.full(self.skin.n_cells, skin_la)
        self.skin.cell_data[FRACTION.vtk] = np.ones(self.skin.n_cells)
        builder.add_potential(
            StableNeoHookeanMembrane.from_pyvista(
                self.skin, name="skin", thickness=thickness_m
            )
        )
        model = builder.finalize()
        self.contact_definition = None
        if contact_config is not None:
            from joint_contact import build_owned_contact

            model.collision, self.contact_definition = build_owned_contact(
                self.mesh, fixed, contact_config
            )
        self.runtime = Equilibrium(
            Forward(model),
            rtol=rtol,
            atol=atol,
            adjoint_rtol=adjoint_rtol,
            max_steps=max_steps,
            forward_method=forward_method,
            newton_linear_rtol=newton_linear_rtol,
            newton_max_steps=newton_max_steps,
        )
        self.base = self.runtime.forward.model.get_materials()
        self.points_t = torch.as_tensor(self.points)
        self.active_t = torch.as_tensor(self.ids, dtype=torch.int64)
        self.jaw_t = torch.as_tensor(
            np.intersect1d(fixed, arrays["mandible_node_ids"]), dtype=torch.int64
        )
        self.observation_t = torch.as_tensor(
            arrays["observation_node_ids"], dtype=torch.int64
        )
        self.weights_t = torch.as_tensor(arrays["observation_weight_normalized"])
        self.targets_t = torch.as_tensor(arrays["target_displacement_m"])
        self.pivot_t = torch.as_tensor(arrays["mandible_pivot_m"])
        self.muscle_tets_t = torch.as_tensor(self.tets[self.ids], dtype=torch.int64)
        self.muscle_mass_t = torch.as_tensor(self.muscle_mass)

    def materials(
        self,
        bulk_stress: torch.Tensor,
        skin_resultant_n_m: torch.Tensor,
        skin_multiplier: torch.Tensor,
        active_stress: torch.Tensor | None = None,
    ):
        values = {name: dict(fields) for name, fields in self.base.items()}
        for index, name in enumerate(BULK_NAMES):
            total = bulk_stress[index].expand(self.mesh.n_cells, 3, 3).contiguous()
            if name == "muscle" and active_stress is not None:
                total = total.index_add(0, self.active_t, active_stress)
            values[name]["active_stress"] = total
        count = self.skin.n_cells
        values["skin"]["baseline_stress"] = (
            (skin_resultant_n_m * 1e-6).expand(count, 2, 2).contiguous()
        )
        values["skin"]["mu"] = self.base["skin"]["mu"] * skin_multiplier
        values["skin"]["lmbda"] = self.base["skin"]["lmbda"] * skin_multiplier
        return values

    def boundary(self, pose: torch.Tensor):
        values = torch.zeros_like(self.points_t).index_copy(
            0,
            self.jaw_t,
            rigid_displacement(self.points_t[self.jaw_t], self.pivot_t, pose),
        )
        return values.flatten()[self.runtime.forward.model.dof_map.fixed_indices]

    def solve(
        self,
        bulk_stress: torch.Tensor,
        skin_resultant_n_m: torch.Tensor,
        skin_multiplier: torch.Tensor,
        active_stress: torch.Tensor | None,
        pose: torch.Tensor,
        seed: torch.Tensor,
        *,
        key: str,
    ):
        return self.runtime.solve(
            self.materials(
                bulk_stress, skin_resultant_n_m, skin_multiplier, active_stress
            ),
            self.boundary(pose),
            seed,
            key=key,
        )

    def neutral_loss(self, u: torch.Tensor):
        surface = (self.weights_t[:, None] * u[self.observation_t].square()).sum() * 1e6
        center = u[self.muscle_tets_t].mean(dim=1)
        muscle = (self.muscle_mass_t[:, None] * center.square()).sum() * 1e6
        return surface / 0.25**2 + muscle / 0.5**2

    def fit_loss(self, u: torch.Tensor, target_index: int):
        target = self.targets_t[target_index]
        squared_scale = (self.weights_t[:, None] * target.square()).sum()
        assert squared_scale > 0
        return (
            self.weights_t[:, None] * (u[self.observation_t] - target).square()
        ).sum() / squared_scale

    def metrics(self, u: torch.Tensor, *, target_index: int | None = None):
        displacement = u.detach().cpu().numpy()
        x = self.points + displacement
        ds = np.transpose(x[self.tets[:, 1:]] - x[self.tets[:, :1]], (0, 2, 1))
        F = ds @ self.dm_inv
        J = np.linalg.det(F)
        w = self.arrays["observation_weight_normalized"]
        obs = self.arrays["observation_node_ids"]
        muscle_centers = displacement[self.tets[self.ids]].mean(axis=1)
        xyz = x[self.skin_tri]
        area = (
            np.linalg.norm(
                np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1
            )
            / 2
        )
        result = {
            "detF_min": float(J.min()),
            "detF_p001": float(np.quantile(J, 0.001)),
            "detF_max": float(J.max()),
            "inverted_tetrahedra": int(np.count_nonzero(J <= 0)),
            "skin_area_ratio_min": float(np.min(area / self.skin_area)),
            "surface_motion_rms_mm": float(
                np.sqrt(np.sum(w[:, None] * displacement[obs] ** 2)) * 1000
            ),
            "muscle_centroid_motion_rms_mm": float(
                np.sqrt(np.sum(self.muscle_mass[:, None] * muscle_centers**2)) * 1000
            ),
            "muscle_deformation_rms": float(
                np.sqrt(
                    np.sum(
                        self.muscle_mass[:, None, None] * (F[self.ids] - np.eye(3)) ** 2
                    )
                )
            ),
        }
        if target_index is not None:
            residual = (
                displacement[obs] - self.arrays["target_displacement_m"][target_index]
            )
            result["area_fit_rms_mm"] = float(
                np.sqrt(np.sum(w[:, None] * residual**2)) * 1000
            )
            result["raw_fit_rms_mm"] = float(
                np.sqrt(np.mean(np.sum(residual**2, axis=1))) * 1000
            )
        return result
