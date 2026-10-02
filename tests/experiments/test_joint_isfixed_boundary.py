"""Regression coverage for the authoritative FEM ``IsFixed`` boundary."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyvista as pv
import torch

from liblaf.apple.forward.dof_map import DofMapBuilder

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import joint_full_skull_contact  # noqa: E402
import joint_physics  # noqa: E402


class _Model:
    def __init__(self, dof_map: Any) -> None:
        self.dof_map = dof_map

    @staticmethod
    def get_materials() -> dict[str, dict[str, Any]]:
        return {}


class _Builder:
    def __init__(self) -> None:
        self.dof = DofMapBuilder()

    def add_vertices(self, mesh: pv.DataSet) -> None:
        self.dof.add_vertices(mesh)

    def add_fixed(self, mesh: pv.DataSet) -> None:
        self.dof.add_fixed(mesh)

    def add_potential(self, potential: Any) -> None:
        del potential

    def finalize(self) -> _Model:
        return _Model(self.dof.finalize())


class _Forward:
    def __init__(self, model: _Model) -> None:
        self.model = model


class _Equilibrium:
    def __init__(self, forward: _Forward, **kwargs: Any) -> None:
        del kwargs
        self.forward = forward


class _Potential:
    @classmethod
    def from_pyvista(cls, *args: Any, **kwargs: Any) -> _Potential:
        del args, kwargs
        return cls()


def _mesh(path: Path) -> tuple[Path, Path]:
    points = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    volume = pv.UnstructuredGrid(
        np.asarray([4, 0, 1, 2, 3]),
        np.asarray([pv.CellType.TETRA]),
        points,
    )
    # Nodes 2 and 3 retain mandible/cranium group membership but are deliberately
    # non-IsFixed.  They must remain free after construction.
    volume.point_data["IsFixed"] = np.asarray([True, True, False, False])
    for name in ("FatFraction", "AponeurosisFraction", "MuscleFraction"):
        volume.cell_data[name] = np.ones(1)
    volume.save(path / "volume.vtu")
    skin = pv.PolyData(points[:3], np.asarray([3, 0, 1, 2]))
    skin.point_data["GlobalPointId"] = np.asarray([0, 1, 2])
    skin.save(path / "skin.vtp")
    return path / "volume.vtu", path / "skin.vtp"


def _arrays() -> dict[str, np.ndarray]:
    return {
        "active_cell_ids": np.asarray([0]),
        "active_effective_volume_m3": np.asarray([1.0]),
        "historical_fixed_node_ids": np.asarray([0, 1]),
        "cranium_node_ids": np.asarray([3]),
        "mandible_node_ids": np.asarray([1, 2]),
        "observation_node_ids": np.asarray([0]),
        "observation_weight_normalized": np.asarray([1.0]),
        "target_displacement_m": np.zeros((1, 1, 3)),
        "mandible_pivot_m": np.zeros(3),
    }


def test_isfixed_controls_original_dofs_and_jaw_subset(
    monkeypatch: Any, tmp_path: Path
) -> None:
    """Group labels retain their role, but do not prescribe original FEM DOFs."""
    volume, skin = _mesh(tmp_path)
    monkeypatch.setattr(joint_physics, "ModelBuilder", _Builder)
    monkeypatch.setattr(joint_physics, "Forward", _Forward)
    monkeypatch.setattr(joint_physics, "Equilibrium", _Equilibrium)
    monkeypatch.setattr(joint_physics, "StableNeoHookeanStress", _Potential)
    monkeypatch.setattr(joint_physics, "StableNeoHookeanMembrane", _Potential)

    physics = joint_physics.JointPhysics(
        volume,
        skin,
        _arrays(),
        bulk_young_mpa={"fat": 0.01, "aponeurosis": 0.01, "muscle": 0.01},
        bulk_nu={"fat": 0.3, "aponeurosis": 0.3, "muscle": 0.3},
    )
    model = physics.runtime.forward.model
    fixed_vertices = np.unique(model.dof_map.fixed_indices.numpy() // 3)
    np.testing.assert_array_equal(fixed_vertices, np.asarray([0, 1]))
    np.testing.assert_array_equal(physics.jaw_t.numpy(), np.asarray([1]))

    pose = torch.tensor(
        [0.0, 0.0, 0.2, 0.03, -0.02, 0.01], dtype=physics.points_t.dtype
    )
    boundary = physics.boundary(pose)
    full = torch.empty(model.dof_map.n_full, dtype=boundary.dtype)
    full[model.dof_map.fixed_indices] = boundary
    full = full.reshape(-1, 3)
    torch.testing.assert_close(full[0], torch.zeros(3, dtype=full.dtype))
    assert torch.linalg.vector_norm(full[1]) > 0
    assert 2 not in fixed_vertices
    assert 3 not in fixed_vertices


def test_full_skull_appends_rigid_obstacles_without_fixing_extra_fem_nodes() -> None:
    """The full-skull extension preserves the original FEM map and fixes additions."""
    original = DofMapBuilder()
    mesh = pv.PolyData(np.zeros((4, 3)))
    mesh.point_data["FixedMask"] = np.repeat(
        np.asarray([[True], [True], [False], [False]]), 3, axis=1
    )
    mesh.point_data["FixedValue"] = np.zeros((4, 3))
    original.add_vertices(mesh)
    original.add_fixed(mesh)
    geometry = SimpleNamespace(fem_node_count=4, full_node_count=9)
    extended = joint_full_skull_contact.extend_dof_map(original.finalize(), geometry)

    fixed_vertices = np.unique(extended.fixed_indices.numpy() // 3)
    np.testing.assert_array_equal(fixed_vertices, np.asarray([0, 1, 4, 5, 6, 7, 8]))
    assert extended.n_free == 6

    geometry = SimpleNamespace(
        fem_node_count=4,
        full_node_count=9,
        mandible_pivot_m=np.zeros(3),
        cranium_points_m=np.asarray([[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]),
        mandible_points_m=np.asarray(
            [[0.0, 0.0, -1.0], [1.0, 0.0, -1.0], [0.0, 1.0, -1.0]]
        ),
        mandible_global_ids=np.asarray([6, 7, 8]),
    )
    adapter = joint_full_skull_contact.FullSkullContactAdapter(
        geometry=geometry, collision=None, contact_definition={}
    )
    pose = torch.tensor([0.0, 0.0, 0.1, 0.01, 0.0, 0.0], dtype=torch.float64)
    boundary = adapter.full_boundary_displacement(
        torch.zeros((4, 3), dtype=torch.float64), torch.tensor([1]), pose
    )
    torch.testing.assert_close(
        boundary[[0, 2, 3, 4, 5]], torch.zeros_like(boundary[:5])
    )
    assert torch.linalg.vector_norm(boundary[1]) > 0
    assert torch.all(torch.linalg.vector_norm(boundary[6:], dim=1) > 0)
