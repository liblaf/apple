"""Focused behavior checks for the fully fixed tetrahedron policy."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "exp/2026/09/23/new-neutral/src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

import mouthopen_tet_policy as policy  # noqa: E402


class _Potential:
    """Minimal CPU material holder used to exercise the replacement mapping."""

    MATERIAL_FIELDS: ClassVar = {"activation_inv": None, "dhdX": None, "dV": None}

    def __init__(self, *, cells: torch.Tensor, name: str) -> None:
        self.cells = cells
        self.name = name
        self.materials: object | None = None
        self.values: dict[str, torch.Tensor] = {}

    @staticmethod
    def material_struct() -> object:
        return object()

    def set_materials(self, values: dict[str, torch.Tensor]) -> None:
        self.values = values


class _Wp:
    vec4i = object()

    @staticmethod
    def to_torch(value: torch.Tensor) -> torch.Tensor:
        return value

    @staticmethod
    def from_torch(value: torch.Tensor, dtype: object) -> torch.Tensor:
        del dtype
        return value


def _physics() -> tuple[Any, dict[str, _Potential], dict[str, dict[str, torch.Tensor]]]:
    points = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    # Cell 0 is fully fixed. Cell 1 shares a fixed face but has one free node.
    tets = np.asarray([[0, 1, 2, 3], [0, 1, 2, 4]], dtype=np.int64)
    cells = torch.as_tensor(tets, dtype=torch.int32)
    registry = {
        name: _Potential(cells=cells.clone(), name=name) for name in policy.BULK_NAMES
    }
    registry["skin"] = _Potential(
        cells=torch.tensor([[0, 1, 2]], dtype=torch.int32), name="skin"
    )
    materials = {
        name: {
            "activation_inv": torch.arange(12, dtype=torch.float64).reshape(2, 6),
            "dhdX": torch.arange(48, dtype=torch.float64).reshape(2, 2, 4, 3),
            "dV": torch.ones((2, 2), dtype=torch.float64),
        }
        for name in policy.BULK_NAMES
    }
    materials["skin"] = {}
    model = SimpleNamespace(
        warp_model=SimpleNamespace(__wrapped__=SimpleNamespace(potentials=registry)),
        dof_map=SimpleNamespace(fixed_values=torch.zeros(12), n_points=len(points)),
        get_materials=lambda: materials,
    )
    base = SimpleNamespace(
        tets=tets,
        points=points,
        ids=np.asarray([0, 1], dtype=np.int64),
        mesh=SimpleNamespace(point_data={"IsFixed": np.asarray([1, 1, 1, 1, 0])}),
    )
    return (
        SimpleNamespace(
            base=base, runtime=SimpleNamespace(forward=SimpleNamespace(model=model))
        ),
        registry,
        materials,
    )


def test_exclusion_filters_bulk_arrays_and_remaps_only_local_active_indices(
    monkeypatch: Any,
) -> None:
    """Original cell IDs stay stable while the active material packing becomes local."""
    physics, registry, _ = _physics()
    monkeypatch.setattr(policy, "StableNeoHookeanActive", _Potential)
    monkeypatch.setattr(policy, "wp", _Wp)

    receipt = policy.exclude_fully_fixed_tetrahedra(physics)

    assert receipt["excluded_tetrahedra"] == 1
    assert receipt["retained_tetrahedra"] == 1
    np.testing.assert_array_equal(physics.base.ids, np.asarray([0, 1]))
    np.testing.assert_array_equal(
        physics.base.retained_active_cell_ids, np.asarray([1])
    )
    torch.testing.assert_close(physics.base.active_t, torch.tensor([0]))
    for name in policy.BULK_NAMES:
        torch.testing.assert_close(
            registry[name].cells, torch.tensor([[0, 1, 2, 4]], dtype=torch.int32)
        )
        torch.testing.assert_close(
            registry[name].values["activation_inv"],
            torch.arange(6, 12, dtype=torch.float64).reshape(1, 6),
        )


def test_geometry_metrics_uses_only_retained_cells_and_reports_inversion() -> None:
    """A removed fixed cell cannot contribute to determinant validity gates."""
    physics, _, _ = _physics()
    physics.base.__dict__["_mouthopen_retained_tetrahedron_ids"] = np.asarray(
        [1], dtype=np.int64
    )
    physics.base.__dict__["_mouthopen_excluded_tetrahedron_ids"] = np.asarray(
        [0], dtype=np.int64
    )
    u = torch.zeros((5, 3), dtype=torch.float64)
    u[4, 2] = -4.0

    metrics = policy.geometry_metrics(physics, u)

    assert set(metrics) == {
        "retained_tetrahedra",
        "excluded_tetrahedra",
        "inverted_tetrahedra",
        "inverted_fraction",
        "inverted_rest_volume_fraction",
        "detF_min",
        "detF_p001",
        "detF_max",
    }
    assert metrics["retained_tetrahedra"] == 1
    assert metrics["excluded_tetrahedra"] == 1
    assert metrics["inverted_tetrahedra"] == 1
    assert metrics["inverted_fraction"] == 1.0
    assert metrics["inverted_rest_volume_fraction"] == 1.0
    assert metrics["detF_min"] == -1.0
    assert metrics["detF_p001"] == -1.0
    assert metrics["detF_max"] == -1.0


def test_all_fixed_cell_has_no_free_gradient_or_hessian_contribution() -> None:
    """Removing an all-fixed element leaves the free Newton system unchanged."""
    fixed = torch.tensor([0, 1, 2, 3])
    free = torch.tensor([4, 5])
    all_fixed_hessian = torch.tensor(
        [
            [3.0, -1.0, 0.0, 0.0],
            [-1.0, 3.0, -1.0, 0.0],
            [0.0, -1.0, 3.0, -1.0],
            [0.0, 0.0, -1.0, 3.0],
        ]
    )
    hessian = torch.zeros((6, 6))
    hessian[fixed[:, None], fixed] = all_fixed_hessian
    gradient = hessian @ torch.arange(6.0)

    torch.testing.assert_close(
        gradient[free], torch.zeros_like(free, dtype=gradient.dtype)
    )
    torch.testing.assert_close(
        hessian[free[:, None], free], torch.zeros((2, 2), dtype=hessian.dtype)
    )
