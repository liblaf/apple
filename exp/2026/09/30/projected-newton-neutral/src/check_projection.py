"""Focused checks for projected_hessian: projection accuracy and GPU topology."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyvista as pv
import torch

ROOT = Path(__file__).resolve().parents[6]
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(ROOT / "exp/2026/09/22/solver-performance/src"),
    str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"),
    str(ROOT / "exp/2026/09/22/neutral-newton/src"),
]
import assembled_fem_hvp
import projected_hessian as ph

MESH = ROOT / "exp/2026/09/23/new-neutral/data/forward-isfixed-001"


def check_projection() -> None:
    torch.manual_seed(0)
    b = 8192
    basis = torch.randn(b, 12, 6, dtype=torch.float64, device="cuda")
    psd = basis @ basis.mT  # rank-6, like an element Hessian's rigid null space
    noise = torch.randn(b, 12, 12, dtype=torch.float64, device="cuda")
    scales = torch.logspace(-3, 3, b, device="cuda", dtype=torch.float64)
    indefinite = (noise + noise.mT) * scales[:, None, None]
    pick = torch.rand(b, device="cuda") < 0.9
    local = torch.where(pick[:, None, None], indefinite, psd)
    none, changed = ph.project_local(local, "none")
    assert changed == 0 and torch.equal(none, 0.5 * (local + local.mT))
    exact = torch.linalg.eigh(local)
    reference = (
        exact.eigenvectors * exact.eigenvalues.clamp_min(0)[:, None, :]
    ) @ exact.eigenvectors.mT
    scale = local.diagonal(dim1=1, dim2=2).abs().amax(1)
    for dtype, limit in ((torch.float64, 1e-12), (torch.float32, 1e-5)):
        ph.STATS["eigh_dtype"] = dtype
        projected, changed = ph.project_local(local, "clamp")
        assert changed == int(pick.sum()), (changed, int(pick.sum()))
        assert torch.equal(projected[~pick], psd[~pick])
        error = float(
            (
                (projected - reference).flatten(1).norm(dim=1)
                / reference.flatten(1).norm(dim=1)
            ).max()
        )
        min_eig = float((torch.linalg.eigvalsh(projected).amin(1) / scale).min())
        assert error < limit and min_eig > -limit, (dtype, error, min_eig)
        print(
            f"projection {dtype}: max relative error {error:.2e}, min scaled eigenvalue {min_eig:.2e}"
        )


class _Model:
    def __init__(self, n_points: int) -> None:
        self.n_points = n_points


def check_topology() -> None:
    volume = pv.read(MESH / "rebased-reference-volume.vtu")
    skin = pv.read(MESH / "rebased-reference-skin.vtp")
    tets = volume.cells_dict[pv.CellType.TETRA].astype(np.int64)
    tris = skin.point_data["GlobalPointId"][skin.faces.reshape(-1, 4)[:, 1:]].astype(
        np.int64
    )
    potentials = {
        "bulk": SimpleNamespace(cells=torch.as_tensor(tets)),
        "skin": SimpleNamespace(cells=torch.as_tensor(tris)),
    }
    timings = {}
    results = {}
    for name, function in (
        ("numpy", assembled_fem_hvp._topology),
        ("gpu", ph._topology_gpu),
    ):
        model = _Model(volume.n_points)
        started = time.perf_counter()
        results[name], hit = function(model, potentials)
        timings[name] = time.perf_counter() - started
        assert not hit
    a, b = results["numpy"], results["gpu"]
    for field in ("keys", "crow", "col"):
        np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    for pa, pb in zip(a.potentials, b.potentials, strict=True):
        np.testing.assert_array_equal(pa.cells, pb.cells)
        np.testing.assert_array_equal(pa.slots, pb.slots)
        assert (pa.name, pa.kind, pa.vertices_per_cell) == (
            pb.name,
            pb.kind,
            pb.vertices_per_cell,
        )
    print(
        f"topology identical: numpy {timings['numpy']:.2f} s, gpu {timings['gpu']:.2f} s"
    )


if __name__ == "__main__":
    check_projection()
    check_topology()
