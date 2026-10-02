"""Nonlinear tetrahedral muscle--fat fixture for the activation-prior study.

Uses the repository's production StableNeoHookeanActive energy, G=F Ainv,
and its implicit adjoint. Only the bottom is fixed; sides and top are free.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import warp as wp

from liblaf.apple.common import FIXED_MASK, FIXED_VALUE

ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "exp/2026/08/31/unreachable-pork-factor-study/src/20-run-pork-3d.py"
spec = importlib.util.spec_from_file_location("activation_prior_pork_base", SOURCE)
assert spec is not None and spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)


def configure():
    assert torch.cuda.is_available(), "CUDA required for the repository Warp adapter"
    torch.set_default_device("cuda")
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(4)
    wp.config.mode = "release"
    wp.init()
    logging.getLogger("liblaf.apple.forward._forward").setLevel(logging.WARNING)
    logging.getLogger("liblaf.apple.inverse._diff_forward").setLevel(logging.WARNING)


def make_mesh(nx=24, ny=10):
    mesh = base.structured_tets((nx, ny, nx), smoke=False)
    bottom = np.isclose(mesh.points[:, 1], 0.0)
    mesh.point_data[FIXED_MASK.vtk] = np.repeat(bottom[:, None], 3, axis=1)
    mesh.point_data[FIXED_VALUE.vtk] = np.zeros_like(mesh.points)
    mesh.point_data["FixedBoundary"] = bottom.astype(np.uint8)
    mesh.point_data["TargetSurface"] = mesh.point_data["TopSurface"].copy()
    return mesh


def array_hash(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype)).encode() + a.tobytes()).hexdigest()


class Physics:
    def __init__(self, nx=24, ny=10, rtol=1e-6, atol=1e-11):
        self.mesh = make_mesh(nx, ny)
        self.nx, self.ny = nx, ny
        self.points = np.asarray(self.mesh.points).copy()
        self.tets = self.mesh.cells.reshape(-1, 5)[:, 1:].copy()
        self.ids = np.flatnonzero(self.mesh.cell_data["Muscle"])
        self.top = np.flatnonzero(self.mesh.point_data["TopSurface"])
        self.centers = self.points[self.tets[self.ids]].mean(axis=1)
        self.dm = np.transpose(
            self.points[self.tets[:, 1:]] - self.points[self.tets[:, :1]], (0, 2, 1)
        )
        self.dm_inv = np.linalg.inv(self.dm)
        self.volumes_all = np.linalg.det(self.dm) / 6
        assert np.all(self.volumes_all > 0)
        self.volumes = self.volumes_all[self.ids]
        self.id_t = torch.as_tensor(self.ids)
        self.top_t = torch.as_tensor(self.top)
        # Exact lumped area weights for this structured triangulated top.
        self.weights = self.surface_weights()
        self.weights_t = torch.as_tensor(self.weights)
        self.forward = base.build_forward(self.mesh, "stable")
        self.forward.optimizer = self.forward.default_optimizer(
            max_steps=10000, rtol=rtol, atol=atol
        )
        from liblaf.peach.linalg.cupy import CupyCG

        from liblaf.apple.inverse import DifferentiableForward

        self.diff = DifferentiableForward(self.forward)
        self.diff.adjoint_solver = CupyCG(maxiter=20000, rtol=1e-7, atol=0.0)
        self.materials = self.forward.model.get_materials()
        base.set_output_material_metadata(self.mesh)
        self.solve_count = 0
        self.rtol, self.atol = rtol, atol

    def surface_weights(self):
        top_lookup = np.full(len(self.points), -1, dtype=int)
        top_lookup[self.top] = np.arange(len(self.top))
        weights = np.zeros(len(self.top))
        for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
            tri = self.tets[:, face]
            mask = np.all(top_lookup[tri] >= 0, axis=1)
            tri = tri[mask]
            p = self.points[tri]
            area = (
                np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
                / 2
            )
            np.add.at(weights, top_lookup[tri].ravel(), np.repeat(area / 3, 3))
        assert abs(weights.sum() - 1) < 1e-10
        return weights

    def solve(self, packed_active, seed=None):
        # The production custom autograd Function returns state.u itself. A
        # fresh detached buffer keeps the next continuation seed from editing
        # that previous Function's output view. Each backward has completed
        # before another solve is allowed by the experiment runner.
        self.forward.state.u = self.forward.state.u.detach().clone()
        if seed is not None:
            with torch.no_grad():
                self.forward.model.update(self.forward.state, torch.as_tensor(seed))
        self.materials["muscle"]["activation_inv"] = torch.zeros(
            (self.mesh.n_cells, 6)
        ).index_copy(0, self.id_t, packed_active)
        with contextlib.redirect_stdout(io.StringIO()):
            u = self.diff.forward(self.materials).clone()
        self.solve_count += 1
        solution = self.diff.last_solution
        assert solution is not None
        state = solution.state.convergence_state
        self.last_forward = {
            "success": bool(solution.success),
            "steps": int(state.step),
            "grad_norm": float(state.grad_norm.detach().cpu()),
            "result": str(solution.result),
        }
        if not solution.success:
            raise RuntimeError(f"Forward failed: {self.last_forward}")
        assert torch.isfinite(u).all()
        return u

    def check_adjoint(self):
        sol = self.diff.last_adjoint_solution
        assert sol is not None and sol.success, f"Adjoint failed: {sol}"
        return {"success": bool(sol.success), "result": str(sol.result)}

    def determinants(self, u, ainv):
        xd = self.points + u
        ds = np.transpose(xd[self.tets[:, 1:]] - xd[self.tets[:, :1]], (0, 2, 1))
        detf = np.linalg.det(ds @ self.dm_inv)
        deta = np.ones(len(self.tets))
        deta[self.ids] = np.linalg.det(ainv)
        detg = detf * deta
        return detf, deta, detg

    def save_mesh(self, path, u=None, ainv=None):
        mesh = self.mesh.copy()
        mesh.point_data["RestPosition"] = self.points
        if u is not None:
            mesh.point_data["Displacement"] = u
            mesh.points = self.points + u
        if ainv is not None:
            field = np.tile(np.eye(3), (len(self.tets), 1, 1))
            field[self.ids] = ainv
            mesh.cell_data["ActivationInverseMatrix"] = field.reshape(-1, 9)
            detf, deta, detg = self.determinants(u, ainv)
            mesh.cell_data["DetF"] = detf
            mesh.cell_data["DetAinv"] = deta
            mesh.cell_data["DetG"] = detg
        mesh.save(path)
