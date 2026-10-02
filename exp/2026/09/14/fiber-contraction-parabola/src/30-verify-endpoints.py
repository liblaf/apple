"""Check endpoint diagnostics for the four guarded inverse runs.

This script never invokes the inverse optimizer. It only reloads the final
checkpoints once all four cases have written their summaries.
"""

# ruff: noqa: E402

from __future__ import annotations

import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.linalg as la
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

LOG = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
INPUT = GROUP / "data/10-comparison-final"

spec = importlib.util.spec_from_file_location(
    "endpoint_physics", GROUP / "src/physics2d.py"
)
assert spec is not None
assert spec.loader is not None
ph = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ph
spec.loader.exec_module(ph)


class ProfileRecord(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("30-endpoint-checks-final")
    psd_relative_tolerance: float = 1.0e-10


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def boundary_nodes(mesh: Any) -> np.ndarray:
    width = mesh.nx + 1
    bottom = np.arange(width)
    right = np.arange(2 * width - 1, (mesh.ny + 1) * width, width)
    top = np.arange((mesh.ny + 1) * width - 2, mesh.ny * width - 1, -1)
    left = np.arange((mesh.ny - 1) * width, 0, -width)
    nodes = np.concatenate((bottom, right, top, left))
    assert len(nodes) == 2 * (mesh.nx + mesh.ny)
    assert len(np.unique(nodes)) == len(nodes)
    return nodes


def shoelace(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return float(
        0.5 * np.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])
    )


def target_area_ratio(mesh: Any, height: float) -> float:
    target = mesh.p.copy()
    top = np.isclose(target[:, 1], 0.1)
    target[top, 1] += height * 4.0 * target[top, 0] * (1.0 - target[top, 0])
    return shoelace(target[boundary_nodes(mesh)]) / shoelace(
        mesh.p[boundary_nodes(mesh)]
    )


def write_vtu(
    path: Path,
    mesh: Any,
    u: np.ndarray,
    B: np.ndarray,
    J: np.ndarray,
    controls: np.ndarray,
    mode: str,
) -> None:
    cells = np.column_stack((np.full(len(mesh.tri), 3), mesh.tri)).ravel()
    points = np.column_stack((mesh.p + u, np.zeros(len(u))))
    grid = pv.UnstructuredGrid(
        cells, np.full(len(mesh.tri), pv.CellType.TRIANGLE), points
    )
    grid.point_data["Displacement"] = np.column_stack((u, np.zeros(len(u))))
    grid.cell_data["J"] = J
    grid.cell_data["MuscleMask"] = mesh.muscle.astype(np.uint8)
    grid.cell_data["Lambda"] = mesh.lam
    grid.cell_data["Mu"] = mesh.mu
    grid.cell_data["ActivationInv"] = np.column_stack(
        (B[:, 0, 0] - 1.0, B[:, 1, 1] - 1.0, B[:, 0, 1])
    )
    extent = np.zeros((len(mesh.tri), 3))
    if mode == "x_contraction":
        extent[mesh.muscle, 0] = controls
    else:
        extent[mesh.muscle] = controls.reshape(-1, 3)
    grid.cell_data["Control"] = extent
    grid.save(path)


def verify_case(
    cfg: Config,
    mesh: Any,
    folder: Path,
    height: float,
    mode: str,
    output: Path,
    forward_tolerance: float,
    inverse_tolerance: float,
) -> dict[str, Any]:
    checkpoint = np.load(folder / "checkpoint.npz")
    controls, u = checkpoint["controls"], checkpoint["u"]
    summary = json.loads((folder / "summary.json").read_text())
    B = ph.matrices(mesh, controls, mode)
    energy, residual, H, J = ph.assemble(mesh, u, B)
    assert H is not None
    U = ph.unpack(mesh, u)
    fixed = mesh.lookup.reshape(-1, 2) < 0
    fixed_error = float(np.max(np.abs(U[fixed])))
    domain_area = float(mesh.area @ J)
    boundary_area = shoelace((mesh.p + U)[boundary_nodes(mesh)])
    symmetric_H = ((H + H.T) * 0.5).tocsc()
    # The 1,980-free-DoF endpoint systems are small enough for a reliable
    # dense symmetric solve; ARPACK did not converge on their near-singular
    # spectra.
    dense_H = symmetric_H.toarray()
    LOG.info("Computing dense smallest Hessian eigenpair for %s", folder.name)
    eigenvalues, eigenvectors = la.eigh(
        dense_H, subset_by_index=[0, 0], driver="evr", check_finite=False
    )
    eigmin = float(eigenvalues[0])
    eigenvector = eigenvectors[:, 0]
    hessian_inf_norm = max(float(np.linalg.norm(dense_H, ord=np.inf)), 1.0)
    eigen_relative_residual = float(
        np.linalg.norm(H @ eigenvector - eigmin * eigenvector) / hessian_inf_norm
    )
    asymmetry = (H - H.T).data
    relative_symmetry_error = float(
        np.max(np.abs(asymmetry), initial=0.0) / hessian_inf_norm
    )
    hessian_scale = max(float(np.max(np.abs(symmetric_H.diagonal()))), 1.0)
    psd_tolerance = cfg.psd_relative_tolerance * hessian_scale
    extent = {
        "minimum": float(controls.min()),
        "maximum": float(controls.max()),
        "maximum_absolute": float(np.max(np.abs(controls))),
        "rms": float(np.sqrt(np.mean(controls**2))),
    }
    record = {
        "case": folder.name,
        "height": height,
        "mode": mode,
        "energy": energy,
        "residual_inf": float(np.linalg.norm(residual, np.inf)),
        "forward_stationarity_evidence": {
            "tolerance": forward_tolerance,
            "passes": bool(np.linalg.norm(residual, np.inf) <= forward_tolerance),
        },
        "inverse_evidence": {
            "optimizer_success": bool(summary["optimizer_success"]),
            "projected_gradient_inf": summary["final"]["projected_gradient_inf"],
            "projected_gradient_tolerance": inverse_tolerance,
            "projected_gradient_passes": bool(
                summary["final"]["projected_gradient_inf"] <= inverse_tolerance
            ),
        },
        "psd_evidence": {
            "smallest_algebraic_eigenvalue": eigmin,
            "eigen_relative_residual": eigen_relative_residual,
            "relative_symmetry_error": relative_symmetry_error,
            "tolerance": psd_tolerance,
            "passes": bool(eigmin >= -psd_tolerance),
        },
        "J": {
            "minimum": float(J.min()),
            "maximum": float(J.max()),
            "fraction_below_0_1": float(np.mean(J < 0.1)),
            "fraction_above_2": float(np.mean(J > 2.0)),
        },
        "boundary": {
            "fixed_displacement_inf": fixed_error,
            "fixed_boundary_passes": bool(fixed_error <= 1.0e-14),
            "area_from_sum_area_J": domain_area,
            "area_from_deformed_boundary": boundary_area,
            "area_difference": domain_area - boundary_area,
        },
        "target_area_ratio_discrete": target_area_ratio(mesh, height),
        "control_extent": extent,
    }
    write_vtu(output / f"{folder.name}.vtu", mesh, U, B, J, controls, mode)
    return record


def main(cfg: Config) -> None:
    protocol = json.loads((INPUT / "protocol.json").read_text())
    mesh = ph.build_mesh(*protocol["mesh"])
    reference_area = shoelace(mesh.p[boundary_nodes(mesh)])
    assert np.isclose(reference_area, mesh.area.sum(), rtol=0.0, atol=1.0e-14)
    expected = [
        (height, mode)
        for height in map(float, protocol["config"]["heights"].split(","))
        for mode in protocol["config"]["modes"].split(",")
    ]
    missing = [
        f"h{round(height * 1000):03d}-{mode}"
        for height, mode in expected
        if not (
            INPUT / f"h{round(height * 1000):03d}-{mode}" / "summary.json"
        ).is_file()
    ]
    if missing:
        message = f"Endpoint checks require all final summaries; missing {missing}"
        raise FileNotFoundError(message)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    records = [
        verify_case(
            cfg,
            mesh,
            INPUT / f"h{round(height * 1000):03d}-{mode}",
            height,
            mode,
            output,
            protocol["config"]["forward_tolerance"],
            protocol["config"]["gradient_tolerance"],
        )
        for height, mode in expected
    ]
    write_json(output / "endpoint-checks.json", records)
    cherries.log_metrics(
        {f"{item['case']}/residual_inf": item["residual_inf"] for item in records}
    )
    LOG.info("Wrote endpoint checks for %d cases to %s", len(records), output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileRecord)
