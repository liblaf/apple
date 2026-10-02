"""Independent endpoint checks for the completed pure-Adam comparison."""

# ruff: noqa: E402

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import physics2d as ph
import pydantic_settings as ps
import scipy.linalg as la
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
INPUT = GROUP / "data/70-adam-comparison"


class ProfileAdamEndpointCheck(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("90-adam-endpoint-checks")
    residual_tolerance: float = 1e-10
    fixed_tolerance: float = 1e-14


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def boundary_nodes(mesh: Any) -> np.ndarray:
    width = mesh.nx + 1
    bottom = np.arange(width)
    right = np.arange(2 * width - 1, (mesh.ny + 1) * width, width)
    top = np.arange((mesh.ny + 1) * width - 2, mesh.ny * width - 1, -1)
    left = np.arange((mesh.ny - 1) * width, 0, -width)
    nodes = np.concatenate((bottom, right, top, left))
    assert len(nodes) == len(np.unique(nodes)) == 2 * (mesh.nx + mesh.ny)
    return nodes


def shoelace(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return float(
        0.5 * np.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])
    )


def pack(mesh: Any, full_u: np.ndarray) -> np.ndarray:
    result = np.empty(mesh.nfree)
    lookup = mesh.lookup
    mask = lookup >= 0
    result[lookup[mask]] = full_u.reshape(-1)[mask]
    return result


def source_hashes(protocol: dict[str, Any]) -> dict[str, bool]:
    checked = {}
    for key, expected in protocol["source_sha256"].items():
        snapshot = INPUT / "source" / Path(key).name
        assert snapshot.is_file(), snapshot
        actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert actual == expected, (key, actual, expected)
        checked[key] = actual == expected
    physics_key = str(Path(ph.__file__).resolve().relative_to(ph.ROOT))
    assert physics_key in protocol["source_sha256"]
    live = hashlib.sha256(Path(ph.__file__).read_bytes()).hexdigest()
    assert live == protocol["source_sha256"][physics_key]
    checked["live_physics2d_matches_run"] = True
    return checked


def verify_case(
    cfg: Config, mesh: Any, folder: Path, height: float, mode: str
) -> dict[str, Any]:
    history = np.load(folder / "history.npz", allow_pickle=False)
    checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
    summary = json.loads((folder / "summary.json").read_text())
    rows = list(csv.DictReader((folder / "trace.csv").open()))
    controls, full_u = checkpoint["controls"], checkpoint["u"]
    assert full_u.shape == mesh.p.shape
    assert np.array_equal(controls, history["controls"][-1])
    assert np.array_equal(full_u, history["u"][-1])
    assert len(rows) == len(history["u"]) == len(history["controls"])
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    assert int(summary["accepted_iterations"]) + 1 == len(rows)
    packed_u = pack(mesh, full_u)
    B = ph.matrices(mesh, controls, mode)
    muscle_B = B[mesh.muscle]
    energy, residual, hessian, J = ph.assemble(mesh, packed_u, B)
    assert hessian is not None
    fixed = mesh.lookup.reshape(-1, 2) < 0
    fixed_error = float(np.max(np.abs(full_u[fixed])))
    dense = ((hessian + hessian.T) * 0.5).toarray()
    LOG.info("Dense smallest Hessian eigenpair: %s", folder.name)
    values, vectors = la.eigh(
        dense, subset_by_index=[0, 0], driver="evr", check_finite=False
    )
    eigmin, vector = float(values[0]), vectors[:, 0]
    hnorm = max(float(np.linalg.norm(dense, np.inf)), 1.0)
    eigen_residual = float(np.linalg.norm(hessian @ vector - eigmin * vector) / hnorm)
    boundary_area = shoelace((mesh.p + full_u)[boundary_nodes(mesh)])
    area_from_J = float(mesh.area @ J)
    result = {
        "case": folder.name,
        "height": height,
        "mode": mode,
        "history": {
            "states": len(rows),
            "last_step": int(rows[-1]["step"]),
            "passes": True,
        },
        "residual_inf": float(np.linalg.norm(residual, np.inf)),
        "forward_tolerance": cfg.residual_tolerance,
        "forward_passes": bool(
            np.linalg.norm(residual, np.inf) <= cfg.residual_tolerance
        ),
        "fixed_displacement_inf": fixed_error,
        "fixed_boundary_passes": bool(fixed_error <= cfg.fixed_tolerance),
        "area_from_J": area_from_J,
        "area_from_boundary": boundary_area,
        "area_difference": area_from_J - boundary_area,
        "J_minimum": float(J.min()),
        "J_maximum": float(J.max()),
        "smallest_algebraic_hessian_eigenvalue": eigmin,
        "hessian_eigen_relative_residual": eigen_residual,
        "x_controls_nonnegative": bool(
            mode != "x_contraction" or np.all(controls >= 0)
        ),
        "control_minimum": float(controls.min()),
        "B": {
            "minimum_determinant": float(np.linalg.det(muscle_B).min()),
            "nonpositive_determinant_fraction": float(
                np.mean(np.linalg.det(muscle_B) <= 0)
            ),
            "minimum_symmetric_eigenvalue": float(np.linalg.eigvalsh(muscle_B).min()),
            "minimum_singular_value": float(
                np.linalg.svd(muscle_B, compute_uv=False).min()
            ),
        },
        "energy": energy,
        "failure_recorded": summary["failure"] is not None,
    }
    assert result["forward_passes"]
    assert result["fixed_boundary_passes"]
    assert abs(result["area_difference"]) <= 1e-12
    assert result["x_controls_nonnegative"]
    return result


def main(cfg: Config) -> None:
    protocol = json.loads((INPUT / "protocol.json").read_text())
    mesh = ph.build_mesh(*protocol["mesh"])
    hashes = source_hashes(protocol)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for height in map(float, protocol["config"]["heights"].split(",")):
        for mode in protocol["config"]["modes"].split(","):
            folder = INPUT / f"h{round(height * 1000):03d}-{mode}"
            records.append(verify_case(cfg, mesh, folder, height, mode))
    report = {"input": str(INPUT), "source_hashes": hashes, "cases": records}
    write_json(output / "endpoint-checks.json", report)
    cherries.log_metrics(
        {f"{item['case']}/residual_inf": item["residual_inf"] for item in records}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileAdamEndpointCheck)
