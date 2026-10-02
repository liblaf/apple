"""Independent checks for the six-case pure-Adam contraction comparison."""

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

import controls2d
import numpy as np
import physics2d as ph
import pydantic_settings as ps
import scipy.linalg as la
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
INPUT = GROUP / "data/100-adam-contraction"
REFERENCE = GROUP / "data/70-adam-comparison"


class ProfileContractionCheck(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("120-contraction-checks")
    residual_tolerance: float = 1e-10


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def boundary_nodes(mesh: Any) -> np.ndarray:
    w = mesh.nx + 1
    nodes = np.concatenate(
        (
            np.arange(w),
            np.arange(2 * w - 1, (mesh.ny + 1) * w, w),
            np.arange((mesh.ny + 1) * w - 2, mesh.ny * w - 1, -1),
            np.arange((mesh.ny - 1) * w, 0, -w),
        )
    )
    assert len(nodes) == len(np.unique(nodes)) == 2 * (mesh.nx + mesh.ny)
    return nodes


def shoelace(points: np.ndarray) -> float:
    shifted = np.roll(points, -1, axis=0)
    return float(
        0.5 * np.sum(points[:, 0] * shifted[:, 1] - points[:, 1] * shifted[:, 0])
    )


def pack(mesh: Any, full: np.ndarray) -> np.ndarray:
    result = np.empty(mesh.nfree)
    mask = mesh.lookup >= 0
    result[mesh.lookup[mask]] = full.reshape(-1)[mask]
    return result


def check_sources(protocol: dict[str, Any]) -> dict[str, bool]:
    result = {}
    for key, expected in protocol["source_sha256"].items():
        snapshot = INPUT / "source" / Path(key).name
        assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == expected
        result[key] = True
    for module in (ph, controls2d):
        key = str(Path(module.__file__).resolve().relative_to(ph.ROOT))
        assert (
            hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            == protocol["source_sha256"][key]
        )
        result[f"live_{Path(module.__file__).stem}_matches_run"] = True
    return result


def cone_extrema(mesh: Any, controls_history: np.ndarray) -> dict[str, float]:
    min_b, min_q, min_sigma, min_det = np.inf, np.inf, np.inf, np.inf
    nonpositive = 0
    total = 0
    for controls in controls_history:
        B = ph.matrices(mesh, controls, "contraction_only")[mesh.muscle]
        assert np.allclose(B, B.swapaxes(1, 2), rtol=0, atol=1e-13)
        effective = B @ B - np.eye(2)
        min_b = min(min_b, float(np.linalg.eigvalsh(B).min()))
        min_q = min(min_q, float(np.linalg.eigvalsh(effective).min()))
        min_sigma = min(min_sigma, float(np.linalg.svd(B, compute_uv=False).min()))
        det = np.linalg.det(B)
        min_det = min(min_det, float(det.min()))
        nonpositive += int(np.sum(det <= 0))
        total += len(det)
    assert min_b >= 1 - 1e-12
    assert min_q >= -1e-12
    assert nonpositive == 0
    return {
        "minimum_eigenvalue_B": min_b,
        "minimum_eigenvalue_effective_BBt_minus_I": min_q,
        "minimum_singular_B": min_sigma,
        "minimum_determinant_B": min_det,
        "nonpositive_determinant_fraction": nonpositive / total,
    }


def finite_difference() -> dict[str, float]:
    mesh = ph.build_mesh(10, 5)
    n = int(mesh.muscle.sum())
    base = np.tile([0.03, 0.02, 0.006], n)
    direction = np.random.default_rng(20260914).normal(size=3 * n)
    direction /= np.linalg.norm(direction)
    B = ph.matrices(mesh, base, "contraction_only")
    state = ph.solve(mesh, B, np.zeros(mesh.nfree), tolerance=1e-11)
    value, du, _ = ph.loss(mesh, state.u, 0.05, "l2")
    gradient, _ = ph.control_gradient(mesh, state, B, du, "contraction_only")
    h = 1e-5
    values = []
    for sign in (-1, 1):
        controls = base + sign * h * direction
        assert np.linalg.eigvalsh(controls2d.symmetric(controls)).min() > 0
        trial = ph.solve(
            mesh,
            ph.matrices(mesh, controls, "contraction_only"),
            state.u,
            tolerance=1e-11,
        )
        values.append(ph.loss(mesh, trial.u, 0.05, "l2")[0])
    fd = (values[1] - values[0]) / (2 * h)
    predicted = float(gradient @ direction)
    relative = abs(fd - predicted) / max(abs(predicted), 1e-12)
    assert relative < 2e-4
    return {
        "objective": value,
        "finite_difference": fd,
        "implicit_gradient": predicted,
        "relative_error": relative,
    }


def verify_case(
    cfg: Config, mesh: Any, folder: Path, height: float, mode: str
) -> dict[str, Any]:
    history = np.load(folder / "history.npz", allow_pickle=False)
    checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
    summary = json.loads((folder / "summary.json").read_text())
    rows = list(csv.DictReader((folder / "trace.csv").open()))
    assert len(rows) == len(history["u"]) == len(history["controls"])
    assert [int(row["step"]) for row in rows] == list(range(len(rows)))
    assert all(
        key in checkpoint
        for key in ("controls", "u", "moment", "variance", "adam_counter")
    )
    assert checkpoint["adam_counter"].item() == len(rows) - 1
    assert np.array_equal(checkpoint["controls"], history["controls"][-1])
    assert np.array_equal(checkpoint["u"], history["u"][-1])
    controls, full = checkpoint["controls"], checkpoint["u"]
    B = ph.matrices(mesh, controls, mode)
    energy, residual, H, J = ph.assemble(mesh, pack(mesh, full), B)
    assert H is not None
    LOG.info("Dense smallest Hessian eigenpair: %s", folder.name)
    dense = ((H + H.T) * 0.5).toarray()
    values, vectors = la.eigh(
        dense, subset_by_index=[0, 0], driver="evr", check_finite=False
    )
    eigmin, vector = float(values[0]), vectors[:, 0]
    hnorm = max(float(np.linalg.norm(dense, np.inf)), 1.0)
    fixed = mesh.lookup.reshape(-1, 2) < 0
    record = {
        "case": folder.name,
        "mode": mode,
        "height": height,
        "states": len(rows),
        "residual_inf": float(np.linalg.norm(residual, np.inf)),
        "fixed_displacement_inf": float(np.max(np.abs(full[fixed]))),
        "area_difference": float(mesh.area @ J)
        - shoelace((mesh.p + full)[boundary_nodes(mesh)]),
        "J_minimum": float(J.min()),
        "J_maximum": float(J.max()),
        "energy": energy,
        "smallest_algebraic_hessian_eigenvalue": eigmin,
        "hessian_eigen_relative_residual": float(
            np.linalg.norm(H @ vector - eigmin * vector) / hnorm
        ),
        "failure_recorded": summary["failure"] is not None,
    }
    if mode == "contraction_only":
        record["cone_history"] = cone_extrema(mesh, history["controls"])
        _, du, _ = ph.loss(mesh, pack(mesh, full), height, "l2")
        raw_gradient, _ = ph.control_gradient(
            mesh, ph.State(pack(mesh, full), energy, residual, H, J, 0), B, du, mode
        )
        record["frobenius_gradient_mapping_inf"] = float(
            np.linalg.norm(
                controls2d.gradient_mapping(controls, raw_gradient, mode), np.inf
            )
        )
    assert record["residual_inf"] <= cfg.residual_tolerance
    assert record["fixed_displacement_inf"] <= 1e-14
    assert abs(record["area_difference"]) <= 1e-12
    return record


def regressions(protocol: dict[str, Any]) -> dict[str, bool]:
    checks = {}
    for height in map(float, protocol["config"]["heights"].split(",")):
        for mode in ("x_contraction", "unconstrained"):
            name = f"h{round(height * 1000):03d}-{mode}"
            new, old = (
                np.load(INPUT / name / "history.npz"),
                np.load(REFERENCE / name / "history.npz"),
            )
            assert np.array_equal(new["u"], old["u"])
            assert np.array_equal(new["controls"], old["controls"])
            checks[name] = True
    return checks


def main(cfg: Config) -> None:
    protocol = json.loads((INPUT / "protocol.json").read_text())
    mesh = ph.build_mesh(*protocol["mesh"])
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    cases = [
        verify_case(
            cfg,
            mesh,
            INPUT / f"h{round(height * 1000):03d}-{mode}",
            height,
            mode,
        )
        for height in map(float, protocol["config"]["heights"].split(","))
        for mode in protocol["config"]["modes"].split(",")
    ]
    report = {
        "source_hashes": check_sources(protocol),
        "controls_gates": controls2d.gates(),
        "contraction_implicit_fd": finite_difference(),
        "x_and_unconstrained_trajectory_regressions": regressions(protocol),
        "cases": cases,
    }
    write_json(output / "contraction-checks.json", report)
    cherries.log_metrics(
        {f"{item['case']}/residual_inf": item["residual_inf"] for item in cases}
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileContractionCheck)
