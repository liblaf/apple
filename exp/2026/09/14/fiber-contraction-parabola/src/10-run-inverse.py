"""Compare fixed-x contraction and free symmetric active strain on a parabola."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import physics2d as ph
import pydantic_settings as ps
import scipy
import scipy.sparse.linalg as spla
from liblaf.cherries import core, plugins, profiles
from optimizer import minimize

from liblaf import cherries

LOG = logging.getLogger(__name__)


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
    output: Path = Path("10-comparison")
    nx: int = 100
    ny: int = 10
    heights: str = "0.05,0.20"
    modes: str = "x_contraction,unconstrained"
    max_iterations: int = 300
    max_evaluations: int = 1500
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 250
    gradient_tolerance: float = 1e-7
    maximum_control_step: float = 0.25
    minimum_control_update: float = 1e-7
    diagnostic_det_floor: float = 1e-6
    validate_derivatives: bool = True


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def validation():
    """Independent centered differences of assembled residual/HVP and adjoint."""
    m = ph.build_mesh(10, 5)
    rng = np.random.default_rng(20260914)
    records = []
    for mode in ("x_contraction", "unconstrained"):
        size = int(m.muscle.sum()) * (1 if mode == "x_contraction" else 3)
        q = (
            np.full(size, 0.025)
            if mode == "x_contraction"
            else rng.normal(0, 0.01, size)
        )
        B = ph.matrices(m, q, mode)
        u = rng.normal(0, 1e-4, m.nfree)
        direction = rng.normal(size=m.nfree)
        direction /= np.linalg.norm(direction)
        h = 1e-6
        energy, residual, H, _ = ph.assemble(m, u, B)
        plus = ph.assemble(m, u + h * direction, B)
        minus = ph.assemble(m, u - h * direction, B)
        fd_energy = (plus[0] - minus[0]) / (2 * h)
        exact_energy = residual @ direction
        energy_relative = abs(fd_energy - exact_energy) / max(abs(exact_energy), 1e-10)
        fd_hvp = (plus[1] - minus[1]) / (2 * h)
        hvp_relative = np.linalg.norm(fd_hvp - H @ direction) / np.linalg.norm(
            H @ direction
        )
        state = ph.solve(m, B, np.zeros(m.nfree), tolerance=1e-11)
        val, du, _ = ph.loss(m, state.u, 0.05, "l2")
        grad, adjoint_residual = ph.control_gradient(m, state, B, du, mode)
        dq = rng.normal(size=size)
        dq /= np.linalg.norm(dq)
        dh = 1e-4
        values = []
        for sign in (-1, 1):
            other = ph.solve(
                m, ph.matrices(m, q + sign * dh * dq, mode), state.u, tolerance=1e-11
            )
            values.append(ph.loss(m, other.u, 0.05, "l2")[0])
        fd_gradient = (values[1] - values[0]) / (2 * dh)
        predicted_gradient = grad @ dq
        implicit_relative = abs(fd_gradient - predicted_gradient) / max(
            abs(predicted_gradient), 1e-12
        )
        assert energy_relative < 1e-5, energy_relative
        assert hvp_relative < 1e-6, hvp_relative
        assert implicit_relative < 2e-4, implicit_relative
        records.append(
            {
                "mode": mode,
                "energy_direction_relative_error": float(energy_relative),
                "hvp_relative_error": float(hvp_relative),
                "implicit_direction_relative_error": float(implicit_relative),
                "adjoint_relative_residual": adjoint_residual,
                "energy": energy,
                "objective": val,
            }
        )
    return records


def run_case(cfg: Config, mesh: Any, mode: str, height: float, output: Path):  # noqa: PLR0915
    name = f"h{round(height * 1000):03d}-{mode}"
    folder = output / name
    folder.mkdir()
    size = int(mesh.muscle.sum()) * (1 if mode == "x_contraction" else 3)
    seed = np.zeros(mesh.nfree)
    rows, histories_u, histories_q = [], [], []
    evaluation_count, forward_iterations = 0, 0
    cache = None
    started = time.monotonic()

    def evaluate(q: np.ndarray):
        nonlocal cache, evaluation_count, forward_iterations
        if cache is not None and np.array_equal(q, cache[0]):
            return cache[1], cache[2]
        evaluation_count += 1
        B = ph.matrices(mesh, q, mode)
        state = ph.solve(
            mesh,
            B,
            seed,
            tolerance=cfg.forward_tolerance,
            max_iterations=cfg.forward_max_iterations,
        )
        value, du, _ = ph.loss(mesh, state.u, height, "l2")
        gradient, adjoint_residual = ph.control_gradient(mesh, state, B, du, mode)
        value /= height**2
        gradient /= height**2
        assert np.isfinite(value)
        assert np.all(np.isfinite(gradient))
        cache = (q.copy(), value, gradient, state, adjoint_residual)
        forward_iterations += state.iterations
        return value, gradient

    def record(q: np.ndarray):
        nonlocal seed
        value, gradient = evaluate(q)
        state = cache[3]
        seed = state.u.copy()
        pg = gradient.copy()
        if mode == "x_contraction":
            pg[(q <= 1e-12) & (gradient > 0)] = 0
        row = {
            "step": len(rows),
            "evaluations": evaluation_count,
            "objective_normalized": value,
            "objective": value * height**2,
            "projected_gradient_inf": float(np.linalg.norm(pg, np.inf)),
            "forward_iterations_total": forward_iterations,
            "last_forward_iterations": state.iterations,
            "adjoint_relative_residual": cache[4],
            "elapsed_seconds": time.monotonic() - started,
            **ph.diagnostics(mesh, state, q, mode, height),
        }
        rows.append(row)
        histories_u.append(ph.unpack(mesh, state.u).copy())
        histories_q.append(q.copy())
        with (folder / "trace.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        np.savez_compressed(folder / "checkpoint.npz", controls=q, u=state.u)
        cherries.set_step(len(rows) - 1)
        cherries.log_metrics({f"{name}/{k}": v for k, v in row.items()})
        if row["step"] % 10 == 0:
            LOG.info(
                "%s step=%d eval=%d fit=%.6g peak=%.6g J=[%.5g,%.5g] pg=%.3g",
                name,
                row["step"],
                evaluation_count,
                row["fit_rms"],
                row["peak_top_uy"],
                row["min_J"],
                row["max_J"],
                row["projected_gradient_inf"],
            )
        if row["min_J"] <= cfg.diagnostic_det_floor:
            return "mesh degeneration diagnostic threshold reached; not stationary"
        return None

    q0 = np.zeros(size)
    record(q0)

    def reject(iteration: int, backtrack: int, alpha: float, reason: str):
        item = {
            "iteration": iteration,
            "backtrack": backtrack,
            "alpha": alpha,
            "reason": reason,
        }
        with (folder / "rejected-trials.jsonl").open("a") as handle:
            handle.write(json.dumps(item) + "\n")
        LOG.info("Rejected equilibrium trial: %s %s", name, item)

    result = minimize(
        evaluate, q0, record, reject, nonnegative=mode == "x_contraction", cfg=cfg
    )
    if not np.array_equal(result.x, histories_q[-1]):
        record(result.x)
    # Resolve final controls independently from the final accepted seed.
    final = ph.solve(
        mesh,
        ph.matrices(mesh, result.x, mode),
        seed,
        tolerance=cfg.forward_tolerance,
        max_iterations=cfg.forward_max_iterations,
    )
    eigmin = float(
        spla.eigsh(final.hessian, k=1, sigma=0, which="LM", return_eigenvectors=False)[
            0
        ]
    )
    np.savez_compressed(
        folder / "history.npz",
        points=mesh.p,
        triangles=mesh.tri,
        muscle=mesh.muscle,
        top=mesh.top,
        u=np.array(histories_u),
        controls=np.array(histories_q),
        height=height,
        mode=mode,
    )
    summary = {
        "name": name,
        "height": height,
        "mode": mode,
        "control_dofs": size,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "accepted_iterations": len(rows) - 1,
        "function_evaluations": int(result.nfev),
        "measured_forward_evaluations": evaluation_count,
        "forward_iterations": forward_iterations,
        "hessian_eigenvalue_nearest_zero": eigmin,
        "inverse_stationarity": rows[-1]["projected_gradient_inf"]
        <= cfg.gradient_tolerance,
        "initial": rows[0],
        "final": rows[-1],
        "wall_seconds": time.monotonic() - started,
    }
    write_json(folder / "summary.json", summary)
    LOG.info("FINISHED %s %s", name, json.dumps(summary))
    return summary


def main(cfg: Config):
    logging.getLogger("liblaf.cherries.plugins.logging").setLevel(logging.WARNING)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    production = ph.ROOT / "src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py"
    # Material import identity is recorded separately from this exact 2D reduction.
    import liblaf.apple.warp.fem._stable_neo_hookean_active as active_material

    assert Path(active_material.__file__).resolve() == production.resolve()
    mesh = ph.build_mesh(cfg.nx, cfg.ny)
    sources = [*Path(__file__).parent.glob("*.py"), ph.MESH_SOURCE, production]
    snapshot = output / "source"
    snapshot.mkdir()
    hashes = {}
    for source in sources:
        key = str(source.relative_to(ph.ROOT))
        hashes[key] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, snapshot / source.name)
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "source_sha256": hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "energy": "mu/2*(||F B||^2-2)-mu*(det(F)-1)+lambda/2*(det(F)-1)^2",
        "reduction": "F3=diag(F2,1), B3=diag(B2,1); exact plane-strain density and derivatives",
        "mesh": [cfg.nx, cfg.ny],
        "triangles": len(mesh.tri),
        "muscle_triangles": int(mesh.muscle.sum()),
        "domain": [1, 0.1],
        "muscle_y": [0.04, 0.06],
        "young_fat_MPa": 0.003,
        "young_muscle_MPa": 0.03,
        "poisson": 0.49,
        "units": "historical length units L; stress MPa; energy per out-of-plane thickness MPa*L^2; residual MPa*L",
        "boundary": "bottom and both sides fixed in x/y; free top and interior",
        "target": "free top-node displacement (0, 4*h*x*(1-x)); same historical top-node L2 loss",
        "controls": {
            "x_contraction": "B=diag(1+a,1), a>=0 per muscle triangle, Axx=1/(1+a)",
            "unconstrained": "B=I+sym(qxx,qyy,qxy), 3 unbounded values per muscle triangle",
        },
        "initialization": "B=I and u=0 for every independent case",
        "regularization": "none",
        "skin": False,
        "forward_policy": "Newton with physical det(F)>1e-8 Armijo trials, same energy for both models; failed equilibrium trial raises and is rejected by the outer search",
        "inverse_policy": "Projected L-BFGS Armijo with loss/h^2, maximum component step 0.25, failed equilibrium trials explicitly logged and backtracked; last accepted equilibrium seeds every trial",
        "diagnostic_stops": "Both modes stop without a convergence claim at accepted min det(F)<=1e-6 or control step infinity norm<=1e-7; these numerical resolution gates are not calibrated physical admissibility bounds",
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(output / "protocol.json", protocol)
    if cfg.validate_derivatives:
        checks = validation()
        write_json(output / "derivative-checks.json", checks)
        LOG.info("Derivative checks: %s", checks)
    completed = []
    for height in map(float, cfg.heights.split(",")):
        for mode in cfg.modes.split(","):
            completed.append(run_case(cfg, mesh, mode, height, output))
            write_json(output / "summary.json", completed)
    LOG.info("Completed all %d cases: %s", len(completed), output)


if __name__ == "__main__":
    cherries.main(
        main, profile="debug" if os.environ.get("DEBUG") == "1" else ProfileRecord
    )
