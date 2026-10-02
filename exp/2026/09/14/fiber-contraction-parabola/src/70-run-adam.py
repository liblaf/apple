"""Historical pure-Adam comparison with the corrected plane-strain physics."""

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

import controls2d
import numpy as np
import physics2d as ph
import pydantic_settings as ps
import scipy
import scipy.sparse.linalg as spla
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

LOG = logging.getLogger(__name__)


class ProfileAdamComparison(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("70-adam-comparison")
    nx: int = 100
    ny: int = 10
    heights: str = "0.05,0.20"
    modes: str = "x_contraction,unconstrained"
    max_updates: int = 1200
    learning_rate: float = 0.03
    lr_decay: float = 0.99
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 250
    diagnostic_det_floor: float = 1e-6


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def adam_step(
    q: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    gradient: np.ndarray,
    counter: int,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """One historical Adam update; ``gradient`` is for the raw top-node MSE."""
    assert counter >= 1
    m = cfg.beta1 * m + (1 - cfg.beta1) * gradient
    v = cfg.beta2 * v + (1 - cfg.beta2) * gradient**2
    rate = cfg.learning_rate * cfg.lr_decay ** (counter - 1)
    mhat = m / (1 - cfg.beta1**counter)
    vhat = v / (1 - cfg.beta2**counter)
    return q - rate * mhat / (np.sqrt(vhat) + cfg.epsilon), m, v, rate


def adam_gate() -> dict[str, float]:
    """Independent two-vector closed-form check of the Adam convention."""
    cfg = Config()
    q = np.array([0.4, -0.3])
    m = np.zeros(2)
    v = np.zeros(2)
    gradients = (np.array([0.2, -0.5]), np.array([-0.4, 0.25]))
    for counter, gradient in enumerate(gradients, start=1):
        proposed, m, v, rate = adam_step(q, m, v, gradient, counter, cfg)
        # Closed form, deliberately separate from the running moment state.
        mh = sum(
            (1 - cfg.beta1) * cfg.beta1 ** (counter - i) * g
            for i, g in enumerate(gradients[:counter], start=1)
        ) / (1 - cfg.beta1**counter)
        vh = sum(
            (1 - cfg.beta2) * cfg.beta2 ** (counter - i) * g**2
            for i, g in enumerate(gradients[:counter], start=1)
        ) / (1 - cfg.beta2**counter)
        expected = q - rate * mh / (np.sqrt(vh) + cfg.epsilon)
        assert np.allclose(proposed, expected, rtol=0, atol=1e-15)
        q = proposed
    return {
        "two_vector_max_abs_error": float(np.max(np.abs(proposed - expected))),
        "final_learning_rate": rate,
    }


def run_case(  # noqa: PLR0915
    cfg: Config, mesh: Any, mode: str, height: float, output: Path
) -> dict[str, Any]:
    name = f"h{round(height * 1000):03d}-{mode}"
    folder = output / name
    folder.mkdir()
    size = int(mesh.muscle.sum()) * (1 if mode == "x_contraction" else 3)
    q = np.zeros(size)
    m, v = np.zeros(size), np.zeros(size)
    seed = np.zeros(mesh.nfree)
    rows: list[dict[str, Any]] = []
    histories_u, histories_q = [], []
    forward_iterations = 0
    started = time.monotonic()
    termination = "Adam update budget reached; not a convergence claim"
    failure: dict[str, Any] | None = None
    failed_forward_evaluations = 0

    for step in range(cfg.max_updates + 1):
        try:
            B = ph.matrices(mesh, q, mode)
            state = ph.solve(
                mesh,
                B,
                seed,
                tolerance=cfg.forward_tolerance,
                max_iterations=cfg.forward_max_iterations,
            )
            raw_loss, raw_gradient_u, _ = ph.loss(mesh, state.u, height, "l2")
            raw_gradient, adjoint_residual = ph.control_gradient(
                mesh, state, B, raw_gradient_u, mode
            )
        except ph.ForwardSolveError as exc:
            failed_forward_evaluations += 1
            failure = {
                "step": step,
                "adam_counter": step,
                "reason": str(exc),
                "last_valid_step": len(rows) - 1,
            }
            np.savez_compressed(
                folder / "failed-proposal.npz",
                controls=q,
                moment=m,
                variance=v,
                adam_counter=step,
            )
            write_json(folder / "failure.json", failure)
            cherries.log_metrics(
                {f"{name}/failed_forward_evaluations": failed_forward_evaluations}
            )
            termination = "proposed Adam state failed forward equilibrium; last valid state retained"
            break
        assert np.isfinite(raw_loss)
        assert np.all(np.isfinite(raw_gradient))
        forward_iterations += state.iterations
        normalized_gradient = raw_gradient / height**2
        projected_normalized = controls2d.gradient_mapping(q, normalized_gradient, mode)
        row = {
            "step": step,
            "adam_counter": step,
            "evaluations": step + 1,
            "objective": float(raw_loss),
            "objective_normalized": float(raw_loss / height**2),
            "raw_loss": float(raw_loss),
            "normalized_loss": float(raw_loss / height**2),
            "raw_gradient_rms": float(np.linalg.norm(raw_gradient) / np.sqrt(size)),
            "raw_gradient_inf": float(np.linalg.norm(raw_gradient, np.inf)),
            "normalized_gradient_rms": float(
                np.linalg.norm(normalized_gradient) / np.sqrt(size)
            ),
            "normalized_gradient_inf": float(
                np.linalg.norm(normalized_gradient, np.inf)
            ),
            "projected_gradient_inf": float(
                np.linalg.norm(projected_normalized, np.inf)
            ),
            "adam_m_rms": float(np.linalg.norm(m) / np.sqrt(size)),
            "adam_v_rms": float(np.linalg.norm(v) / np.sqrt(size)),
            "update_learning_rate": None,
            "control_update_inf": 0.0,
            "control_update_rms": 0.0,
            "forward_iterations_total": forward_iterations,
            "last_forward_iterations": state.iterations,
            "adjoint_relative_residual": adjoint_residual,
            "elapsed_seconds": time.monotonic() - started,
            **ph.diagnostics(mesh, state, q, mode, height),
        }
        rows.append(row)
        histories_u.append(ph.unpack(mesh, state.u).copy())
        histories_q.append(q.copy())
        seed = state.u.copy()
        # This is sufficient to restart from every valid endpoint: moments are
        # those in force on arrival at the current Adam state.
        np.savez_compressed(
            folder / "checkpoint.npz",
            controls=q,
            u=histories_u[-1],
            moment=m,
            variance=v,
            adam_counter=step,
        )
        if step % 50 == 0:
            LOG.info(
                "%s Adam step=%d raw=%.6g normalized=%.6g Jmin=%.5g",
                name,
                step,
                raw_loss,
                row["objective_normalized"],
                row["min_J"],
            )
        cherries.set_step(step)
        if step % 50 == 0:
            cherries.log_metrics(
                {
                    f"{name}/raw_loss": row["raw_loss"],
                    f"{name}/normalized_loss": row["normalized_loss"],
                    f"{name}/min_J": row["min_J"],
                    f"{name}/projected_gradient_inf": row["projected_gradient_inf"],
                }
            )
        if row["min_J"] <= cfg.diagnostic_det_floor:
            termination = (
                "mesh degeneration diagnostic threshold reached; not stationary"
            )
            cherries.log_metrics({f"{name}/diagnostic_stop": 1})
            break
        if step == cfg.max_updates:
            break
        proposed, m_next, v_next, rate = adam_step(q, m, v, raw_gradient, step + 1, cfg)
        proposed = controls2d.project(proposed, mode)
        row["update_learning_rate"] = rate
        row["control_update_inf"] = float(np.linalg.norm(proposed - q, np.inf))
        row["control_update_rms"] = float(np.linalg.norm(proposed - q) / np.sqrt(size))
        m, v, q = m_next, v_next, proposed

    assert rows, name
    with (folder / "trace.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
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
    final_B = ph.matrices(mesh, histories_q[-1], mode)
    final = ph.solve(
        mesh,
        final_B,
        seed,
        tolerance=cfg.forward_tolerance,
        max_iterations=cfg.forward_max_iterations,
    )
    eigmin = float(
        spla.eigsh(final.hessian, k=1, sigma=0, which="LM", return_eigenvectors=False)[
            0
        ]
    )
    summary = {
        "name": name,
        "height": height,
        "mode": mode,
        "control_dofs": size,
        "optimizer": "pure historical Adam",
        "optimizer_success": False,
        "optimizer_message": termination,
        "accepted_iterations": len(rows) - 1,
        "function_evaluations": len(rows) + failed_forward_evaluations,
        "successful_forward_evaluations": len(rows),
        "failed_forward_evaluations": failed_forward_evaluations,
        "measured_forward_evaluations": len(rows),
        "forward_iterations": forward_iterations,
        "hessian_eigenvalue_nearest_zero": eigmin,
        "inverse_stationarity": False,
        "initial": rows[0],
        "final": rows[-1],
        "failure": failure,
        "wall_seconds": time.monotonic() - started,
    }
    write_json(folder / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    logging.getLogger("liblaf.cherries.plugins.logging").setLevel(logging.WARNING)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    mesh = ph.build_mesh(cfg.nx, cfg.ny)
    production = ph.ROOT / "src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py"
    sources = [
        Path(__file__),
        Path(ph.__file__),
        Path(controls2d.__file__),
        ph.MESH_SOURCE,
        production,
    ]
    snapshot = output / "source"
    snapshot.mkdir()
    hashes = {}
    for source in sources:
        hashes[str(source.relative_to(ph.ROOT))] = hashlib.sha256(
            source.read_bytes()
        ).hexdigest()
        shutil.copy2(source, snapshot / source.name)
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "source_sha256": hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "energy": "mu/2*(||F B||^2-2)-mu*(det(F)-1)+lambda/2*(det(F)-1)^2",
        "mesh": [cfg.nx, cfg.ny],
        "target": "free top-node displacement (0, 4*h*x*(1-x)); historical top-node L2",
        "controls": {
            "x_contraction": "B=diag(1+a,1), a>=0, projection after each Adam update",
            "unconstrained": "B=I+sym(qxx,qyy,qxy), unbounded",
            "contraction_only": "B=I+S, S positive semidefinite; Frobenius eigenvalue projection after each raw-coordinate Adam update",
        },
        "inverse_policy": "pure Adam on raw top-node L2 loss: lr=.03*.99^(t-1), betas=(.9,.999), eps=1e-8; no outer line search, gradient clipping, step caps, monotonic filter, or refinement",
        "projection_policy": "feasibility projection only: x nonnegative or matrix Frobenius PSD; raw Adam moments unchanged by projection; no gradient clipping",
        "contraction_gradient_diagnostic": "unit-step Frobenius gradient mapping with offdiagonal gradient g_xy/2; not an Adam update or a packed-coordinate Euclidean projection",
        "diagnostic_stops": "min physical det(F)<=1e-6 stops without a convergence claim",
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "adam-gate.json", adam_gate())
    write_json(output / "contraction-gates.json", controls2d.gates())
    summaries = []
    for height in map(float, cfg.heights.split(",")):
        for mode in cfg.modes.split(","):
            summaries.append(run_case(cfg, mesh, mode, height, output))
            write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileAdamComparison)
