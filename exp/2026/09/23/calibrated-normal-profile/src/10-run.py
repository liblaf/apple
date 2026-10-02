"""Neutral-start activation by smoothness by data-loss factorial comparison."""

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

import calibrated_study as ns
import numpy as np
import pydantic_settings as ps
import scipy
from experiment import Profile

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("10-comparison")
    height: float = 0.20
    modes: str = ",".join(ns.MODES)
    max_updates: int = 1200
    learning_rate: float = 0.03
    lr_decay: float = 0.99
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 250
    history_stride: int = 10


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def sha(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_trace(folder: Path, rows: list[dict]):
    with (folder / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def run_case(  # noqa: PLR0915
    cfg: Config,
    mesh: Any,
    mode: str,
    height: float,
    variant: str,
    kind: str,
    beta: float,
    scales: dict,
    smooth_weight: float,
    folder: Path,
):
    folder.mkdir()
    name = f"{folder.parent.name}/{variant}"
    q = ns.am.initialize(int(mesh.muscle.sum()), mode)
    q0 = q.copy()
    seed = np.zeros(mesh.nfree)
    m, v = np.zeros_like(q), np.zeros_like(q)
    rows, us, qs, steps = [], [], [], []
    start = time.monotonic()
    failure, first_B_step_rms = None, None
    total_forward = 0
    for step in range(cfg.max_updates + 1):
        try:
            state, B, gradient, values = ns.evaluate(
                mesh,
                q,
                mode,
                height,
                kind,
                beta,
                smooth_weight,
                scales,
                seed,
                tolerance=cfg.forward_tolerance,
                max_iterations=cfg.forward_max_iterations,
            )
        except ns.ph.ForwardSolveError as exc:
            failure = {"step": step, "reason": str(exc), "last_valid_step": step - 1}
            write_json(folder / "failure.json", failure)
            np.savez_compressed(
                folder / "failed-proposal.npz",
                controls=q,
                moment=m,
                variance=v,
                step=step,
            )
            LOG.error("%s failed at update %d: %s", name, step, exc)  # noqa: TRY400
            break
        assert np.all(np.isfinite(gradient))
        total_forward += state.iterations
        row = {
            "step": step,
            **values,
            "elapsed_seconds": time.monotonic() - start,
            "forward_iterations_total": total_forward,
            "update_learning_rate": cfg.learning_rate * cfg.lr_decay**step,
            "projected_gradient_ratio": values["projected_gradient_inf"]
            / (
                rows[0]["projected_gradient_inf"]
                if rows
                else values["projected_gradient_inf"]
            ),
        }
        rows.append(row)
        full_u = ns.ph.unpack(mesh, state.u)
        if step == 0:
            assert np.count_nonzero(q) == 0
            assert np.count_nonzero(full_u) == 0
            assert np.array_equal(B, np.broadcast_to(np.eye(2), B.shape))
            assert np.count_nonzero(m) == np.count_nonzero(v) == 0
            np.savez_compressed(
                folder / "initial-state.npz",
                controls=q,
                u=full_u,
                B=B,
                moment=m,
                variance=v,
                step=0,
            )
        if step % cfg.history_stride == 0 or step == cfg.max_updates:
            us.append(full_u.copy())
            qs.append(q.copy())
            steps.append(step)
        # Preserve every accepted state, including the one before any failed solve.
        np.savez_compressed(
            folder / "checkpoint.npz",
            controls=q,
            u=full_u,
            B=B,
            moment=m,
            variance=v,
            step=step,
        )
        seed = state.u.copy()
        if step % 100 == 0 or step == cfg.max_updates:
            LOG.info(
                "%s step=%d J/J20=%.6f normal=%.3fdeg minJ=%.5g",
                name,
                step,
                values["objective_normalized"],
                values["normal_angle_rms_deg"],
                values["min_J"],
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    f"{name}/{key}": values[key]
                    for key in (
                        "objective_normalized",
                        "normal_angle_rms_deg",
                        "position_loss_normalized",
                        "min_J",
                        "projected_gradient_inf",
                    )
                }
            )
            write_trace(folder, rows)
        if values["min_J"] <= 1e-6:
            failure = {
                "step": step,
                "reason": "diagnostic minimum physical J threshold reached",
                "last_valid_step": step,
            }
            write_json(folder / "failure.json", failure)
            break
        if step == cfg.max_updates:
            break
        counter = step + 1
        m = cfg.beta1 * m + (1 - cfg.beta1) * gradient
        v = cfg.beta2 * v + (1 - cfg.beta2) * gradient**2
        proposed = ns.am.project(
            q
            - cfg.learning_rate
            * cfg.lr_decay**step
            * (m / (1 - cfg.beta1**counter))
            / (np.sqrt(v / (1 - cfg.beta2**counter)) + cfg.epsilon),
            mode,
        )
        if step == 0:
            delta = (
                ns.gs.study.matrices(mesh, proposed, mode)[mesh.muscle] - B[mesh.muscle]
            )
            first_B_step_rms = float(np.sqrt(np.mean(np.sum(delta**2, axis=(1, 2)))))
        q = proposed
    assert rows
    with np.load(folder / "checkpoint.npz") as final:
        if not steps or steps[-1] != rows[-1]["step"]:
            us.append(final["u"])
            qs.append(final["controls"])
            steps.append(rows[-1]["step"])
    np.savez_compressed(
        folder / "history.npz",
        points=mesh.p,
        triangles=mesh.tri,
        muscle=mesh.muscle,
        top=mesh.top,
        top_all=ns.gs.top_nodes(mesh),
        u=np.asarray(us),
        controls=np.asarray(qs),
        steps=np.asarray(steps),
        height=height,
        mode=mode,
        kind=kind,
        beta=beta,
        smooth_weight=smooth_weight,
    )
    write_trace(folder, rows)
    window = min(25, len(rows) - 1)
    summary = {
        "name": name,
        "mode": mode,
        "height": height,
        "variant": variant,
        "kind": kind,
        "beta": beta,
        "activation_smoothness_weight": smooth_weight,
        "dofs_per_element": ns.am.dofs(mode),
        "control_dofs": len(q0),
        "initialization": "neutral q=u=0, B=I, Adam m=v=0",
        "accepted_iterations": rows[-1]["step"],
        "failure": failure,
        "initial": rows[0],
        "final": rows[-1],
        "first_B_step_rms": first_B_step_rms,
        "position_loss_weight": 1.0,
        "last_window_updates": window,
        "last_window_objective_relative_decline": (
            rows[-1 - window]["objective"] - rows[-1]["objective"]
        )
        / rows[-1 - window]["objective"],
        "wall_seconds": time.monotonic() - start,
        "inverse_stationarity": False,
        "optimizer_message": "forward/geometry diagnostic stop"
        if failure
        else "fixed Adam budget; convergence not established",
    }
    write_json(folder / "summary.json", summary)
    return summary


def main(cfg: Config):
    logging.getLogger("liblaf.cherries.plugins.logging").setLevel(logging.WARNING)
    LOG.info("Starting all factorial runs from neutral activation and displacement")
    assert cfg.height == ns.HEIGHT
    assert cfg.history_stride > 0
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    gate = ns.GROUP / "data/05-verification/checks.json"
    checks = json.loads(gate.read_text())
    assert checks["passed"]
    for source in ns.numerical_sources():
        assert checks["source_sha256"][str(source.resolve())] == sha(source)
    snapshot = output / "source"
    snapshot.mkdir()
    sources = list(
        dict.fromkeys(
            [
                *ns.numerical_sources(),
                Path(__file__),
                Path(__file__).with_name("experiment.py"),
                Path(__file__).with_name("05-verify.py"),
            ]
        )
    )
    hashes = {}
    for source in sources:
        relative = source.relative_to(ns.ROOT)
        hashes[str(relative)] = sha(source)
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    shutil.copy2(gate, output / "input-gates.json")
    shutil.copy2(ns.GROUP / "docs/00-protocol.md", output / "protocol.md")
    mesh = ns.ph.build_mesh(100, 10)
    scales = ns.ns.normalization(mesh, cfg.height)
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "source_sha256": hashes,
        "gate_sha256": sha(gate),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "initialization": "Every run independently q=u=m=v=0, B=I; no checkpoint inputs",
        "mesh": [100, 10],
        "active_triangles": int(mesh.muscle.sum()),
        "objective": "L2 + coefficient * normal + smooth_weight * height^2 * R",
        "normal_weights": "fixed reference edge length / reference span",
        "normal_coefficient": ns.NORMAL_COEFFICIENT,
        "calibration": {
            "position_rms": ns.POSITION_SCALE,
            "normal_angle_deg": ns.ANGLE_SCALE_DEG,
        },
        "smoothness": "0 or 1; R=mean muscle-neighbor squared Frobenius tensor difference",
        "normalizers": scales,
        "variants": ns.VARIANTS,
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(output / "protocol.json", protocol)
    summaries = []
    for mode in cfg.modes.split(","):
        assert mode in ns.MODES
        cell = output / mode
        cell.mkdir()
        for variant, smooth_weight, kind, beta in ns.VARIANTS:
            result = run_case(
                cfg,
                mesh,
                mode,
                cfg.height,
                variant,
                kind,
                beta,
                scales,
                smooth_weight,
                cell / variant,
            )
            summaries.append(result)
            write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
