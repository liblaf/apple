"""Pure gradient-matching comparison with saved L2 runs and corrected 2D physics."""

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

import gradient_study as study
import numpy as np
import pydantic_settings as ps
import scipy
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

am = study.am
LOG = logging.getLogger(__name__)


class ProfileGradientStudy(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(
            plugins.Comet(run=run, disabled=os.environ.get("DEBUG") == "1")
        )
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("10-gradient")
    nx: int = 100
    ny: int = 10
    heights: str = "0.05,0.20"
    modes: str = "unconstrained,contraction_only,learned_direction,x_contraction"
    max_updates: int = 1200
    learning_rate: float = 0.03
    lr_decay: float = 0.99
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 250
    history_stride: int = 10
    save_history: bool = True


def write_json(path: Path, value: Any):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_case(  # noqa: PLR0915
    cfg: Config, mesh: Any, mode: str, height: float, output: Path
):
    name = f"h{round(height * 1000):03d}-{mode}"
    folder = output / name
    folder.mkdir()
    q = am.initialize(int(mesh.muscle.sum()), mode)
    m, v = np.zeros_like(q), np.zeros_like(q)
    seed = np.zeros(mesh.nfree)
    rows, us, qs, steps = [], [], [], []
    start = time.monotonic()
    failure = None
    total_forward = 0
    for step in range(cfg.max_updates + 1):
        try:
            state, B, gradient, values = study.evaluate(
                mesh,
                q,
                mode,
                height,
                seed,
                tolerance=cfg.forward_tolerance,
                max_iterations=cfg.forward_max_iterations,
            )
        except study.ph.ForwardSolveError as exc:
            failure = {"step": step, "reason": str(exc), "last_valid_step": step - 1}
            write_json(folder / "failure.json", failure)
            np.savez_compressed(
                folder / "failed-proposal.npz",
                controls=q,
                moment=m,
                variance=v,
                step=step,
            )
            LOG.exception("%s failed at update %d", name, step)
            break
        assert np.all(np.isfinite(gradient))
        total_forward += state.iterations
        row = {
            "step": step,
            **values,
            "elapsed_seconds": time.monotonic() - start,
            "forward_iterations_total": total_forward,
            "update_learning_rate": cfg.learning_rate * cfg.lr_decay**step,
        }
        rows.append(row)
        full_u = study.ph.unpack(mesh, state.u)
        if cfg.save_history and (
            step % cfg.history_stride == 0 or step == cfg.max_updates
        ):
            us.append(full_u.copy())
            qs.append(q.copy())
            steps.append(step)
        # Preserve the last valid complete optimizer state, including early failure.
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
                "%s step=%d G/h2=%.6f R=%.5g minJ=%.5g",
                name,
                step,
                values["normalized_loss"],
                values["roughness"],
                values["min_J"],
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    f"{name}/{key}": values[key]
                    for key in (
                        "normalized_loss",
                        "normalized_position_loss",
                        "roughness",
                        "min_J",
                        "projected_gradient_inf",
                    )
                }
            )
            with (folder / "trace.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
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
        q = am.project(
            q
            - cfg.learning_rate
            * cfg.lr_decay**step
            * (m / (1 - cfg.beta1**counter))
            / (np.sqrt(v / (1 - cfg.beta2**counter)) + cfg.epsilon),
            mode,
        )
    assert rows
    final_ckpt = np.load(folder / "checkpoint.npz")
    if not steps or steps[-1] != rows[-1]["step"]:
        us.append(final_ckpt["u"])
        qs.append(final_ckpt["controls"])
        steps.append(rows[-1]["step"])
    np.savez_compressed(
        folder / "history.npz",
        points=mesh.p,
        triangles=mesh.tri,
        muscle=mesh.muscle,
        top=mesh.top,
        top_all=study.top_nodes(mesh),
        u=np.asarray(us),
        controls=np.asarray(qs),
        steps=np.asarray(steps),
        height=height,
        mode=mode,
        loss_kind="gradient_only",
    )
    with (folder / "trace.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "name": name,
        "mode": mode,
        "height": height,
        "loss_kind": "gradient_only",
        "position_loss_weight": 0,
        "activation_smoothness_weight": 0,
        "control_dofs": len(q),
        "dofs_per_element": am.dofs(mode),
        "accepted_iterations": rows[-1]["step"],
        "failure": failure,
        "initial": rows[0],
        "final": rows[-1],
        "wall_seconds": time.monotonic() - start,
        "inverse_stationarity": False,
        "optimizer_message": "forward/geometry diagnostic stop"
        if failure
        else "fixed Adam budget; not demonstrated convergence",
    }
    write_json(folder / "summary.json", summary)
    return summary


def main(cfg: Config):
    logging.getLogger("liblaf.cherries.plugins.logging").setLevel(logging.WARNING)
    assert cfg.history_stride > 0
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    mesh = study.ph.build_mesh(cfg.nx, cfg.ny)
    sources = [
        *Path(__file__).parent.glob("*.py"),
        Path(study.ph.__file__),
        study.ph.MESH_SOURCE,
        Path(am.__file__),
        Path(study.study.__file__),
    ]
    snapshot = output / "source"
    snapshot.mkdir()
    hashes = {}
    for source in sources:
        relative = source.relative_to(study.ROOT)
        hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, snapshot / source.name)
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "source_sha256": hashes,
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "energy": "mu/2*(||F B||^2-2)-mu*(det(F)-1)+lambda/2*(det(F)-1)^2",
        "fixed_material": {"E_muscle": 0.03, "E_fat": 0.003, "nu": 0.49},
        "muscle_band_y": [0.04, 0.06],
        "active_triangles": int(mesh.muscle.sum()),
        "muscle_neighbor_edges": len(mesh.edges),
        "data_loss": "sum full top edges ||e[j]-e[i]||2^2 / dx / span; e=u-(0,4*h*x*(1-x)); both vector components",
        "regularizer": "none; activation roughness is diagnostic only",
        "objective": "gradient matching only; positional L2 weight=0, activation smoothness weight=0",
        "anchoring": "full top chain includes fixed zero-displacement corners; sides and bottom unchanged",
        "reference_derivatives": "fixed reference x, sampled target on identical nodes",
        "baseline": "exp/2026/09/15/activation-direction-smoothness/data/tune-w0; source hashes checked by 20-verify.py",
        "initialization": "B=I all modes; learned s=0, theta=0 (x axis), both trainable; no random seed",
        "optimizer": "raw objective Adam; same lr/betas/epsilon; feasibility projection after each update; no inverse line search",
        "constraints": {
            "unconstrained": "B=I+sym(qxx,qyy,qxy)",
            "contraction_only": "B=I+S, S PSD via eigenvalue clipping",
            "learned_direction": "B=I+s*n(theta)*n(theta)^T; s>=0; theta free",
            "x_contraction": "B=diag(1+s,1), s>=0",
        },
        "thread_env": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    }
    write_json(output / "protocol.json", protocol)
    write_json(output / "control-gates.json", am.gates())
    summaries = []
    for height in map(float, cfg.heights.split(",")):
        for mode in cfg.modes.split(","):
            summaries.append(run_case(cfg, mesh, mode, height, output))
            write_json(output / "summary.json", summaries)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileGradientStudy)
