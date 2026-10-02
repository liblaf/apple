"""Continue original Adam through forward nonconvergence using its last iterate."""

# ruff: noqa: PLR0915

from __future__ import annotations

import collections
import csv
import hashlib
import importlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import activation_models as am
import inexact_forward as forward
import numpy as np
import pydantic_settings as ps
import scipy.linalg as la
import study

from liblaf import cherries

runner = importlib.import_module("10-run")
LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
MODE = "unconstrained"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("120-inexact-continuation")
    base_case: Path = Path("tune-w0/h200-unconstrained-w0")
    height: float = 0.2
    max_updates: int = 1200
    forward_tolerance: float = 1e-10
    forward_max_iterations: int = 250
    history_stride: int = 10
    reset_forward_on_failure: bool = False


def evaluate(mesh: Any, q: np.ndarray, seed: np.ndarray, cfg: Config):
    B = study.matrices(mesh, q, MODE)
    outcome = forward.solve(
        mesh,
        B,
        seed,
        tolerance=cfg.forward_tolerance,
        max_iterations=cfg.forward_max_iterations,
    )
    state = outcome.state
    loss, g_u, _ = study.ph.loss(mesh, state.u, cfg.height, "l2")
    gradient, adjoint_residual = study.ph.control_gradient(mesh, state, B, g_u, MODE)
    assert np.all(np.isfinite(gradient))
    values = study.ph.diagnostics(mesh, state, q, MODE, cfg.height)
    R, _ = am.smoothness(B[mesh.muscle], np.asarray(mesh.edges))
    eig = np.linalg.eigvalsh(B[mesh.muscle])
    values.update(
        {
            "raw_loss": loss,
            "normalized_loss": loss / cfg.height**2,
            "objective": loss,
            "objective_normalized": loss / cfg.height**2,
            "roughness": R,
            "tensor_neighbor_rms": float(np.sqrt(R)),
            "forward_converged": outcome.converged,
            "forward_termination": outcome.termination,
            "forward_iterations": state.iterations,
            "forward_energy": state.energy,
            "gradient_is_equilibrium_implicit": outcome.converged,
            "adjoint_relative_residual": adjoint_residual,
            "gradient_inf": float(np.linalg.norm(gradient, np.inf)),
            "gradient_rms": float(np.linalg.norm(gradient) / np.sqrt(len(q))),
            "min_eigenvalue_B": float(eig.min()),
            "min_det_B": float(np.linalg.det(B[mesh.muscle]).min()),
            "nonpositive_det_B_fraction": float(
                np.mean(np.linalg.det(B[mesh.muscle]) <= 0)
            ),
        }
    )
    return outcome, B, gradient, values


def main(cfg: Config):
    assert cfg.history_stride > 0
    base = GROUP / "data" / cfg.base_case
    base_summary = json.loads((base / "summary.json").read_text())
    assert base_summary["height"] == cfg.height
    assert base_summary["mode"] == MODE
    assert base_summary["smooth_weight"] == 0.0
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    folder = out / base.name
    folder.mkdir()
    mesh = study.ph.build_mesh(100, 10)
    original = json.loads((base.parent / "protocol.json").read_text())
    settings = original["config"]
    source_dir = out / "source"
    source_dir.mkdir()
    hashes = {}
    for source in dict.fromkeys(
        [
            Path(__file__),
            Path(sys.modules["__main__"].__file__).resolve(),
            Path(forward.__file__),
            Path(study.__file__),
            Path(am.__file__),
            Path(study.ph.__file__),
            study.ph.MESH_SOURCE,
            Path(runner.__file__),
        ]
    ):
        key = str(source.relative_to(study.ROOT))
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if source in [
            Path(study.__file__),
            Path(am.__file__),
            Path(study.ph.__file__),
            study.ph.MESH_SOURCE,
        ]:
            assert original["source_sha256"][key] == digest
        hashes[key] = digest
        shutil.copy2(source, source_dir / source.name)
    with np.load(base / "checkpoint.npz") as checkpoint:
        q, m, v = (checkpoint[k].copy() for k in ["controls", "moment", "variance"])
        start_step = int(checkpoint["step"])
        seed = checkpoint["u"].ravel()[mesh.lookup >= 0]
    assert start_step == base_summary["accepted_iterations"]
    _, _, exact_gradient, exact_values = study.evaluate(
        mesh, q, MODE, cfg.height, 0.0, seed
    )
    outcome, B, gradient, values = evaluate(mesh, q, seed, cfg)
    assert outcome.converged
    np.testing.assert_array_equal(gradient, exact_gradient)
    assert values["raw_loss"] == exact_values["raw_loss"]
    protocol = {
        "config": cfg.model_dump(mode="json"),
        "original_protocol": str(base.parent / "protocol.json"),
        "source_sha256": hashes,
        "base_checkpoint_sha256": hashlib.sha256(
            (base / "checkpoint.npz").read_bytes()
        ).hexdigest(),
        "start_step": start_step,
        "forward_failure_policy": "Return last assembled accepted finite Newton iterate and continue Adam; never use a rejected Armijo trial.",
        "next_forward_seed_policy": (
            "After a failed solve, initialize the next forward solve at rest (u=0); after a converged solve, warm-start from its displacement. Preserve activation and Adam state."
            if cfg.reset_forward_on_failure
            else "Warm-start from the last finite iterate after every solve, including failures."
        ),
        "unchanged": "energy, material parameters, L2 objective, zero smoothness, free symmetric activation, full muscle band, original Adam moments and learning-rate schedule, positive-J forward Armijo rule",
        "relaxed": "No outer stop for forward failure, physical J<=1e-6, or a numerical plateau. No inverse backtracking or extra loss/Hessian acceptance gate.",
        "gradient_interpretation": "At nonconverged forward states the adjoint is an approximate off-equilibrium linearization, not the exact gradient of the equilibrium-constrained inverse problem.",
        "invalid_numerics_policy": "Fail visibly on nonfinite state/gradient, inverted accepted iterate, or unusable adjoint; no fabricated gradient.",
        "equilibrium_replay_gradient_max_abs_error": 0.0,
    }
    runner.write_json(out / "protocol.json", protocol)
    beta1, beta2 = settings["beta1"], settings["beta2"]
    rate, decay, eps = (
        settings["learning_rate"],
        settings["lr_decay"],
        settings["epsilon"],
    )
    rows, us, qs, steps, statuses = [], [], [], [], []
    seeds, seed_reset_flags = [], []
    seed_was_reset = False
    reset_count = 0
    counts = collections.Counter()
    start = time.monotonic()
    last_log = start
    for step in range(start_step, cfg.max_updates + 1):
        if step != start_step:
            outcome, B, gradient, values = evaluate(mesh, q, seed, cfg)
        counts[outcome.termination] += 1
        reset_count += int(seed_was_reset)
        state = outcome.state
        reset_next_seed = cfg.reset_forward_on_failure and not outcome.converged
        next_seed = np.zeros_like(state.u) if reset_next_seed else state.u.copy()
        row = {
            "step": step,
            **values,
            "elapsed_seconds": time.monotonic() - start,
            "update_learning_rate": rate * decay**step,
            "forward_seed_inf": float(np.linalg.norm(seed, np.inf)),
            "forward_seed_was_reset": seed_was_reset,
            "next_forward_seed_reset": reset_next_seed,
        }
        rows.append(row)
        full_u = study.ph.unpack(mesh, state.u)
        np.savez_compressed(
            folder / "checkpoint.npz",
            controls=q,
            moment=m,
            variance=v,
            u=full_u,
            B=B,
            step=step,
            forward_converged=outcome.converged,
            forward_residual=values["force_residual_inf"],
            next_forward_seed=next_seed,
            next_forward_seed_reset=reset_next_seed,
        )
        with (folder / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        if step % cfg.history_stride == 0 or step in {start_step, cfg.max_updates}:
            us.append(full_u.copy())
            qs.append(q.copy())
            steps.append(step)
            statuses.append(outcome.converged)
            seeds.append(seed.copy())
            seed_reset_flags.append(seed_was_reset)
            np.savez_compressed(
                folder / "history.npz",
                points=mesh.p,
                triangles=mesh.tri,
                muscle=mesh.muscle,
                top=mesh.top,
                u=np.asarray(us),
                controls=np.asarray(qs),
                steps=np.asarray(steps),
                forward_converged=np.asarray(statuses),
                forward_seeds=np.asarray(seeds),
                forward_seed_was_reset=np.asarray(seed_reset_flags),
                height=cfg.height,
                mode=MODE,
                smooth_weight=0.0,
            )
        runner.write_json(
            out / "progress.json",
            {
                "step": step,
                "normalized_loss": values["normalized_loss"],
                "min_J": values["min_J"],
                "residual": values["force_residual_inf"],
                "forward_converged": outcome.converged,
                "forward_status_counts": dict(counts),
                "forward_resets_used": reset_count,
                "elapsed_seconds": time.monotonic() - start,
            },
        )
        if step <= 265 or step % 50 == 0 or time.monotonic() - last_log > 45:
            LOG.info(
                "step=%d L2/h2=%.7g residual=%.3g minJ=%.3g status=%s",
                step,
                values["normalized_loss"],
                values["force_residual_inf"],
                values["min_J"],
                outcome.termination,
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    k: row[k]
                    for k in [
                        "normalized_loss",
                        "min_J",
                        "force_residual_inf",
                        "forward_converged",
                    ]
                }
            )
            last_log = time.monotonic()
        if step == cfg.max_updates:
            break
        counter = step + 1
        m = beta1 * m + (1 - beta1) * gradient
        v = beta2 * v + (1 - beta2) * gradient**2
        q = am.project(
            q
            - rate
            * decay**step
            * (m / (1 - beta1**counter))
            / (np.sqrt(v / (1 - beta2**counter)) + eps),
            MODE,
        )
        assert np.all(np.isfinite(q))
        if step == start_step:
            with np.load(base / "failed-proposal.npz") as failed:
                for actual, expected in [
                    (q, failed["controls"]),
                    (m, failed["moment"]),
                    (v, failed["variance"]),
                ]:
                    np.testing.assert_array_equal(actual, expected)
        seed = next_seed
        seed_was_reset = reset_next_seed
    eigenvalue = float(
        la.eigh(
            ((state.hessian + state.hessian.T) * 0.5).toarray(),
            subset_by_index=[0, 0],
            eigvals_only=True,
            driver="evr",
        )[0]
    )
    summary = {
        "name": base.name,
        "mode": MODE,
        "height": cfg.height,
        "smooth_weight": 0.0,
        "control_dofs": len(q),
        "start_step": start_step,
        "completed_updates": step,
        "accepted_iterations": step,
        "forward_status_counts": dict(counts),
        "forward_resets_used": reset_count,
        "final_next_seed_reset": reset_next_seed,
        "initial": rows[0],
        "final": rows[-1],
        "failure": None
        if outcome.converged
        else {
            "reason": "final forward iterate not converged",
            "residual": values["force_residual_inf"],
        },
        "outer_budget_complete": step == cfg.max_updates,
        "final_forward_converged": outcome.converged,
        "final_smallest_hessian_eigenvalue": eigenvalue,
        "inverse_stationarity": False,
        "interpretation": "Continuation uses approximate adjoints at failed forward states; losses of those states are not equilibrium-fit values.",
        "wall_seconds": time.monotonic() - start,
    }
    runner.write_json(folder / "summary.json", summary)
    runner.write_json(out / "summary.json", [summary])
    LOG.info(
        "Finished %d updates; final forward converged=%s, statuses=%s",
        step,
        outcome.converged,
        dict(counts),
    )


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileActivationStudy)
