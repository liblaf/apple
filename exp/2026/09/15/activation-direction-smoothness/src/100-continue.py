"""Resume the stopped free-activation case using backtracked Adam proposals."""

# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pydantic_settings as ps
import scipy.linalg as la
import study

from liblaf import cherries

runner = importlib.import_module("10-run")
LOG = logging.getLogger(__name__)
GROUP = Path(__file__).resolve().parents[1]
BASE = GROUP / "data/tune-w0/h200-unconstrained-w0"
MODE = "unconstrained"
HEIGHT = 0.2


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("100-backtracked-continuation")
    max_updates: int = 1200
    max_backtracks: int = 20
    minimum_J: float = 1e-6


def pack(mesh: Any, full_u: np.ndarray):
    return full_u.ravel()[mesh.lookup >= 0]


def stable(state: Any):
    hessian = ((state.hessian + state.hessian.T) * 0.5).toarray()
    try:
        la.cholesky(hessian, lower=True, check_finite=True)
    except la.LinAlgError:
        return False
    return True


def save_state(
    output: Path,
    mesh: Any,
    step: int,
    q: np.ndarray,
    m: np.ndarray,
    v: np.ndarray,
    state: Any,
    B: np.ndarray,
):
    np.savez_compressed(
        output / "checkpoint.npz",
        controls=q,
        moment=m,
        variance=v,
        u=study.ph.unpack(mesh, state.u),
        B=B,
        step=step,
    )


def main(cfg: Config):
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    mesh = study.ph.build_mesh(100, 10)
    protocol = json.loads((BASE.parent / "protocol.json").read_text())
    settings = protocol["config"]
    for source in (
        Path(study.__file__),
        Path(am.__file__),
        Path(study.ph.__file__),
        study.ph.MESH_SOURCE,
    ):
        key = str(source.relative_to(study.ROOT))
        assert (
            hashlib.sha256(source.read_bytes()).hexdigest()
            == protocol["source_sha256"][key]
        )
    source_dir = output / "source"
    source_dir.mkdir()
    for source in (
        Path(__file__),
        Path(runner.__file__),
        Path(study.__file__),
        Path(am.__file__),
        Path(study.ph.__file__),
        study.ph.MESH_SOURCE,
    ):
        shutil.copy2(source, source_dir / source.name)
    with np.load(BASE / "checkpoint.npz") as checkpoint:
        q, m, v = (checkpoint[k].copy() for k in ("controls", "moment", "variance"))
        step = int(checkpoint["step"])
        seed = pack(mesh, checkpoint["u"])
    assert step == 262
    start = time.monotonic()
    state, B, gradient, values = study.evaluate(mesh, q, MODE, HEIGHT, 0.0, seed)
    assert stable(state)
    beta1, beta2 = settings["beta1"], settings["beta2"]
    rate, decay, epsilon = (
        settings["learning_rate"],
        settings["lr_decay"],
        settings["epsilon"],
    )
    m_next = beta1 * m + (1 - beta1) * gradient
    v_next = beta2 * v + (1 - beta2) * gradient**2
    proposed = am.project(
        q
        - rate
        * decay**step
        * (m_next / (1 - beta1 ** (step + 1)))
        / (np.sqrt(v_next / (1 - beta2 ** (step + 1))) + epsilon),
        MODE,
    )
    with np.load(BASE / "failed-proposal.npz") as failed:
        for actual, expected in [
            (proposed, failed["controls"]),
            (m_next, failed["moment"]),
            (v_next, failed["variance"]),
        ]:
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        replay_error = float(np.max(np.abs(proposed - failed["controls"])))
    runner.write_json(
        output / "protocol.json",
        {
            "config": cfg.model_dump(mode="json"),
            "base_checkpoint": str(BASE / "checkpoint.npz"),
            "base_checkpoint_sha256": hashlib.sha256(
                (BASE / "checkpoint.npz").read_bytes()
            ).hexdigest(),
            "base_protocol": str(BASE.parent / "protocol.json"),
            "physics_source_hashes_match_original": True,
            "recreated_failed_proposal_max_abs_error": replay_error,
            "changes": "On failed forward/geometry/stability/loss gate halve the inverse Adam proposal from the same valid state. Moments update once per accepted iteration. Stop if direction is non-descending or all trial fractions are rejected.",
            "acceptance": "Same 1e-10 forward tolerance; J>1e-6 (original diagnostic guard); positive Hessian Cholesky; outer Armijo with c1=1e-4 on the same raw L2. No new objective term.",
            "same_parameters": {
                k: settings[k]
                for k in ["learning_rate", "lr_decay", "beta1", "beta2", "epsilon"]
            },
        },
    )
    rows = [
        {"step": step, **values, "alpha": 0.0, "backtracks": 0, "elapsed_seconds": 0.0}
    ]
    qs, us, steps = [q.copy()], [study.ph.unpack(mesh, state.u)], [step]
    trials = []
    stop_reason = "iteration_budget"
    save_state(output, mesh, step, q, m, v, state, B)
    while step < cfg.max_updates:
        counter = step + 1
        m_next = beta1 * m + (1 - beta1) * gradient
        v_next = beta2 * v + (1 - beta2) * gradient**2
        delta = (
            -rate
            * decay**step
            * (m_next / (1 - beta1**counter))
            / (np.sqrt(v_next / (1 - beta2**counter)) + epsilon)
        )
        slope = float(gradient @ delta)
        if slope >= 0:
            stop_reason = "Adam momentum direction is not descending; reducing its length cannot enforce Armijo"
            break
        accepted = False
        for backtrack in range(cfg.max_backtracks + 1):
            alpha = 0.5**backtrack
            proposal = am.project(q + alpha * delta, MODE)
            trial = {"proposed_step": counter, "backtrack": backtrack, "alpha": alpha}
            try:
                trial_state, trial_B, trial_gradient, trial_values = study.evaluate(
                    mesh, proposal, MODE, HEIGHT, 0.0, state.u
                )
            except study.ph.ForwardSolveError as exc:
                trial.update(accepted=False, reason=str(exc))
                trials.append(trial)
                continue
            assert np.all(np.isfinite(trial_gradient))
            trial.update(
                min_J=trial_values["min_J"],
                normalized_loss=trial_values["normalized_loss"],
                residual=trial_values["force_residual_inf"],
            )
            if trial_values["min_J"] <= cfg.minimum_J:
                reason = "minimum physical J guard"
            elif (
                trial_values["objective"]
                > values["objective"] + 1e-4 * alpha * slope + 1e-15
            ):
                reason = "outer Armijo loss condition"
            elif not stable(trial_state):
                reason = "nonpositive equilibrium Hessian"
            else:
                reason = "accepted"
                accepted = True
            trial.update(accepted=accepted, reason=reason)
            trials.append(trial)
            if accepted:
                q, m, v = proposal, m_next, v_next
                state, B, gradient, values = (
                    trial_state,
                    trial_B,
                    trial_gradient,
                    trial_values,
                )
                step = counter
                break
        if not accepted:
            stop_reason = "no accepted inverse trial within backtracking budget"
            break
        rows.append(
            {
                "step": step,
                **values,
                "alpha": alpha,
                "backtracks": backtrack,
                "elapsed_seconds": time.monotonic() - start,
            }
        )
        qs.append(q.copy())
        us.append(study.ph.unpack(mesh, state.u))
        steps.append(step)
        save_state(output, mesh, step, q, m, v, state, B)
        runner.write_json(
            output / "progress.json",
            {
                "step": step,
                "normalized_loss": values["normalized_loss"],
                "min_J": values["min_J"],
                "alpha": alpha,
            },
        )
        if step <= 270 or step % 20 == 0:
            LOG.info(
                "step=%d L2/h2=%.9f minJ=%.5g alpha=%.5g",
                step,
                values["normalized_loss"],
                values["min_J"],
                alpha,
            )
            cherries.set_step(step)
            cherries.log_metrics(
                {
                    "normalized_loss": values["normalized_loss"],
                    "min_J": values["min_J"],
                    "alpha": alpha,
                }
            )
        if (
            np.array_equal(us[-1], us[-2])
            and rows[-1]["raw_loss"] == rows[-2]["raw_loss"]
        ):
            stop_reason = (
                "no displacement or loss change resolved at the forward tolerance"
            )
            break
    with (output / "trace.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    runner.write_json(output / "trials.json", trials)
    np.savez_compressed(
        output / "history.npz",
        points=mesh.p,
        triangles=mesh.tri,
        muscle=mesh.muscle,
        top=mesh.top,
        u=np.asarray(us),
        controls=np.asarray(qs),
        steps=np.asarray(steps),
    )
    eigenvalue = float(
        la.eigh(
            ((state.hessian + state.hessian.T) * 0.5).toarray(),
            subset_by_index=[0, 0],
            eigvals_only=True,
            driver="evr",
        )[0]
    )
    assert eigenvalue > 0
    summary = {
        "mode": MODE,
        "height": HEIGHT,
        "smooth_weight": 0.0,
        "start_step": 262,
        "last_step": step,
        "accepted_new_updates": step - 262,
        "stop_reason": stop_reason,
        "initial": rows[0],
        "final": rows[-1],
        "smallest_hessian_eigenvalue": eigenvalue,
        "rejected_trials": sum(not t["accepted"] for t in trials),
        "wall_seconds": time.monotonic() - start,
        "inverse_converged": False,
    }
    runner.write_json(output / "summary.json", summary)
    LOG.info("Continuation finished: %s", summary)


if __name__ == "__main__":
    cherries.main(main, profile=runner.ProfileActivationStudy)
