"""Run the complete sequential MouthOpen activation restriction chain."""

# ruff: noqa: C901, E402, PLR0912, PLR0915
from __future__ import annotations

import copy
import datetime as dt
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

GROUP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mouthopen_reuse_56", GROUP / "src/56-fit-mouthopen-reuse.py"
)
assert spec is not None
assert spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = adapter
spec.loader.exec_module(adapter)
FIT, base = adapter.FIT, adapter.FIT.base

from activation_models import (
    controls_from_matrix,
    learned_controls_from_fixed,
    matrices,
    project_,
)
from experiment import Profile
from run_support import (
    NUMERICAL_FAILURES,
    _check_usable,
    append_jsonl,
    metrics,
    release_zero_axes,
    save_state,
)
from stress_study import L_REF_MM, SMOOTH_LENGTH_M, StressStudy

from liblaf import cherries

LOG = logging.getLogger(__name__)
STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")


class Config(FIT.Config):
    output: Path = Path("70-mouthopen-four-stage")
    search_shift_policy: str = "reuse"


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def save_optimizer(
    path: Path,
    q: torch.Tensor,
    axes: torch.Tensor | None,
    adam: torch.optim.Adam,
    seed: np.ndarray,
    attempt: int,
    accepted_attempt: int,
    mode: str,
) -> None:
    temporary = path.with_suffix(".tmp.pt")
    torch.save(
        {
            "q": q.detach().cpu(),
            "fixed_axes": None if axes is None else axes.detach().cpu(),
            "optimizer": adam.state_dict(),
            "u": seed,
            "attempt": attempt,
            "accepted_attempt": accepted_attempt,
            "mode": mode,
            "activation_model": "strain",
        },
        temporary,
    )
    temporary.replace(path)


def verify_frozen_sources(folder: Path) -> None:
    records = json.loads((folder / "source-manifest.json").read_text())
    for item in records.values():
        assert base.record(Path(item["path"]))["sha256"] == item["sha256"]
        assert base.record(Path(item["source"]))["sha256"] == item["sha256"]


def initialize(
    mode: str,
    parent_path: Path,
    n_active: int,
) -> tuple[torch.Tensor, torch.Tensor | None, np.ndarray, torch.Tensor]:
    with np.load(parent_path) as parent:
        if mode == "symmetric6":
            assert float(parent["fraction"]) == 1.0
            previous = torch.zeros((n_active, 3, 3))
            q, axes = controls_from_matrix(previous, mode)
            seed = parent["displacement"].copy()
        else:
            expected_parent_mode = STAGES[STAGES.index(mode) - 1]
            assert str(parent["mode"]) == expected_parent_mode
            previous = torch.as_tensor(parent["S"].copy())
            seed = parent["u"].copy()
            assert bool(parent["solver_valid"])
            if mode == "rankone_learned":
                assert str(parent["mode"]) == "rankone_fixed"
                q = learned_controls_from_fixed(
                    torch.as_tensor(parent["q"].copy()),
                    torch.as_tensor(parent["fixed_axes"].copy()),
                )
                axes = None
                torch.testing.assert_close(
                    matrices(q, mode), previous, rtol=2e-14, atol=2e-14
                )
            else:
                q, axes = controls_from_matrix(previous, mode)
    return q, axes, seed, previous


def endpoint_diagnostic(
    study: StressStudy,
    folder: Path,
    mode: str,
    q: torch.Tensor,
    axes: torch.Tensor | None,
    seed: np.ndarray,
    cfg: Config,
    summary: dict,
) -> None:
    try:
        result = study.evaluate(
            q,
            mode,
            axes,
            seed,
            cfg.normal_weight,
            cfg.smooth_weight,
            component_gradients=True,
        )
        assert result["solver_valid"]
    except NUMERICAL_FAILURES as error:
        summary["gradient_diagnostic_failure"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        base.write(
            folder / "gradient-balance.json",
            {
                "status": "unavailable",
                "checkpoint": summary["final_checkpoint"],
                "failure": summary["gradient_diagnostic_failure"],
            },
        )
    else:
        np.savez_compressed(
            folder / "gradient-components.npz",
            l2_tensor_gradient=result["l2_tensor_gradient"].numpy(force=True),
            regularizer_tensor_gradient=result["regularizer_tensor_gradient"].numpy(
                force=True
            ),
            active_volume_weights=study.active_weights,
            smooth_weight=cfg.smooth_weight,
        )
        base.write(
            folder / "gradient-balance.json",
            {
                "status": "available",
                **metrics(result),
                "checkpoint": summary["final_checkpoint"],
                "components": base.record(folder / "gradient-components.npz"),
                "metric": "full symmetric S covectors; dual normalized effective-active-volume norm",
                "ratio": "norm(eta dR/dS) / norm(dL2/dS); normal term excluded",
                "displacement_change_from_saved_m": float(
                    np.max(np.abs(result["u"] - seed))
                ),
                "forward": result["forward"],
                "l2_adjoint": result["l2_adjoint"],
                "adjoint": result["adjoint"],
            },
        )


def run_stage(
    study: StressStudy,
    folder: Path,
    mode: str,
    parent_path: Path,
    cfg: Config,
    chain: dict,
    chain_out: Path,
) -> dict:
    folder.mkdir(exist_ok=False)
    controls, axes, seed, previous = initialize(
        mode, parent_path, len(study.physics.ids)
    )
    q = torch.nn.Parameter(controls.detach().clone())
    axes = None if axes is None else axes.detach().clone()
    adam = torch.optim.Adam([q], lr=cfg.learning_rate, eps=1e-8, betas=(0.9, 0.999))
    assert len(adam.state) == 0
    initial_s = matrices(q, mode, axes).detach()
    np.savez_compressed(
        folder / "initialization.npz",
        q=q.detach().cpu().numpy(),
        fixed_axes=np.empty((0, 3)) if axes is None else axes.detach().cpu().numpy(),
        S=initial_s.cpu().numpy(),
        B=initial_s.cpu().numpy() + np.eye(3),
        u_seed=seed,
        mode=mode,
    )
    save_optimizer(folder / "optimizer-initial.pt", q, axes, adam, seed, 0, -1, mode)
    summary: dict[str, Any] = {
        "schema": "mouthopen-four-stage-fit-v1",
        "mode": mode,
        "status": "running",
        "activation_model": "strain",
        "started_at": now(),
        "config": cfg.model_dump(mode="json"),
        "initialization": {
            "fresh_adam": True,
            "parent_checkpoint": base.record(parent_path),
            "controls": base.record(folder / "initialization.npz"),
            "optimizer": base.record(folder / "optimizer-initial.pt"),
            "projection_max_abs_change": float((initial_s - previous).abs().max()),
            "projection_frobenius_rms": float(
                (initial_s - previous).square().sum((-2, -1)).mean().sqrt()
            ),
            "zero_tensor_cells": int((initial_s.square().sum((-2, -1)) == 0).sum()),
            "zero_axis_release": "after strict evaluations; only exactly-zero amplitudes, total tensor gradient",
        },
        "parent_checkpoint": base.record(parent_path),
        "inputs": chain["inputs"],
        "chain_mesh": base.record(chain_out / "mesh.npz"),
        "material_spec": study.physics.material_spec,
        "solver_policy": chain["solver_policy"],
        "attempted_updates": 0,
        "optimizer_updates": 0,
        "skipped_updates": 0,
        "inversion_limit_cells": int(
            cfg.inversion_fraction_limit * study.physics.mesh.n_cells
        ),
    }
    base.freeze(folder)
    base.write(folder / "summary.json", summary)
    result = None
    best = float("inf")
    accepted_attempt = -1
    started = time.perf_counter()
    for attempt in range(cfg.steps + 1):
        previous_q = q.detach().clone()
        previous_adam = copy.deepcopy(adam.state_dict())
        try:
            if result is not None:
                q.grad = result["gradient"].clone()
                adam.step()
                q.grad = None
                project_(q, mode)
            candidate = study.evaluate(
                q,
                mode,
                axes,
                seed,
                cfg.normal_weight,
                cfg.smooth_weight,
                component_gradients=False,
            )
            assert candidate["solver_valid"]
            _check_usable(q, candidate)
        except NUMERICAL_FAILURES as error:
            with torch.no_grad():
                q.copy_(previous_q)
            q.grad = None
            adam.load_state_dict(previous_adam)
            if result is not None:
                for group in adam.param_groups:
                    group["lr"] *= 0.5
                summary["skipped_updates"] += 1
            summary["last_failure"] = {
                "attempt": attempt,
                "type": type(error).__name__,
                "message": str(error),
                "receipt": getattr(error, "receipt", None),
            }
            append_jsonl(
                folder / "proposals.jsonl",
                {**summary["last_failure"], "accepted": False},
            )
            LOG.warning("%s attempt %d rejected: %s", mode, attempt, error)
            if result is None:
                summary["status"] = "initial_gradient_failed"
                base.write(folder / "summary.json", summary)
                break
        else:
            candidate["zero_amplitude_axis_updates"] = release_zero_axes(
                q, mode, candidate, adam
            )
            if result is not None:
                summary["optimizer_updates"] += 1
            result = candidate
            seed = result["u"].copy()
            accepted_attempt = attempt
            row = {
                "attempt": attempt,
                "optimizer_updates": summary["optimizer_updates"],
                "elapsed_seconds": time.perf_counter() - started,
                "learning_rate": adam.param_groups[0]["lr"],
                **metrics(result),
            }
            row["orientation_valid"] = row["inverted_all_cells"] == 0
            cherries.set_step(STAGES.index(mode) * (cfg.steps + 1) + attempt)
            cherries.log_metrics(
                {
                    f"{mode}/{k}": v
                    for k, v in row.items()
                    if isinstance(v, (int, float))
                }
            )
            append_jsonl(folder / "trace.jsonl", row)
            append_jsonl(
                folder / "solver-receipts.jsonl",
                {
                    "attempt": attempt,
                    "forward": result["forward"],
                    "adjoint": result["adjoint"],
                },
            )
            save_state(folder / "last.npz", q, axes, result, mode, attempt, "strain")
            if attempt == 0 or attempt % 50 == 0 or attempt == cfg.steps:
                save_state(
                    folder / f"step-{attempt:04d}.npz",
                    q,
                    axes,
                    result,
                    mode,
                    attempt,
                    "strain",
                )
            if row["objective"] < best:
                best = row["objective"]
                summary["best_attempt"] = attempt
                save_state(
                    folder / "best-objective.npz",
                    q,
                    axes,
                    result,
                    mode,
                    attempt,
                    "strain",
                )
            summary["last_metrics"] = row
            if attempt == 0:
                summary["initial_metrics"] = row
            LOG.info(
                "%s %d/%d: fit %.4f mm, normal %.4f deg, %d inverted",
                mode,
                attempt,
                cfg.steps,
                row["fit_rms_mm"],
                row["normal_angle_rms_deg"],
                row["inverted_all_cells"],
            )
        if result is not None:
            save_optimizer(
                folder / "optimizer-latest.pt",
                q,
                axes,
                adam,
                seed,
                attempt,
                accepted_attempt,
                mode,
            )
        summary["attempted_updates"] = attempt
        summary["elapsed_seconds"] = time.perf_counter() - started
        base.write(folder / "summary.json", summary)
        chain["stages"][mode] = summary
        chain["updated_at"] = now()
        base.write(chain_out / "chain-status.json", chain)
    else:
        summary["status"] = "completed_attempt_budget"
    if result is not None:
        summary["final_checkpoint"] = base.record(folder / "last.npz")
        summary["orientation_valid"] = result["inverted_all_cells"] == 0
        summary["solver_converged"] = result["solver_valid"]
        endpoint_diagnostic(study, folder, mode, q, axes, seed, cfg, summary)
    summary["ended_at"] = now()
    summary["elapsed_seconds"] = time.perf_counter() - started
    base.freeze(folder)
    verify_frozen_sources(folder)
    base.write(folder / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    assert cfg.steps == 200
    assert cfg.search_shift_policy == "reuse"
    assert cfg.normal_weight == 1.0
    assert cfg.smooth_weight == 7.2e-6
    out = cherries.output(cfg.output / "chain-status.json", mkdir=True).parent
    assert not (out / "protocol.json").exists(), out
    parent_summary_path = cherries.input(cfg.forward / "summary.json")
    parent_path = cherries.input(cfg.forward / "final.npz")
    prepared_path = cherries.input(GROUP / "data/10-mandible/prepared.npz")
    parent = json.loads(parent_summary_path.read_text())
    assert parent["completed_pose_fraction"] == 1.0
    assert base.record(parent_path)["sha256"] == parent["final_checkpoint"]["sha256"]
    with np.load(parent_path) as z, np.load(prepared_path) as prepared:
        pose, pivot = z["pose"].copy(), prepared["pivot"].copy()
        np.testing.assert_array_equal(pose, prepared["pose"])
    adapter.install_reuse_policy()
    FIT.stress_study.FIXTURE = cfg.fixture
    FIT.stress_study.FacePhysics = FIT.make_physics(cfg, pose, pivot)
    study = StressStudy(
        activation_model="strain",
        atol=cfg.force_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_newton_steps=cfg.max_newton_steps,
        newton_linear_max_steps=cfg.linear_max_steps,
    )
    assert len(study.physics.ids) == 288172
    assert len(study.physics.graph[0]) == 501313
    assert study.physics.diff.require_convergence
    assert study.physics.forward.optimizer.require_convergence
    study.save_geometry(out / "mesh.npz")
    base.freeze(out)
    chain = {
        "schema": "mouthopen-four-stage-chain-v1",
        "status": "running",
        "started_at": now(),
        "updated_at": now(),
        "stage_sequence": list(STAGES),
        "current_stage": None,
        "completed": [],
        "stages": {},
        "inputs": {
            path.name: base.record(path)
            for path in (
                parent_summary_path,
                parent_path,
                prepared_path,
                cfg.fixture / "volume.vtu",
                cfg.fixture / "skin.vtp",
            )
        },
        "config": cfg.model_dump(mode="json"),
        "solver_policy": "strict primal and unshifted adjoint; Newton shift reuse; rollback+halve lr on numerical failure; limited finite inversions; contact off",
    }
    base.write(
        out / "protocol.json",
        {
            **chain,
            "mesh": base.record(out / "mesh.npz"),
            "source_manifest": base.record(out / "source-manifest.json"),
            "steps_per_stage": cfg.steps,
            "fresh_adam_per_stage": True,
            "transfers": [
                "zero S at full-jaw equilibrium",
                "PSD projection of stage-1 endpoint",
                "largest nonnegative eigenmode of stage-2 endpoint",
                "parent amplitude and axes, including zero-amplitude axes",
            ],
            "l_ref_mm": L_REF_MM,
            "smooth_length_m": SMOOTH_LENGTH_M,
            "material_spec": study.physics.material_spec,
            "runtime": {
                "python": sys.version,
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        },
    )
    stat = Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(") ", 1)[1].split()
    base.write(
        out / "process.json",
        {
            "pid": os.getpid(),
            "startticks": int(stat[19]),
            "started_at": now(),
            "cmdline": Path(f"/proc/{os.getpid()}/cmdline")
            .read_bytes()
            .decode()
            .strip("\0")
            .split("\0"),
            "script": base.record(Path(__file__)),
        },
    )
    try:
        for mode in STAGES:
            chain["current_stage"] = mode
            base.write(out / "chain-status.json", chain)
            verify_frozen_sources(out)
            summary = run_stage(study, out / mode, mode, parent_path, cfg, chain, out)
            chain["stages"][mode] = summary
            if summary["status"] != "completed_attempt_budget":
                chain["status"] = "blocked_by_stage_failure"
                break
            assert summary["attempted_updates"] == cfg.steps
            assert (
                summary["optimizer_updates"] + summary["skipped_updates"] == cfg.steps
            )
            chain["completed"].append(mode)
            parent_path = out / mode / "last.npz"
        else:
            chain["status"] = "completed_attempt_budgets"
    except BaseException as error:
        chain["status"] = "failed"
        chain["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        chain["ended_at"] = now()
        base.write(out / "chain-status.json", chain)
        base.freeze(out)
        verify_frozen_sources(out)
    cherries.log_metrics(
        {
            "completed_stages": len(chain["completed"]),
            "attempts": sum(s["attempted_updates"] for s in chain["stages"].values()),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
