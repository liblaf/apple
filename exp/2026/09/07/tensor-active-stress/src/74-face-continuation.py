"""Checkpoint-preserving face continuations with explicit optimizer settings."""

# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import copy
import csv
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from continuation_helpers import (
    BASE,
    adam_metrics,
    initial_replay_metrics,
    load_checkpoint,
    projected_gradient_metrics,
    rms,
    tensor_rms_mpa,
    verify_parent_sources,
)
from experiment_profile import ProfileCometNoCommit
from face_physics import FacePhysics, configure
from tensor_controls import project

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
COMPLETED = False
LOG = logging.getLogger(__name__)
WEAK_SMOOTHNESS_WEIGHT = 1.4762928671047126
FULL_SMOOTHNESS_WEIGHT = 5.9051714684188505
RANK_WEIGHT = 119.62893380703123


class Config(BASE.Config):
    output_dir: Path = cherries.output("74-face-continuation", mkdir=True)
    resume: Path = HERE / "data/21-psd/optimizer-latest.pt"
    steps: int = 16
    checkpoint_interval: int = 16
    phase: Literal[
        "baseline-probe", "calibrated-probe", "fit-continuation", "regularization"
    ] = "baseline-probe"
    calibration_receipt: Path | None = None
    projected_gradient_eta: float = 1.0
    resume_intermediate: bool = False


def run(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    assert cfg.model == "tensor"
    assert cfg.steps > 0
    assert 0 < cfg.checkpoint_interval <= 16
    assert cfg.learning_rate > 0
    assert cfg.adam_eps > 0
    assert cfg.projected_gradient_eta == 1.0
    assert min(cfg.smoothness_weight, cfg.rank_weight, cfg.magnitude_weight) >= 0
    if cfg.resume_intermediate:
        assert cfg.phase == "regularization"
    if cfg.phase != "regularization":
        assert cfg.smoothness_weight == cfg.rank_weight == cfg.magnitude_weight == 0
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"output directory must be empty: {out}"
    state = load_checkpoint(cfg.resume, cfg, allow_intermediate=cfg.resume_intermediate)
    start_step = int(state["step"])
    parent = json.loads((cfg.resume.parent / "provenance.json").read_text())
    parent_summary = json.loads((cfg.resume.parent / "summary.json").read_text())
    assert parent_summary["status"] in {
        "completed_fixed_budget",
        "completed_fixed_budget_continuation",
    }
    parent_endpoint = state["parent_endpoint"]
    if not cfg.resume_intermediate:
        assert parent_summary["primary_endpoint"]["step"] == start_step
        assert parent_summary["primary_endpoint"]["solver_valid"] is True
    parent_weights = tuple(
        state["config"][key]
        for key in ("smoothness_weight", "rank_weight", "magnitude_weight")
    )
    if cfg.phase != "regularization":
        assert parent_weights == (0, 0, 0)
    else:
        parent_smoothness, parent_rank, parent_magnitude = parent_weights
        assert parent_rank == parent_magnitude == 0
        assert cfg.magnitude_weight == 0
        if parent_smoothness == 0:
            assert cfg.rank_weight == 0
            assert cfg.smoothness_weight in {
                0,
                WEAK_SMOOTHNESS_WEIGHT,
                FULL_SMOOTHNESS_WEIGHT,
            }
        else:
            assert parent_smoothness in {
                WEAK_SMOOTHNESS_WEIGHT,
                FULL_SMOOTHNESS_WEIGHT,
            }
            assert cfg.smoothness_weight == parent_smoothness
            assert cfg.rank_weight in {0, RANK_WEIGHT}
    if cfg.phase in {"baseline-probe", "calibrated-probe"}:
        assert start_step == 64
        assert cfg.steps == 16
    elif cfg.phase == "fit-continuation":
        assert (start_step, cfg.steps) in {(80, 64), (144, 64), (208, 48)}
    else:
        assert cfg.steps == 16
        assert cfg.checkpoint_interval == 1
    if cfg.phase == "calibrated-probe":
        assert cfg.calibration_receipt is not None
    if cfg.phase in {"fit-continuation", "regularization"}:
        assert cfg.learning_rate == state["config"]["learning_rate"]
        assert cfg.adam_eps == state["config"]["adam_eps"]
    parent_uses_calibrated_settings = (
        state["config"]["learning_rate"] != 0.3 or state["config"]["adam_eps"] != 0.01
    )
    if cfg.phase == "fit-continuation" and parent_uses_calibrated_settings:
        assert cfg.calibration_receipt is not None
    BASE.write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = BASE.archive(out, cfg)
    verify_parent_sources(parent, provenance)
    calibration = None
    frozen_gradient = None
    frozen_gradient_record = None
    if cfg.calibration_receipt is not None:
        calibration = json.loads(cfg.calibration_receipt.read_text())
        assert calibration["status"] == "frozen_before_probe_runs"
        assert cfg.adam_eps == calibration["selected"]["adam_eps"]
        assert cfg.learning_rate == calibration["selected"]["learning_rate"]
        if cfg.phase == "calibrated-probe":
            assert calibration["parent_checkpoint"]["sha256"] == BASE.sha256(cfg.resume)
            gradient_path = cfg.calibration_receipt.with_name("initial-gradient.npz")
            with np.load(gradient_path) as saved:
                assert int(saved["step"]) == start_step
                assert np.array_equal(saved["q"], state["q"].numpy())
                q_sha256 = hashlib.sha256(saved["q"].tobytes()).hexdigest()
                gradient = saved["gradient"].copy()
            assert q_sha256 == calibration["initial_control_sha256"]
            assert gradient.shape == state["q"].shape
            assert np.isfinite(gradient).all()
            gradient_sha256 = hashlib.sha256(gradient.tobytes()).hexdigest()
            assert gradient_sha256 == calibration["gradient_sha256"]
            frozen_gradient = torch.as_tensor(gradient, device="cuda")
            frozen_gradient_record = {
                "path": str(gradient_path.resolve()),
                "sha256": BASE.sha256(gradient_path),
                "gradient_sha256": gradient_sha256,
                "q_sha256": q_sha256,
                "step": start_step,
                "q_matches_parent_checkpoint": True,
            }
    if cfg.phase == "baseline-probe":
        assert cfg.learning_rate == state["config"]["learning_rate"]
        assert cfg.adam_eps == state["config"]["adam_eps"]
    resume_receipt = {
        "parent_checkpoint": {
            "path": str(cfg.resume.resolve()),
            "sha256": BASE.sha256(cfg.resume),
        },
        "parent_summary": {
            "path": str((cfg.resume.parent / "summary.json").resolve()),
            "sha256": BASE.sha256(cfg.resume.parent / "summary.json"),
        },
        "parent_global_step": start_step,
        "requested_additional_updates": cfg.steps,
        "requested_final_global_step": start_step + cfg.steps,
        "controls_sha256": hashlib.sha256(state["q"].numpy().tobytes()).hexdigest(),
        "seed_displacement_sha256": hashlib.sha256(
            np.asarray(state["u"]).tobytes()
        ).hexdigest(),
        "optimizer_state_policy": "restore moments and counter exactly, then replace only the explicitly configured learning rate and epsilon",
        "old_learning_rate": state["optimizer"]["param_groups"][0]["lr"],
        "old_adam_eps": state["optimizer"]["param_groups"][0]["eps"],
        "new_learning_rate": cfg.learning_rate,
        "new_adam_eps": cfg.adam_eps,
        "checkpoint_evidence": state["checkpoint_evidence"],
        "calibration_receipt": {
            "path": str(cfg.calibration_receipt.resolve()),
            "sha256": BASE.sha256(cfg.calibration_receipt),
        }
        if cfg.calibration_receipt
        else None,
    }
    if frozen_gradient_record is not None:
        resume_receipt["frozen_initial_gradient"] = frozen_gradient_record
    BASE.write_json(out / "resume.json", resume_receipt)
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="tensor")
    assert len(physics.ids) == 288235
    assert physics.n_regions == 103
    assert np.array_equal(physics.ids, state["active_ids"])
    objective = BASE.Objective(physics, cfg)
    q = torch.nn.Parameter(state["q"].to(device="cuda").clone())
    initial_q = q.detach().clone()
    optimizer = torch.optim.Adam([q], lr=cfg.learning_rate, eps=cfg.adam_eps)
    optimizer.load_state_dict(state["optimizer"])
    optimizer.param_groups[0]["lr"] = cfg.learning_rate
    optimizer.param_groups[0]["eps"] = cfg.adam_eps
    assert int(optimizer.state[q]["step"]) == start_step
    seed = np.asarray(state["u"]).copy()
    maximum = cfg.stress_cap_mpa / cfg.stress_reference_mpa
    trace: list[dict] = []
    best = last = None
    started = time.perf_counter()
    cumulative = 0.0
    expected_first_q = None
    first_update_replay = None
    frozen_gradient_comparison = None
    projection = {
        "projection_rms": 0.0,
        "projected_negative_eigenvalue_fraction": 0.0,
        "projected_upper_eigenvalue_fraction": 0.0,
        "unprojected_update_rms": 0.0,
        "actual_update_rms": 0.0,
        "physical_update_rms_mpa": 0.0,
        "physical_update_frobenius_max_mpa": 0.0,
    }
    try:
        for local_step in range(cfg.steps + 1):
            step = start_step + local_step
            cherries.set_step(step)
            current = objective(q, seed)
            assert q.grad is not None
            if local_step == 0 and cfg.phase == "calibrated-probe":
                assert calibration is not None
                assert frozen_gradient is not None
                difference = q.grad.detach() - frozen_gradient
                frozen_rms = rms(frozen_gradient)
                assert frozen_rms > 0
                recomputed_rms = rms(q.grad.detach())
                frozen_gradient_comparison = {
                    "frozen_gradient_rms": frozen_rms,
                    "recomputed_gradient_rms": recomputed_rms,
                    "difference_rms": rms(difference),
                    "difference_max_abs": float(difference.abs().max()),
                    "relative_difference_rms": rms(difference) / frozen_rms,
                    "recomputed_gradient_sha256": hashlib.sha256(
                        q.grad.detach().cpu().numpy().tobytes()
                    ).hexdigest(),
                }
                replay_q = torch.nn.Parameter(q.detach().clone())
                replay_opt = torch.optim.Adam(
                    [replay_q], lr=cfg.learning_rate, eps=cfg.adam_eps
                )
                replay_opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                replay_q.grad = frozen_gradient.clone()
                replay_opt.step()
                project(replay_q, maximum)
                expected_first_q = replay_q.detach().clone()
            scalar = {
                key: value
                for key, value in current.items()
                if key not in {"u", "forward", "adjoint", "component_gradients"}
            }
            stationarity = projected_gradient_metrics(
                q, q.grad, maximum, cfg.projected_gradient_eta
            )
            row = {
                "step": step,
                "local_step": local_step,
                **scalar,
                **BASE.metrics(physics, q, current, cfg),
                **projection,
                **stationarity,
                **adam_metrics(optimizer),
                "cumulative_physical_update_rms_mpa": cumulative,
                "net_physical_update_rms_mpa": tensor_rms_mpa(
                    q.detach() - initial_q, cfg.stress_reference_mpa
                ),
                "projected_gradient_relative_to_start": 1.0
                if not trace
                else stationarity["projected_gradient_mapping_rms"]
                / trace[0]["projected_gradient_mapping_rms"],
                "forward_steps": current["forward"]["steps"],
                "forward_grad_norm": current["forward"]["grad_norm"],
                "solver_valid": True,
                "elapsed_s": time.perf_counter() - started,
            }
            trace.append(row)
            if local_step == 0:
                replay = initial_replay_metrics(row, parent_endpoint)
                BASE.write_json(out / "initial-replay.json", replay)
                assert replay["passed"], replay
            last = current
            with (out / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(trace)
            with (out / "solver-receipts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "local_step": local_step,
                            "forward": current["forward"],
                            "adjoint": current["adjoint"],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            if best is None or row["objective"] < best["objective"]:
                best = dict(row)
                BASE.snapshot(
                    out / "best.npz", physics, q, current, step, cfg, mesh=False
                )
            BASE.snapshot(
                out / "latest.npz", physics, q, current, step, cfg, mesh=False
            )
            if local_step % cfg.checkpoint_interval == 0 or local_step == cfg.steps:
                BASE.snapshot(
                    out / f"step-{step:04d}.npz",
                    physics,
                    q,
                    current,
                    step,
                    cfg,
                    mesh=True,
                )
                torch.save(
                    {
                        "step": step,
                        "q": q.detach().cpu(),
                        "u": current["u"],
                        "optimizer": optimizer.state_dict(),
                        "config": cfg.model_dump(mode="json"),
                    },
                    out / f"optimizer-step-{step:04d}.pt",
                )
                (out / "optimizer-latest.pt").write_bytes(
                    (out / f"optimizer-step-{step:04d}.pt").read_bytes()
                )
            if cfg.record_component_gradients:
                np.savez_compressed(
                    out / f"gradients-{step:04d}.npz", **current["component_gradients"]
                )
            BASE.write_json(out / "progress.json", row)
            cherries.log_metrics(
                {
                    "face": {
                        key: value
                        for key, value in row.items()
                        if isinstance(value, (int, float))
                    }
                }
            )
            output = {
                key: row[key]
                for key in (
                    "step",
                    "local_step",
                    "area_fit_rms_mm",
                    "area_motion_rms_mm",
                    "physical_update_rms_mpa",
                    "projected_gradient_mapping_rms",
                    "inverted_tetrahedra",
                    "forward_steps",
                    "elapsed_s",
                )
            }
            print(json.dumps(output), flush=True)
            LOG.info("Continuation %s", output)
            if local_step == cfg.steps:
                break
            seed = current["u"]
            before = q.detach().clone()
            optimizer.step()
            unprojected_rms = rms(q.detach() - before)
            projection = project(q, maximum)
            delta = q.detach() - before
            physical_step = tensor_rms_mpa(delta, cfg.stress_reference_mpa)
            cumulative += physical_step
            if local_step == 0 and expected_first_q is not None:
                assert calibration is not None
                control_error = float((q.detach() - expected_first_q).abs().max())
                declared_physical = calibration["selected"]["physical_update_rms_mpa"]
                relative_physical_error = abs(physical_step / declared_physical - 1)
                first_update_replay = {
                    "control_max_abs_error": control_error,
                    "control_max_abs_tolerance": 1e-10,
                    "relative_physical_update_error": relative_physical_error,
                    "relative_physical_update_tolerance": 1e-5,
                    "declared_physical_update_rms_mpa": declared_physical,
                    "actual_physical_update_rms_mpa": physical_step,
                    "frozen_gradient": frozen_gradient_record,
                    "frozen_vs_recomputed_gradient": frozen_gradient_comparison,
                }
                assert control_error < 1e-10
                assert relative_physical_error < 1e-5
                expected_first_q = None
            projection.update(
                {
                    "unprojected_update_rms": unprojected_rms,
                    "actual_update_rms": rms(delta),
                    "physical_update_rms_mpa": physical_step,
                    "physical_update_frobenius_max_mpa": float(
                        cfg.stress_reference_mpa * delta.square().sum(-1).sqrt().max()
                    ),
                }
            )
    except BaseException as error:
        BASE.write_json(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "attempted_global_step": start_step + len(trace),
                "last_valid_step": trace[-1]["step"] if trace else None,
                "last_forward": getattr(physics, "last_forward", None),
                "first_update_replay": first_update_replay,
                "frozen_vs_recomputed_gradient": frozen_gradient_comparison,
            },
        )
        with (out / "failed-trial.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                q=q.detach().cpu().numpy(),
                u=physics.forward.state.u.detach().cpu().numpy(),
            )
        raise
    assert last is not None
    assert best is not None
    final_step = start_step + cfg.steps
    BASE.snapshot(out / "final.npz", physics, q, last, final_step, cfg, mesh=True)
    with np.load(out / "best.npz") as saved:
        BASE.snapshot(
            out / "best.npz",
            physics,
            torch.as_tensor(saved["q"]),
            {"u": saved["u"]},
            int(saved["step"]),
            cfg,
            mesh=True,
        )
    summary = {
        "status": "completed_fixed_budget_continuation",
        "inverse_convergence_claimed": False,
        "config": cfg.model_dump(mode="json"),
        "resume": resume_receipt,
        "provenance": provenance,
        "materials": physics.material_spec,
        "forward_solver": physics.forward_tolerance,
        "adjoint_solver": {
            "relative_tolerance": 5e-4,
            "max_steps": 10000,
            "implementations": ["CupyCG", "CupyMinRes"],
        },
        "mesh": {
            "active_tetrahedra": len(physics.ids),
            "scalar_controls": q.numel(),
            "graph_edges": len(physics.graph[0]),
            "regions": physics.n_regions,
        },
        "constraint": {
            "kind": "spectral PSD box",
            "reference_second_piola_stress_cap_mpa": cfg.stress_cap_mpa,
        },
        "geometry_rejection_enabled": False,
        "fiber_directions_used": False,
        "primary_endpoint": trace[-1],
        "initial_endpoint": trace[0],
        "first_update_replay": first_update_replay,
        "frozen_vs_recomputed_gradient": frozen_gradient_comparison,
        "best_objective_endpoint": best,
        "endpoint_policy": "actual final global step is primary; best total-objective state is secondary",
        "physical_update_definition": "uniform-cell RMS Frobenius norm of ΔQ in MPa; cumulative value is the sum of per-update RMS values",
        "stationarity_definition": "RMS of (Z - projection(Z - eta * gradient)) / eta at fixed eta; a local constrained diagnostic, not a capacity certificate",
        "wall_s": time.perf_counter() - started,
    }
    BASE.write_json(out / "summary.json", summary)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(run, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
