"""Refit one nonnegative strength per tetrahedron with fixed contraction axes."""

# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path

import comet_ml  # noqa: F401  # Import before Torch for Comet instrumentation.
import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics
from study_physics import FacePhysics, configure

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
FORWARD = GROUP / "data/10-forward"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("40-fixed-directions", mkdir=True)
    steps: int = 200
    checkpoint_interval: int = 20
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    resume: Path | None = None


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def array_digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pack(s: torch.Tensor, axes: torch.Tensor) -> torch.Tensor:
    """Pack B-I in historical unscaled xx,yy,zz,xy,yz,xz coordinates."""
    x, y, z = axes.unbind(-1)
    return s[:, None] * torch.stack((x * x, y * y, z * z, x * y, y * z, x * z), dim=-1)


def fields(s: np.ndarray, axes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projector = axes[:, :, None] * axes[:, None, :]
    return np.eye(3) + s[:, None, None] * projector, (2 * s + s * s)[
        :, None, None
    ] * projector


def chain_check() -> dict:
    axes = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 2.0, 3.0], [-2.0, 1.0, 1.0]],
        device="cpu",
        dtype=torch.float64,
    )
    axes = axes / torch.linalg.vector_norm(axes, dim=-1, keepdim=True)
    s = torch.tensor(
        [0.0, 0.3, 0.0], device="cpu", dtype=torch.float64, requires_grad=True
    )
    weights = torch.arange(1, 19, device="cpu", dtype=torch.float64).reshape(3, 6)
    q = pack(s, axes)
    (gradient,) = torch.autograd.grad((q * weights).sum(), s)
    basis = axes[:, :, None] * axes[:, None, :]
    expected = torch.stack(
        [
            sum(
                weights[i, k] * basis[i, row, col]
                for k, (row, col) in enumerate(
                    ((0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2))
                )
            )
            for i in range(3)
        ]
    )
    assert torch.allclose(gradient, expected, rtol=0, atol=1e-14)
    assert torch.all(gradient[[0, 2]].abs() > 0)
    return {
        "status": "passed",
        "max_gradient_error": float((gradient - expected).abs().max()),
        "zero_strength_derivatives": gradient[[0, 2]].tolist(),
    }


def objective(
    physics: FacePhysics,
    s: torch.Tensor,
    axes: torch.Tensor,
    seed: np.ndarray,
    *,
    backward: bool,
) -> dict:
    if backward:
        s.grad = None
    tick = time.perf_counter()
    u_t = physics.solve(pack(s, axes), seed)
    forward = dict(physics.last_forward)
    assert forward["success"], f"Unsuccessful forward solve: {forward}"
    target = torch.as_tensor(physics.target[physics.top])
    loss = (u_t[physics.top_t] - target).square().mean() * 1e6
    assert torch.isfinite(loss)
    result = {
        "objective_mm2": float(loss.detach()),
        "u": u_t.detach().cpu().numpy().copy(),
        "forward": forward,
    }
    if backward:
        loss.backward()
        adjoint = physics.check_adjoint()
        assert s.grad is not None
        assert torch.isfinite(s.grad).all()
        result["gradient"] = s.grad.detach().cpu().numpy().copy()
        result["adjoint"] = adjoint
    result["objective_seconds"] = time.perf_counter() - tick
    return result


def finite_difference_check(
    physics: FacePhysics,
    s: torch.Tensor,
    axes: torch.Tensor,
    initial: dict,
    chain: dict,
    out: Path,
) -> dict:
    """Compare complete scalar implicit gradients to independent forward solves."""
    s0 = s.detach().cpu().numpy().copy()
    grad = initial["gradient"]
    rng = np.random.default_rng(20260914)
    region_sign = rng.choice(np.array([-1.0, 1.0]), size=physics.n_regions)
    directions = {
        "signed_by_muscle": s0 * region_sign[physics.region],
        "gradient_sign": s0 * np.sign(grad),
    }
    checks = []
    for name, direction in directions.items():
        exact = float(np.dot(grad, direction))
        assert abs(exact) > 1e-6, "Directional check has insufficient signal"
        slopes = []
        for epsilon in (0.005, 0.0025):
            losses = []
            receipts = []
            for sign in (-1, 1):
                trial = s0 + sign * epsilon * direction
                assert np.min(trial) >= 0
                LOG.info(
                    "Gradient audit %s, epsilon %.4g, sign %+d", name, epsilon, sign
                )
                result = objective(
                    physics, torch.as_tensor(trial), axes, initial["u"], backward=False
                )
                losses.append(result["objective_mm2"])
                receipts.append(result["forward"])
            slope = (losses[1] - losses[0]) / (2 * epsilon)
            slopes.append(slope)
            checks.append(
                {
                    "direction": name,
                    "epsilon": epsilon,
                    "adjoint_derivative": exact,
                    "centered_difference": slope,
                    "relative_error": abs(slope - exact) / abs(exact),
                    "loss_minus": losses[0],
                    "loss_plus": losses[1],
                    "forward": receipts,
                }
            )
            write_json(
                out / "gradient-validation.json",
                {"status": "running", "chain_check": chain, "checks": checks},
            )
        assert abs(slopes[-1] - exact) <= 0.02 * abs(exact), checks[-1]
        assert abs(slopes[-1] - slopes[0]) <= 0.02 * abs(exact), (
            "FD step-size instability"
        )
    result = {
        "status": "passed",
        "chain_check": chain,
        "checks": checks,
        "finite_difference_tolerance": "2% relative slope error and 2% two-epsilon agreement",
        "zero_strength_scope": "Multiplicative face perturbations exclude zero-strength cells; synthetic chain check covers exact-zero derivative",
        "seed_policy": "Every perturbed solve starts at the identical replayed dominant-only displacement",
    }
    write_json(out / "gradient-validation.json", result)
    return result


def metrics(
    physics: FacePhysics,
    diagnostics: StudyMetrics,
    s: np.ndarray,
    result: dict,
    initial_s: np.ndarray,
    raw_gradient: np.ndarray,
) -> dict:
    u = result["u"]
    pred, target = u[physics.top], physics.target[physics.top]
    error = pred - target
    mapping = s - np.maximum(s - raw_gradient, 0)
    kkt = np.where(s > 0, raw_gradient, np.minimum(raw_gradient, 0))
    det = physics.detf(u)
    fraction = np.asarray(physics.mesh.cell_data["MuscleFraction"])
    pure = (
        (fraction >= 1 - 1e-12)
        & (np.asarray(physics.mesh.cell_data["FatFraction"]) == 0)
        & (np.asarray(physics.mesh.cell_data["AponeurosisFraction"]) == 0)
    )
    fixed = np.asarray(physics.mesh.point_data["FixedMask"], dtype=bool)
    fixed_value = np.asarray(physics.mesh.point_data["FixedValue"])
    fixed_error = float(np.max(np.abs(u[fixed] - fixed_value[fixed])))
    assert fixed_error < 1e-14
    row = {
        "objective_mm2": result["objective_mm2"],
        "uniform_fit_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "uniform_motion_rms_mm": float(
            1000 * np.sqrt(np.mean(np.sum(pred**2, axis=1)))
        ),
        "area_weighted_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(physics.weights * np.sum(error**2, axis=1)))
        ),
        "area_weighted_motion_rms_mm": float(
            1000 * np.sqrt(np.sum(physics.weights * np.sum(pred**2, axis=1)))
        ),
        "target_projection": float(np.sum(pred * target) / np.sum(target**2)),
        "raw_gradient_rms": float(np.sqrt(np.mean(raw_gradient**2))),
        "projected_gradient_rms": float(np.sqrt(np.mean(mapping**2))),
        "projected_gradient_max": float(np.max(np.abs(mapping))),
        "kkt_residual_rms": float(np.sqrt(np.mean(kkt**2))),
        "kkt_residual_max": float(np.max(np.abs(kkt))),
        "s_min": float(s.min()),
        "s_max": float(s.max()),
        "s_change_from_initial_rms": float(np.sqrt(np.mean((s - initial_s) ** 2))),
        "zero_strength_cells": int(np.sum(s == 0)),
        "initially_zero_now_positive_cells": int(np.sum((initial_s == 0) & (s > 0))),
        "detF_min": float(det.min()),
        "detF_max": float(det.max()),
        "inverted_all_cells": int(np.sum(det <= 0)),
        "inverted_active_cells": int(np.sum(det[physics.ids] <= 0)),
        "inverted_pure_muscle_cells": int(np.sum((det <= 0) & pure)),
        "active_volume_weighted_rms_detF_minus_one": float(
            np.sqrt(np.average((det[physics.ids] - 1) ** 2, weights=physics.volumes))
        ),
        "fixed_max_error_m": fixed_error,
        "forward_steps": result["forward"]["steps"],
        "forward_grad_norm": result["forward"]["grad_norm"],
        "objective_seconds": result["objective_seconds"],
    }
    for quantile in (0, 50, 90, 99, 100):
        row[f"active_axial_stretch_p{quantile}"] = float(
            np.percentile(1 / (1 + s), quantile)
        )
    row.update(diagnostics.evaluate_surface(u[diagnostics.skin_ids]))
    assert math.isclose(
        row["uniform_fit_rms_mm"] ** 2, 3 * row["objective_mm2"], rel_tol=1e-12
    )
    return row


def save_state(path: Path, state: dict, active_ids: np.ndarray, axes_hash: str) -> None:
    np.savez_compressed(
        path,
        s=state["s"],
        u=state["u"],
        step=state["step"],
        active_ids=active_ids,
        axes_sha256=np.array(axes_hash),
        solver_valid=True,
        physical_volume_energy=True,
    )


def archive_sources(out: Path) -> dict:
    records = {}
    for name, module in tuple(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or not file.endswith(".py"):
            continue
        path = Path(file).resolve()
        if path.parent == GROUP / "src":
            relative = Path("experiment") / path.name
        elif name.startswith(("liblaf.apple", "liblaf.peach")):
            relative = Path("runtime") / Path(*name.split(".")).with_suffix(".py")
        else:
            continue
        destination = out / "sources" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        records[name] = {**record(path), "snapshot": str(destination)}
    return records


def main(cfg: Config) -> None:
    assert cfg.steps > 0
    assert cfg.checkpoint_interval > 0
    assert cfg.learning_rate > 0
    assert cfg.adam_eps > 0
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Output directory must be empty: {out}"
    configure()
    chain = chain_check()
    physics = FacePhysics(FIXTURE, activation_model="raw6")
    diagnostics = StudyMetrics(FIXTURE)
    with np.load(FORWARD / "baseline-replay.npz", allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert np.array_equal(saved["active_ids"], physics.ids)
        assert np.array_equal(saved["rest_points"], physics.points)
        full_z = saved["Z"].copy()
        full_u = saved["u"].copy()
    eigenvalues, eigenvectors = np.linalg.eigh(full_z)
    axes_np = eigenvectors[:, :, -1].copy()
    axes_hash = array_digest(axes_np)
    initial_s = np.sqrt(1 + np.maximum(eigenvalues[:, -1], 0)) - 1
    with np.load(FORWARD / "dominant-only.npz", allow_pickle=False) as saved:
        assert bool(saved["solver_valid"])
        assert bool(saved["physical_volume_energy"])
        assert np.array_equal(saved["active_ids"], physics.ids)
        assert np.array_equal(saved["rest_points"], physics.points)
        seed = saved["u"].copy()
        initial_b, initial_z = fields(initial_s, axes_np)
        assert np.max(np.abs(initial_b - saved["B"])) < 1e-12
        assert np.max(np.abs(initial_z - saved["Z"])) < 1e-12
    assert np.max(np.abs(np.linalg.norm(axes_np, axis=1) - 1)) < 1e-14
    axes = torch.as_tensor(axes_np.copy())
    assert not axes.requires_grad
    s = torch.nn.Parameter(torch.as_tensor(initial_s.copy()))
    assert s.shape == (len(physics.ids),)
    optimizer = torch.optim.Adam(
        [s],
        lr=cfg.learning_rate,
        eps=cfg.adam_eps,
        betas=(0.9, 0.999),
        weight_decay=0,
        amsgrad=False,
    )
    start_step = 0
    trace = []
    best = None
    parent = None
    elapsed_offset = 0.0
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location="cpu", weights_only=False)
        assert checkpoint["axes_sha256"] == axes_hash
        assert np.array_equal(checkpoint["active_ids"], physics.ids)
        assert checkpoint["parameterization"] == "B=I+s*n*nT;s>=0;fixed-reference-axes"
        with torch.no_grad():
            s.copy_(checkpoint["s"].to(s))
        optimizer.load_state_dict(checkpoint["optimizer"])
        assert optimizer.param_groups[0]["lr"] == cfg.learning_rate
        assert optimizer.param_groups[0]["eps"] == cfg.adam_eps
        s.grad = checkpoint["gradient"].to(s)
        seed = checkpoint["u"].copy()
        best = checkpoint["best"]
        start_step = int(checkpoint["step"])
        assert cfg.steps > start_step
        trace = json.loads((cfg.resume.parent / "trace.json").read_text())
        trace = [row for row in trace if row["step"] <= start_step]
        assert trace[-1]["step"] == start_step
        elapsed_offset = trace[-1]["elapsed_seconds"]
        shutil.copyfile(
            cfg.resume.parent / "initialization.npz", out / "initialization.npz"
        )
        shutil.copyfile(
            cfg.resume.parent / "gradient-validation.json",
            out / "gradient-validation.json",
        )
        parent = {
            **record(cfg.resume),
            "step": start_step,
            "restored": "s, cached gradient, u seed, Adam moments and counter, best state, trace",
        }
    else:
        np.savez_compressed(
            out / "initialization.npz",
            axes=axes_np,
            initial_s=initial_s,
            active_ids=physics.ids,
            rest_points=physics.points,
            eigenvalues_Z0=eigenvalues,
            axes_sha256=np.array(axes_hash),
        )
    full_error = full_u[physics.top] - physics.target[physics.top]
    reference = {
        "uniform_fit_rms_mm": float(
            1000 * np.sqrt(np.mean(np.sum(full_error**2, axis=1)))
        ),
        "area_weighted_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(physics.weights * np.sum(full_error**2, axis=1)))
        ),
    }
    protocol = {
        "question": "Can scalar strengths along frozen maximum-contraction axes recover the fitted target?",
        "parameterization": "B_i=I+s_i*n_i*n_iT; one s_i>=0 per active tetrahedron; axes fixed in reference configuration",
        "trainable_parameter_count": int(s.numel()),
        "frozen_axes_sha256": axes_hash,
        "initialization": "s0=sqrt(1+max(lambda_max(Z_full),0))-1; u from saved dominant-only equilibrium; fresh scalar Adam unless resuming",
        "unsupported_positive_axis_cells": int(np.sum(initial_s == 0)),
        "near_repeated_top_axis_cells": int(
            np.sum(
                eigenvalues[:, -1] - eigenvalues[:, -2]
                <= 1e-6 * np.maximum(1, np.abs(eigenvalues[:, -1]))
            )
        ),
        "near_repeated_gap_rule": "lambda_max-lambda_second<=1e-6*max(1,abs(lambda_max))",
        "zero_strength_policy": "Keep largest-eigenvalue axis of original full Z even if it is nonpositive; scalar may activate it; do not interpret those axes as inferred positive contraction directions",
        "objective": "uniform finite-IsFace Cartesian component MSE times1e6; no regularizers",
        "optimizer": {
            "name": "Adam",
            "learning_rate": cfg.learning_rate,
            "eps": cfg.adam_eps,
            "betas": [0.9, 0.999],
            "steps": cfg.steps,
            "weight_decay": 0,
            "projection": "After each Adam proposal clamp s to>=0, preserving moments; no upper cap or loss-based rejection",
        },
        "stationarity": "Report raw gradient, KKT residual and unit-step gradient mapping s-max(s-g,0); a completed budget is not a convergence certificate",
        "materials": physics.material_spec,
        "forward_tolerance": physics.forward_tolerance,
        "reference_full_fit": reference,
        "parent": parent,
        "inputs": {
            "full": record(FORWARD / "baseline-replay.npz"),
            "dominant": record(FORWARD / "dominant-only.npz"),
            **{
                name: record(FIXTURE / name)
                for name in ("volume.vtu", "skin.vtp", "summary.json")
            },
        },
        "runtime": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "gpu": torch.cuda.get_device_name(),
            "command": [sys.executable, *sys.argv],
            "cwd": str(Path.cwd()),
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
        "failure_policy": "Stop visibly on nonfinite values or unsuccessful forward/adjoint; keep last valid optimizer checkpoint and best solved state",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(
        out / "summary.json",
        {
            "status": "initializing",
            "last_evaluated_step": start_step,
            "reference_full_fit": reference,
        },
    )
    status = "running"
    failure = None
    last_state = None
    started = time.perf_counter()
    step = start_step
    clamp_count = 0
    projection_rms = 0.0
    try:
        if cfg.resume is None:
            LOG.info(
                "Initial fixed-axis forward/adjoint: %d scalar controls", s.numel()
            )
            initial = objective(physics, s, axes, seed, backward=True)
            saved_optimizer = physics.forward.optimizer
            saved_tolerance = physics.forward_tolerance
            cg, minres = physics.diff.adjoint_solver.solvers
            saved_adjoint_rtol, saved_adjoint_tol = cg.rtol, minres.tol
            try:
                physics.forward.optimizer = physics.forward.default_optimizer(
                    max_steps=5000, rtol=5e-6, atol=1e-12
                )
                physics.forward_tolerance = {
                    **saved_tolerance,
                    "rtol": 5e-6,
                    "atol": 1e-12,
                }
                cg.rtol = minres.tol = 5e-6
                LOG.info("Tight equilibrium/adjoint center for finite-difference audit")
                audit_initial = objective(physics, s, axes, initial["u"], backward=True)
                relative_gradient_change = float(
                    np.linalg.norm(audit_initial["gradient"] - initial["gradient"])
                    / np.linalg.norm(audit_initial["gradient"])
                )
                validation = finite_difference_check(
                    physics, s, axes, audit_initial, chain, out
                )
                validation["audit_forward_tolerance"] = physics.forward_tolerance
                validation["audit_adjoint_relative_tolerance"] = 5e-6
                validation["normal_vs_tight_gradient_relative_difference"] = (
                    relative_gradient_change
                )
                validation["normal_vs_tight_objective_difference_mm2"] = (
                    initial["objective_mm2"] - audit_initial["objective_mm2"]
                )
                assert relative_gradient_change < 0.02, (
                    "Normal-tolerance gradient differs materially from tight audit"
                )
                write_json(out / "gradient-validation.json", validation)
            finally:
                physics.forward.optimizer = saved_optimizer
                physics.forward_tolerance = saved_tolerance
                cg.rtol, minres.tol = saved_adjoint_rtol, saved_adjoint_tol
            s.grad = torch.as_tensor(initial["gradient"].copy())
            protocol["sources"] = archive_sources(out)
            write_json(out / "protocol.json", protocol)
            result = initial
            first_step = 0
        else:
            protocol["sources"] = archive_sources(out)
            write_json(out / "protocol.json", protocol)
            optimizer.step()
            with torch.no_grad():
                before = s.clone()
                s.clamp_min_(0)
                clamp_count = int((before < 0).sum())
                projection_rms = float((s - before).square().mean().sqrt())
            first_step = start_step + 1
        for step in range(first_step, cfg.steps + 1):
            tick = time.perf_counter()
            cherries.set_step(step)
            if step != 0 or cfg.resume is not None:
                result = objective(physics, s, axes, seed, backward=True)
            s_np = s.detach().cpu().numpy().copy()
            assert np.min(s_np) >= 0
            assert array_digest(axes.detach().cpu().numpy()) == axes_hash
            row = {
                "step": step,
                **metrics(
                    physics, diagnostics, s_np, result, initial_s, result["gradient"]
                ),
                "clamped_proposal_cells": clamp_count,
                "projection_rms": projection_rms,
                "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
            }
            if not trace:
                initial_projected_gradient = row["projected_gradient_rms"]
            else:
                initial_projected_gradient = trace[0]["projected_gradient_rms"]
            row["relative_projected_gradient"] = (
                row["projected_gradient_rms"] / initial_projected_gradient
            )
            last_state = {
                "step": step,
                "s": s_np,
                "u": result["u"],
                "metrics": row.copy(),
            }
            if best is None or row["objective_mm2"] < best["metrics"]["objective_mm2"]:
                best = last_state
            row["best_step"] = best["step"]
            trace.append(row)
            write_json(out / "trace.json", trace)
            with (out / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(trace[0]))
                writer.writeheader()
                writer.writerows(trace)
            with (out / "solver-receipts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "forward": result["forward"],
                            "adjoint": result["adjoint"],
                        }
                    )
                    + "\n"
                )
            if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                save_state(
                    out / f"step-{step:04d}.npz", last_state, physics.ids, axes_hash
                )
                save_state(out / "best.npz", best, physics.ids, axes_hash)
                save_state(out / "last.npz", last_state, physics.ids, axes_hash)
            temporary = out / "optimizer-latest.tmp"
            torch.save(
                {
                    "step": step,
                    "s": s.detach().cpu(),
                    "gradient": s.grad.detach().cpu(),
                    "u": result["u"],
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "active_ids": physics.ids,
                    "axes_sha256": axes_hash,
                    "parameterization": "B=I+s*n*nT;s>=0;fixed-reference-axes",
                },
                temporary,
            )
            temporary.replace(out / "optimizer-latest.pt")
            if step % 5 == 0 or step == cfg.steps:
                cherries.log_metrics(
                    {
                        key: row[key]
                        for key in (
                            "objective_mm2",
                            "uniform_fit_rms_mm",
                            "area_weighted_fit_rms_mm",
                            "area_weighted_motion_rms_mm",
                            "projected_gradient_rms",
                            "inverted_all_cells",
                        )
                    }
                )
            LOG.info(
                "Fixed axes %d/%d: fit %.6f mm, area fit %.6f mm, motion %.6f mm, PG %.3e; forward %d, %.1fs",
                step,
                cfg.steps,
                row["uniform_fit_rms_mm"],
                row["area_weighted_fit_rms_mm"],
                row["area_weighted_motion_rms_mm"],
                row["projected_gradient_rms"],
                row["forward_steps"],
                time.perf_counter() - tick,
            )
            write_json(
                out / "summary.json",
                {
                    "status": "running"
                    if step < cfg.steps
                    else "completed_budget_not_stationarity_certified",
                    "last_evaluated_step": step,
                    "best_step": best["step"],
                    "best_metrics": best["metrics"],
                    "last_metrics": row,
                    "reference_full_fit": reference,
                    "inverse_stationarity_claimed": False,
                    "elapsed_seconds": row["elapsed_seconds"],
                    "parent": parent,
                },
            )
            if step == cfg.steps:
                status = "completed_budget_not_stationarity_certified"
                break
            seed = result["u"]
            optimizer.step()
            with torch.no_grad():
                before = s.clone()
                s.clamp_min_(0)
                clamp_count = int((before < 0).sum())
                projection_rms = float((s - before).square().mean().sqrt())
    except BaseException as error:
        status = "failed_before_completion"
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "step": step,
            "last_forward": getattr(physics, "last_forward", None),
        }
        write_json(out / "failure.json", failure)
        np.savez_compressed(
            out / "failed-proposal.npz",
            s=s.detach().cpu().numpy(),
            step=step,
            axes_sha256=np.array(axes_hash),
            solver_valid=False,
        )
        raise
    finally:
        if best is not None:
            save_state(out / "best.npz", best, physics.ids, axes_hash)
        if last_state is not None:
            save_state(out / "last.npz", last_state, physics.ids, axes_hash)
        write_json(
            out / "summary.json",
            {
                "status": status,
                "last_evaluated_step": None
                if last_state is None
                else last_state["step"],
                "best_step": None if best is None else best["step"],
                "best_metrics": None if best is None else best["metrics"],
                "last_metrics": None if last_state is None else last_state["metrics"],
                "reference_full_fit": reference,
                "inverse_stationarity_claimed": False,
                "elapsed_seconds": elapsed_offset + time.perf_counter() - started,
                "failure": failure,
                "parent": parent,
            },
        )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
