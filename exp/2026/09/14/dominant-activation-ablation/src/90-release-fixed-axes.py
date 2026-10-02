"""Continue the fixed-400 state with learned three-component uniaxial controls."""

# ruff: noqa: C901, PLR0912, PLR0915

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import comet_ml  # noqa: F401
import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from study_metrics import StudyMetrics
from study_physics import FacePhysics, configure

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
FIXED = GROUP / "data/42-fixed-directions-400"
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("90-released-axes", mkdir=True)
    steps: int = 200
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    checkpoint_interval: int = 20


def record(path: Path) -> dict:
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def pack(v: torch.Tensor) -> torch.Tensor:
    """Pack vvT as the historical xx,yy,zz,xy,yz,xz coordinates of B-I."""
    x, y, z = v.unbind(-1)
    return torch.stack((x * x, y * y, z * z, x * y, y * z, x * z), dim=-1)


def fields(v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = v[:, :, None] * v[:, None, :]
    s = np.sum(v * v, axis=1)
    return c, np.eye(3) + c, (2 + s)[:, None, None] * c


def chain_check() -> dict:
    v = torch.tensor(
        [[0.0, 0.0, 0.0], [0.3, -0.7, 1.1], [2.0, 0.5, -0.2]],
        device="cpu",
        dtype=torch.float64,
        requires_grad=True,
    )
    weights = torch.arange(1, 19, device="cpu", dtype=torch.float64).reshape(3, 6)
    (actual,) = torch.autograd.grad((pack(v) * weights).sum(), v)
    x, y, z = v.detach().unbind(-1)
    gxx, gyy, gzz, gxy, gyz, gxz = weights.unbind(-1)
    expected = torch.stack(
        (
            2 * x * gxx + y * gxy + z * gxz,
            2 * y * gyy + x * gxy + z * gyz,
            2 * z * gzz + y * gyz + x * gxz,
        ),
        dim=-1,
    )
    error = float((actual - expected).abs().max())
    assert error < 2e-14
    assert torch.equal(actual[0], torch.zeros(3, device="cpu", dtype=torch.float64))
    return {
        "status": "passed",
        "max_abs_error": error,
        "exact_zero_gradient": actual[0].tolist(),
    }


def objective(
    physics: FacePhysics, v: torch.Tensor, seed: np.ndarray, *, backward: bool
) -> dict:
    tick = time.perf_counter()
    if backward:
        v.grad = None
    u = physics.solve(pack(v), seed)
    forward = dict(physics.last_forward)
    assert forward["success"], forward
    target = torch.as_tensor(physics.target[physics.top])
    loss = (u[physics.top_t] - target).square().mean() * 1e6
    assert torch.isfinite(loss)
    result = {
        "objective_mm2": float(loss.detach()),
        "u": u.detach().cpu().numpy().copy(),
        "forward": forward,
    }
    if backward:
        loss.backward()
        result["adjoint"] = physics.check_adjoint()
        assert v.grad is not None
        assert torch.isfinite(v.grad).all()
        result["gradient"] = v.grad.detach().cpu().numpy().copy()
    result["objective_seconds"] = time.perf_counter() - tick
    return result


def gradient_audit(
    physics: FacePhysics, v: torch.Tensor, initial: dict, out: Path
) -> dict:
    original_optimizer, original_tolerance = (
        physics.forward.optimizer,
        physics.forward_tolerance,
    )
    cg, minres = physics.diff.adjoint_solver.solvers
    old_cg, old_minres = cg.rtol, minres.tol
    try:
        physics.forward.optimizer = physics.forward.default_optimizer(
            max_steps=5000, rtol=5e-6, atol=1e-12
        )
        physics.forward_tolerance = {**original_tolerance, "rtol": 5e-6, "atol": 1e-12}
        cg.rtol = minres.tol = 5e-6
        center = objective(physics, v, initial["u"], backward=True)
        v0 = v.detach().cpu().numpy().copy()
        norms = np.linalg.norm(v0, axis=1)
        unit = np.divide(
            v0, norms[:, None], out=np.zeros_like(v0), where=norms[:, None] > 0
        )
        gradient = center["gradient"]
        radial_component = np.sum(gradient * unit, axis=1)
        tangent = gradient - radial_component[:, None] * unit
        tangent_norm = np.linalg.norm(tangent, axis=1)
        tangent_unit = np.divide(
            tangent,
            tangent_norm[:, None],
            out=np.zeros_like(tangent),
            where=tangent_norm[:, None] > 0,
        )
        directions = {
            "radial": v0 * np.sign(radial_component)[:, None],
            "rotational": norms[:, None] * tangent_unit,
        }
        checks = []
        for name, direction in directions.items():
            exact = float(np.sum(gradient * direction))
            assert abs(exact) > 1e-6, (name, exact)
            slopes = []
            for epsilon in [0.005, 0.0025]:
                losses, receipts = [], []
                for sign in [-1, 1]:
                    LOG.info(
                        "Gradient audit %s epsilon %.4g sign %+d", name, epsilon, sign
                    )
                    trial = torch.as_tensor(v0 + sign * epsilon * direction)
                    result = objective(physics, trial, center["u"], backward=False)
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
                    {"status": "running", "checks": checks},
                )
            assert abs(slopes[-1] - exact) / abs(exact) < 0.02, checks[-1]
            assert abs(slopes[-1] - slopes[0]) / abs(exact) < 0.02, checks[-2:]
        relative_gradient_difference = float(
            np.linalg.norm(center["gradient"] - initial["gradient"])
            / np.linalg.norm(center["gradient"])
        )
        assert relative_gradient_difference < 0.02
        audit = {
            "status": "passed",
            "checks": checks,
            "normal_vs_tight_gradient_relative_difference": relative_gradient_difference,
            "normal_vs_tight_objective_difference_mm2": initial["objective_mm2"]
            - center["objective_mm2"],
            "tolerance": "2% slope error and two-epsilon agreement",
            "forward_tolerance": physics.forward_tolerance,
            "adjoint_relative_tolerance": 5e-6,
            "trial_seed": "identical tight center displacement for every trial",
            "zero_cells": "excluded by zero-length radial/tangent directions; exact-zero chain check recorded separately",
        }
        write_json(out / "gradient-validation.json", audit)
        return audit
    finally:
        physics.forward.optimizer, physics.forward_tolerance = (
            original_optimizer,
            original_tolerance,
        )
        cg.rtol, minres.tol = old_cg, old_minres
        v.grad = torch.as_tensor(initial["gradient"].copy())


def metrics(
    physics: FacePhysics,
    diagnostics: StudyMetrics,
    v: np.ndarray,
    result: dict,
    initial_axes: np.ndarray,
    initial_s: np.ndarray,
) -> dict:
    s = np.sum(v * v, axis=1)
    norms = np.sqrt(s)
    active = (initial_s > 0) & (s > 0)
    axes = np.divide(v, norms[:, None], out=np.zeros_like(v), where=norms[:, None] > 0)
    dot = np.clip(np.abs(np.sum(axes[active] * initial_axes[active], axis=1)), 0, 1)
    angles = np.degrees(np.arccos(dot))
    grad = result["gradient"]
    radial = np.sum(grad * axes, axis=1)[:, None] * axes
    tangent = grad - radial
    u = result["u"]
    prediction, target = u[physics.top], physics.target[physics.top]
    error = prediction - target
    det = physics.detf(u)
    fraction = np.asarray(physics.mesh.cell_data["MuscleFraction"])
    pure = (
        (fraction >= 1 - 1e-12)
        & (np.asarray(physics.mesh.cell_data["FatFraction"]) == 0)
        & (np.asarray(physics.mesh.cell_data["AponeurosisFraction"]) == 0)
    )
    fixed = np.asarray(physics.mesh.point_data["FixedMask"], dtype=bool)
    fixed_values = np.asarray(physics.mesh.point_data["FixedValue"])
    fixed_error = float(np.max(np.abs(u[fixed] - fixed_values[fixed])))
    assert fixed_error < 1e-14
    i, j, edge_weight = physics.graph
    pair_active = active[i] & active[j]
    pair_dot = np.clip(np.sum(axes[i] * axes[j], axis=1), -1, 1)
    projector_diff2 = 2 * (1 - pair_dot**2)
    c, _, z = fields(v)
    row = {
        "objective_mm2": result["objective_mm2"],
        "uniform_fit_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(error**2, axis=1)))),
        "area_weighted_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(physics.weights * np.sum(error**2, axis=1)))
        ),
        "area_weighted_motion_rms_mm": float(
            1000 * np.sqrt(np.sum(physics.weights * np.sum(prediction**2, axis=1)))
        ),
        "target_projection": float(np.sum(prediction * target) / np.sum(target**2)),
        "gradient_rms": float(np.sqrt(np.mean(grad**2))),
        "radial_gradient_rms": float(np.sqrt(np.mean(radial**2))),
        "tangential_gradient_rms": float(np.sqrt(np.mean(tangent**2))),
        "direction_drift_mean_deg": float(np.mean(angles)),
        "direction_drift_volume_weighted_mean_deg": float(
            np.average(angles, weights=physics.volumes[active])
        ),
        "direction_drift_max_deg": float(np.max(angles)),
        "axis_neighbor_projector_jump_rms": float(
            np.sqrt(
                np.average(
                    projector_diff2[pair_active], weights=edge_weight[pair_active]
                )
            )
        ),
        "C_neighbor_jump_rms": float(
            np.sqrt(
                np.average(np.sum((c[i] - c[j]) ** 2, axis=(1, 2)), weights=edge_weight)
            )
        ),
        "Z_neighbor_jump_rms": float(
            np.sqrt(
                np.average(np.sum((z[i] - z[j]) ** 2, axis=(1, 2)), weights=edge_weight)
            )
        ),
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
    for p in [50, 90, 99]:
        row[f"direction_drift_p{p}_deg"] = float(np.percentile(angles, p))
    for angle in [15, 30, 60]:
        row[f"direction_drift_volume_fraction_above_{angle}deg"] = float(
            np.average(angles > angle, weights=physics.volumes[active])
        )
    for p in [0, 50, 90, 99, 100]:
        row[f"active_axial_stretch_p{p}"] = float(np.percentile(1 / (1 + s), p))
    row.update(diagnostics.evaluate_surface(u[diagnostics.skin_ids]))
    assert math.isclose(
        row["uniform_fit_rms_mm"] ** 2, 3 * row["objective_mm2"], rel_tol=1e-12
    )
    return row


def save_state(path: Path, state: dict, physics: FacePhysics) -> None:
    c, b, z = fields(state["v"])
    np.savez_compressed(
        path,
        v=state["v"],
        u=state["u"],
        step=state["step"],
        source_fixed_step=400,
        active_ids=physics.ids,
        rest_points=physics.points,
        C=c,
        B=b,
        Z=z,
        solver_valid=True,
        physical_volume_energy=True,
        physical_noninverted=state["metrics"]["inverted_all_cells"] == 0,
        parameterization="B=I+vvT; learned reference axis and strength",
    )


def archive_sources(out: Path) -> dict:
    sources = {}
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
        sources[name] = {**record(path), "snapshot": str(destination)}
    return sources


def main(cfg: Config) -> None:
    assert cfg.steps > 0
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    configure()
    chain = chain_check()
    verified = json.loads((GROUP / "data/62-verification-400/receipt.json").read_text())
    for key in ["best", "initialization", "summary"]:
        expected = verified["inputs"][key]
        assert record(Path(expected["path"])) == expected
    with np.load(FIXED / "best.npz", allow_pickle=False) as saved:
        assert saved["step"].item() == 400
        assert saved["solver_valid"].item()
        assert saved["physical_volume_energy"].item()
        initial_s, seed, ids = saved["s"], saved["u"], saved["active_ids"]
    with np.load(FIXED / "initialization.npz", allow_pickle=False) as saved:
        axes, rest = saved["axes"], saved["rest_points"]
        assert np.array_equal(ids, saved["active_ids"])
    physics = FacePhysics(FIXTURE, activation_model="raw6")
    diagnostics = StudyMetrics(FIXTURE)
    assert np.array_equal(ids, physics.ids)
    assert np.array_equal(rest, physics.points)
    v0 = np.sqrt(initial_s)[:, None] * axes
    _, b, z = fields(v0)
    projector = axes[:, :, None] * axes[:, None, :]
    b_error = float(
        np.max(np.abs(b - (np.eye(3) + initial_s[:, None, None] * projector)))
    )
    z_error = float(
        np.max(np.abs(z - (2 * initial_s + initial_s**2)[:, None, None] * projector))
    )
    assert b_error < 1e-13
    assert z_error < 1e-12
    inactive = initial_s == 0
    assert np.all(v0[inactive] == 0)
    v = torch.nn.Parameter(torch.as_tensor(v0.copy()))
    optimizer = torch.optim.Adam(
        [v],
        lr=cfg.learning_rate,
        eps=cfg.adam_eps,
        betas=(0.9, 0.999),
        weight_decay=0,
        amsgrad=False,
    )
    parent_summary = json.loads((FIXED / "summary.json").read_text())
    conversion = {
        "B_max_abs_error": b_error,
        "Z_max_abs_error": z_error,
        "source_fixed_step": 400,
        "trainable_coordinates": v.numel(),
        "exact_zero_inactive_cells": int(np.sum(inactive)),
        "initial_u_exact_source": True,
        "initial_s_axis_hash": verified["frozen_axis_contract"]["axes_sha256"],
    }
    protocol = {
        "question": "Does releasing directions from the fixed-400 solution improve fitting without recreating incoherence and element distortion?",
        "parameterization": "B=I+vvT; 3 real v components per active tetrahedron",
        "initialization": "v0=sqrt(s_fixed400)*n_fixed; exact saved u seed; no random perturbation",
        "zero_policy": "2305 exact-zero v remain inactive because d(vvT)/dv=0 there; active directions are released",
        "objective": "uniform finite-IsFace Cartesian component MSE times1e6; no regularizers or added barriers",
        "optimizer": {
            "name": "Adam",
            "learning_rate": cfg.learning_rate,
            "eps": cfg.adam_eps,
            "betas": [0.9, 0.999],
            "weight_decay": 0,
            "steps": cfg.steps,
            "moments": "fresh; scalar moments are not transported to vector coordinates",
            "projection": "none",
        },
        "conversion": conversion,
        "materials": physics.material_spec,
        "forward_tolerance": physics.forward_tolerance,
        "inputs": {
            key: verified["inputs"][key]
            for key in ["best", "initialization", "summary", "full"]
        },
        "fixture": {
            name: record(FIXTURE / name)
            for name in ["volume.vtu", "skin.vtp", "summary.json"]
        },
        "chain_check": chain,
        "sources": archive_sources(out),
        "failure_policy": "Stop on nonfinite or unsuccessful forward/adjoint; retain valid checkpoints and failed proposal. Record inversions without changing objective; save first-inverted and best-noninverted separately.",
        "metrics": "Direction angles use abs(n dot n0) on nonzero axes; tensor/projector jumps use same-muscle shared-face weights. Roughness is existing frozen StudyMetrics.",
    }
    write_json(out / "protocol.json", protocol)
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    converted_state = {
        "v": v0,
        "u": seed,
        "step": 0,
        "metrics": parent_summary["best_metrics"],
    }
    save_state(out / "initial-converted.npz", converted_state, physics)
    trace, best, last, best_noninverted = [], None, None, None
    status, failure = "initializing", None
    step = 0
    started = time.perf_counter()
    write_json(out / "summary.json", {"status": status, "conversion": conversion})
    try:
        LOG.info(
            "Exact fixed-400 conversion: B error %.3g, Z error %.3g; %d vector controls",
            b_error,
            z_error,
            v.numel(),
        )
        result = objective(physics, v, seed, backward=True)
        conversion["replay_u_rms_change_mm"] = float(
            1000 * np.sqrt(np.mean((result["u"] - seed) ** 2))
        )
        conversion["replay_objective_change_mm2"] = (
            result["objective_mm2"] - parent_summary["best_metrics"]["objective_mm2"]
        )
        assert abs(conversion["replay_objective_change_mm2"]) < 0.001
        LOG.info(
            "Warm-start replay fit %.6f mm; checking radial and rotational gradients",
            math.sqrt(3 * result["objective_mm2"]),
        )
        audit = gradient_audit(physics, v, result, out)
        protocol["gradient_validation"] = audit
        protocol["sources"] = archive_sources(out)
        write_json(out / "protocol.json", protocol)
        first_inverted_saved = False
        for step in range(cfg.steps + 1):
            cherries.set_step(step)
            if step > 0:
                result = objective(physics, v, seed, backward=True)
            current = v.detach().cpu().numpy().copy()
            assert np.all(current[inactive] == 0)
            assert np.all(result["gradient"][inactive] == 0)
            row = {
                "step": step,
                **metrics(physics, diagnostics, current, result, axes, initial_s),
                "elapsed_seconds": time.perf_counter() - started,
            }
            row["relative_gradient_rms"] = row["gradient_rms"] / (
                trace[0]["gradient_rms"] if trace else row["gradient_rms"]
            )
            last = {"step": step, "v": current, "u": result["u"], "metrics": row.copy()}
            if best is None or row["objective_mm2"] < best["metrics"]["objective_mm2"]:
                best = last
            if row["inverted_all_cells"] == 0 and (
                best_noninverted is None
                or row["objective_mm2"] < best_noninverted["metrics"]["objective_mm2"]
            ):
                best_noninverted = last
            if row["inverted_all_cells"] > 0 and not first_inverted_saved:
                save_state(out / "first-inverted.npz", last, physics)
                first_inverted_saved = True
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
            if step == 0:
                save_state(out / "initial.npz", last, physics)
            if step % cfg.checkpoint_interval == 0 or step in [128, cfg.steps]:
                save_state(out / f"step-{step:04d}.npz", last, physics)
                save_state(out / "best.npz", best, physics)
                save_state(out / "last.npz", last, physics)
                if best_noninverted is not None:
                    save_state(out / "best-noninverted.npz", best_noninverted, physics)
            temporary = out / "optimizer-latest.tmp"
            torch.save(
                {
                    "step": step,
                    "v": v.detach().cpu(),
                    "gradient": v.grad.detach().cpu(),
                    "u": result["u"],
                    "optimizer": optimizer.state_dict(),
                    "best": best,
                    "best_noninverted": best_noninverted,
                    "active_ids": ids,
                    "initial_v": v0,
                    "initial_axes": axes,
                    "initial_s": initial_s,
                    "parameterization": "B=I+vvT; learned-axis",
                    "source_fixed_step": 400,
                },
                temporary,
            )
            temporary.replace(out / "optimizer-latest.pt")
            if step % 5 == 0 or step == cfg.steps:
                cherries.log_metrics(
                    {
                        key: row[key]
                        for key in [
                            "uniform_fit_rms_mm",
                            "area_weighted_fit_rms_mm",
                            "direction_drift_p50_deg",
                            "direction_drift_p90_deg",
                            "inverted_all_cells",
                            "gradient_rms",
                        ]
                    }
                )
            LOG.info(
                "Released %d/%d: fit %.6f mm, area %.6f, drift median/p90 %.3f/%.3f deg, inversions %d; forward %d",
                step,
                cfg.steps,
                row["uniform_fit_rms_mm"],
                row["area_weighted_fit_rms_mm"],
                row["direction_drift_p50_deg"],
                row["direction_drift_p90_deg"],
                row["inverted_all_cells"],
                row["forward_steps"],
            )
            status = (
                "running"
                if step < cfg.steps
                else "completed_budget_not_stationarity_certified"
            )
            write_json(
                out / "summary.json",
                {
                    "status": status,
                    "last_evaluated_step": step,
                    "best_step": best["step"],
                    "best_metrics": best["metrics"],
                    "last_metrics": row,
                    "conversion": conversion,
                    "best_noninverted_step": None
                    if best_noninverted is None
                    else best_noninverted["step"],
                    "parent_fixed_metrics": parent_summary["best_metrics"],
                },
            )
            if step == cfg.steps:
                break
            seed = result["u"]
            optimizer.step()
    except BaseException as error:
        status = "failed_before_completion"
        failure = {
            "step": step,
            "type": type(error).__name__,
            "message": str(error),
            "last_forward": getattr(physics, "last_forward", None),
        }
        write_json(out / "failure.json", failure)
        np.savez_compressed(
            out / "failed-proposal.npz",
            v=v.detach().cpu().numpy(),
            step=step,
            solver_valid=False,
        )
        raise
    finally:
        for name, state in [
            ("best", best),
            ("last", last),
            ("best-noninverted", best_noninverted),
        ]:
            if state is not None:
                save_state(out / f"{name}.npz", state, physics)
        write_json(
            out / "summary.json",
            {
                "status": status,
                "last_evaluated_step": None if last is None else last["step"],
                "best_step": None if best is None else best["step"],
                "best_metrics": None if best is None else best["metrics"],
                "last_metrics": None if last is None else last["metrics"],
                "best_noninverted_step": None
                if best_noninverted is None
                else best_noninverted["step"],
                "best_noninverted_metrics": None
                if best_noninverted is None
                else best_noninverted["metrics"],
                "conversion": conversion,
                "parent_fixed_metrics": parent_summary["best_metrics"],
                "failure": failure,
                "elapsed_seconds": time.perf_counter() - started,
                "inverse_stationarity_claimed": False,
            },
        )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
