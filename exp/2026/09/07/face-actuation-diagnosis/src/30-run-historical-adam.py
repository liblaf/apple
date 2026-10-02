"""Matched June-style no-skin Raw6/Raw6-S face inversions."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from historical_adam_physics import (
    ROOT as REPO,
)
from historical_adam_physics import (
    FacePhysics,
    ForwardConvergenceError,
    active_graph,
    configure,
)

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
AREF = -math.log(0.8)
LOSS_SCALE = 1.0e6
HISTORICAL_ENDPOINT = HERE / "data/11-historical-no-skin/final.npz"
HISTORICAL_SUMMARY = HERE / "data/11-historical-no-skin/summary.json"


class AdjointConvergenceError(RuntimeError):
    """The declared adjoint solve failed at an otherwise valid equilibrium."""

    def __init__(self, receipt: dict[str, Any]) -> None:
        self.receipt = receipt
        super().__init__(f"adjoint failed: {receipt}")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = HERE / "data/12-historical-fixture"
    output_dir: Path = HERE / "data/30-historical-adam-raw6"
    steps: int = 200
    checkpoint_interval: int = 10
    inverse_lr: float = 0.3
    adam_eps: float = 0.01
    smoothness_weight: float = 0.0
    smooth_length: float = 0.005
    forward_rtol: float = 5e-4
    forward_atol: float = 1e-10
    adjoint_rtol: float = 5e-4
    preflight: bool = False


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=lambda item: (
                item.item() if isinstance(item, np.generic) else str(item)
            ),
        )
        + "\n"
    )


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def activation_matrix(q: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    shape = (q.shape[0], 3, 3)
    if isinstance(q, torch.Tensor):
        h = torch.zeros(shape, dtype=q.dtype, device=q.device)
        eye = torch.eye(3, dtype=q.dtype, device=q.device)
    else:
        h = np.zeros(shape, dtype=q.dtype)
        eye = np.eye(3, dtype=q.dtype)
    h[:, 0, 0], h[:, 1, 1], h[:, 2, 2] = q[:, 0], q[:, 1], q[:, 2]
    h[:, 0, 1] = h[:, 1, 0] = q[:, 3]
    h[:, 1, 2] = h[:, 2, 1] = q[:, 4]
    h[:, 0, 2] = h[:, 2, 0] = q[:, 5]
    return h + eye


def graph_arrays(mesh: pv.UnstructuredGrid) -> dict[str, np.ndarray]:
    active = np.flatnonzero(np.asarray(mesh.cell_data["ActivationMask"], dtype=bool))
    tets = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)[:, 1:]
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=np.int64)
    region_values = np.unique(muscle_id[active])
    region_map = np.full(muscle_id.max() + 1, -1, dtype=np.int64)
    region_map[region_values] = np.arange(len(region_values))
    region = region_map[muscle_id[active]]
    fraction = np.asarray(mesh.cell_data["MuscleFraction"], dtype=np.float64)
    i, j, weight = active_graph(
        np.asarray(mesh.points, dtype=np.float64),
        tets,
        active,
        region,
        fraction,
    )
    volume = (
        np.asarray(mesh.cell_data["Volume"], dtype=np.float64)[active]
        * fraction[active]
    )
    return {
        "active": active,
        "region": region,
        "i": i,
        "j": j,
        "weight": weight,
        "volume": volume,
    }


def smoothness_numpy(
    q: np.ndarray, graph: dict[str, np.ndarray], length: float
) -> float:
    diff = q[graph["i"]] - q[graph["j"]]
    packed_frobenius_weights = np.asarray((1, 1, 1, 2, 2, 2), dtype=np.float64)
    return float(
        length**2
        * np.sum(
            graph["weight"][:, None] * packed_frobenius_weights[None, :] * diff**2 / 1.5
        )
        / graph["volume"].sum()
        / AREF**2
    )


def validate_config(cfg: Config) -> None:
    expected = {
        "steps": 200,
        "checkpoint_interval": 10,
        "inverse_lr": 0.3,
        "adam_eps": 0.01,
        "smooth_length": 0.005,
        "forward_rtol": 5e-4,
        "forward_atol": 1e-10,
        "adjoint_rtol": 5e-4,
    }
    actual = {key: getattr(cfg, key) for key in expected}
    if actual != expected:
        raise ValueError(
            f"matched historical constants changed: {actual} != {expected}"
        )
    if cfg.smoothness_weight not in {0.0, 5e-4}:
        raise ValueError("use the declared baseline weight 0 or candidate weight 5e-4")


def preflight(cfg: Config) -> dict[str, Any]:
    validate_config(cfg)
    required = [
        cfg.fixture / "volume.vtu",
        cfg.fixture / "skin.vtp",
        cfg.fixture / "summary.json",
        HISTORICAL_ENDPOINT,
        HISTORICAL_SUMMARY,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    mesh = pv.read(cfg.fixture / "volume.vtu")
    if not isinstance(mesh, pv.UnstructuredGrid):
        raise TypeError("historical fixture is not an unstructured grid")
    graph = graph_arrays(mesh)
    fixed = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    top = np.asarray(mesh.point_data["IsFace"], dtype=bool) & np.isfinite(
        np.asarray(mesh.point_data["Smile"])
    ).all(axis=1)
    if not (
        len(graph["active"]) == 288_235
        and len(np.unique(graph["region"])) == 103
        and fixed.shape == (228_660, 3)
        and np.count_nonzero(fixed) == 81_108
        and np.count_nonzero(top) == 15_302
    ):
        raise ValueError("historical fixture cardinality changed")
    saved = np.load(HISTORICAL_ENDPOINT)
    q = np.asarray(saved["q"], dtype=np.float64)
    if q.shape != (288_235, 6):
        raise ValueError("historical endpoint control shape changed")
    graph_energy = smoothness_numpy(q, graph, cfg.smooth_length)
    historical = json.loads(HISTORICAL_SUMMARY.read_text())
    fit = float(historical["recorded_june_metrics"]["best_fit_rms_mm"]) ** 2 / 3.0
    candidate_weight = 5e-4
    candidate_penalty = candidate_weight * graph_energy
    candidate_ratio = candidate_penalty / fit
    report = {
        "status": "cpu_preflight_passed",
        "gpu_used": False,
        "config": cfg.model_dump(mode="json"),
        "fixture": {
            "points": int(mesh.n_points),
            "tetrahedra": int(mesh.n_cells),
            "active_tetrahedra": len(graph["active"]),
            "scalar_controls": int(6 * len(graph["active"])),
            "muscle_regions": len(np.unique(graph["region"])),
            "within_same_MuscleId_shared_face_edges": len(graph["i"]),
            "fixed_vertices": int(np.count_nonzero(fixed[:, 0])),
            "target_vertices": int(np.count_nonzero(top)),
        },
        "material_contract": {
            "fat": {"E_MPa": 0.003, "nu": 0.49},
            "muscle": {"E_MPa": 0.03, "nu": 0.49},
            "aponeurosis": {"E_MPa": 0.1, "nu": 0.35},
            "lame_convention": "classical lambda passed directly to Stable energies, matching June",
            "skin_energy": 0.0,
        },
        "objective_contract": {
            "data": "mean over finite IsFace vertices and x/y/z components of squared displacement residual, multiplied by 1e6",
            "smoothness": "L^2 sum_e[(area/distance)*harmonic(MuscleFraction)*(||H_i-H_j||F^2/1.5)] / [sum_t(volume*MuscleFraction)*(-log(0.8))^2]",
            "smooth_length_m": cfg.smooth_length,
            "historical_saved_graph_energy": graph_energy,
            "historical_saved_data_objective_mm2": fit,
            "candidate_weight_mm2": candidate_weight,
            "candidate_penalty_at_historical_saved_field_mm2": candidate_penalty,
            "candidate_penalty_over_historical_saved_data_objective": candidate_ratio,
            "selection": (
                "training-target heuristic: 5e-4 makes the saved-field penalty "
                f"{100 * candidate_ratio:.1f}% of its residual objective, comparable "
                "without dominating"
            ),
        },
        "hashes": {str(path.resolve()): sha256(path) for path in required},
    }
    output = cfg.output_dir.with_name(cfg.output_dir.name + "-preflight.json")
    write_json(output, report)
    return report


class Objective:
    def __init__(self, physics: FacePhysics, cfg: Config) -> None:
        self.physics = physics
        self.cfg = cfg
        self.i = torch.as_tensor(physics.graph[0], dtype=torch.long)
        self.j = torch.as_tensor(physics.graph[1], dtype=torch.long)
        self.edge_weight = torch.as_tensor(physics.graph[2])
        self.volume_sum = torch.as_tensor(float(physics.volumes.sum()))
        self.target = torch.as_tensor(physics.target[physics.top])
        self.packed_frobenius_weights = torch.as_tensor((1, 1, 1, 2, 2, 2))

    def smoothness(self, q: torch.Tensor) -> torch.Tensor:
        diff = q[self.i] - q[self.j]
        return (
            self.cfg.smooth_length**2
            * (
                self.edge_weight[:, None]
                * self.packed_frobenius_weights[None, :]
                * diff.square()
                / 1.5
            ).sum()
            / self.volume_sum
            / AREF**2
        )

    def __call__(
        self, q: torch.Tensor, seed: np.ndarray, *, backward: bool = True
    ) -> dict[str, Any]:
        if q.grad is not None:
            q.grad = None
        u = self.physics.solve(q, seed)
        residual = u[self.physics.top_t] - self.target
        data = residual.square().mean() * LOSS_SCALE
        smooth = self.smoothness(q)
        total = data + self.cfg.smoothness_weight * smooth
        adjoint = None
        if backward:
            total.backward()
            solution = self.physics.diff.last_adjoint_solution
            adjoint = {
                "success": bool(solution is not None and solution.success),
                "result": None if solution is None else str(solution.result),
            }
            if q.grad is None or not torch.isfinite(q.grad).all():
                raise AdjointConvergenceError(
                    {"success": False, "result": "nonfinite_or_missing_gradient"}
                )
        return {
            "objective": float(total.detach()),
            "data_objective_mm2": float(data.detach()),
            "smoothness": float(smooth.detach()),
            "smoothness_penalty_mm2": float(
                (self.cfg.smoothness_weight * smooth).detach()
            ),
            "u": u.detach().cpu().numpy(),
            "forward": dict(self.physics.last_forward),
            "adjoint": adjoint,
        }


def endpoint_metrics(physics: FacePhysics, q: torch.Tensor, result: dict[str, Any]):
    u = result["u"]
    target = physics.target[physics.top]
    pred = u[physics.top]
    residual = pred - target
    target_rms = np.linalg.norm(target) / math.sqrt(len(target))
    fit_rms = np.linalg.norm(residual) / math.sqrt(len(target))
    motion_rms = np.linalg.norm(pred) / math.sqrt(len(target))
    projection = float(np.sum(pred * target) / np.sum(target**2))
    orthogonal = pred - projection * target
    deformation_det = physics.detf(u)
    ainv = np.asarray(activation_matrix(q.detach().cpu().numpy()))
    eig = np.linalg.eigvalsh(ainv)
    return {
        "target_rms_mm": 1000.0 * target_rms,
        "fit_rms_mm": 1000.0 * fit_rms,
        "fit_rms_over_D": fit_rms / target_rms,
        "motion_rms_mm": 1000.0 * motion_rms,
        "target_projection_amplitude": projection,
        "target_projection_residual_over_D": np.linalg.norm(orthogonal)
        / math.sqrt(len(target))
        / target_rms,
        "detF_min": float(deformation_det.min()),
        "detF_max": float(deformation_det.max()),
        "inverted_tetrahedra": int(np.count_nonzero(deformation_det <= 0.0)),
        "activation_eigen_min": float(eig.min()),
        "activation_eigen_max": float(eig.max()),
        "non_spd_active_tetrahedra": int(np.count_nonzero(eig[:, 0] <= 0.0)),
    }


def snapshot(
    path: Path,
    q: torch.Tensor,
    result: dict[str, Any],
    step: int,
) -> None:
    q_numpy = q.detach().cpu().numpy()
    np.savez_compressed(
        path,
        q=q_numpy,
        u=result["u"],
        Ainv=activation_matrix(q_numpy),
        step=np.asarray(step, dtype=np.int64),
        forward_success=np.asarray(result["forward"]["success"], dtype=np.bool_),
        adjoint_success=np.asarray(result["adjoint"]["success"], dtype=np.bool_),
        solver_valid=np.asarray(
            result["forward"]["success"] and result["adjoint"]["success"],
            dtype=np.bool_,
        ),
    )


def archive_sources(out: Path, cfg: Config) -> dict[str, Any]:
    source_dir = out / "sources"
    source_dir.mkdir()
    names = [
        "12-prepare-historical-fixture.py",
        "30-run-historical-adam.py",
        "experiment_profile.py",
        "historical_adam_physics.py",
    ]
    sources = {}
    for name in names:
        path = Path(__file__).parent / name
        shutil.copy2(path, source_dir / name)
        sources[name] = sha256(path)
    inputs = {
        name: sha256(cfg.fixture / name)
        for name in ("volume.vtu", "skin.vtp", "summary.json")
    }
    provenance = {
        "sources": sources,
        "inputs": inputs,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": os.sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    write_json(out / "provenance.json", provenance)
    return provenance


def run(cfg: Config) -> None:
    validate_config(cfg)
    if cfg.preflight:
        report = preflight(cfg)
        print(json.dumps(report, indent=2))
        return
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {out}")
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = archive_sources(out, cfg)
    configure()
    physics = FacePhysics(
        cfg.fixture,
        skin_factor=0.0,
        fat_factor=1.0,
        muscle_factor=1.0,
        rtol=cfg.forward_rtol,
        atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        soft_nu=0.49,
        fat_nu=0.49,
        fat_model="stable",
        target_name="Smile",
        target_scale=1.0,
    )
    if len(physics.ids) != 288_235 or physics.n_regions != 103:
        raise ValueError("runtime historical activation domain changed")
    objective = Objective(physics, cfg)
    q = torch.nn.Parameter(torch.zeros((len(physics.ids), 6)))
    optimizer = torch.optim.Adam([q], lr=cfg.inverse_lr, eps=cfg.adam_eps)
    start = time.perf_counter()
    trace: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    optimizer_events: list[dict[str, Any]] = []
    current = objective(q, np.zeros_like(physics.points))
    if not (current["forward"]["success"] and current["adjoint"]["success"]):
        write_json(
            out / "initial-solver-failure.json",
            {"forward": current["forward"], "adjoint": current["adjoint"]},
        )
        raise RuntimeError("initial equilibrium or adjoint solve was unsuccessful")
    best = {"step": 0, "q": q.detach().clone(), "result": current}
    last_solver_valid: dict[str, Any] | None = None
    status = "fixed_budget_completed_not_stationarity_certified"
    optimizer_updates = 0
    consecutive_solver_failures = 0
    try:
        for step in range(cfg.steps + 1):
            metrics = endpoint_metrics(physics, q, current)
            gradient_rms = float(
                torch.linalg.vector_norm(q.grad) / math.sqrt(q.numel())
            )
            row = {
                "step": step,
                "objective": current["objective"],
                "data_objective_mm2": current["data_objective_mm2"],
                "smoothness": current["smoothness"],
                "smoothness_penalty_mm2": current["smoothness_penalty_mm2"],
                "gradient_rms": gradient_rms,
                "elapsed_s": time.perf_counter() - start,
                "forward_steps": current["forward"]["steps"],
                "forward_grad_norm": current["forward"]["grad_norm"],
                **metrics,
            }
            solver_valid = bool(
                current["forward"]["success"] and current["adjoint"]["success"]
            )
            row["forward_success"] = bool(current["forward"]["success"])
            row["adjoint_success"] = bool(current["adjoint"]["success"])
            row["solver_valid"] = solver_valid
            if solver_valid:
                last_solver_valid = {"step": step, "row": dict(row)}
            consecutive_solver_failures = (
                0 if solver_valid else consecutive_solver_failures + 1
            )
            trace.append(row)
            receipts.append(
                {
                    "step": step,
                    "forward": current["forward"],
                    "adjoint": current["adjoint"],
                }
            )
            with (out / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(row), lineterminator="\n"
                )
                writer.writeheader()
                writer.writerows(trace)
            with (out / "solver-receipts.jsonl").open("w") as stream:
                for receipt in receipts:
                    stream.write(json.dumps(receipt, sort_keys=True) + "\n")
            cherries.log_metrics(
                {
                    "historical_adam/objective": row["objective"],
                    "historical_adam/fit_rms_mm": row["fit_rms_mm"],
                    "historical_adam/detF_min": row["detF_min"],
                    "historical_adam/gradient_rms": row["gradient_rms"],
                },
                step=step,
            )
            if solver_valid and current["objective"] < best["result"]["objective"]:
                best = {"step": step, "q": q.detach().clone(), "result": current}
            if step % cfg.checkpoint_interval == 0:
                snapshot(out / f"step-{step:04d}.npz", q, current, step)
            snapshot(out / "latest.npz", q, current, step)
            if step == cfg.steps:
                if consecutive_solver_failures:
                    status = (
                        "fixed_budget_completed_with_unresolved_solver_failures_"
                        "best_valid_retained"
                    )
                break
            if consecutive_solver_failures >= 3:
                with torch.no_grad():
                    q.copy_(best["q"])
                optimizer.param_groups[0]["lr"] *= 0.5
                optimizer.state.clear()
                event = {
                    "step": step,
                    "event": "three_solver_failures_restore_best_halve_lr_reset_adam_moments",
                    "restored_step": best["step"],
                    "new_learning_rate": optimizer.param_groups[0]["lr"],
                }
                optimizer_events.append(event)
                consecutive_solver_failures = 0
                write_json(out / "optimizer-events.json", optimizer_events)
                current = objective(q, best["result"]["u"])
                continue
            accepted_q = q.detach().clone()
            accepted = current
            optimizer.step()
            optimizer_updates += 1
            try:
                current = objective(q, accepted["u"])
            except (ForwardConvergenceError, AdjointConvergenceError) as error:
                with torch.no_grad():
                    q.copy_(accepted_q)
                receipt = {
                    "attempted_step": step + 1,
                    "type": type(error).__name__,
                    "message": str(error),
                    "receipt": error.receipt,
                }
                failures.append(receipt)
                write_json(out / "numerical-failures.json", failures)
                current = accepted
                status = "numerical_failure_best_valid_retained"
                break
    except KeyboardInterrupt:
        status = "interrupted_best_valid_retained"

    best_q = best["q"]
    best_result = best["result"]
    assert last_solver_valid is not None
    snapshot(out / "final.npz", best_q, best_result, best["step"])
    physics.save_mesh(
        out / "final.vtu",
        best_result["u"],
        activation_matrix(best_q).detach().cpu().numpy(),
    )
    summary = {
        "schema_version": 1,
        "status": status,
        "convergence": {
            "claimed": False,
            "label": status,
            "declared_steps": cfg.steps,
            "last_evaluated_step": trace[-1]["step"],
            "last_solver_valid_step": last_solver_valid["step"],
            "best_valid_step": best["step"],
            "best_endpoint_policy": (
                "finite objective with successful forward and adjoint solves; this is "
                "stricter than the June forward-success-only BestState selection"
            ),
        },
        "config": cfg.model_dump(mode="json"),
        "objective": {
            "data": "uniform Cartesian MSE times 1e6, exactly matching June",
            "smoothness_weight_mm2": cfg.smoothness_weight,
            "smoothness_graph": "within identical MuscleId shared tetrahedral faces only",
            "smoothness_length_m": cfg.smooth_length,
            "smoothness_normalization": "finite-volume conductance divided by active muscle-fraction volume and (-log(0.8))^2",
        },
        "optimizer": {
            "implementation": "torch.optim.Adam",
            "learning_rate": cfg.inverse_lr,
            "epsilon": cfg.adam_eps,
            "state_updates": optimizer_updates,
            "failure_semantics": "isolated unsuccessful forward/adjoint receipts remain visible and still advance Adam; three consecutive failures restore the best solver-valid state, halve lr, clear Adam moments, and skip that update, matching the June mandatory-baseline safety branch",
            "events": optimizer_events,
        },
        "materials": physics.material_spec,
        "solver": {
            "forward": physics.forward_tolerance,
            "adjoint": {
                "implementations": ["CupyCG", "CupyMinRes"],
                "max_steps": 10_000,
                "rtol": cfg.adjoint_rtol,
                "atol": 0.0,
            },
        },
        "mesh": {
            "points": len(physics.points),
            "tetrahedra": len(physics.tets),
            "active_tetrahedra": len(physics.ids),
            "scalar_controls": int(q.numel()),
            "muscle_regions": physics.n_regions,
            "within_same_MuscleId_shared_face_edges": len(physics.graph[0]),
            "fixed_vertices": int(
                np.count_nonzero(np.asarray(physics.mesh.point_data["IsFixed"], bool))
            ),
            "target_vertices": len(physics.top),
        },
        "best": {**trace[best["step"]], "output": "final.npz and final.vtu"},
        "last_solver_valid": last_solver_valid["row"],
        "last_evaluated": trace[-1],
        "numerical_failures": failures,
        "geometry_rejection_enabled": False,
        "provenance": provenance,
        "wall_s": time.perf_counter() - start,
    }
    write_json(out / "summary.json", summary)
    for name in (
        "config.json",
        "provenance.json",
        "trace.csv",
        "solver-receipts.jsonl",
        "final.npz",
        "final.vtu",
        "summary.json",
    ):
        cherries.log_output(out / name)


if __name__ == "__main__":
    cherries.main(
        run, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
