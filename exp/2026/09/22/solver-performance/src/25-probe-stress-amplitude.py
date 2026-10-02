"""Evaluate scaled fixed Smile stresses without changing the inverse trajectory.

Each scale starts from the recorded 20-update Hybrid endpoint's displacement,
keeps its jaw fixed, and solves the unchanged full-bone-and-eye contact model to
the production accepted-force tolerance.  No backward pass or optimizer update
is performed: this is an objective/feasibility amplitude probe only.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(EXPERIMENT / "src"), str(SOURCE_GROUP / "src")]

from joint_common import sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_fields import activation_regularizers, project_activation_
from remote_paths import install_loader_path_relocation
from smile_collision import audit_collision_state, audit_required_collision

SMILE = "Smile"
SMILE_INDEX = 12
FORCE_ATOL = 1.5192003475221146e-10


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_module(EXPERIMENT / "src/10-benchmark.py", "amplitude_benchmark")


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/smile-amplitude-probe-001"
    source_root: Path = EXPERIMENT.parents[4]
    origin_metadata: Path | None = None
    source_run: Path = EXPERIMENT / "data/smile-hybrid-comp07-production-003"
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    calibration_source: Path = SOURCE_GROUP / "data/expression-fitting-007"
    forward_wall_seconds: float | None = None
    ipc_threads: int = 8
    scales: str = "1,3,10"
    hybrid_floor: float = 1e-7


def record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def load_fitter() -> Any:
    return load_module(SOURCE_GROUP / "src/93-fit-expressions.py", "amplitude_fitter")


def calibration_weight(cfg: Config) -> float:
    value = json.loads((cfg.calibration_source / "calibration.json").read_text())
    protocol = json.loads((cfg.calibration_source / "protocol.json").read_text())
    assert value["success"] and value["expression_count"] == 36
    assert protocol["inputs_manifest_sha256"] == sha256(
        cfg.inputs_dir / "manifest.json"
    )
    return float(value["strong_weight"])


def make_fitter(cfg: Config, runner: Any, output: Path) -> Any:
    fit_cfg = runner.Config(
        _cli_parse_args=False,
        output_dir=output,
        inputs_dir=cfg.inputs_dir,
        calibration_source=None,
        forward_atol=FORCE_ATOL,
        adjoint_rtol=1e-7,
        ipc_threads=cfg.ipc_threads,
        pose_first=False,
        pose_collision=True,
    )
    fitter = runner.Fitter(fit_cfg)
    from accelerated_solvers import accelerate_runtime

    fitter.runtime = accelerate_runtime(
        fitter.runtime,
        "hybrid_diag",
        rest_points=fitter.physics.points,
        wall_seconds=cfg.forward_wall_seconds,
        linear_rtol=1e-3,
        max_newton_steps=100,
        newton_switch_atol=cfg.hybrid_floor,
    )
    fitter.physics.runtime = fitter.runtime
    fitter.smooth_weight = calibration_weight(cfg)
    assert fitter.runtime.tolerances["atol"] == FORCE_ATOL
    return fitter


def constraint_receipt(q: torch.Tensor, runner: Any, fitter: Any) -> dict[str, Any]:
    projected = q.detach().clone()
    project_activation_(projected, runner.CAP)
    projection_error = float((projected - q).abs().max())
    roughness = runner.neighbor_rms(q.detach(), fitter.graph)
    return {
        "spectral_projection_max_abs_change": projection_error,
        "spectral_cap_satisfied": projection_error <= 1e-12,
        "neighbor_rms": roughness,
        "neighbor_rms_budget": fitter.cfg.neighbor_rms_budget,
        "neighbor_budget_satisfied": roughness <= fitter.cfg.neighbor_rms_budget,
    }


def objective_metrics(
    fitter: Any, runner: Any, q: torch.Tensor, jaw: torch.Tensor, u: torch.Tensor
) -> dict[str, Any]:
    data, mse = fitter.data_loss(u, SMILE_INDEX)
    reg = activation_regularizers(q, fitter.graph)
    smooth = fitter.smooth_weight * reg["smoothness"]
    magnitude = fitter.cfg.magnitude_weight * reg["magnitude"]
    pose = fitter.cfg.jaw_weight * jaw.square().sum() / 6
    objective = data + smooth + magnitude + pose
    shape = fitter.physics.metrics(u, target_index=SMILE_INDEX)
    return {
        "objective": float(objective.detach()),
        "data": float(data.detach()),
        "fit_rms_mm": float(mse.detach().sqrt()) * 1000,
        "smoothness": float(reg["smoothness"].detach()),
        "weighted_smoothness": float(smooth.detach()),
        "magnitude": float(reg["magnitude"].detach()),
        "weighted_magnitude": float(magnitude.detach()),
        "weighted_jaw_prior": float(pose.detach()),
        "shape": shape,
        "gradients_or_error_correction_computed": False,
    }


def evaluate_scale(
    cfg: Config, runner: Any, state: dict[str, Any], scale: float
) -> dict[str, Any]:
    q = state["activation"].to("cuda") * scale
    jaw = state["jaw_normalized"].to("cuda").detach().clone()
    seed = state["displacement_m"].to("cuda").detach().clone()
    fitter = make_fitter(cfg, runner, cfg.output_dir / "scratch" / f"scale-{scale:g}")
    required = audit_required_collision(fitter.physics)
    constraints = constraint_receipt(q, runner, fitter)
    result: dict[str, Any] = {
        "scale": scale,
        "constraint": constraints,
        "collision_required": required,
        "jaw_fixed_from_step20": float(jaw[0]),
        "backward_or_adjoint": "not evaluated; objective-only probe",
    }
    if not (
        constraints["spectral_cap_satisfied"]
        and constraints["neighbor_budget_satisfied"]
    ):
        result.update(success=False, status="constraint_violation_not_solved")
        return result
    started = time.perf_counter()
    try:
        with torch.no_grad():
            u = fitter.solve(q, jaw, seed, jaw, f"Smile-amplitude-{scale:g}")
            metrics = objective_metrics(fitter, runner, q, jaw, u)
        axis = torch.as_tensor(
            fitter.physics.base.arrays["mandible_frame_world"][:, 0],
            device="cuda",
            dtype=jaw.dtype,
        )
        collision = audit_collision_state(
            fitter.physics, u, runner.hinge_pose(jaw, axis)
        )
        forward = copy.deepcopy(fitter.runtime.last_forward)
        gates = {
            "force": bool(forward["success"] and forward["grad_norm"] <= FORCE_ATOL),
            "contact": bool(
                forward["contact"]["contact_numerically_valid"]
                and collision["state_feasible"]
            ),
            "no_inversions": metrics["shape"]["inverted_tetrahedra"] == 0,
        }
        result.update(
            success=all(gates.values()),
            status="evaluated" if all(gates.values()) else "physical_gate_failed",
            seconds=time.perf_counter() - started,
            forward=forward,
            collision_state=collision,
            gates=gates,
            metrics=metrics,
        )
    except ForwardConvergenceError as error:
        result.update(
            success=False,
            status="forward_failed",
            seconds=time.perf_counter() - started,
            failure=str(error),
            forward=copy.deepcopy(fitter.runtime.last_forward),
        )
    return result


def main(cfg: Config) -> None:
    assert cfg.forward_wall_seconds is None or cfg.forward_wall_seconds > 0
    assert cfg.ipc_threads == 8 and cfg.hybrid_floor == 1e-7
    scales = tuple(float(value) for value in cfg.scales.split(","))
    assert scales == (1.0, 3.0, 10.0)
    latest = cfg.source_run / "arms/hybrid_diag/expressions/Smile/latest.pt"
    assert latest.is_file() and cfg.inputs_dir.is_dir()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    BENCHMARK.archive_benchmark_sources(cfg)
    install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    configure_cuda()
    runner = load_fitter()
    state = torch.load(latest, map_location="cpu", weights_only=False)
    assert state["expression"] == SMILE and state["accepted_steps"] == 20
    assert float(state["jaw_normalized"][0]) == 0.0
    source = {
        "checkpoint": record(latest),
        "source_summary": record(cfg.source_run / "summary.json"),
        "inputs_manifest": record(cfg.inputs_dir / "manifest.json"),
        "calibration": record(cfg.calibration_source / "calibration.json"),
        "script": record(Path(__file__)),
    }
    rows = []
    for scale in scales:
        row = evaluate_scale(cfg, runner, state, scale)
        rows.append(row)
        write_json(cfg.output_dir / "progress.json", {"source": source, "rows": rows})
        cherries.log_metrics(
            {
                f"amplitude/{scale:g}/success": float(row["success"]),
                f"amplitude/{scale:g}/seconds": row.get("seconds", 0.0),
            }
        )
    summary = {
        "schema": "smile-fixed-stress-amplitude-probe-v1",
        "success": all(row["success"] for row in rows),
        "scope": "fixed step-20 Hybrid Smile stress direction, fixed jaw/materials, fresh full-contact equilibria; no optimizer update, no backward or adjoint",
        "source": source,
        "config": cfg.model_dump(mode="json"),
        "production_force_atol": FORCE_ATOL,
        "ipc_threads_actual": int(ipctk.get_num_threads()),
        "rows": rows,
    }
    write_json(cfg.output_dir / "summary.json", summary)


if __name__ == "__main__":
    cherries.main(main, profile=BENCHMARK.ProfilePerformance)
