# ruff: noqa: CPY001, E402, EM101, PLR0915, PT018, TRY003
"""Bounded Smile forward/adjoint tolerance sensitivity at one fixed proposal.

Every arm reconstructs fresh bone-and-eye contact physics and begins from the
same frozen neutral displacement.  It is an accuracy probe, not an optimizer
trajectory or a cross-GPU speed benchmark.
"""

from __future__ import annotations

import copy
import hashlib
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

from accelerated_solvers import accelerate_runtime
from joint_common import sha256, write_json
from joint_equilibrium import ForwardConvergenceError, configure_cuda
from joint_expression_inputs import EyeExpressionInputs
from joint_fields import project_activation_
from remote_paths import install_loader_path_relocation
from smile_collision import audit_collision_state, audit_required_collision

SMILE = "Smile"
SMILE_INDEX = 12
ATOLS = (1e-8, 1e-9, 1.5192003475221146e-10, 1e-12)


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BENCHMARK = load_module(EXPERIMENT / "src/10-benchmark.py", "tolerance_benchmark")


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/smile-tolerance-probe-001"
    source_root: Path = EXPERIMENT.parents[4]
    origin_metadata: Path | None = None
    pilot_dir: Path = EXPERIMENT / "data/smile-hybrid-comp07-pilot-001"
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    calibration_source: Path = SOURCE_GROUP / "data/expression-fitting-007"
    forward_wall_seconds: float | None = None
    adjoint_rtol: float = 1e-7
    linear_rtol: float = 1e-3
    max_newton_steps: int = 100
    ipc_threads: int = 8
    proposal_scale: float = 1.0
    atols: str = "1e-8,1e-9,1.5192003475221146e-10,1e-12"
    reference_atol: float = 1e-12
    include_pncg: bool = False


def file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": sha256(path)}


def tensor_sha256(value: torch.Tensor) -> str:
    value = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(str(tuple(value.shape)).encode())
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_fitter() -> Any:
    return load_module(SOURCE_GROUP / "src/93-fit-expressions.py", "tolerance_fitter")


def first_proposal(
    initial: dict[str, Any], runner: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.nn.Parameter(initial["activation"].clone())
    jaw = torch.nn.Parameter(initial["jaw_normalized"].clone())
    optimizer = torch.optim.Adam((q, jaw), lr=0.003)
    q.grad, jaw.grad = initial["gradient_q"].clone(), initial["gradient_jaw"].clone()
    optimizer.step()
    project_activation_(q, runner.CAP)
    with torch.no_grad():
        jaw.clamp_(runner.HINGE_MIN, runner.HINGE_MAX)
    return q.detach(), jaw.detach()


def calibration_weight(cfg: Config) -> tuple[float, dict[str, Any]]:
    calibration = cfg.calibration_source / "calibration.json"
    protocol = cfg.calibration_source / "protocol.json"
    values = json.loads(calibration.read_text())
    parent = json.loads(protocol.read_text())
    assert values["success"] and values["expression_count"] == 36
    assert parent["inputs_manifest_sha256"] == sha256(cfg.inputs_dir / "manifest.json")
    return float(values["strong_weight"]), {
        "calibration": file_record(calibration),
        "protocol": file_record(protocol),
    }


def make_fitter(cfg: Config, runner: Any, atol: float, method: str) -> Any:
    fit_cfg = runner.Config(
        _cli_parse_args=False,
        output_dir=cfg.output_dir / "scratch" / f"{method}-{atol:.3e}",
        inputs_dir=cfg.inputs_dir,
        calibration_source=None,
        forward_atol=atol,
        adjoint_rtol=cfg.adjoint_rtol,
        ipc_threads=cfg.ipc_threads,
        max_backtracks=12,
        pose_first=False,
        pose_collision=True,
    )
    fitter = runner.Fitter(fit_cfg)
    fitter.smooth_weight, _ = calibration_weight(cfg)
    if method == "newton_diag":
        fitter.runtime = accelerate_runtime(
            fitter.runtime,
            "newton_diag",
            rest_points=fitter.physics.points,
            wall_seconds=cfg.forward_wall_seconds,
            linear_rtol=cfg.linear_rtol,
            max_newton_steps=cfg.max_newton_steps,
        )
        fitter.physics.runtime = fitter.runtime
    assert fitter.runtime.tolerances["atol"] == atol
    assert fitter.runtime.tolerances["adjoint_rtol"] == cfg.adjoint_rtol
    return fitter


def scalar_metrics(
    candidate: dict[str, Any], initial_objective: float
) -> dict[str, Any]:
    metrics = candidate["metrics"]
    decrease = initial_objective - metrics["objective"]
    correction = metrics["primal_objective_correction_estimate"]
    return {
        "objective": metrics["objective"],
        "fit_rms_mm": metrics["fit_rms_mm"],
        "primal_objective_correction_estimate": correction,
        "initial_to_proposal_objective_decrease": decrease,
        "correction_over_abs_trial_decrease": abs(correction)
        / max(abs(decrease), 1e-300),
        "forward": metrics["forward"],
        "adjoint": metrics["adjoint"],
        "shape": metrics["shape"],
    }


def solve_one(
    cfg: Config,
    runner: Any,
    *,
    method: str,
    atol: float,
    q: torch.Tensor,
    jaw: torch.Tensor,
    seed: torch.Tensor,
    initial_objective: float,
) -> dict[str, Any]:
    fitter = make_fitter(cfg, runner, atol, method)
    collision = audit_required_collision(fitter.physics)
    q_i = q.detach().clone().requires_grad_()
    jaw_i = jaw.detach().clone().requires_grad_()
    started = time.perf_counter()
    try:
        candidate = fitter.evaluate(
            SMILE_INDEX, q_i, jaw_i, seed, torch.zeros_like(jaw_i)
        )
    except ForwardConvergenceError as error:
        return {
            "success": False,
            "method": method,
            "forward_atol": atol,
            "wall_seconds": time.perf_counter() - started,
            "failure": str(error),
            "forward": copy.deepcopy(fitter.runtime.last_forward),
        }
    metrics = scalar_metrics(candidate, initial_objective)
    shape = metrics["shape"]
    forward = metrics["forward"]
    axis = torch.as_tensor(
        fitter.physics.base.arrays["mandible_frame_world"][:, 0],
        device=jaw_i.device,
        dtype=jaw_i.dtype,
    )
    collision_state = audit_collision_state(
        fitter.physics,
        candidate["displacement_m"],
        runner.hinge_pose(jaw_i.detach(), axis),
    )
    return {
        "success": bool(
            forward["success"]
            and forward["grad_norm"] <= atol
            and shape["inverted_tetrahedra"] == 0
            and forward["contact"]["contact_numerically_valid"]
            and collision_state["state_feasible"]
        ),
        "method": method,
        "forward_atol": atol,
        "wall_seconds": time.perf_counter() - started,
        "collision_required": collision,
        "collision_state": collision_state,
        "displacement_m": candidate["displacement_m"].detach().cpu(),
        "gradient_q": candidate["gradient_q"].detach().cpu(),
        "gradient_jaw": candidate["gradient_jaw"].detach().cpu(),
        **metrics,
    }


def comparison_to_tight(
    row: dict[str, Any], tight: dict[str, Any], weights: torch.Tensor, obs: torch.Tensor
) -> dict[str, float]:
    delta = row["displacement_m"] - tight["displacement_m"]
    skin_rms_m = (weights * delta[obs].square().sum(-1)).sum().sqrt()

    def vector_error(
        value: torch.Tensor, reference: torch.Tensor
    ) -> tuple[float, float]:
        norm = torch.linalg.vector_norm(reference)
        relative = float(
            torch.linalg.vector_norm(value - reference) / max(float(norm), 1e-300)
        )
        cosine = float(
            torch.dot(value.flatten(), reference.flatten())
            / max(float(torch.linalg.vector_norm(value) * norm), 1e-300)
        )
        return relative, cosine

    q_relative, q_cosine = vector_error(row["gradient_q"], tight["gradient_q"])
    jaw_relative, jaw_cosine = vector_error(row["gradient_jaw"], tight["gradient_jaw"])
    return {
        "skin_weighted_rms_difference_mm": float(skin_rms_m) * 1000,
        "skin_max_node_difference_mm": float(
            torch.linalg.vector_norm(delta[obs], dim=-1).max()
        )
        * 1000,
        "objective_relative_difference": abs(row["objective"] - tight["objective"])
        / max(abs(tight["objective"]), 1e-300),
        "activation_gradient_relative_l2": q_relative,
        "activation_gradient_cosine": q_cosine,
        "jaw_gradient_relative_l2": jaw_relative,
        "jaw_gradient_cosine": jaw_cosine,
        "definitions": {
            "skin": "area-weighted RMS and max Euclidean displacement over observed skin nodes, millimetres",
            "objective": "dimensionless normalized data plus fixed regularizers",
            "activation_gradient": "L2 over all dimensionless six-component active-stress coordinates",
            "jaw_gradient": "L2 over the one normalized hinge-angle coordinate",
        },
    }


def jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {"tensor_sha256": tensor_sha256(value), "shape": list(value.shape)}
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def main(cfg: Config) -> None:
    assert cfg.forward_wall_seconds is None or cfg.forward_wall_seconds > 0
    assert cfg.adjoint_rtol == 1e-7
    assert cfg.ipc_threads > 0
    assert 0 < cfg.proposal_scale <= 1
    atols = tuple(float(value) for value in cfg.atols.split(","))
    assert atols and all(atol > 0 for atol in atols)
    assert cfg.reference_atol > 0
    assert cfg.pilot_dir.is_dir() and cfg.inputs_dir.is_dir()
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    BENCHMARK.archive_benchmark_sources(cfg)
    install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    configure_cuda()
    runner = load_fitter()
    initial_path = cfg.pilot_dir / "arms/hybrid_diag/expressions/Smile/initial.pt"
    seed_path = cfg.pilot_dir / "shared-neutral-init.pt"
    initial = torch.load(initial_path, map_location="cuda", weights_only=False)
    seed = torch.load(seed_path, map_location="cuda", weights_only=False)[
        "displacement_m"
    ]
    q, jaw = first_proposal(initial, runner)
    q, jaw = q * cfg.proposal_scale, jaw * cfg.proposal_scale
    assert q.is_cuda and jaw.is_cuda and seed.is_cuda
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    assert inputs.expression_names[SMILE_INDEX] == SMILE
    weight = torch.as_tensor(
        inputs.arrays["observation_weight_normalized"], device="cpu"
    )
    obs = torch.as_tensor(
        inputs.arrays["observation_node_ids"], device="cpu", dtype=torch.long
    )
    initial_objective = float(initial["metrics"]["objective"])
    source = {
        "script": file_record(Path(__file__)),
        "accelerated_solvers": file_record(EXPERIMENT / "src/accelerated_solvers.py"),
        "fitter": file_record(SOURCE_GROUP / "src/93-fit-expressions.py"),
        "initial_checkpoint": file_record(initial_path),
        "shared_seed": file_record(seed_path),
        "inputs_manifest": file_record(cfg.inputs_dir / "manifest.json"),
        "inputs_state": file_record(cfg.inputs_dir / "state.npz"),
        "calibration": calibration_weight(cfg)[1],
        "proposal": {
            "activation_tensor_sha256": tensor_sha256(q),
            "jaw_tensor_sha256": tensor_sha256(jaw),
        },
    }
    rows: list[dict[str, Any]] = []
    # Tight reference comes first so every comparison is against one endpoint.
    tight = solve_one(
        cfg,
        runner,
        method="newton_diag",
        atol=cfg.reference_atol,
        q=q,
        jaw=jaw,
        seed=seed,
        initial_objective=initial_objective,
    )
    if not tight["success"]:
        write_json(
            cfg.output_dir / "summary.json",
            {"success": False, "source": source, "tight": jsonable(tight)},
        )
        raise RuntimeError("tight direct-Newton reference failed")
    rows.append(tight)
    for atol in atols:
        if atol == cfg.reference_atol:
            continue
        row = solve_one(
            cfg,
            runner,
            method="newton_diag",
            atol=atol,
            q=q,
            jaw=jaw,
            seed=seed,
            initial_objective=initial_objective,
        )
        if row["success"]:
            row["comparison_to_tight"] = comparison_to_tight(row, tight, weight, obs)
        rows.append(row)
        write_json(
            cfg.output_dir / "progress.json", jsonable({"source": source, "rows": rows})
        )
        cherries.log_metrics(
            {
                f"newton/{atol:.3e}/success": float(row["success"]),
                f"newton/{atol:.3e}/wall_seconds": row["wall_seconds"],
            }
        )
    if cfg.include_pncg:
        for atol in ATOLS[:2]:
            row = solve_one(
                cfg,
                runner,
                method="original_pncg",
                atol=atol,
                q=q,
                jaw=jaw,
                seed=seed,
                initial_objective=initial_objective,
            )
            if row["success"]:
                row["comparison_to_tight"] = comparison_to_tight(
                    row, tight, weight, obs
                )
            rows.append(row)
            write_json(
                cfg.output_dir / "progress.json",
                jsonable({"source": source, "rows": rows}),
            )
    summary = {
        "schema": "smile-forward-tolerance-sensitivity-v1",
        "success": all(row["success"] for row in rows),
        "scope": "fixed first projected-Adam Smile active-stress proposal and common neutral seed; fresh direct Newton solves; not cross-GPU speed or an inverse-fit trajectory",
        "reference": "fresh newton_diag solve at 1e-12",
        "source": source,
        "config": cfg.model_dump(mode="json"),
        "ipc_threads_actual": int(ipctk.get_num_threads()),
        "rows": jsonable(rows),
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "tolerance_probe/success": float(summary["success"]),
            "tolerance_probe/arms": len(rows),
        }
    )
    if not summary["success"]:
        raise RuntimeError("one or more tolerance arms failed")


if __name__ == "__main__":
    cherries.main(main, profile=BENCHMARK.ProfilePerformance)
