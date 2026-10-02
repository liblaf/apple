# ruff: noqa: E402, PLR0915, SLF001
"""Profile complete saved-state Smile updates, with sequential GPU ownership."""

from __future__ import annotations

import copy
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
JOINT = GROUP.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(GROUP / "src"), str(JOINT / "src")]
import accelerated_solvers
import gpu_contact
import joint_equilibrium
from adjoint_tolerance_common import (
    build_fixed_state_context,
    file_record,
    fixed_state_output,
    load_protocol_weights,
    tensor_sha256,
)
from inverse_timing import install_inverse_timing
from joint_common import write_json
from joint_equilibrium import configure_cuda
from remote_paths import install_loader_path_relocation
from smile_collision import audit_required_collision


class Config(cherries.BaseConfig):
    output_dir: Path
    source_root: Path = GROUP.parents[4]
    inputs_dir: Path = JOINT / "data/expression-inputs-002"
    cases: str = "regularized,zero_smoothing"
    variants: str = "historical,candidate"
    ipc_threads: int = 8
    forward_atol: float = 1e-8
    newton_switch_atol: float = 1e-7
    forward_wall_seconds: float | None = None
    cuda_sync: bool = True


def prior_adjoint(
    cfg: Config, checkpoint: Path, output: Path
) -> tuple[Any, torch.Tensor, dict]:
    """Reconstruct a common previous-state warm adjoint, excluded from timing."""
    started = time.perf_counter()
    context = build_fixed_state_context(
        checkpoint_path=checkpoint,
        inputs_dir=cfg.inputs_dir,
        output_dir=output,
        forward_atol=cfg.forward_atol,
        adjoint_rtol=1e-7,
        ipc_threads=cfg.ipc_threads,
    )
    key = "profile-prior-adjoint"
    saved = fixed_state_output(context, key=key)
    loss, _ = context.fitter.data_loss(saved[: context.fem_node_count], context.index)
    torch.autograd.grad(loss, (context.q, context.jaw))
    warm = context.fitter.runtime.warm_adjoints[key].detach().clone()
    torch.cuda.synchronize()
    receipt = {
        "seconds": time.perf_counter() - started,
        "scope": "reconstructed previous-state adjoint at CPU rtol1e-7; excluded from update; shared across variants",
        "sha256": tensor_sha256(warm),
        "adjoint": copy.deepcopy(context.fitter.runtime.last_adjoint),
        "forward_count": context.fitter.runtime.forward_count,
    }
    assert receipt["forward_count"] == 0
    return context.runner, warm, receipt


def install_stage_hooks(installed: Any, fitter: Any, runner: Any) -> None:
    timer = installed.timer
    timer.patch(fitter, "evaluate", "evaluate")
    timer.patch(fitter.runtime, "primal", "forward")
    timer.patch(joint_equilibrium._Implicit, "backward", "adjoint")
    timer.patch(accelerated_solvers.AcceptedForcePncg, "minimize", "coarse_pncg")
    timer.patch(accelerated_solvers, "safeguarded_newton", "newton")
    timer.patch(accelerated_solvers, "pcg", "newton_cg")
    timer.patch(fitter.runtime.solver, "solve", "adjoint_linear")
    timer.patch(torch.autograd, "grad", "autograd")
    timer.patch(torch.optim.Adam, "step", "adam_step")
    timer.patch(runner, "project_activation_", "stress_projection")
    timer.patch(runner, "activation_regularizers", "regularizers")
    timer.patch(runner, "neighbor_rms", "roughness_metric")
    timer.patch(runner, "stationarity", "stationarity_metric")
    timer.patch(fitter.physics, "metrics", "shape_metrics")
    timer.patch(fitter, "data_loss", "skin_loss")
    timer.patch(fitter, "contact_gate", "contact_gate")
    timer.patch(fitter, "publish", "status_write")
    timer.patch(runner, "atomic_torch", "checkpoint_write")
    timer.patch(runner, "append", "log_append")
    timer.patch(torch, "load", "checkpoint_read")
    timer.patch(gpu_contact.GpuContactHessian, "hess_prod", "gpu_contact/hess_prod")
    timer.patch(gpu_contact.GpuContactHessian, "_upload", "gpu_contact/upload")
    timer.patch(gpu_contact.GpuContactHessian, "_assemble", "gpu_contact/assembly")
    assert not timer.missing, timer.missing


def profile_one(
    cfg: Config,
    runner: Any,
    checkpoint: Path,
    warm: torch.Tensor,
    run_dir: Path,
    directory: Path,
    variant: str,
) -> dict:
    setup_started = time.perf_counter()
    expression_dir = directory / "expressions/Smile"
    expression_dir.mkdir(parents=True)
    shutil.copy2(checkpoint, expression_dir / "latest.pt")
    old = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert not old["inverse_converged"]
    assert old["outer_step_policy"] == "full_adam"
    weights = load_protocol_weights(run_dir)
    assert weights["magnitude_weight"] == weights["jaw_weight"] == 0
    assert weights["learning_rate"] == 0.3
    candidate = variant == "candidate"
    fitter = runner.Fitter(
        runner.Config(
            _cli_parse_args=False,
            output_dir=directory,
            inputs_dir=cfg.inputs_dir,
            calibration_source=None,
            forward_atol=cfg.forward_atol,
            adjoint_rtol=1e-4 if candidate else 1e-7,
            ipc_threads=cfg.ipc_threads,
            learning_rate=0.3,
            magnitude_weight=0.0,
            jaw_weight=0.0,
            outer_step_policy="full_adam",
            trial_prescreen=False,
            pose_first=False,
            pose_collision=True,
        )
    )
    fitter.smooth_weight = weights["smoothness_weight"]
    runtime = accelerated_solvers.accelerate_runtime(
        fitter.runtime,
        "hybrid_diag",
        rest_points=fitter.physics.points,
        wall_seconds=cfg.forward_wall_seconds,
        linear_rtol=1e-3,
        max_newton_steps=100,
        newton_switch_atol=cfg.newton_switch_atol,
        shift_policy="reset",
    )
    fitter.runtime = runtime
    fitter.physics.runtime = runtime
    runtime.warm_adjoints["Smile"] = warm.clone()
    assert tensor_sha256(runtime.warm_adjoints["Smile"]) == tensor_sha256(warm)
    gpu = (
        gpu_contact.install_adjoint_gpu_contact(runtime, fitter.physics)
        if candidate
        else None
    )
    coverage = audit_required_collision(fitter.physics)
    original_evaluate = fitter.evaluate

    def evaluate_with_geometry_gate(*args: Any, **kwargs: Any) -> dict:
        value = original_evaluate(*args, **kwargs)
        assert value["metrics"]["shape"]["inverted_tetrahedra"] == 0
        return value

    fitter.evaluate = evaluate_with_geometry_gate
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - setup_started
    installed = install_inverse_timing(runtime.forward.model, cuda_sync=cfg.cuda_sync)
    success = False
    error = None
    started = time.perf_counter()
    try:
        install_stage_hooks(installed, fitter, runner)
        with installed.timer.scope("inverse_update"):
            fitter.fit_step(12)
        success = True
    except Exception as failure:
        error = {"type": type(failure).__name__, "message": str(failure)}
        raise
    finally:
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        profiling = installed.report()
        installed.uninstall()
        gpu_counts = (
            None if gpu is None else {"uploads": gpu.uploads, "products": gpu.products}
        )
        if gpu is not None:
            gpu.uninstall()
        write_json(
            directory / "timing.json",
            {
                "success": success,
                "failure": error,
                "variant": variant,
                "setup_seconds_excluded": setup_seconds,
                "outer_profile_wall_seconds": seconds,
                "profiling": profiling,
                "coverage": coverage,
                "gpu_contact": gpu_counts,
                "forward": copy.deepcopy(runtime.last_forward),
                "adjoint": copy.deepcopy(runtime.last_adjoint),
            },
        )
    new = torch.load(
        expression_dir / "latest.pt", map_location="cpu", weights_only=False
    )
    assert new["accepted_steps"] == old["accepted_steps"] + 1
    assert new["accepted_fraction"] == 1.0
    assert new["metrics"]["shape"]["inverted_tetrahedra"] == 0
    if candidate:
        assert gpu_counts["uploads"] > 0
        assert gpu_counts["products"] > 0
    receipt = {
        "success": True,
        "variant": variant,
        "input_checkpoint": file_record(checkpoint),
        "output_checkpoint": file_record(expression_dir / "latest.pt"),
        "timing": file_record(directory / "timing.json"),
        "from_step": old["accepted_steps"],
        "to_step": new["accepted_steps"],
        "fit_rms_mm": new["metrics"]["fit_rms_mm"],
        "weights": weights,
    }
    del fitter, runtime, old, new
    gc.collect()
    torch.cuda.empty_cache()
    return receipt


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists()
    cases = cfg.cases.split(",")
    variants = cfg.variants.split(",")
    assert set(cases) <= {"regularized", "zero_smoothing"}
    assert set(variants) <= {"historical", "candidate"}
    assert len(cases) == len(set(cases))
    assert len(variants) == len(set(variants))
    cfg.output_dir.mkdir(parents=True)
    install_loader_path_relocation(source_root=cfg.source_root)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    source_dir = cfg.output_dir / "sources"
    source_dir.mkdir()
    sources = [
        Path(__file__),
        GROUP / "src/inverse_timing.py",
        GROUP / "src/accelerated_solvers.py",
        GROUP / "src/gpu_contact.py",
        GROUP / "src/adjoint_tolerance_common.py",
        JOINT / "src/93-fit-expressions.py",
        JOINT / "src/joint_equilibrium.py",
        JOINT / "src/joint_expression_equilibrium.py",
    ]
    for path in sources:
        shutil.copy2(path, source_dir / path.name)
    protocol = {
        "schema": "complete-inverse-update-duration-profile-v1",
        "config": cfg.model_dump(mode="json"),
        "sources": {p.name: file_record(p) for p in sources},
        "scope": "One full projected-Adam update from saved step15, including checkpoint I/O; setup and reconstructed prior adjoint excluded; no outer rejection",
        "warm_start": "A common previous-state CPU1e-7 adjoint is reconstructed once per checkpoint, because original warm adjoints were not checkpointed",
        "timing_limit": "Synchronized hierarchical scopes perturb overlap; use exclusive times for additive totals and inclusive times only for nested explanations",
        "runs": [],
    }
    write_json(cfg.output_dir / "protocol.json", protocol)
    for case in cases:
        name = (
            "smile-fit-adam03-unconditional-004"
            if case == "regularized"
            else "smile-fit-adam03-no-smoothness-005"
        )
        run_dir = GROUP / "data" / name
        checkpoint = (
            run_dir / "arms/hybrid_diag/expressions/Smile/comparison-step-00015.pt"
        )
        case_dir = cfg.output_dir / case
        case_dir.mkdir()
        runner, warm, warm_receipt = prior_adjoint(
            cfg, checkpoint, case_dir / "warm-setup"
        )
        write_json(case_dir / "warm-start.json", warm_receipt)
        torch.save(warm.cpu(), case_dir / "warm-adjoint.pt")
        for variant in variants:
            cherries.set_step(len(protocol["runs"]))
            result = profile_one(
                cfg, runner, checkpoint, warm, run_dir, case_dir / variant, variant
            )
            protocol["runs"].append({"case": case, **result})
            write_json(cfg.output_dir / "progress.json", protocol)
            print(
                json.dumps({"case": case, "variant": variant, "success": True}),
                flush=True,
            )
        del warm
    protocol["success"] = True
    write_json(cfg.output_dir / "summary.json", protocol)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main)
