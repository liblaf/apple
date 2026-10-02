# ruff: noqa: BLE001, E402, PLR0911
"""Matched saved-state primal and implicit-adjoint solver benchmark.

The script deliberately reconstructs a fresh eye-inclusive physics object for
every solver/case.  It never resumes or writes into the 21 September fitting
runs.  Its only input mutations are byte-for-byte frozen copies beneath this
experiment's Cherries output directory.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path.insert(0, str(SOURCE_GROUP / "src"))

from joint_common import sha256, write_json
from joint_equilibrium import configure_cuda
from joint_expression_equilibrium import (
    FeasibleExpressionProblem,
    install_expression_runtime,
)
from joint_expression_inputs import EyeExpressionInputs
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem

LOG = logging.getLogger(__name__)
RUN006 = SOURCE_GROUP / "data/expression-fitting-006/expressions/MouthOpen/initial.pt"
RUN007 = SOURCE_GROUP / "data/expression-fitting-007/expressions/MouthOpen/initial.pt"


class ProfilePerformance(profiles.Profile):
    """Cherries evidence without requiring a Git checkout on a compute host."""

    def init(self) -> core.Run:
        os.environ.setdefault("COMET_AUTO_LOG_GIT_METADATA", "false")
        os.environ.setdefault("COMET_AUTO_LOG_GIT_PATCH", "false")
        run = core.run
        run.plugins.register(
            plugins.Comet(run=run, disabled=os.environ.get("DEBUG") == "1")
        )
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/solver-performance-001"
    source_root: Path = EXPERIMENT.parents[4]
    origin_metadata: Path | None = None
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    run006_initial: Path = RUN006
    run007_initial: Path = RUN007
    methods: str = "baseline,cached_pncg,newton_diag"
    cases: str = "collision_off_pose,contact_expression"
    collision_off_jaw_degrees: float = 1.0
    contact_jaw_degrees: float = 0.03125
    wall_seconds: float | None = None
    linear_rtol: float = 1e-3
    max_newton_steps: int = 100
    ipc_threads: int | None = None


def archive_benchmark_sources(cfg: Config) -> dict[str, Any]:
    """Copy the exact runtime sources and record a Git-independent provenance.

    The optional staging metadata comes from the originating checkout.  It is
    evidence only; no remote command assumes that a .git directory exists.
    """
    target = cfg.output_dir / "sources"
    assert not target.exists()
    sources = (
        (EXPERIMENT / "src", "solver-performance"),
        (cfg.source_root / "src/liblaf/apple", "apple"),
        (
            cfg.source_root / "exp/2026/09/21/joint-activation-material-mandible/src",
            "joint-experiment",
        ),
        (
            cfg.source_root / "exp/2026/09/07/tensor-active-stress/src",
            "tensor-reference",
        ),
    )
    for source, name in sources:
        assert source.is_dir(), source
        shutil.copytree(
            source, target / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    origin = None
    if cfg.origin_metadata is not None:
        origin = json.loads(cfg.origin_metadata.read_text())
    receipt = {
        "source_root": str(cfg.source_root.resolve()),
        "origin_metadata": origin,
        "git_required_on_runner": False,
        "sources": {
            str(path.relative_to(target)): sha256(path)
            for path in sorted(target.rglob("*.py"))
        },
    }
    write_json(cfg.output_dir / "provenance.json", receipt)
    return receipt


def load_runner() -> Any:
    """Load fitting helpers without constructing ``Fitter`` or launching a fit."""
    path = SOURCE_GROUP / "src/93-fit-expressions.py"
    spec = importlib.util.spec_from_file_location("solver_benchmark_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensor_sha256(value: torch.Tensor) -> str:
    array = np.ascontiguousarray(value.detach().cpu().to(torch.float64).numpy())
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode())
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return jsonable(value.detach().cpu().item())
        return {"tensor_sha256": tensor_sha256(value), "shape": list(value.shape)}
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, np.ndarray):
        return {
            "array_sha256": hashlib.sha256(value.tobytes()).hexdigest(),
            "shape": list(value.shape),
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def synchronized_time(operation: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def gpu_snapshot() -> dict[str, Any]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "perf_counter_seconds": time.perf_counter(),
        "returncode": result.returncode,
        "query": result.stdout.strip()
        if result.returncode == 0
        else result.stderr.strip(),
    }


def freeze(path: Path, directory: Path) -> dict[str, str]:
    """Copy an immutable source input and prove it did not change while copying."""
    path = path.resolve()
    assert path.is_file()
    before = sha256(path)
    destination = directory / path.name
    shutil.copy2(path, destination)
    after = sha256(path)
    copied = sha256(destination)
    assert before == after == copied, path
    return {"source": str(path), "copy": str(destination), "sha256": before}


def parse_names(value: str, *, allowed: set[str]) -> tuple[str, ...]:
    names = tuple(item.strip() for item in value.split(",") if item.strip())
    assert names, names
    assert len(set(names)) == len(names), names
    assert set(names) <= allowed, names
    return names


def load_checkpoint(path: Path) -> dict[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    assert value["expression"] == "MouthOpen"
    assert value["fit_stage"] == "pose_only"
    assert value["activation"].shape[1] == 6
    assert value["jaw_normalized"].shape == (1,)
    return value


def make_fixture(
    cfg: Config, case: str, frozen: dict[str, dict[str, str]]
) -> tuple[EyeExpressionInputs, Any, dict[str, Any], torch.Tensor, torch.Tensor]:
    """Build a fresh runtime and one specified immutable saved-state proposal."""
    inputs = EyeExpressionInputs.load(cfg.inputs_dir)
    checkpoint_record = frozen[case]
    checkpoint = load_checkpoint(Path(checkpoint_record["copy"]))
    physics, _baseline = inputs.build_physics()
    runtime = physics.runtime
    runtime.tolerances["atol"] = 1.5192003475221146e-10
    runtime.tolerances["adjoint_rtol"] = 1e-7
    # ``build_physics`` installs the accepted-force contact runtime itself.
    assert type(runtime).__name__ == "ExpressionEquilibrium"
    if case == "collision_off_pose":
        from joint_collision_off_pose import install_collision_off_pose_runtime

        runtime = install_collision_off_pose_runtime(physics)
        jaw = checkpoint["jaw_normalized"].clone() + cfg.collision_off_jaw_degrees / 10
    elif case == "contact_expression":
        runtime = install_expression_runtime(physics)
        jaw = checkpoint["jaw_normalized"].clone() + cfg.contact_jaw_degrees / 10
    else:
        raise ValueError(case)
    q = checkpoint["activation"].clone()
    assert not bool(torch.count_nonzero(q))
    return inputs, physics, checkpoint, q, jaw


def operation_counts(receipt: dict[str, Any]) -> dict[str, Any]:
    """Keep owned counters verbatim while exposing common baseline fields."""
    result = {
        key: jsonable(receipt[key])
        for key in (
            "operation_counts",
            "steps",
            "hvp_count",
            "gradient_count",
            "line_search_trial_count",
            "linear_matvec_count",
            "newton_steps",
            "coarse_steps",
            "coarse_threshold",
            "coarse_terminal_force",
            "rejected_contact_trials",
            "counts",
        )
        if key in receipt
    }
    trace = receipt.get("trace")
    if isinstance(trace, list):
        result["trace_entries"] = len(trace)
        result["trace_linear_matvec_count"] = sum(
            int(row.get("linear_matvec_count", 0))
            for row in trace
            if isinstance(row, dict)
        )
        result["trace_line_search_trials"] = sum(
            len(row.get("trials", ())) for row in trace if isinstance(row, dict)
        )
    return result


def warm_kernels(
    physics: Any,
    runtime: Any,
    *,
    q: torch.Tensor,
    jaw: torch.Tensor,
    seed: torch.Tensor,
    seed_jaw: torch.Tensor,
    runner: Any,
) -> None:
    """Compile the owned initial-state kernels outside every timed solver arm."""
    model = runtime.forward.model
    pose = runner.hinge_pose(
        jaw,
        torch.as_tensor(physics.arrays["mandible_frame_world"][:, 0], device="cuda"),
    )
    seed_pose = runner.hinge_pose(
        seed_jaw,
        torch.as_tensor(physics.arrays["mandible_frame_world"][:, 0], device="cuda"),
    )
    model.set_materials(
        physics.expression_materials(
            skin_multiplier=torch.ones((), device="cuda", dtype=torch.float64),
            active_stress=runner.activation_stresses_mpa(
                q.detach(), runner.REFERENCE_MPA
            ),
        )
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    full_seed = physics.full_skull.extend_seed(seed.detach(), seed_pose.detach())
    state = model.State(u=model.dof_map.to_full(model.dof_map.to_free(full_seed)))
    collision = model.collision
    if collision is None:
        problem = ForwardProblem(model=model)
    else:
        state.collision = collision.state_at(state.u)
        problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.95)
    direction = torch.ones_like(model.dof_map.to_free(state.u))
    with torch.no_grad():
        problem.fun(state)
        problem.grad(state)
        problem.hess_diag(state)
        problem.hess_quad(state, direction)
        problem.hess_prod(state, direction)
    torch.cuda.synchronize()


def run_one(
    cfg: Config,
    *,
    case: str,
    method: str,
    frozen: dict[str, dict[str, str]],
    runner: Any,
) -> dict[str, Any]:
    inputs, physics, checkpoint, q_cpu, jaw_cpu = make_fixture(cfg, case, frozen)
    reference = physics.runtime
    if method != "original":
        from accelerated_solvers import accelerate_runtime

        physics.runtime = accelerate_runtime(
            reference,
            method,
            rest_points=physics.points,
            wall_seconds=cfg.wall_seconds,
            linear_rtol=cfg.linear_rtol,
            max_newton_steps=cfg.max_newton_steps,
        )
    runtime = physics.runtime
    q = q_cpu.to(device="cuda", dtype=torch.float64).requires_grad_()
    jaw = jaw_cpu.to(device="cuda", dtype=torch.float64).requires_grad_()
    seed = checkpoint["displacement_m"].to(device="cuda", dtype=torch.float64)
    seed_jaw = checkpoint["jaw_normalized"].to(device="cuda", dtype=torch.float64)
    index = int(checkpoint["expression_index"])
    scale = torch.as_tensor(
        inputs.arrays["expression_displacement_m"][index], device="cuda"
    )
    weights = torch.as_tensor(
        inputs.arrays["observation_weight_normalized"], device="cuda"
    )
    observation = torch.as_tensor(
        inputs.arrays["observation_node_ids"], device="cuda", dtype=torch.long
    )
    target = torch.as_tensor(
        inputs.arrays["target_total_displacement_m"][index], device="cuda"
    )
    scale2 = (weights * scale.square().sum(-1)).sum()
    warm_kernels(
        physics,
        runtime,
        q=q,
        jaw=jaw,
        seed=seed,
        seed_jaw=seed_jaw,
        runner=runner,
    )
    before = gpu_snapshot()
    torch.cuda.reset_peak_memory_stats()
    try:
        displacement, forward_seconds = synchronized_time(
            lambda: physics.solve(
                skin_multiplier=torch.ones((), device="cuda", dtype=torch.float64),
                active_stress=runner.activation_stresses_mpa(q, runner.REFERENCE_MPA),
                pose=runner.hinge_pose(
                    jaw,
                    torch.as_tensor(
                        inputs.arrays["mandible_frame_world"][:, 0], device="cuda"
                    ),
                ),
                seed=seed.detach().clone(),
                seed_pose=runner.hinge_pose(
                    seed_jaw,
                    torch.as_tensor(
                        inputs.arrays["mandible_frame_world"][:, 0], device="cuda"
                    ),
                ),
                key=f"solver-performance/{case}/{method}",
            )
        )
        loss = (
            weights * (displacement[observation] - target).square().sum(-1)
        ).sum() / scale2
        (gradient_q, gradient_jaw), adjoint_seconds = synchronized_time(
            lambda: torch.autograd.grad(loss, (q, jaw))
        )
        receipt = copy.deepcopy(runtime.last_forward)
        adjoint = copy.deepcopy(runtime.last_adjoint)
        residual = runtime.last_problem.grad(runtime.forward.state)
        torch.cuda.synchronize()
        output_path = cfg.output_dir / "outputs" / f"{case}-{method}.pt"
        output_path.parent.mkdir(exist_ok=True)
        temporary = output_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "displacement": displacement.detach().cpu(),
                "gradient_q": gradient_q.detach().cpu(),
                "gradient_jaw": gradient_jaw.detach().cpu(),
            },
            temporary,
        )
        temporary.replace(output_path)
        geometry = physics.metrics(displacement.detach(), target_index=index)
        return {
            "success": True,
            "case": case,
            "method": method,
            "checkpoint_sha256": frozen[case]["sha256"],
            "proposal": {
                "activation_sha256": tensor_sha256(q),
                "seed_displacement_sha256": tensor_sha256(seed),
                "seed_jaw_normalized": float(seed_jaw[0]),
                "target_jaw_normalized": float(jaw[0]),
                "target_jaw_degrees": float(jaw[0]) * 10,
            },
            "forward_wall_seconds": forward_seconds,
            "adjoint_wall_seconds": adjoint_seconds,
            "forward": jsonable(receipt),
            "adjoint": jsonable(adjoint),
            "operation_counts": operation_counts(receipt),
            "terminal_force_norm": float(torch.linalg.vector_norm(residual)),
            "displacement_sha256": tensor_sha256(displacement),
            "gradient_q_sha256": tensor_sha256(gradient_q),
            "gradient_jaw_sha256": tensor_sha256(gradient_jaw),
            "output_path": str(output_path),
            "output": {
                "loss": float(loss),
                "displacement_norm_m": float(torch.linalg.vector_norm(displacement)),
                "gradient_q_norm": float(torch.linalg.vector_norm(gradient_q)),
                "gradient_jaw": float(gradient_jaw[0]),
            },
            "geometry": jsonable(geometry),
            "validity_gate": {
                "zero_inverted_tetrahedra": geometry["inverted_tetrahedra"] == 0,
            },
            "ipc_threads": int(ipctk.get_num_threads()),
            "gpu": {
                "before": before,
                "after": gpu_snapshot(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            },
        }
    except Exception as error:  # benchmark evidence must retain failed arms
        torch.cuda.synchronize()
        receipt = getattr(error, "receipt", runtime.last_forward)
        return {
            "success": False,
            "case": case,
            "method": method,
            "checkpoint_sha256": frozen[case]["sha256"],
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
                "receipt": jsonable(receipt),
            },
            "gpu": {"before": before, "after": gpu_snapshot()},
        }


def compare(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    if not baseline["success"] or not candidate["success"]:
        return {"comparable": False, "reason": "one_or_both_arms_failed"}
    validity_gate = {
        "baseline_zero_inverted_tetrahedra": baseline["validity_gate"][
            "zero_inverted_tetrahedra"
        ],
        "candidate_zero_inverted_tetrahedra": candidate["validity_gate"][
            "zero_inverted_tetrahedra"
        ],
    }
    if not validity_gate["baseline_zero_inverted_tetrahedra"]:
        return {
            "comparable": False,
            "reason": "baseline_invalid_geometry",
            "validity_gate": validity_gate,
        }
    if not validity_gate["candidate_zero_inverted_tetrahedra"]:
        return {
            "comparable": False,
            "reason": "candidate_invalid_geometry",
            "validity_gate": validity_gate,
        }
    baseline_output = torch.load(baseline["output_path"], weights_only=False)
    candidate_output = torch.load(candidate["output_path"], weights_only=False)

    def difference(name: str) -> dict[str, float]:
        delta = candidate_output[name] - baseline_output[name]
        return {
            "l2": float(torch.linalg.vector_norm(delta)),
            "linf": float(delta.abs().max()),
            "relative_l2": float(torch.linalg.vector_norm(delta))
            / max(float(torch.linalg.vector_norm(baseline_output[name])), 1e-300),
        }

    return {
        "comparable": True,
        "forward_wall_speedup": baseline["forward_wall_seconds"]
        / candidate["forward_wall_seconds"],
        "adjoint_wall_speedup": baseline["adjoint_wall_seconds"]
        / candidate["adjoint_wall_seconds"],
        "terminal_force_ratio": candidate["terminal_force_norm"]
        / baseline["terminal_force_norm"],
        "output_hashes_equal": {
            "displacement": baseline["displacement_sha256"]
            == candidate["displacement_sha256"],
            "gradient_q": baseline["gradient_q_sha256"]
            == candidate["gradient_q_sha256"],
            "gradient_jaw": baseline["gradient_jaw_sha256"]
            == candidate["gradient_jaw_sha256"],
        },
        "numerical_difference": {
            "displacement_m": difference("displacement"),
            "gradient_q": difference("gradient_q"),
            "gradient_jaw": difference("gradient_jaw"),
        },
        "equivalence_gate": {
            "maximum_displacement_difference_m_at_most_1e-6": difference(
                "displacement"
            )["linf"]
            <= 1e-6,
            "activation_gradient_relative_l2_at_most_1e-3": difference("gradient_q")[
                "relative_l2"
            ]
            <= 1e-3,
            "jaw_gradient_relative_l2_at_most_1e-3": difference("gradient_jaw")[
                "relative_l2"
            ]
            <= 1e-3,
        },
        "validity_gate": validity_gate,
    }


def main(cfg: Config) -> None:
    methods = parse_names(
        cfg.methods,
        allowed={
            "baseline",
            "cached_pncg",
            "exact_pncg",
            "newton_diag",
            "newton_block",
            "hybrid_diag",
            "hybrid_block",
        },
    )
    cases = parse_names(cfg.cases, allowed={"collision_off_pose", "contact_expression"})
    assert cfg.wall_seconds is None or cfg.wall_seconds > 0
    assert 0 < cfg.linear_rtol < 1
    assert cfg.max_newton_steps > 0
    assert cfg.ipc_threads is None or cfg.ipc_threads > 0
    assert cfg.source_root.is_dir(), cfg.source_root
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_benchmark_sources(cfg)
    frozen_dir = cfg.output_dir / "frozen-inputs"
    frozen_dir.mkdir()
    (frozen_dir / "run007").mkdir(exist_ok=True)
    (frozen_dir / "run006").mkdir(exist_ok=True)
    frozen = {
        "collision_off_pose": freeze(cfg.run007_initial, frozen_dir / "run007"),
        "contact_expression": freeze(cfg.run006_initial, frozen_dir / "run006"),
    }
    (frozen_dir / "inputs").mkdir(exist_ok=True)
    inputs_manifest = cfg.inputs_dir / "manifest.json"
    input_state = cfg.inputs_dir / "state.npz"
    frozen["inputs_manifest"] = freeze(inputs_manifest, frozen_dir / "inputs")
    frozen["inputs_state"] = freeze(input_state, frozen_dir / "inputs")
    write_json(cfg.output_dir / "frozen-inputs.json", frozen)
    from remote_paths import install_loader_path_relocation

    install_loader_path_relocation(source_root=cfg.source_root)
    if cfg.ipc_threads is not None:
        ipctk.set_num_threads(cfg.ipc_threads)
    configure_cuda()
    runner = load_runner()
    rows: list[dict[str, Any]] = []
    for case in cases:
        baseline = run_one(
            cfg, case=case, method="original", frozen=frozen, runner=runner
        )
        rows.append(baseline)
        write_json(cfg.output_dir / "results.json", rows)
        for method in methods:
            row = run_one(cfg, case=case, method=method, frozen=frozen, runner=runner)
            row["comparison_to_baseline"] = compare(baseline, row)
            rows.append(row)
            write_json(cfg.output_dir / "results.json", rows)
            LOG.info("%s %s: success=%s", case, method, row["success"])
    summary = {
        "schema": "saved-state-solver-performance-benchmark-v1",
        "success": all(
            row["success"]
            and row.get("geometry", {}).get("inverted_tetrahedra", 0) == 0
            and all(
                row.get("comparison_to_baseline", {})
                .get("equivalence_gate", {})
                .values()
            )
            for row in rows
        ),
        "scope": {
            "primal": "fresh runtime, fixed-material MouthOpen saved-state replay",
            "adjoint": "implicit adjoint through fixed jaw and activation probe",
            "timing": "CUDA-synchronized wall time; runtime construction excluded",
            "warmup": "fun, grad, Hessian diagonal, quadratic, and product kernels at the owned initial state are synchronized before each timed arm",
            "inputs": "source checkpoints copied with pre/post-copy SHA-256 checks",
            "old_runs_mutated": False,
        },
        "protocol": {
            "original": "existing AcceptedForcePncg runtime without accelerated wrapper",
            "baseline": "accelerated wrapper in cache-disabled baseline mode",
            "methods": ["original", *methods],
            "cases": list(cases),
            "wall_seconds_per_solver": cfg.wall_seconds,
            "linear_rtol": cfg.linear_rtol,
            "max_newton_steps": cfg.max_newton_steps,
            "collision_off_pose": {
                "checkpoint": "run007 initial.pt",
                "activation": "zero",
                "jaw_degrees": cfg.collision_off_jaw_degrees,
            },
            "contact_expression": {
                "checkpoint": "run006 initial.pt",
                "activation": "zero",
                "jaw_degrees": cfg.contact_jaw_degrees,
            },
            "ipc_threads_requested": cfg.ipc_threads,
            "ipc_threads_actual": int(ipctk.get_num_threads()),
        },
        "frozen_inputs": frozen,
        "results": rows,
        "source_sha256": sha256(Path(__file__)),
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "solver_benchmark/success": float(summary["success"]),
            "solver_benchmark/arms": len(rows),
        }
    )
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    cherries.main(main, profile=ProfilePerformance)
