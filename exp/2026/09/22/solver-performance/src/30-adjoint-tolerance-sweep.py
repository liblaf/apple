# ruff: noqa: E402, EM101, TRY003
"""Fixed-state Smile implicit-adjoint tolerance replay.

This experiment deliberately never invokes a primal optimizer.  Each arm builds
fresh full skull, mandible, and eye contact physics, then evaluates the owned
implicit backward at a checkpointed full displacement.  The only changed
quantity is the actual CuPy adjoint linear-solver tolerance.
"""

from __future__ import annotations

import copy
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Literal

import ipctk
import torch
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(EXPERIMENT / "src"), str(SOURCE_GROUP / "src")]

from adjoint_tolerance_common import (
    build_fixed_state_context,
    file_record,
    fixed_state_output,
    fixed_state_physical_status,
    install_adjoint_tolerance,
    jsonable,
    load_protocol_weights,
    restore_adjoint_tolerance,
    tensor_sha256,
)
from joint_equilibrium import configure_cuda
from joint_fields import activation_regularizers, project_activation_
from remote_paths import install_loader_path_relocation
from smile_collision import audit_required_collision


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


BASELINE_RUN = EXPERIMENT / "data/smile-fit-adam03-unconditional-004"
ZERO_RUN = EXPERIMENT / "data/smile-fit-adam03-no-smoothness-005"


class Config(cherries.BaseConfig):
    output_dir: Path = EXPERIMENT / "data/adjoint-tolerance-sweep-001"
    source_root: Path = EXPERIMENT.parents[4]
    origin_metadata: Path | None = None
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    baseline_checkpoint: Path = (
        BASELINE_RUN / "arms/hybrid_diag/expressions/Smile/comparison-step-00020.pt"
    )
    zero_step15_checkpoint: Path = (
        ZERO_RUN / "arms/hybrid_diag/expressions/Smile/comparison-step-00015.pt"
    )
    zero_latest_checkpoint: Path = (
        ZERO_RUN / "arms/hybrid_diag/expressions/Smile/latest.pt"
    )
    # A single checkpoint restricts the run to just that state, useful for a
    # bounded continuation or a remote smoke test.
    checkpoint: Path | None = None
    tolerances: str = "1e-8,1e-7,1e-6,1e-5,1e-4"
    reference_rtol: float = 1e-8
    forward_atol: float = 1e-8
    ipc_threads: int = 8
    gpu_contact: bool = False
    gpu_contact_scope: Literal["all", "adjoint"] = "adjoint"


def write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def parse_tolerances(value: str, reference: float) -> tuple[float, ...]:
    tolerances = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    assert tolerances
    assert all(math.isfinite(item) and item > 0 for item in tolerances)
    assert len(set(tolerances)) == len(tolerances), tolerances
    assert reference in tolerances, (reference, tolerances)
    return (reference, *(item for item in tolerances if item != reference))


def vector_comparison(value: torch.Tensor, reference: torch.Tensor) -> dict[str, Any]:
    """Stable L2/cosine comparison, including a documented zero-reference case."""
    value = value.detach().flatten().to(dtype=torch.float64)
    reference = reference.detach().flatten().to(dtype=torch.float64)
    value_l2 = float(torch.linalg.vector_norm(value))
    reference_l2 = float(torch.linalg.vector_norm(reference))
    difference_l2 = float(torch.linalg.vector_norm(value - reference))
    if reference_l2 == 0.0:
        relative_l2 = 0.0 if difference_l2 == 0.0 else None
        relative_definition = (
            "zero-reference: 0 for exact equality, otherwise undefined"
        )
    else:
        relative_l2 = difference_l2 / reference_l2
        relative_definition = "||value-reference||_2 / ||reference||_2"
    if value_l2 == 0.0 or reference_l2 == 0.0:
        cosine = 1.0 if value_l2 == reference_l2 == 0.0 else None
        cosine_definition = (
            "undefined when exactly one norm is zero; 1 for two zero vectors"
        )
    else:
        cosine = float(torch.dot(value, reference) / (value_l2 * reference_l2))
        cosine_definition = "dot(value,reference)/(||value||_2 ||reference||_2)"
    return {
        "value_l2": value_l2,
        "reference_l2": reference_l2,
        "difference_l2": difference_l2,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "relative_definition": relative_definition,
        "cosine_definition": cosine_definition,
    }


def optimizer_update(
    context: Any,
    *,
    data_gradient_q: torch.Tensor,
    data_gradient_jaw: torch.Tensor,
    weights: dict[str, float],
) -> dict[str, Any]:
    """Replay the checkpoint's next projected Adam proposal from frozen moments."""
    q = torch.nn.Parameter(context.q.detach().clone())
    jaw = torch.nn.Parameter(context.jaw.detach().clone())
    reg = activation_regularizers(q, context.fitter.graph)
    prior = (
        weights["smoothness_weight"] * reg["smoothness"]
        + weights["magnitude_weight"] * reg["magnitude"]
        + weights["jaw_weight"] * jaw.square().sum() / 6
    )
    prior_q, prior_jaw = torch.autograd.grad(prior, (q, jaw))
    q.grad = data_gradient_q.detach().clone() + prior_q
    jaw.grad = data_gradient_jaw.detach().clone() + prior_jaw
    optimizer = torch.optim.Adam((q, jaw), lr=weights["learning_rate"])
    optimizer.load_state_dict(copy.deepcopy(context.checkpoint["optimizer"]))
    actual_lr = float(optimizer.param_groups[0]["lr"])
    assert actual_lr == weights["learning_rate"], (actual_lr, weights)
    optimizer.step()
    with torch.no_grad():
        project_activation_(q, context.runner.CAP)
        jaw.clamp_(context.runner.HINGE_MIN, context.runner.HINGE_MAX)
    return {
        "actual_learning_rate": actual_lr,
        "weights": weights,
        "q_update": (q.detach() - context.q.detach()).cpu(),
        "jaw_update": (jaw.detach() - context.jaw.detach()).cpu(),
        "total_gradient_q": q.grad.detach().cpu(),
        "total_gradient_jaw": jaw.grad.detach().cpu(),
        "prior_gradient_q": prior_q.detach().cpu(),
        "prior_gradient_jaw": prior_jaw.detach().cpu(),
    }


def replay_one(
    cfg: Config,
    *,
    checkpoint_path: Path,
    run_dir: Path,
    rtol: float,
) -> dict[str, Any]:
    """One fresh fixed-state VJP.  No forward/primal solve is reachable here."""
    context = build_fixed_state_context(
        checkpoint_path=checkpoint_path,
        inputs_dir=cfg.inputs_dir,
        output_dir=cfg.output_dir
        / "scratch"
        / checkpoint_path.stem
        / f"rtol-{rtol:.0e}",
        forward_atol=cfg.forward_atol,
        adjoint_rtol=rtol,
        ipc_threads=cfg.ipc_threads,
    )
    runtime = context.fitter.runtime
    assert runtime.forward_count == 0, "context initialization must not run a primal"
    collision = audit_required_collision(context.fitter.physics)
    physical_status = fixed_state_physical_status(context)
    # The requested tolerance replaces runtime.solver, so install the
    # adjoint-only wrapper only after the replacement is in place.
    prior = install_adjoint_tolerance(runtime, rtol)
    gpu = None
    if cfg.gpu_contact:
        if cfg.gpu_contact_scope == "all":
            from gpu_contact import install_gpu_contact

            gpu = install_gpu_contact(context.fitter.physics)
        else:
            from gpu_contact import install_adjoint_gpu_contact

            gpu = install_adjoint_gpu_contact(runtime, context.fitter.physics)
    try:
        # The output is a saved full displacement.  Its tail contains the fixed
        # skull/eyes; the data VJP assigns that tail zero gradient automatically.
        saved = fixed_state_output(context, key=f"fixed-adjoint/{checkpoint_path.name}")
        loss, mse = context.fitter.data_loss(
            saved[: context.fem_node_count], context.index
        )
        torch.cuda.synchronize()
        started = time.perf_counter()
        gradient_q, gradient_jaw = torch.autograd.grad(loss, (context.q, context.jaw))
        torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        assert runtime.forward_count == 0, "fixed-state replay must not run a primal"
        receipt = copy.deepcopy(runtime.last_adjoint)
        if gpu is not None and receipt["result"] != "zero right-hand side":
            uploads = gpu.uploads if hasattr(gpu, "uploads") else gpu.adapter.uploads
            products = (
                gpu.products if hasattr(gpu, "products") else gpu.adapter.products
            )
            assert uploads > 0, ("GPU contact upload missing", uploads)
            assert products > 0, ("GPU contact product missing", products)
        update = optimizer_update(
            context,
            data_gradient_q=gradient_q,
            data_gradient_jaw=gradient_jaw,
            weights=load_protocol_weights(run_dir),
        )
        return {
            "success": True,
            "rtol_requested": rtol,
            "checkpoint": file_record(checkpoint_path),
            "checkpoint_accepted_steps": int(context.checkpoint["accepted_steps"]),
            "checkpoint_inverse_converged": bool(
                context.checkpoint["inverse_converged"]
            ),
            "initialization_seconds": context.initialization_seconds,
            "total_adjoint_vjp_seconds": total_seconds,
            "owned_adjoint_seconds": receipt["seconds"],
            "data_loss": float(loss),
            "fit_rms_mm": float(mse.sqrt()) * 1000,
            "data_gradient_q": gradient_q.detach().cpu(),
            "data_gradient_jaw": gradient_jaw.detach().cpu(),
            "adjoint": receipt,
            "optimizer_update": update,
            "collision_required": collision,
            "gpu_contact": {
                "enabled": gpu is not None,
                "scope": cfg.gpu_contact_scope if gpu is not None else None,
                "backend": (
                    "CUDA sparse CSR owned IPC contact Hessian"
                    if gpu is not None
                    else "owned CPU contact Hessian"
                ),
                "uploads": 0
                if gpu is None
                else (gpu.uploads if hasattr(gpu, "uploads") else gpu.adapter.uploads),
                "products": 0
                if gpu is None
                else (
                    gpu.products if hasattr(gpu, "products") else gpu.adapter.products
                ),
            },
            "fixed_state_physical_status": physical_status,
            "no_primal": {
                "forward_count": runtime.forward_count,
                "saved_fem_displacement_sha256": tensor_sha256(
                    context.checkpoint["displacement_m"]
                ),
                "saved_full_displacement_sha256": tensor_sha256(
                    context.full_displacement
                ),
                "appended_rigid_nodes": int(
                    context.full_displacement.shape[0] - context.fem_node_count
                ),
                "hessian": "owned unshifted model.hess_prod; no damping or shift introduced",
            },
        }
    finally:
        if gpu is not None:
            gpu.uninstall()
        restore_adjoint_tolerance(runtime, prior)


def checkpoint_arms(cfg: Config) -> list[tuple[str, Path, Path]]:
    if cfg.checkpoint is not None:
        checkpoint = cfg.checkpoint.resolve()
        # Infer the run protocol convention from the normal arms path.  A
        # user-supplied checkpoint outside these runs must sit under a run root.
        run_dir = checkpoint.parents[4]
        assert (run_dir / "protocol.json").is_file(), run_dir
        return [(checkpoint.stem, checkpoint, run_dir)]
    return [
        ("baseline-hybrid-step20", cfg.baseline_checkpoint, BASELINE_RUN),
        ("zero-smoothness-step15", cfg.zero_step15_checkpoint, ZERO_RUN),
        ("zero-smoothness-latest19", cfg.zero_latest_checkpoint, ZERO_RUN),
    ]


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Prevent progress JSON from duplicating multi-million-element tensors."""
    result = copy.deepcopy(row)
    for key in ("data_gradient_q", "data_gradient_jaw"):
        result[key] = jsonable(result[key])
    update = result["optimizer_update"]
    for key in (
        "q_update",
        "jaw_update",
        "total_gradient_q",
        "total_gradient_jaw",
        "prior_gradient_q",
        "prior_gradient_jaw",
    ):
        update[key] = jsonable(update[key])
    return result


def add_reference_comparisons(rows: list[dict[str, Any]]) -> None:
    reference = next(row for row in rows if row["rtol_requested"] == 1e-8)
    for row in rows:
        row["comparison_to_reference"] = {
            "data_gradient_q": vector_comparison(
                row["data_gradient_q"], reference["data_gradient_q"]
            ),
            "data_gradient_jaw": vector_comparison(
                row["data_gradient_jaw"], reference["data_gradient_jaw"]
            ),
            "total_gradient_q": vector_comparison(
                row["optimizer_update"]["total_gradient_q"],
                reference["optimizer_update"]["total_gradient_q"],
            ),
            "total_gradient_jaw": vector_comparison(
                row["optimizer_update"]["total_gradient_jaw"],
                reference["optimizer_update"]["total_gradient_jaw"],
            ),
            "next_projected_adam_q_update": vector_comparison(
                row["optimizer_update"]["q_update"],
                reference["optimizer_update"]["q_update"],
            ),
            "next_projected_adam_jaw_update": vector_comparison(
                row["optimizer_update"]["jaw_update"],
                reference["optimizer_update"]["jaw_update"],
            ),
        }


def save_raw_vectors(
    output_dir: Path, label: str, row: dict[str, Any]
) -> dict[str, Any]:
    """Persist each exact vector set used in the tolerance comparisons."""
    path = output_dir / "vectors" / label / f"rtol-{row['rtol_requested']:.0e}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": "smile-fixed-state-adjoint-vectors-v1",
            "rtol": row["rtol_requested"],
            "checkpoint": row["checkpoint"],
            "data_gradient_q": row["data_gradient_q"],
            "data_gradient_jaw": row["data_gradient_jaw"],
            "total_gradient_q": row["optimizer_update"]["total_gradient_q"],
            "total_gradient_jaw": row["optimizer_update"]["total_gradient_jaw"],
            "next_projected_adam_q_update": row["optimizer_update"]["q_update"],
            "next_projected_adam_jaw_update": row["optimizer_update"]["jaw_update"],
        },
        path,
    )
    return file_record(path)


def main(cfg: Config) -> None:
    tolerances = parse_tolerances(cfg.tolerances, cfg.reference_rtol)
    assert cfg.ipc_threads > 0
    assert cfg.forward_atol > 0
    # This experiment's protocol and comparison labels declare a 1e-8 reference.
    assert cfg.reference_rtol == 1e-8
    arms = checkpoint_arms(cfg)
    assert all(path.is_file() and run_dir.is_dir() for _, path, run_dir in arms)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_dir = cfg.output_dir / "sources"
    archive_dir.mkdir()
    source_paths = {
        "script": Path(__file__),
        "helper": EXPERIMENT / "src/adjoint_tolerance_common.py",
        "joint_equilibrium": SOURCE_GROUP / "src/joint_equilibrium.py",
        "fitter": SOURCE_GROUP / "src/93-fit-expressions.py",
    }
    archived = {}
    for label, source_path in source_paths.items():
        destination = archive_dir / source_path.name
        shutil.copy2(source_path, destination)
        archived[label] = {
            "source": file_record(source_path),
            "archived": file_record(destination),
        }
    if cfg.gpu_contact:
        source_path = EXPERIMENT / "src/gpu_contact.py"
        destination = archive_dir / source_path.name
        shutil.copy2(source_path, destination)
        archived_gpu_contact = {
            "source": file_record(source_path),
            "archived": file_record(destination),
        }
    else:
        archived_gpu_contact = None
    install_loader_path_relocation(source_root=cfg.source_root)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    source = {
        **archived,
        "gpu_contact": archived_gpu_contact,
        "inputs_manifest": file_record(cfg.inputs_dir / "manifest.json"),
        "inputs_state": file_record(cfg.inputs_dir / "state.npz"),
        "arms": {
            label: {
                "checkpoint": file_record(path),
                "protocol": file_record(run_dir / "protocol.json"),
            }
            for label, path, run_dir in arms
        },
    }
    summary: dict[str, Any] = {
        "schema": "smile-fixed-state-adjoint-tolerance-sweep-v1",
        "success": False,
        "scope": "fixed saved Smile displacement; no primal solve; full skull/mandible/eye collision model; data-gradient and next projected-Adam sensitivity only",
        "config": cfg.model_dump(mode="json"),
        "source": source,
        "tolerances": list(tolerances),
        "reference_rtol": cfg.reference_rtol,
        "ipc_threads_actual": int(ipctk.get_num_threads()),
        "arms": {},
    }
    for label, checkpoint_path, run_dir in arms:
        rows: list[dict[str, Any]] = []
        # Reference comes first.  Every tolerance starts a fresh runtime with
        # an identical zero adjoint initial condition.
        for rtol in tolerances:
            cherries.set_step(len(rows))
            row = replay_one(
                cfg, checkpoint_path=checkpoint_path, run_dir=run_dir, rtol=rtol
            )
            row["raw_vectors"] = save_raw_vectors(cfg.output_dir, label, row)
            rows.append(row)
            if rtol == cfg.reference_rtol:
                assert row["success"]
            if rtol != cfg.reference_rtol:
                add_reference_comparisons(rows)
            summary["arms"][label] = {"rows": [compact_row(item) for item in rows]}
            write_json(cfg.output_dir / "progress.json", summary)
            cherries.log_metrics(
                {
                    f"{label}/{rtol:.0e}/adjoint_seconds": row["owned_adjoint_seconds"],
                    f"{label}/{rtol:.0e}/hvp_matvec_calls": row["adjoint"][
                        "hvp_matvec_calls"
                    ],
                    f"{label}/{rtol:.0e}/relative_residual": row["adjoint"][
                        "relative_residual"
                    ],
                }
            )
        add_reference_comparisons(rows)
        summary["arms"][label] = {
            "rows": [compact_row(item) for item in rows],
            "reference_vectors": rows[0]["raw_vectors"],
        }
    summary["success"] = all(
        len(arm["rows"]) == len(tolerances)
        and all(row["success"] for row in arm["rows"])
        for arm in summary["arms"].values()
    )
    write_json(cfg.output_dir / "summary.json", summary)
    if not summary["success"]:
        raise RuntimeError("one or more adjoint replay arms failed")


if __name__ == "__main__":
    cherries.main(main, profile=ProfilePerformance)
