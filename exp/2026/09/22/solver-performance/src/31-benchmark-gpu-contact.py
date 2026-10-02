# ruff: noqa: E402, PLR0915
"""Validate an opt-in CUDA sparse IPC contact-Hessian product on one saved state."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

from liblaf import cherries

EXPERIMENT = Path(__file__).resolve().parent.parent
SOURCE_GROUP = EXPERIMENT.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(EXPERIMENT / "src"), str(SOURCE_GROUP / "src")]

from adjoint_tolerance_common import build_fixed_state_context, file_record
from gpu_contact import install_gpu_contact
from joint_equilibrium import configure_cuda
from remote_paths import install_loader_path_relocation


class Config(cherries.BaseConfig):
    checkpoint: Path = (
        EXPERIMENT
        / "data/smile-fit-adam03-unconditional-004/arms/hybrid_diag/expressions/Smile/latest.pt"
    )
    source_root: Path = EXPERIMENT.parents[4]
    inputs_dir: Path = SOURCE_GROUP / "data/expression-inputs-002"
    output_dir: Path = EXPERIMENT / "data/gpu-contact-benchmark-001"
    forward_atol: float = 1e-8
    adjoint_rtol: float = 1e-7
    ipc_threads: int = 8
    samples: int = 5


def timed_hvp(
    model: object, state: object, direction: torch.Tensor
) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = model.hess_prod(state, direction)  # type: ignore[attr-defined]
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def relative_l2(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(left - right) / torch.linalg.vector_norm(right)
    )


def main(cfg: Config) -> None:
    assert torch.cuda.is_available()
    assert cfg.samples >= 2
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    install_loader_path_relocation(source_root=cfg.source_root)
    configure_cuda()
    context = build_fixed_state_context(
        checkpoint_path=cfg.checkpoint,
        inputs_dir=cfg.inputs_dir,
        output_dir=cfg.output_dir / "fitter",
        forward_atol=cfg.forward_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        ipc_threads=cfg.ipc_threads,
    )
    model = context.fitter.runtime.forward.model
    model.set_materials(context.materials)
    model.dof_map.fixed_values = context.fixed_values
    state = model.State(u=context.full_displacement.detach().clone())
    assert model.collision is not None
    state.collision = model.collision.state_at(state.u)
    directions = [
        torch.randn(
            state.u.shape,
            device=state.u.device,
            dtype=state.u.dtype,
            generator=torch.Generator(device="cuda").manual_seed(seed),
        )
        for seed in (7, 11, 13)
    ]
    cpu_pairs = [timed_hvp(model, state, direction) for direction in directions]
    cpu_values = [value for value, _seconds in cpu_pairs]
    cpu_cached_seconds = [
        timed_hvp(model, state, directions[0])[1] for _ in range(cfg.samples)
    ]
    installed = install_gpu_contact(model)
    gpu_values = []
    first_seconds = None
    for index, direction in enumerate(directions):
        value, seconds = timed_hvp(model, state, direction)
        if index == 0:
            first_seconds = seconds
        gpu_values.append(value)
    errors = [
        relative_l2(gpu, cpu) for cpu, gpu in zip(cpu_values, gpu_values, strict=True)
    ]
    for cpu, gpu in zip(cpu_values, gpu_values, strict=True):
        torch.testing.assert_close(gpu, cpu, rtol=1e-10, atol=1e-11)
    assert installed.adapter.uploads == 1
    assert installed.adapter.products == len(directions)
    cached_seconds = [
        timed_hvp(model, state, directions[0])[1] for _ in range(cfg.samples)
    ]
    assert installed.adapter.uploads == 1
    # A bounded nonzero free-DOF update must replace the exact matrix.
    updated = state.u.detach().clone().flatten()
    free = model.dof_map.free_indices.to(device=updated.device)
    updated[free[0]] += 1e-9
    model.update(state, updated.reshape_as(state.u))
    cpu_updated = torch.zeros_like(state.u)
    model.warp_model.hess_prod(state.u, directions[0], cpu_updated)
    # ``OwnedContact.hess_prod`` owns an ``OwnedContactState``.  The public
    # model dispatcher supplies this nested state; do the same when invoking
    # the retained CPU implementation directly for the post-update oracle.
    assert state.collision is not None
    installed.adapter.original_hess_prod(
        state.collision, state.u, directions[0], cpu_updated
    )
    gpu_updated, _ = timed_hvp(model, state, directions[0])
    torch.testing.assert_close(gpu_updated, cpu_updated, rtol=1e-10, atol=1e-11)
    assert installed.adapter.uploads == 2
    # A second owned state cannot reuse the first state's cached CSR matrix.
    old_state = state
    state = model.State(u=context.full_displacement.detach().clone())
    state.collision = model.collision.state_at(state.u)
    timed_hvp(model, state, directions[1])
    assert installed.adapter.uploads == 3
    timed_hvp(model, old_state, directions[2])
    assert installed.adapter.uploads == 4
    receipt = {
        "schema": "gpu-contact-hessian-benchmark-v1",
        "success": True,
        "scope": "Fixed saved state only; no primal solve or optimizer update.",
        "checkpoint": file_record(cfg.checkpoint),
        "inputs": {
            "manifest": file_record(cfg.inputs_dir / "manifest.json"),
            "state": file_record(cfg.inputs_dir / "state.npz"),
        },
        "directions": len(directions),
        "comparison": {
            "rtol": 1e-10,
            "atol": 1e-11,
            "whole_model_hvp": True,
            "relative_l2": errors,
            "updated_state_relative_l2": relative_l2(gpu_updated, cpu_updated),
        },
        "timing_seconds": {
            "cpu_first_samples": [seconds for _value, seconds in cpu_pairs],
            "cpu_cached_samples": cpu_cached_seconds,
            "first_upload_and_hvp": first_seconds,
            "gpu_cached_samples": cached_seconds,
        },
        "cache": {
            "uploads_after_first_state": 1,
            "uploads_after_update_and_two_owned_states": installed.adapter.uploads,
            "cached_products": installed.adapter.products,
        },
    }
    (cfg.output_dir / "summary.json").write_text(json.dumps(receipt, indent=2) + "\n")
    cherries.log_metrics(
        {
            "gpu_contact/first_seconds": first_seconds,
            "gpu_contact/cached_mean_seconds": sum(cached_seconds)
            / len(cached_seconds),
            "gpu_contact/uploads": installed.adapter.uploads,
        }
    )
    cherries.log_output(cfg.output_dir)
    installed.uninstall()


if __name__ == "__main__":
    cherries.main(main)
