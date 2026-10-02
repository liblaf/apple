# Copyright (c) 2026 liblaf
# ruff: noqa: EM101, PLR0915, PT018, TRY003
"""Measure PCG conditioning at one immutable accepted contact-transition state."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
RUN = GROUP / "src/52-run-fixed-activation-contact.py"
sys.path[:0] = [str(GROUP / "src"), str(PARENT / "src")]
spec = importlib.util.spec_from_file_location("fixed_contact_run52", RUN)
assert spec is not None
assert spec.loader is not None
RUN52 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = RUN52
spec.loader.exec_module(RUN52)

from accelerated_solvers import LinearRejection, pcg  # noqa: E402
from experiment import Profile  # noqa: E402
from fixed_reference_contact import (  # noqa: E402
    attach_fixed_reference_contact,
    build_fixed_reference_contact,
)
from joint_expression_equilibrium import FeasibleExpressionProblem  # noqa: E402
from stress_physics import (  # noqa: E402
    FacePhysics,
    configure,
    strain_to_activation_inv,
)


class Config(cherries.BaseConfig):
    source: Path = GROUP / "data/52-fixed-activation-contact"
    fixture: Path = GROUP / "data/50-fixed-reference/fixture"
    output: Path = Path("71-transition-pcg-probe")
    checkpoint_index: int = 73
    rtol: float = 1e-3
    caps: tuple[int, int] = (1000, 3000)
    normalized_shift: float = 1e-3
    wall_seconds: float = 120.0
    minimum_free_bytes: int = 2_000_000_000
    memory_fraction: float = 0.04


def record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": RUN52.BASE.record(path)["sha256"]}


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.caps == (1000, 3000)
    assert 0 < cfg.rtol < 1 and cfg.normalized_shift > 0
    assert cfg.wall_seconds <= 120 and 0 < cfg.memory_fraction <= 0.04
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    free_before, total_bytes = torch.cuda.mem_get_info()
    if free_before < cfg.minimum_free_bytes:
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "schema": "transition-pcg-conditioning-probe-v1",
                    "status": "skipped_insufficient_free_vram",
                    "gpu_memory_before_bytes": {
                        "free": free_before,
                        "total": total_bytes,
                    },
                    "minimum_free_bytes": cfg.minimum_free_bytes,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
        return
    torch.cuda.set_per_process_memory_fraction(cfg.memory_fraction)
    torch.cuda.reset_peak_memory_stats()
    summary_path = cfg.source / "summary.json"
    summary = json.loads(summary_path.read_text())
    row = summary["accepted"][cfg.checkpoint_index]
    assert row["phase"] == "transition"
    checkpoint = Path(row["checkpoint"]["path"])
    assert record(checkpoint)["sha256"] == row["checkpoint"]["sha256"]
    with np.load(checkpoint, allow_pickle=False) as saved:
        u = saved["u_full"].copy()
        beta = float(saved["fraction"])
        alpha = float(saved["alpha"])
        pose = saved["pose"].copy()
    assert alpha == 1 - beta
    configure()
    physics = FacePhysics(cfg.fixture, activation_model="strain", atol=1e-8)
    with np.load(cfg.source / "endpoints.npz", allow_pickle=False) as endpoints:
        mouth = endpoints["S_mouthopen"].copy()
        smile = endpoints["S_smile"].copy()
        full_pose = endpoints["pose_mouthopen"].copy()
        pivot = endpoints["pivot"].copy()
    np.testing.assert_allclose(pose, alpha * full_pose, rtol=0, atol=1e-14)
    reference = build_fixed_reference_contact(
        physics.mesh, stiffness_mpa=1.3544, dhat_m=1e-4, minimum_distance_m=1e-8
    )
    values = reference.boundary_values(
        torch.as_tensor(physics.points), torch.as_tensor(pose), torch.as_tensor(pivot)
    )
    model = attach_fixed_reference_contact(physics.forward.model, reference, values)
    activation = (1 - beta) * mouth + beta * smile
    physics.materials["muscle"]["activation_inv"] = torch.zeros(
        (physics.mesh.n_cells, 6), device="cuda"
    ).index_copy(0, physics.id_t, strain_to_activation_inv(torch.as_tensor(activation)))
    model.set_materials(physics.materials)
    state = model.State(
        u=torch.as_tensor(u), collision=reference.contact.state_at(torch.as_tensor(u))
    )
    torch.testing.assert_close(
        state.u.flatten()[model.dof_map.fixed_indices],
        model.dof_map.fixed_values,
        rtol=0,
        atol=1e-12,
    )
    problem = FeasibleExpressionProblem(model=model, collision_step_safety=0.95)
    hessian = RUN52.ProjectedContactSearch(problem, "gpu_contact")
    rhs = -problem.grad(state)
    diagonal = hessian.hess_diag(state)
    assert bool(torch.isfinite(diagonal).all())
    scale = float(diagonal.abs().mean())
    assert scale > 0
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for normalized_shift in (0.0, cfg.normalized_shift):
        shift = normalized_shift * scale
        for cap in cfg.caps:
            products = 0
            elapsed_start = time.perf_counter()

            def matvec(direction: torch.Tensor) -> torch.Tensor:
                nonlocal products
                if time.perf_counter() - started >= cfg.wall_seconds:
                    raise TimeoutError("declared total probe wall budget exhausted")
                products += 1
                return hessian.hess_prod(state, direction)

            def precondition(
                residual: torch.Tensor, shift: float = shift
            ) -> torch.Tensor:
                return residual / (diagonal + shift).abs()

            try:
                _, receipt = pcg(
                    matvec, precondition, rhs, rtol=cfg.rtol, max_steps=cap
                )
                result: dict[str, Any] = {"status": "converged", **receipt}
            except (LinearRejection, TimeoutError) as error:
                result = {"status": "rejected_or_budgeted", "reason": str(error)}
            results.append(
                {
                    "cap": cap,
                    "normalized_shift": normalized_shift,
                    "shift": shift,
                    "elapsed_seconds": time.perf_counter() - elapsed_start,
                    "hessian_products": products,
                    **result,
                }
            )
            if time.perf_counter() - started >= cfg.wall_seconds:
                break
        if time.perf_counter() - started >= cfg.wall_seconds:
            break
    free_after, _ = torch.cuda.mem_get_info()
    output = {
        "schema": "transition-pcg-conditioning-probe-v1",
        "status": "completed",
        "scope": "local conditioning probe at an immutable accepted checkpoint; it is not a replay of the unsaved pre-Newton state that exhausted the original PCG budget",
        "inputs": {
            "source_summary": record(summary_path),
            "checkpoint": record(checkpoint),
            "endpoints": record(cfg.source / "endpoints.npz"),
        },
        "checkpoint": {
            "index": cfg.checkpoint_index,
            "phase": row["phase"],
            "beta": beta,
            "alpha": alpha,
        },
        "operator": "exact active bulk plus PSD CLAMP IPC contact-search Hessian; absolute diagonal preconditioner",
        "rhs_free_force_l2": float(torch.linalg.vector_norm(rhs)),
        "diagonal_scale_mean_abs": scale,
        "rtol": cfg.rtol,
        "results": results,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu_memory_bytes": {
            "free_before": free_before,
            "free_after": free_after,
            "total": total_bytes,
            "peak_allocated": int(torch.cuda.max_memory_allocated()),
        },
    }
    (out / "summary.json").write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            "probe/elapsed_seconds": output["elapsed_seconds"],
            "probe/peak_allocated_bytes": output["gpu_memory_bytes"]["peak_allocated"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
