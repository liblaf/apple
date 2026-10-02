"""Compare exact CUDA contact HVP backends on the pruned active-strain mesh."""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import ipctk
import numpy as np
import torch

from liblaf import cherries
from liblaf.apple.forward._problem import ForwardProblem
from liblaf.apple.forward.hessian._problem import HessianProblem

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
sys.path.insert(0, str(PARENT / "src"))
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402
from stress_physics import (  # noqa: E402
    FacePhysics,
    configure,
    strain_to_activation_inv,
)
from transition_contact import build_self_contact  # noqa: E402


class Config(cherries.BaseConfig):
    fixture: Path = PARENT / "data/30-pruned-fixture"
    output: Path = cherries.output("11-sparse-contact-check/summary.json", mkdir=True)
    directions: int = 3
    seed: int = 20260930
    stiffness_mpa: float = 0.0012
    minimum_distance_m: float = 1e-8


def _elapsed(fn: Callable[[], torch.Tensor]) -> tuple[torch.Tensor, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = fn()
    torch.cuda.synchronize()
    return result, time.perf_counter() - started


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.directions == 3
    assert cfg.stiffness_mpa > 0
    assert cfg.minimum_distance_m > 0
    configure()
    torch.cuda.reset_peak_memory_stats()
    physics = FacePhysics(cfg.fixture, activation_model="strain")
    model = physics.forward.model
    contact, contact_receipt = build_self_contact(
        physics.mesh,
        stiffness_mpa=cfg.stiffness_mpa,
        minimum_distance_m=cfg.minimum_distance_m,
    )
    model.collision = contact
    zero_s = torch.zeros((len(physics.ids), 3, 3), device="cuda")
    physics.materials["muscle"]["activation_inv"] = torch.zeros(
        (physics.mesh.n_cells, 6), device="cuda"
    ).index_copy(0, physics.id_t, strain_to_activation_inv(zero_s))
    model.set_materials(physics.materials)
    u = torch.zeros_like(torch.as_tensor(physics.points))
    state = model.State(u=u, collision=contact.state_at(u))
    positions = (contact.vertices + u[contact.indices]).numpy(force=True)
    state.collision.hess = contact.potential.hessian(
        collisions=state.collision.collisions,
        mesh=contact.collision_mesh,
        X=positions,
        project_hessian_to_psd=ipctk.PSDProjectionMethod.CLAMP,
    )
    problem = ForwardProblem(model=model)
    gpu_contact = HessianProblem(problem, "gpu_contact")
    gpu_sparse = HessianProblem(problem, "gpu_sparse")
    diagonal_contact, diagonal_contact_seconds = _elapsed(
        lambda: gpu_contact.hess_diag(state)
    )
    try:
        diagonal_sparse, diagonal_sparse_seconds = _elapsed(
            lambda: gpu_sparse.hess_diag(state)
        )
    except TypeError as error:
        summary = {
            "schema": "gpu-contact-versus-sparse-hvp-v1",
            "status": "unavailable",
            "reason": str(error),
            "fixture": str(cfg.fixture),
            "device": str(u.device),
            "dtype": str(u.dtype),
            "contact": contact_receipt,
            "activation": "zero active-strain S at rest",
            "contact_hessian": "IPC physical barrier Hessian with PSDProjectionMethod.CLAMP",
            "gpu_contact_diagonal_seconds": diagonal_contact_seconds,
            "gpu_sparse_backend": "does not support this active-strain bulk material set",
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
        }
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        cherries.log_metrics({"comparison/gpu_sparse_available": 0})
        return
    diagonal_relative_error = float(
        torch.linalg.vector_norm(diagonal_contact - diagonal_sparse)
        / torch.linalg.vector_norm(diagonal_contact)
    )
    assert diagonal_relative_error <= 1e-12

    generator = torch.Generator(device="cuda").manual_seed(cfg.seed)
    rows = []
    maximum_relative_error = 0.0
    for index in range(cfg.directions):
        direction = torch.randn(problem.grad(state).shape, generator=generator)
        direction /= torch.linalg.vector_norm(direction)
        contact_product, contact_seconds = _elapsed(
            lambda direction=direction: gpu_contact.hess_prod(state, direction)
        )
        sparse_product, sparse_seconds = _elapsed(
            lambda direction=direction: gpu_sparse.hess_prod(state, direction)
        )
        relative_error = float(
            torch.linalg.vector_norm(contact_product - sparse_product)
            / torch.linalg.vector_norm(contact_product)
        )
        assert relative_error <= 1e-9
        maximum_relative_error = max(maximum_relative_error, relative_error)
        rows.append(
            {
                "index": index,
                "contact_seconds": contact_seconds,
                "sparse_seconds": sparse_seconds,
                "relative_hvp_error": relative_error,
                "contact_hvp_l2": float(torch.linalg.vector_norm(contact_product)),
            }
        )
    report_contact = gpu_contact.report()
    report_sparse = gpu_sparse.report()
    summary = {
        "schema": "gpu-contact-versus-sparse-hvp-v1",
        "fixture": str(cfg.fixture),
        "device": str(u.device),
        "dtype": str(u.dtype),
        "contact": contact_receipt,
        "activation": "zero active-strain S at rest",
        "contact_hessian": "IPC physical barrier Hessian with PSDProjectionMethod.CLAMP",
        "directions": rows,
        "maximum_relative_hvp_error": maximum_relative_error,
        "diagonal_relative_error": diagonal_relative_error,
        "diagonal_seconds": {
            "gpu_contact": diagonal_contact_seconds,
            "gpu_sparse": diagonal_sparse_seconds,
        },
        "backend_report": {
            "gpu_contact": report_contact,
            "gpu_sparse": report_sparse,
        },
        "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated()),
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    cherries.log_metrics(
        {
            "comparison/max_relative_hvp_error": maximum_relative_error,
            "comparison/diagonal_relative_error": diagonal_relative_error,
            "comparison/contact_seconds_mean": float(
                np.mean([row["contact_seconds"] for row in rows])
            ),
            "comparison/sparse_seconds_mean": float(
                np.mean([row["sparse_seconds"] for row in rows])
            ),
            "comparison/cuda_peak_memory_bytes": summary["cuda_peak_memory_bytes"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
