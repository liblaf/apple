"""Validate derivatives of area-weighted standard IPC on the contact fixture."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json

from liblaf import cherries


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/positive-ipc-contact-validation-002"


def evaluate(
    contact: object,
    state: object,
    u: torch.Tensor,
) -> tuple[float, torch.Tensor, int]:
    energy = float(contact.fun(state, u))
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    return energy, gradient, len(state.collisions)


def main(cfg: Config) -> None:
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    spec = importlib.util.spec_from_file_location(
        "contact_fixture", Path(__file__).with_name("53-validate-full-skull-contact.py")
    )
    assert spec is not None
    assert spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    config = fixture.contact_config()
    config["collision_set_type"] = "IPC"
    adapter = fixture.build_full_skull_contact(fixture.synthetic_geometry(), config)
    contact = adapter.collision
    generator = torch.Generator(device="cpu").manual_seed(76)
    u = torch.zeros((adapter.geometry.full_node_count, 3), dtype=torch.float64)
    p = torch.randn(u.shape, generator=generator, dtype=torch.float64)
    p /= torch.linalg.vector_norm(p)
    energy, gradient = fixture.energy_and_gradient(contact, u)
    state = contact.state_at(u)
    assert energy > 0
    assert all(state.collisions[i].weight >= 0 for i in range(len(state.collisions)))
    hp = torch.zeros_like(u)
    contact.hess_prod(state, u, p, hp)
    slope = float(torch.sum(gradient * p))
    rows = []
    for epsilon in (3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6):
        up = u + epsilon * p
        um = u - epsilon * p
        rebuilt_plus = contact.state_at(up)
        rebuilt_minus = contact.state_at(um)
        eplus, gplus, nplus = evaluate(contact, rebuilt_plus, up)
        eminus, gminus, nminus = evaluate(contact, rebuilt_minus, um)
        frozen_eplus, frozen_gplus, frozen_nplus = evaluate(contact, state, up)
        frozen_eminus, frozen_gminus, frozen_nminus = evaluate(contact, state, um)

        def errors(
            plus_energy: float,
            minus_energy: float,
            plus_gradient: torch.Tensor,
            minus_gradient: torch.Tensor,
            step: float = epsilon,
        ) -> dict[str, float]:
            fd = (plus_energy - minus_energy) / (2 * step)
            hpfd = (plus_gradient - minus_gradient) / (2 * step)
            return {
                "energy_directional_finite_difference": fd,
                "gradient_relative_error": abs(fd - slope)
                / max(abs(fd), abs(slope), 1e-30),
                "hessian_product_relative_error": float(
                    torch.linalg.vector_norm(hpfd - hp) / torch.linalg.vector_norm(hp)
                ),
            }

        rebuilt = errors(eplus, eminus, gplus, gminus)
        frozen = errors(
            frozen_eplus,
            frozen_eminus,
            frozen_gplus,
            frozen_gminus,
        )
        rows.append(
            {
                "epsilon": epsilon,
                "rebuilt": rebuilt,
                "frozen_central_stencil": frozen,
                "collision_counts": {
                    "central": len(state.collisions),
                    "rebuilt_plus": nplus,
                    "rebuilt_minus": nminus,
                    "frozen_plus": frozen_nplus,
                    "frozen_minus": frozen_nminus,
                },
                "rebuilt_minus_frozen": {
                    "plus_energy": eplus - frozen_eplus,
                    "minus_energy": eminus - frozen_eminus,
                    "plus_gradient_relative": float(
                        torch.linalg.vector_norm(gplus - frozen_gplus)
                        / torch.linalg.vector_norm(frozen_gplus)
                    ),
                    "minus_gradient_relative": float(
                        torch.linalg.vector_norm(gminus - frozen_gminus)
                        / torch.linalg.vector_norm(frozen_gminus)
                    ),
                },
            }
        )
    balance = float(
        torch.linalg.vector_norm(gradient.sum(dim=0))
        / torch.linalg.vector_norm(gradient)
    )
    assert balance < 1e-12
    repeat_energies = [fixture.energy_and_gradient(contact, u)[0] for _ in range(5)]
    resolved = next(row for row in rows if row["epsilon"] == 1e-4)
    rebuilt_passed = (
        resolved["rebuilt"]["gradient_relative_error"] < 2e-5
        and resolved["rebuilt"]["hessian_product_relative_error"] < 2e-5
    )
    frozen_passed = (
        resolved["frozen_central_stencil"]["gradient_relative_error"] < 2e-5
        and resolved["frozen_central_stencil"]["hessian_product_relative_error"] < 2e-5
    )
    success = rebuilt_passed and frozen_passed and balance < 1e-12
    summary = {
        "schema": "joint-positive-ipc-contact-validation-v1",
        "success": success,
        "status": "passed_step_size_and_stencil_resolved_derivative_validation"
        if success
        else "failed_step_size_and_stencil_resolved_derivative_validation",
        "config": config,
        "precision": {
            "torch_default_dtype": str(torch.get_default_dtype()),
            "torch_dtype": str(u.dtype),
            "numpy_dtype": str(np.asarray(contact.vertices.numpy(force=True)).dtype),
            "python_float_bits": 64,
            "reason": (
                "Collision.fun wraps the Python-float IPC energy with torch.as_tensor; "
                "the global default must be float64 to avoid quantizing energy while "
                "the NumPy-derived gradient and Hessian remain float64."
            ),
        },
        "energy": energy,
        "energy_repeat_range": max(repeat_energies) - min(repeat_energies),
        "analytic_directional_gradient": slope,
        "hessian_product_norm": float(torch.linalg.vector_norm(hp)),
        "derivative_checks": rows,
        "resolved_gate": {
            "epsilon": 1e-4,
            "gradient_relative_error_limit": 2e-5,
            "hessian_product_relative_error_limit": 2e-5,
            "rebuilt_passed": rebuilt_passed,
            "frozen_passed": frozen_passed,
        },
        "force_balance_relative": balance,
        "interpretation": (
            "All rows are retained to expose truncation and cancellation. The gate uses "
            "the pre-observed resolved 1e-4 step; smaller-step energy differences are "
            "not required to improve monotonically. Frozen-stencil and rebuilt-state "
            "results distinguish collision-set changes from finite-difference scale."
        ),
        "implementation_sha256": {
            str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
            str(Path(__file__).with_name("joint_contact.py").resolve()): sha256(
                Path(__file__).with_name("joint_contact.py").resolve()
            ),
            str(
                Path(__file__).with_name("joint_full_skull_contact.py").resolve()
            ): sha256(
                Path(__file__).with_name("joint_full_skull_contact.py").resolve()
            ),
        },
    }
    write_json(
        cfg.output_dir / "summary.json",
        summary,
    )
    cherries.log_output(cfg.output_dir)
    assert success, summary["resolved_gate"]


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
