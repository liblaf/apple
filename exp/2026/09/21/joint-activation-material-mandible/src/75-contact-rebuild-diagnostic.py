"""Diagnose full-source contact rebuild repeatability and collision weights on CPU."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_contact import OwnedContactState
from joint_full_skull_contact import build_full_skull_contact, load_full_skull_geometry

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    run_dir: Path = GROUP / "data/simple-skin-forward-003"
    geometry: Path = GROUP / "data/simple-skin-forward-inputs-001/geometry.npz"
    geometry_audit: Path = (
        GROUP / "data/simple-skin-forward-inputs-001/geometry-audit.json"
    )
    output_dir: Path = cherries.output("contact-rebuild-diagnostic", mkdir=True)
    repeats: int = 3


def contact_config() -> dict[str, Any]:
    return {
        "schema": "joint-full-source-bone-contact-v1",
        "enabled": True,
        "surface_selection": "pure-soft-vs-complete-source-bones",
        "attachment_policy": "no-source-triangle-exclusions",
        "friction": "frictionless",
        "dhat_m": 1.0e-4,
        "stiffness_mpa": 0.01,
    }


def collision_types() -> dict[str, Any]:
    enum = ipctk.NormalCollisions.CollisionSetType
    result = {
        "ipc": enum.IPC,
        "improved_max_approx": enum.IMPROVED_MAX_APPROX,
    }
    if hasattr(enum, "MAX_APPROX"):
        result["max_approx"] = enum.MAX_APPROX
    return result


def rebuilt_state(
    contact: Any, u: torch.Tensor, collision_type: Any
) -> OwnedContactState:
    state = OwnedContactState()
    state.collisions.use_area_weighting = True
    state.collisions.collision_set_type = collision_type
    contact.update(state, u)
    return state


def state_row(contact: Any, u: torch.Tensor, collision_type: Any) -> dict[str, Any]:
    started = time.perf_counter()
    state = rebuilt_state(contact, u, collision_type)
    build_seconds = time.perf_counter() - started
    energy = float(contact.fun(state, u))
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    force_balance = torch.linalg.vector_norm(gradient.sum(dim=0))
    gradient_norm = torch.linalg.vector_norm(gradient)
    diagnostics = contact.diagnostics(state, u)
    weights = np.asarray(
        [state.collisions[index].weight for index in range(len(state.collisions))],
        dtype=np.float64,
    )
    assert len(weights) == diagnostics["active_contact_count"]
    return {
        "build_seconds": build_seconds,
        "candidate_count": len(state.candidates),
        "collision_count": len(state.collisions),
        "energy_mpa_m3": energy,
        "gradient_norm_mpa_m2": float(gradient_norm),
        "gradient_max_abs_mpa_m2": float(torch.amax(torch.abs(gradient))),
        "force_balance_relative": float(
            force_balance / gradient_norm.clamp_min(1.0e-300)
        ),
        "minimum_active_distance_m": diagnostics["minimum_active_distance_m"],
        "contact_numerically_valid": diagnostics["contact_numerically_valid"],
        "weights": {
            "minimum": float(weights.min()) if len(weights) else None,
            "maximum": float(weights.max()) if len(weights) else None,
            "sum": float(weights.sum()),
            "negative_count": int(np.count_nonzero(weights < 0)),
            "zero_count": int(np.count_nonzero(weights == 0)),
            "nonfinite_count": int(np.count_nonzero(~np.isfinite(weights))),
        },
    }


def directional_gradient_check(
    contact: Any,
    u: torch.Tensor,
    soft_ids: np.ndarray,
    collision_type: Any,
) -> dict[str, Any]:
    state = rebuilt_state(contact, u, collision_type)
    gradient = torch.zeros_like(u)
    contact.grad(state, u, gradient)
    generator = torch.Generator(device="cpu").manual_seed(75)
    direction = torch.zeros_like(u)
    soft_direction = torch.randn((len(soft_ids), 3), generator=generator, dtype=u.dtype)
    soft_direction /= torch.amax(torch.abs(soft_direction))
    direction[torch.as_tensor(soft_ids)] = soft_direction
    analytic = float(torch.sum(gradient * direction))
    rows = []
    for step in (3.0e-9, 1.0e-9):
        plus_u = u + step * direction
        minus_u = u - step * direction
        plus = rebuilt_state(contact, plus_u, collision_type)
        minus = rebuilt_state(contact, minus_u, collision_type)
        finite_difference = float(
            (contact.fun(plus, plus_u) - contact.fun(minus, minus_u)) / (2 * step)
        )
        relative_error = abs(finite_difference - analytic) / max(
            abs(finite_difference), abs(analytic), 1.0e-30
        )
        rows.append(
            {
                "step_m": step,
                "analytic_mpa_m2": analytic,
                "finite_difference_mpa_m2": finite_difference,
                "relative_error": relative_error,
            }
        )
    return {
        "direction": "seed75 random soft-surface displacement; infinity norm one",
        "rows": rows,
        "maximum_relative_error": max(row["relative_error"] for row in rows),
    }


def spread(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    values = [float(row[key]) for row in rows]
    return {
        "minimum": min(values),
        "maximum": max(values),
        "range": max(values) - min(values),
    }


def synthetic_ccd_cap_check() -> dict[str, Any]:
    vertices_t0 = np.asarray(
        [
            [0.0, 0.0, 5.0e-5],
            [-1.0e-3, -1.0e-3, 0.0],
            [1.0e-3, -1.0e-3, 0.0],
            [0.0, 1.0e-3, 0.0],
        ],
        dtype=np.float64,
    )
    vertices_t1 = vertices_t0.copy()
    vertices_t1[0, 2] = -5.0e-5
    faces = np.asarray([[1, 2, 3]], dtype=np.int32)
    mesh = ipctk.CollisionMesh(vertices_t0, ipctk.edges(faces), faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.asarray([0, 1, 1, 1], dtype=np.int32)
    )
    mesh.init_adjacencies()
    candidates = ipctk.Candidates()
    candidates.build(
        mesh=mesh,
        vertices_t0=vertices_t0,
        vertices_t1=vertices_t1,
        inflation_radius=0.0,
        broad_phase=ipctk.LBVH(),
    )
    rows = []
    for max_iterations in (1, 10, 100, 100_000, 10_000_000):
        ccd = ipctk.TightInclusionCCD(
            tolerance=1.0e-6,
            max_iterations=max_iterations,
            conservative_rescaling=0.8,
        )
        started = time.perf_counter()
        try:
            fraction = float(
                candidates.compute_collision_free_stepsize(
                    mesh,
                    vertices_t0,
                    vertices_t1,
                    min_distance=0.0,
                    narrow_phase_ccd=ccd,
                )
            )
            endpoint = vertices_t0 + fraction * (vertices_t1 - vertices_t0)
            collision_free = bool(
                ipctk.is_step_collision_free(
                    mesh,
                    vertices_t0,
                    endpoint,
                    min_distance=0.0,
                    broad_phase=ipctk.LBVH(),
                    narrow_phase_ccd=ipctk.TightInclusionCCD(),
                )
            )
            row = {
                "max_iterations": max_iterations,
                "status": "returned",
                "fraction": fraction,
                "verified_collision_free": collision_free,
            }
        except Exception as error:  # noqa: BLE001 - diagnostic records native policy
            row = {
                "max_iterations": max_iterations,
                "status": "raised",
                "exception_type": type(error).__name__,
                "exception": str(error),
            }
        row["seconds"] = time.perf_counter() - started
        rows.append(row)
    return {
        "geometry": "point crosses a static triangle from +50 to -50 micrometres",
        "rows": rows,
    }


def main(cfg: Config) -> None:
    if cfg.repeats != 3:
        message = "the declared diagnostic uses exactly three rebuilds"
        raise ValueError(message)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    provenance = archive_sources(cfg.output_dir)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    geometry = load_full_skull_geometry(cfg.geometry, cfg.geometry_audit)
    adapter = build_full_skull_contact(geometry, contact_config())
    contact = adapter.collision
    protocol = json.loads((cfg.run_dir / "protocol.json").read_text())
    assert protocol["mechanics"]["contact"] == contact_config()
    checkpoints = []
    displacement_by_step: dict[int, torch.Tensor] = {}
    types = collision_types()
    for step in (0, 200, 600, 1000):
        checkpoint_path = cfg.run_dir / f"checkpoint-step-{step:05d}.npz"
        receipt_path = checkpoint_path.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        assert receipt["sha256"] == sha256(checkpoint_path)
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            fem_u = np.asarray(archive["displacement_m"])
        assert fem_u.shape == (geometry.fem_node_count, 3)
        u = torch.from_numpy(
            np.concatenate(
                (
                    fem_u,
                    np.zeros(
                        (geometry.cranium_node_count + geometry.mandible_node_count, 3),
                        dtype=np.float64,
                    ),
                )
            )
        )
        displacement_by_step[step] = u
        variants = {}
        for name, collision_type in types.items():
            rows = [state_row(contact, u, collision_type) for _ in range(cfg.repeats)]
            variants[name] = {
                "rows": rows,
                "energy_spread": spread(rows, "energy_mpa_m3"),
                "gradient_norm_spread": spread(rows, "gradient_norm_mpa_m2"),
                "collision_count_values": sorted(
                    {row["collision_count"] for row in rows}
                ),
                "candidate_count_values": sorted(
                    {row["candidate_count"] for row in rows}
                ),
            }
        checkpoints.append(
            {
                "step": step,
                "checkpoint": {
                    "path": str(checkpoint_path.resolve()),
                    "sha256": receipt["sha256"],
                    "stored_contact": receipt.get("contact"),
                },
                "variants": variants,
            }
        )
    cap_check = synthetic_ccd_cap_check()
    ipc_directional = directional_gradient_check(
        contact,
        displacement_by_step[0],
        geometry.soft_global_ids,
        types["ipc"],
    )
    capped_rows = [
        row for row in cap_check["rows"] if row["max_iterations"] < 10_000_000
    ]
    capped_policy_safe = all(
        row["status"] == "raised" or row.get("verified_collision_free") is True
        for row in capped_rows
    )
    improved_1000 = checkpoints[-1]["variants"]["improved_max_approx"]["rows"]
    ipc_1000 = checkpoints[-1]["variants"]["ipc"]["rows"]
    summary = {
        "schema": "joint-contact-rebuild-diagnostic-v1",
        "success": True,
        "status": "completed_cpu_contact_rebuild_diagnostic",
        "scope": "no equilibrium solve; exact saved displacements; CPU IPC only",
        "runtime": {
            "ipctk_version": ipctk.__version__,
            "ipctk_threads": ipctk.get_num_threads(),
            "torch_threads": torch.get_num_threads(),
            "collision_set_enum_members": sorted(
                ipctk.NormalCollisions.CollisionSetType.__members__
            ),
            "max_approx_available": hasattr(
                ipctk.NormalCollisions.CollisionSetType, "MAX_APPROX"
            ),
        },
        "inputs": {
            "run_protocol": str((cfg.run_dir / "protocol.json").resolve()),
            "run_protocol_sha256": sha256(cfg.run_dir / "protocol.json"),
            "geometry": geometry.binding_receipt(),
            "contact_config": contact_config(),
            "provenance": provenance,
        },
        "checkpoints": checkpoints,
        "tight_inclusion_iteration_cap": cap_check,
        "tight_inclusion_capped_policy_safe_on_synthetic": capped_policy_safe,
        "ipc_full_source_directional_gradient": ipc_directional,
        "conclusions": {
            "improved_step1000_has_negative_energy": any(
                row["energy_mpa_m3"] < 0 for row in improved_1000
            ),
            "improved_step1000_has_negative_weights": any(
                row["weights"]["negative_count"] > 0 for row in improved_1000
            ),
            "ipc_step1000_has_negative_energy": any(
                row["energy_mpa_m3"] < 0 for row in ipc_1000
            ),
            "ipc_step1000_has_negative_weights": any(
                row["weights"]["negative_count"] > 0 for row in ipc_1000
            ),
            "positive_replacement_validated": all(
                row["energy_mpa_m3"] >= 0
                and row["weights"]["negative_count"] == 0
                and math.isfinite(row["gradient_norm_mpa_m2"])
                for checkpoint in checkpoints
                for row in checkpoint["variants"]["ipc"]["rows"]
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "improved_step1000_min_energy": min(
                row["energy_mpa_m3"] for row in improved_1000
            ),
            "ipc_step1000_min_energy": min(row["energy_mpa_m3"] for row in ipc_1000),
            "ipc_positive_replacement": float(
                summary["conclusions"]["positive_replacement_validated"]
            ),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
