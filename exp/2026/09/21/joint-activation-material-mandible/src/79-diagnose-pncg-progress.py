"""Summarize stopped PNCG traces without evaluating the mechanics model."""

from __future__ import annotations

import json
import math
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_full_skull_contact import build_full_skull_contact, load_full_skull_geometry

from liblaf import cherries


class Config(cherries.BaseConfig):
    capped_run: Path = GROUP / "data/simple-skin-forward-006"
    aggressive_run: Path = GROUP / "data/simple-skin-forward-007"
    output_dir: Path = GROUP / "data/pncg-progress-diagnostic-001"


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[round(fraction * (len(ordered) - 1))]

    return {
        "minimum": ordered[0],
        "p10": at(0.1),
        "median": at(0.5),
        "p90": at(0.9),
        "maximum": ordered[-1],
    }


def analyze(path: Path) -> dict[str, Any]:
    protocol_path = path / "protocol.json"
    summary_path = path / "summary.json"
    trace_path = path / "trace.jsonl"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    steps = [row for row in rows if row["step"] > 0]
    exact = [row for row in rows if "accepted_state_free_force_norm" in row]
    cap = float(protocol["solver"]["max_step_norm_m"])
    actual_norms = [
        abs(float(row["line_search"]["alpha"])) * float(row["direction_inf_norm"])
        for row in steps
    ]
    positive_curvature = [
        row for row in steps if float(row["direction_hessian_quadratic"]) > 0
    ]
    raw_newton_displacements = [
        -float(row["directional_slope"])
        / float(row["direction_hessian_quadratic"])
        * float(row["direction_inf_norm"])
        for row in positive_curvature
    ]
    contacts = [row["contact"] for row in exact if "contact" in row]
    force_values = [float(row["accepted_state_free_force_norm"]) for row in exact]
    force_fit = None
    if len(exact) >= 3 and all(value > 0 for value in force_values):
        x = [float(row["step"]) for row in exact]
        y = [math.log(value) for value in force_values]
        mean_x = statistics.mean(x)
        mean_y = statistics.mean(y)
        slope = sum(
            (x_i - mean_x) * (y_i - mean_y) for x_i, y_i in zip(x, y, strict=True)
        ) / sum((x_i - mean_x) ** 2 for x_i in x)
        target = float(summary["force_threshold"])
        projected = math.log(target / force_values[-1]) / slope if slope < 0 else None
        force_fit = {
            "log_force_slope_per_step": slope,
            "multiplicative_factor_per_step": math.exp(slope),
            "projected_additional_steps_to_threshold": projected,
            "projection_is_descriptive_only": True,
        }
    return {
        "path": str(path.resolve()),
        "hashes": {
            "protocol": sha256(protocol_path),
            "summary": sha256(summary_path),
            "trace": sha256(trace_path),
        },
        "status": summary["status"],
        "accepted_steps": int(summary["accepted_steps"]),
        "wall_seconds": float(summary["wall_seconds"]),
        "solver": protocol["solver"],
        "initial_exact_force_norm": force_values[0],
        "terminal_exact_force_norm": float(summary["final_free_force_norm"]),
        "minimum_sampled_exact_force_norm": min(force_values),
        "maximum_sampled_exact_force_norm": max(force_values),
        "sampled_force_reduction_ratio": force_values[-1] / force_values[0],
        "force_fit": force_fit,
        "accepted_step_inf_norm_m": quantiles(actual_norms),
        "norm_cap_binding_fraction": sum(
            abs(value - cap) <= max(1e-15, 1e-10 * cap) for value in actual_norms
        )
        / len(actual_norms),
        "armijo_step_histogram": {
            str(key): count
            for key, count in sorted(
                Counter(int(row["line_search"]["step"]) for row in steps).items()
            )
        },
        "negative_raw_curvature_count": sum(
            float(row["direction_hessian_quadratic"]) < 0 for row in steps
        ),
        "damping_factor": quantiles(
            [float(row["hessian_damping_factor"]) for row in steps]
        ),
        "raw_positive_curvature_newton_displacement_inf_m": quantiles(
            raw_newton_displacements
        ),
        "step_seconds": quantiles([float(row["step_seconds"]) for row in steps]),
        "contact": {
            "minimum_sampled_gap_m": min(
                float(value["minimum_active_distance_m"]) for value in contacts
            ),
            "minimum_recorded_ccd_fraction": min(
                float(value["ccd_minimum_inner_fraction"]) for value in contacts
            ),
            "all_sampled_numerically_valid": all(
                value["contact_numerically_valid"] is True for value in contacts
            ),
        },
        "terminal_metrics": summary["metrics"],
    }


def synthetic_clearance_check(clearance_m: float) -> dict[str, Any]:
    vertices = np.asarray(
        [
            [0.0, 0.0, 5.0e-5],
            [-1.0e-3, -1.0e-3, 0.0],
            [1.0e-3, -1.0e-3, 0.0],
            [0.0, 1.0e-3, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray([[1, 2, 3]], dtype=np.int32)
    mesh = ipctk.CollisionMesh(vertices, ipctk.edges(faces), faces)
    mesh.can_collide = ipctk.make_vertex_patches_filter(
        np.asarray([0, 1, 1, 1], dtype=np.int32)
    )
    mesh.init_adjacencies()
    ccd = ipctk.TightInclusionCCD(
        tolerance=1.0e-6, max_iterations=1000, conservative_rescaling=0.8
    )
    rows = []
    for iteration in range(8):
        endpoint = vertices.copy()
        endpoint[0, 2] = -5.0e-5
        candidates = ipctk.Candidates()
        candidates.build(
            mesh=mesh,
            vertices_t0=vertices,
            vertices_t1=endpoint,
            inflation_radius=0.0,
            broad_phase=ipctk.LBVH(),
        )
        fraction = float(
            candidates.compute_collision_free_stepsize(
                mesh,
                vertices,
                endpoint,
                min_distance=clearance_m,
                narrow_phase_ccd=ccd,
            )
        )
        vertices = vertices + fraction * (endpoint - vertices)
        gap = float(vertices[0, 2])
        rows.append({"iteration": iteration + 1, "fraction": fraction, "gap_m": gap})
        assert gap >= clearance_m - 1e-15
    return {
        "clearance_m": clearance_m,
        "barrier_dmin_m": 0.0,
        "rows": rows,
        "minimum_gap_m": min(row["gap_m"] for row in rows),
    }


def stopped_checkpoint_clearance_check(
    path: Path, clearance_m: float
) -> dict[str, Any]:
    protocol = json.loads((path / "protocol.json").read_text())
    summary = json.loads((path / "summary.json").read_text())
    geometry_receipt = protocol["inputs"]["geometry"]["geometry"]
    geometry = load_full_skull_geometry(
        Path(geometry_receipt["geometry_path"]),
        Path(geometry_receipt["audit_path"]),
    )
    contact_config = protocol["mechanics"]["contact"]
    assert contact_config["dhat_m"] == 1e-4
    adapter = build_full_skull_contact(geometry, contact_config)
    contact = adapter.collision
    assert contact.dmin == 0.0
    contact.min_distance = clearance_m
    checkpoint = Path(summary["checkpoint"]["path"])
    assert sha256(checkpoint) == summary["checkpoint"]["sha256"]
    with np.load(checkpoint, allow_pickle=False) as archive:
        fem_u = np.asarray(archive["displacement_m"], dtype=np.float64)
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
    state = contact.state_at(u)
    diagnostics = contact.diagnostics(state, u)
    assert diagnostics["minimum_active_distance_m"] > clearance_m
    zero_fraction = float(contact.max_step_size(state, u, torch.zeros_like(u)))
    assert zero_fraction == 1.0
    return {
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "clearance_m": clearance_m,
        "barrier_dmin_m": float(contact.dmin),
        "barrier_energy_unchanged": True,
        "minimum_active_distance_m": diagnostics["minimum_active_distance_m"],
        "clearance_inactive_at_checkpoint": True,
        "zero_step_ccd_fraction": zero_fraction,
    }


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    torch.set_default_device("cpu")
    torch.set_default_dtype(torch.float64)
    capped = analyze(cfg.capped_run)
    aggressive = analyze(cfg.aggressive_run)
    clearance_m = 1e-8
    clearance = {
        "definition": (
            "CCD-only minimum separation; the IPC barrier dmin, dhat, stiffness, "
            "energy, gradient, and Hessian remain unchanged"
        ),
        "without_clearance": synthetic_clearance_check(0.0),
        "ten_nanometres": synthetic_clearance_check(clearance_m),
        "stopped_checkpoint": stopped_checkpoint_clearance_check(
            cfg.capped_run, clearance_m
        ),
    }
    success = (
        capped["status"] == "interrupted"
        and aggressive["status"] == "interrupted"
        and capped["contact"]["all_sampled_numerically_valid"]
        and aggressive["contact"]["all_sampled_numerically_valid"]
    )
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "joint-pncg-progress-diagnostic-v1",
            "success": success,
            "status": "completed_stopped_trace_diagnostic",
            "scope": "CPU-only analysis of stopped accepted-step traces; no mechanics evaluation",
            "runs": {"ten_micrometre_cap": capped, "half_millimetre_cap": aggressive},
            "ccd_clearance_proposal": clearance,
            "conclusions": {
                "ten_micrometre_cap_dominated": capped["norm_cap_binding_fraction"]
                == 1.0,
                "half_millimetre_run_contact_limited": (
                    aggressive["contact"]["minimum_sampled_gap_m"] < 1e-12
                    and aggressive["maximum_sampled_exact_force_norm"] > 1e-3
                ),
                "negative_curvature_observed_only_after_aggressive_steps": (
                    capped["negative_raw_curvature_count"] == 0
                    and aggressive["negative_raw_curvature_count"] > 0
                ),
                "conditioning_not_estimated": True,
            },
        },
    )
    cherries.log_output(cfg.output_dir)
    assert success


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
