# ruff: noqa: PLR0915
"""Manual facial-muscle activation sweep with and without the skin membrane."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path

import activation_models as am
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from face_physics import ROOT, FacePhysics, ForwardConvergenceError, configure

from liblaf import cherries
from liblaf.apple.warp.model import WarpModel, WarpModelAdapter

LOG = logging.getLogger(__name__)

PATTERNS: dict[str, tuple[int, ...]] = {
    "smile-elevators": (57, 58, 63, 64, 142, 143, 218, 219, 283, 284),
    "risorius": (73, 74),
    "orbicularis-oris": (254,),
}
CONTRACTIONS = (0.10, 0.30, 0.50)
SKIN_FACTORS = (0.0, 0.12)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = (
        Path(__file__).resolve().parents[2]
        / "face-activation-materials/data/10-fixture"
    )
    output_dir: Path = cherries.output("10-manual-activation", mkdir=True)
    fat_factor: float = 1.0
    muscle_factor: float = 0.8
    soft_nu: float = 0.46
    skin_nu: float = 0.46
    fat_model: str = "stable"
    fat_nu: float = 0.49
    forward_rtol: float = 1e-5
    forward_atol: float = 1e-12
    target_name: str = "Smile"


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )


def quantiles(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values)
    assert values.size
    assert np.isfinite(values).all()
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q01", "q05", "median", "q95", "q99", "max"),
            np.quantile(values, (0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)),
            strict=True,
        )
    }


def weighted_stats(values: np.ndarray, weights: np.ndarray) -> dict[str, object]:
    values = np.asarray(values)
    weights = np.asarray(weights)
    assert values.shape[0] == weights.shape[0]
    assert np.isfinite(values).all()
    assert np.isfinite(weights).all()
    assert np.all(weights >= 0)
    assert weights.sum() > 0
    normalized = weights / weights.sum()
    return {
        "mean": np.sum(
            values * normalized.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0
        ).tolist(),
        "rms": float(np.sqrt(np.sum(normalized * np.sum(values**2, axis=-1))))
        if values.ndim > 1
        else float(np.sqrt(np.sum(normalized * values**2))),
    }


def deformation_gradients(p: FacePhysics, u: np.ndarray) -> np.ndarray:
    x = p.points + u
    ds = np.transpose(x[p.tets[:, 1:]] - x[p.tets[:, :1]], (0, 2, 1))
    return ds @ p.dm_inv


def component_mechanics(p: FacePhysics, u: np.ndarray) -> dict[str, object]:
    """Return potential energy and force diagnostics in raw and SI units."""
    u_t = torch.as_tensor(u)
    fixed_mask = np.asarray(p.mesh.point_data["FixedMask"], dtype=bool)
    result: dict[str, object] = {}
    total_grad = torch.zeros_like(u_t)
    total_energy = 0.0
    potentials = p.forward.model.warp_model.__wrapped__.potentials
    for name, potential in potentials.items():
        adapter = WarpModelAdapter(WarpModel({name: potential}))
        grad = torch.zeros_like(u_t)
        energy = float(adapter.fun(u_t).detach().cpu())
        adapter.grad(u_t, grad)
        grad_np = grad.detach().cpu().numpy()
        assert math.isfinite(energy)
        assert np.isfinite(grad_np).all()
        total_energy += energy
        total_grad += grad
        reaction = -np.where(fixed_mask, grad_np, 0.0).sum(axis=0) * 1e6
        assert reaction.shape == (3,)
        result[name] = {
            "energy_raw_MPa_m3": energy,
            "energy_J": energy * 1e6,
            "force_gradient_l2_raw_MPa_m2": float(np.linalg.norm(grad_np)),
            "force_gradient_l2_N": float(np.linalg.norm(grad_np) * 1e6),
            "constraint_reaction_resultant_N": reaction.tolist(),
            "constraint_reaction_resultant_norm_N": float(np.linalg.norm(reaction)),
        }
    total_np = total_grad.detach().cpu().numpy()
    free = total_np[~fixed_mask]
    reaction = -np.where(fixed_mask, total_np, 0.0).sum(axis=0) * 1e6
    assert reaction.shape == (3,)
    result["total"] = {
        "energy_raw_MPa_m3": total_energy,
        "energy_J": total_energy * 1e6,
        "free_force_gradient_l2_N": float(np.linalg.norm(free) * 1e6),
        "free_force_gradient_linf_N": float(np.abs(free).max() * 1e6),
        "constraint_reaction_resultant_N": reaction.tolist(),
        "constraint_reaction_resultant_norm_N": float(np.linalg.norm(reaction)),
        "unit_conversion": "Coordinates are metres and moduli are MPa: energy_raw*1e6 is J and gradient_raw*1e6 is N.",
    }
    return result


def case_diagnostics(
    p: FacePhysics,
    u: np.ndarray,
    ainv: np.ndarray,
    selected_local: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    selected_cells = p.ids[selected_local]
    selected_tets = p.tets[selected_cells]
    selected_vertices = np.unique(selected_tets)
    f_all = deformation_gradients(p, u)
    f = f_all[selected_cells]
    g = f @ ainv[selected_local]
    f_stretch = np.linalg.svd(f, compute_uv=False)
    g_stretch = np.linalg.svd(g, compute_uv=False)
    fibers = np.asarray(p.mesh.cell_data["ActivationFiber"])[selected_cells]
    fibers /= np.linalg.norm(fibers, axis=1, keepdims=True)
    fiber_stretch = np.linalg.norm(np.einsum("nij,nj->ni", f, fibers), axis=1)
    detf = np.linalg.det(f_all)
    detf_selected = detf[selected_cells]
    muscle_fraction = np.asarray(p.mesh.cell_data["MuscleFraction"])
    weights = p.volumes_all[selected_cells] * muscle_fraction[selected_cells]
    centroids = u[selected_tets].mean(axis=1)

    surface_u = u[p.top]
    surface_target = p.target[p.top]
    projection = float(np.sum(p.weights[:, None] * surface_u * surface_target) / p.D**2)
    projected = projection * surface_target
    surface_residual = surface_u - projected
    lip = np.asarray(p.mesh.point_data["IsLip"], dtype=bool)
    lip_ids = np.flatnonzero(lip)
    lip_u = u[lip_ids]
    lip_xy = p.points[lip_ids, :2]
    lip_center = lip_xy.mean(axis=0)
    radial = lip_xy - lip_center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    radial_motion = np.sum(lip_u[:, :2] * radial, axis=1)
    fixed_mask = np.asarray(p.mesh.point_data["FixedMask"], dtype=bool)
    fixed_vertices = np.any(fixed_mask[selected_vertices], axis=1)

    diagnostics: dict[str, object] = {
        "physical_deformation": {
            "interpretation": "F maps rest tetrahedra to the solved state; det(F) is the physical local volume ratio.",
            "detF_all": quantiles(detf),
            "detF_selected": quantiles(detf_selected),
            "inverted_tets_all": int(np.count_nonzero(detf <= 0)),
            "inverted_tets_selected": int(np.count_nonzero(detf_selected <= 0)),
            "principal_stretches_F_selected": {
                f"sigma_{index + 1}": quantiles(f_stretch[:, index])
                for index in range(3)
            },
            "fiber_stretch_F_selected": quantiles(fiber_stretch),
            "fiber_stretch_F_selected_fraction_volume_weighted_mean": float(
                np.sum(weights * fiber_stretch) / weights.sum()
            ),
        },
        "elastic_deformation": {
            "interpretation": "G=F@A_inv is the elastic deformation evaluated by the active muscle material.",
            "detG_selected": quantiles(np.linalg.det(g)),
            "principal_stretches_G_selected": {
                f"sigma_{index + 1}": quantiles(g_stretch[:, index])
                for index in range(3)
            },
        },
        "selected_muscle_motion": {
            "cell_count": len(selected_cells),
            "vertex_count": len(selected_vertices),
            "fixed_vertex_count": int(fixed_vertices.sum()),
            "fixed_vertex_fraction": float(fixed_vertices.mean()),
            "fixed_incident_cell_count": int(
                np.count_nonzero(
                    np.any(np.any(fixed_mask[selected_tets], axis=2), axis=1)
                )
            ),
            "fraction_weighted_volume_m3": float(weights.sum()),
            "centroid_displacement": weighted_stats(centroids, weights),
            "centroid_displacement_rms_mm": float(
                1000
                * np.sqrt(
                    np.sum(weights * np.sum(centroids**2, axis=1)) / weights.sum()
                )
            ),
            "vertex_displacement_rms_mm": float(
                1000 * np.sqrt(np.mean(np.sum(u[selected_vertices] ** 2, axis=1)))
            ),
            "vertex_displacement_max_mm": float(
                1000 * np.linalg.norm(u[selected_vertices], axis=1).max()
            ),
        },
        "surface_motion": {
            "observation_vertex_count": len(p.top),
            "weighted_mean_vector_mm": (
                1000 * np.sum(p.weights[:, None] * surface_u, axis=0)
            ).tolist(),
            "weighted_rms_mm": float(
                1000 * np.sqrt(np.sum(p.weights[:, None] * surface_u**2))
            ),
            "weighted_max_mm": float(1000 * np.linalg.norm(surface_u, axis=1).max()),
            "smile_target_rms_mm": float(1000 * p.D),
            "smile_target_projection_amplitude": projection,
            "orthogonal_residual_over_target": float(
                np.sqrt(np.sum(p.weights[:, None] * surface_residual**2)) / p.D
            ),
        },
        "lip_motion": {
            "vertex_count": len(lip_ids),
            "mean_vector_mm": (1000 * lip_u.mean(axis=0)).tolist(),
            "rms_mm": float(1000 * np.sqrt(np.mean(np.sum(lip_u**2, axis=1)))),
            "max_mm": float(1000 * np.linalg.norm(lip_u, axis=1).max()),
            "mean_outward_radial_xy_mm": float(1000 * radial_motion.mean()),
            "radial_xy_rms_mm": float(1000 * np.sqrt(np.mean(radial_motion**2))),
        },
        "mechanics": component_mechanics(p, u),
    }
    arrays = {
        "F_selected": f,
        "F_principal_stretches_selected": f_stretch,
        "G_selected": g,
        "G_principal_stretches_selected": g_stretch,
        "fiber_stretch_F_selected": fiber_stretch,
        "selected_cell_ids": selected_cells,
        "selected_active_local_ids": selected_local,
        "selected_vertex_ids": selected_vertices,
        "selected_centroid_displacement": centroids,
        "surface_vertex_ids": p.top,
        "surface_displacement": surface_u,
        "surface_target": surface_target,
        "surface_weights": p.weights,
        "lip_vertex_ids": lip_ids,
        "lip_displacement": lip_u,
        "lip_radial_xy_motion": radial_motion,
    }
    return diagnostics, arrays


def save_case(
    p: FacePhysics,
    path: Path,
    u: np.ndarray,
    q: np.ndarray,
    ainv: np.ndarray,
    selected_local: np.ndarray,
    diagnostics: dict[str, object],
    arrays: dict[str, np.ndarray],
) -> None:
    path.mkdir(parents=True, exist_ok=False)
    p.save_mesh(path / "state.vtu", u, ainv)
    mesh = pv.read(path / "state.vtu")
    selected_cells = p.ids[selected_local]
    pattern = np.zeros(mesh.n_cells, dtype=np.int8)
    pattern[selected_cells] = 1
    control = np.zeros(mesh.n_cells)
    control[p.ids] = q
    surface_weight = np.zeros(mesh.n_points)
    surface_weight[p.top] = p.weights
    mesh.cell_data["ManualPattern"] = pattern
    mesh.cell_data["ManualFiberAmplitude"] = control
    mesh.point_data["SurfaceObservationWeight"] = surface_weight
    mesh.save(path / "state.vtu")
    np.savez_compressed(path / "state.npz", q=q, u=u, Ainv=ainv, **arrays)
    write_json(path / "diagnostics.json", diagnostics)


def static_preflight(fixture: Path) -> dict[str, object]:
    mesh = pv.read(fixture / "volume.vtu")
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    active = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    muscle_id = np.asarray(mesh.cell_data["MuscleId"], dtype=int)
    fixed = np.asarray(mesh.point_data["FixedMask"], dtype=bool)
    rows = {}
    for name, ids in PATTERNS.items():
        selected = active & np.isin(muscle_id, ids)
        assert selected.any()
        vertices = np.unique(tets[selected])
        rows[name] = {
            "muscle_ids": list(ids),
            "active_cell_count": int(selected.sum()),
            "vertex_count": len(vertices),
            "fixed_vertex_count": int(np.any(fixed[vertices], axis=1).sum()),
            "fixed_incident_cell_count": int(
                np.count_nonzero(np.any(np.any(fixed[tets[selected]], axis=2), axis=1))
            ),
        }
    return {
        "n_tets": int(mesh.n_cells),
        "n_vertices": int(mesh.n_points),
        "active_tets": int(active.sum()),
        "fixed_vertices": int(np.any(fixed, axis=1).sum()),
        "cut_added_fixed_vertices": int(
            np.count_nonzero(mesh.point_data["CutBoundaryAddedFixed"])
        ),
        "patterns": rows,
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Choose an empty output directory"
    assert cfg.forward_rtol == 1e-5
    assert cfg.forward_atol == 1e-12
    assert cfg.fat_nu == 0.49
    preflight = static_preflight(cfg.fixture)
    write_json(out / "preflight.json", preflight)
    write_json(out / "config.json", cfg.model_dump(mode="json"))

    source_dir = out / "sources"
    source_dir.mkdir()
    sources = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, source_dir / path.name)
        sources[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    input_files = ("volume.vtu", "skin.vtp", "summary.json")
    write_json(
        out / "provenance.json",
        {
            "sources": sources,
            "inputs": {
                name: hashlib.sha256((cfg.fixture / name).read_bytes()).hexdigest()
                for name in input_files
            },
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "control_definition": {
                "patterns": PATTERNS,
                "contractions": CONTRACTIONS,
                "skin_factors": SKIN_FACTORS,
                "a_from_contraction": "a=-log(1-c); A_inv has eigenvalues exp(a), exp(-a/2), exp(-a/2)",
            },
        },
    )

    configure()
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    control_hashes: dict[str, str] = {}
    for skin_factor in SKIN_FACTORS:
        p = FacePhysics(
            cfg.fixture,
            skin_factor=skin_factor,
            fat_factor=cfg.fat_factor,
            muscle_factor=cfg.muscle_factor,
            rtol=cfg.forward_rtol,
            atol=cfg.forward_atol,
            soft_nu=cfg.soft_nu,
            skin_nu=cfg.skin_nu,
            target_name=cfg.target_name,
            fat_model=cfg.fat_model,
            fat_nu=cfg.fat_nu,
        )
        active_muscle_id = np.asarray(p.mesh.cell_data["MuscleId"], dtype=int)[p.ids]
        for pattern_name, muscle_ids in PATTERNS.items():
            selected_local = np.flatnonzero(np.isin(active_muscle_id, muscle_ids))
            assert (
                len(selected_local)
                == preflight["patterns"][pattern_name]["active_cell_count"]
            )
            seed = np.zeros_like(p.points)
            for contraction in CONTRACTIONS:
                amplitude = -math.log1p(-contraction)
                q = np.zeros(len(p.ids))
                q[selected_local] = amplitude
                q_hash = hashlib.sha256(q.tobytes()).hexdigest()
                key = f"{pattern_name}/c{round(100 * contraction):02d}"
                previous = control_hashes.setdefault(key, q_hash)
                assert previous == q_hash, (
                    "Skin comparisons must use identical controls"
                )
                q_t = torch.as_tensor(q[:, None])
                ainv_t, _ = am.matrices(q_t, "F", p.fibers)
                ainv = ainv_t.detach().cpu().numpy()
                assert np.allclose(np.linalg.det(ainv), 1.0, rtol=1e-12, atol=1e-12)
                case_name = f"skin-{skin_factor:.2f}/{pattern_name}/c{round(100 * contraction):02d}"
                case_path = out / case_name
                LOG.info(
                    "Solving %s: %d cells, contraction %.0f%%, a=%.9f",
                    case_name,
                    len(selected_local),
                    100 * contraction,
                    amplitude,
                )
                try:
                    u = p.solve(am.packed(ainv_t), seed).detach().cpu().numpy()
                except ForwardConvergenceError as error:
                    failure = {
                        "case": case_name,
                        "error": str(error),
                        "forward": p.last_forward,
                    }
                    case_path.mkdir(parents=True, exist_ok=False)
                    write_json(case_path / "failure.json", failure)
                    write_json(out / "failure.json", failure)
                    raise
                assert np.isfinite(u).all()
                diagnostics, arrays = case_diagnostics(p, u, ainv, selected_local)
                diagnostics.update(
                    {
                        "case": case_name,
                        "skin_factor": skin_factor,
                        "pattern": pattern_name,
                        "muscle_ids": muscle_ids,
                        "contraction": contraction,
                        "fiber_amplitude_a": amplitude,
                        "activation_inverse": {
                            "interpretation": "A_inv is a prescribed stress-free active-map parameter, separate from final F.",
                            "determinant": quantiles(np.linalg.det(ainv)),
                            "control_sha256": q_hash,
                        },
                        "forward": p.last_forward,
                    }
                )
                save_case(
                    p,
                    case_path,
                    u,
                    q,
                    ainv,
                    selected_local,
                    diagnostics,
                    arrays,
                )
                physical = diagnostics["physical_deformation"]
                surface = diagnostics["surface_motion"]
                muscle = diagnostics["selected_muscle_motion"]
                lip = diagnostics["lip_motion"]
                row = {
                    "case": case_name,
                    "skin_factor": skin_factor,
                    "pattern": pattern_name,
                    "contraction": contraction,
                    "fiber_amplitude_a": amplitude,
                    "selected_cells": len(selected_local),
                    "forward_steps": p.last_forward["steps"],
                    "forward_grad_norm": p.last_forward["grad_norm"],
                    "detF_min_all": physical["detF_all"]["min"],
                    "inverted_tets_all": physical["inverted_tets_all"],
                    "fiber_stretch_F_median": physical["fiber_stretch_F_selected"][
                        "median"
                    ],
                    "muscle_centroid_rms_mm": muscle["centroid_displacement_rms_mm"],
                    "surface_rms_mm": surface["weighted_rms_mm"],
                    "smile_projection": surface["smile_target_projection_amplitude"],
                    "lip_rms_mm": lip["rms_mm"],
                    "lip_radial_outward_mean_mm": lip["mean_outward_radial_xy_mm"],
                    "wall_s": time.perf_counter() - start,
                }
                rows.append(row)
                with (out / "trace.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(rows)
                cherries.log_metrics(
                    {
                        "manual/surface_rms_mm": row["surface_rms_mm"],
                        "manual/smile_projection": row["smile_projection"],
                        "manual/detF_min": row["detF_min_all"],
                        "manual/fiber_stretch_median": row["fiber_stretch_F_median"],
                    },
                    step=len(rows),
                )
                seed = u
                LOG.info("Completed %s: %s", case_name, row)

    summary = {
        "status": "completed",
        "case_count": len(rows),
        "expected_case_count": len(SKIN_FACTORS) * len(PATTERNS) * len(CONTRACTIONS),
        "preflight": preflight,
        "materials": {
            "fat_nu": cfg.fat_nu,
            "fat_model": cfg.fat_model,
            "muscle_factor": cfg.muscle_factor,
            "skin_factors": SKIN_FACTORS,
        },
        "solver": {
            "max_steps": 10000,
            "rtol": cfg.forward_rtol,
            "atol": cfg.forward_atol,
            "continuation": "Within each skin/pattern branch: 10%, then 30%, then 50%; each pattern starts from rest.",
            "geometry_rejection": False,
            "success_requirement": "Strict finite equilibrium; distorted and inverted finite solved states are retained.",
        },
        "historical_coverage_context": {
            "current_active_tets": preflight["active_tets"],
            "june_active_tets": 288235,
            "current_fixed_vertices": preflight["fixed_vertices"],
            "june_fixed_vertices": 27036,
            "current_cut_added_fixed_vertices": preflight["cut_added_fixed_vertices"],
            "interpretation": "Counts diagnose changed actuator and boundary coverage; June values come from the previously audited baseline and are not inferred from this solve.",
        },
        "control_hashes_identical_across_skin": control_hashes,
        "rows": rows,
        "wall_s": time.perf_counter() - start,
    }
    assert len(rows) == summary["expected_case_count"]
    write_json(out / "summary.json", summary)
    LOG.info("Completed all %d manual activation cases", len(rows))


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
