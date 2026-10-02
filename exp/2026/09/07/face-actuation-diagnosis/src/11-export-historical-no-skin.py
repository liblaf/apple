"""Export the saved June no-skin face endpoint in the diagnosis schema.

This is a readback and metric conversion only.  It does not run equilibrium,
replay the controls, or establish a fresh-rest branch.
"""

# ruff: noqa: C901, EM101, EM102, PERF401, PLR0912, PLR0915, TRY003

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
REPO = ROOT.parents[4]
JUNE = REPO / "exp/2026/06/17/human-face-smile-prestrain-v2"
CURRENT = REPO / "exp/2026/09/07/face-activation-materials"
SOURCE = JUNE / "data/20-human-face-smile-no-skin-lr3.vtu"
SOURCE_TARGET = JUNE / "data/20-human-face-smile-no-skin-lr3-target.vtu"
SOURCE_SUMMARY = JUNE / "data/20-human-face-smile-no-skin-lr3-summary.json"
SOURCE_TRACE = JUNE / "data/20-human-face-smile-no-skin-lr3-trace.jsonl"
FIXTURE = CURRENT / "data/10-fixture/volume.vtu"
SKIN = CURRENT / "data/10-fixture/skin.vtp"
OUTPUT = ROOT / "data/11-historical-no-skin"
CURRENT_RUNNER = ROOT / "src/20-run-face-inverse.py"
CURRENT_RAW6_CONFIG = ROOT / "data/20-raw6-no-skin/config.json"


def digest(path: Path) -> dict[str, str | int]:
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def tetrahedra(mesh: pv.UnstructuredGrid) -> np.ndarray:
    cells = np.asarray(mesh.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        raise ValueError("historical baseline must contain only tetrahedra")
    return cells[:, 1:]


def triangles(mesh: pv.PolyData) -> np.ndarray:
    faces = np.asarray(mesh.faces, dtype=np.int64).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("fixture skin must contain only triangles")
    return faces[:, 1:]


def activation_matrix(q: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"expected packed six-DOF offsets, got {q.shape}")
    h = np.zeros((len(q), 3, 3), dtype=np.float64)
    h[:, 0, 0], h[:, 1, 1], h[:, 2, 2] = q[:, 0], q[:, 1], q[:, 2]
    h[:, 0, 1] = h[:, 1, 0] = q[:, 3]
    h[:, 1, 2] = h[:, 2, 1] = q[:, 4]
    h[:, 0, 2] = h[:, 2, 0] = q[:, 5]
    return h + np.eye(3)


def surface_weights(
    volume: pv.UnstructuredGrid, skin: pv.PolyData, top: np.ndarray
) -> np.ndarray:
    volume_ids = np.asarray(volume.point_data["GlobalPointId"], dtype=np.int64)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    order = np.argsort(volume_ids)
    global_triangles = skin_ids[triangles(skin)]
    locations = np.searchsorted(volume_ids[order], global_triangles)
    if np.any(locations >= len(order)):
        raise ValueError("skin point is absent from the volume")
    tri = order[locations]
    if not np.array_equal(volume_ids[tri], global_triangles):
        raise ValueError("skin-to-volume GlobalPointId mapping failed")
    xyz = np.asarray(volume.points, dtype=np.float64)[tri]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2.0
    )
    weights = np.zeros(volume.n_points, dtype=np.float64)
    np.add.at(weights, tri.ravel(), np.repeat(area / 3.0, 3))
    result = weights[top]
    if np.any(result < 0.0) or result.sum() <= 0.0:
        raise ValueError("target surface weights are invalid")
    return result / result.sum()


def detf(rest: np.ndarray, current: np.ndarray, tets: np.ndarray) -> np.ndarray:
    result = np.empty(len(tets), dtype=np.float64)
    for start in range(0, len(tets), 100_000):
        stop = min(start + 100_000, len(tets))
        ids = tets[start:stop]
        dm = np.transpose(rest[ids[:, 1:]] - rest[ids[:, :1]], (0, 2, 1))
        ds = np.transpose(current[ids[:, 1:]] - current[ids[:, :1]], (0, 2, 1))
        if np.any(np.linalg.det(dm) <= 0.0):
            raise ValueError("rest mesh contains a nonpositive tetrahedron")
        result[start:stop] = np.linalg.det(ds @ np.linalg.inv(dm))
    return result


def region_coverage(
    volume: pv.UnstructuredGrid,
    historical: np.ndarray,
    retained: np.ndarray,
    q_full: np.ndarray,
    tets: np.ndarray,
) -> list[dict[str, Any]]:
    muscle_id = np.asarray(volume.cell_data["MuscleId"], dtype=np.int64)
    names = np.asarray(volume.field_data["MuscleName"]).astype(str)
    volume_weight = np.asarray(
        volume.cell_data["Volume"], dtype=np.float64
    ) * np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)
    q2 = (
        q_full[:, 0] ** 2
        + q_full[:, 1] ** 2
        + q_full[:, 2] ** 2
        + 2.0 * (q_full[:, 3] ** 2 + q_full[:, 4] ** 2 + q_full[:, 5] ** 2)
    )
    added_fixed = np.asarray(volume.point_data["CutBoundaryAddedFixed"], dtype=bool)
    incident = np.any(added_fixed[tets], axis=1)
    total_energy = float(np.sum(volume_weight[historical] * q2[historical]))
    if total_energy <= 0.0:
        raise ValueError("historical activation energy is not positive")
    rows: list[dict[str, Any]] = []
    for value in np.unique(muscle_id[historical]):
        selected = historical & (muscle_id == value)
        weight = volume_weight[selected]
        energy = float(np.sum(weight * q2[selected]))
        rows.append(
            {
                "muscle_id": int(value),
                "name": str(names[value]),
                "current_screen": "retained"
                if np.any(retained & selected)
                else "excluded",
                "cells": int(np.count_nonzero(selected)),
                "muscle_fraction_volume_m3": float(weight.sum()),
                "historical_activation_frobenius_rms": float(
                    math.sqrt(energy / weight.sum())
                ),
                "historical_activation_energy_share": energy / total_energy,
                "tetrahedra_incident_to_added_fixed_vertices": int(
                    np.count_nonzero(selected & incident)
                ),
            }
        )
    return rows


def markdown(summary: dict[str, Any]) -> str:
    metrics = summary["common_area_weighted_metrics"]
    fixed = summary["fixed_boundary_comparison"]
    lines = [
        "# Historical no-skin baseline",
        "",
        "This directory converts the saved June 17 no-skin endpoint into the current diagnosis artifact schema. It performs no forward solve, control replay, or fresh-rest check. The June optimizer saved its best endpoint at step 194 after a 200-step budget; the run reported six failed forward evaluations and did not report inverse convergence.",
        "",
        "The common September rest-surface weights give the saved endpoint an area-weighted fit RMS of "
        f"{metrics['fit_rms_mm']:.6f} mm against a {metrics['target_rms_mm']:.6f} mm target, with {metrics['motion_rms_mm']:.6f} mm motion RMS and target projection amplitude {metrics['target_projection_amplitude']:.6f}. The original uniform-vertex RMS was {summary['recorded_june_metrics']['best_fit_rms_mm']:.6f} mm. The area-weighted and uniform values measure the same saved displacement with different vertex weights.",
        "",
        f"The saved state contains {metrics['inverted_tetrahedra']:,} inverted tetrahedra (minimum det(F) {metrics['detF_min']:.6g}) and {metrics['non_spd_active_tetrahedra']:,} active tensors with a nonpositive minimum eigenvalue. These are recorded endpoint properties. They do not invalidate the fit measurement, and this export does not claim the state satisfies the newer geometric checks.",
        "",
        "## Semantic differences from the September screen",
        "",
        "| Setting | June saved no-skin endpoint | Current diagnosis Raw6 no-skin | Archived September screen |",
        "|---|---:|---:|---:|",
        f"| Active tetrahedra | {summary['activation_domain']['historical_active_tetrahedra']:,} | {summary['activation_domain']['current_screen_active_tetrahedra']:,} | {summary['activation_domain']['current_screen_active_tetrahedra']:,} |",
        f"| Scalar activation controls | {summary['activation_domain']['historical_scalar_dofs']:,} | {summary['activation_domain']['current_raw6_scalar_dofs']:,} | {summary['activation_domain']['current_raw6_scalar_dofs']:,} |",
        f"| Muscle regions represented | {summary['activation_domain']['historical_regions']:,} | {summary['activation_domain']['current_screen_regions']:,} | {summary['activation_domain']['current_screen_regions']:,} |",
        f"| Fixed vertices | {fixed['historical_fixed_vertices']:,} | {fixed['current_fixed_vertices']:,} | {fixed['current_fixed_vertices']:,} |",
        "| Skin energy | absent; zero skin triangles | absent (`skin_factor=0`) | corrected zero-prestrain membrane, factor 0.12 |",
        "| Muscle material | E=30 kPa, nu=0.49 | E=24 kPa, nu=0.46 | E=24 kPa, nu=0.46 |",
        "| Fit weighting | uniform observed vertices and Cartesian components | rest-triangle-area weights | rest-triangle-area weights |",
        "| Outer optimizer | Adam, lr 0.3, eps 0.01 | volume-preconditioned L-BFGS direction with Armijo backtracking and 0.1 per-tet trust radius | volume-preconditioned L-BFGS direction with Armijo backtracking |",
        "| Objective scaling | Cartesian MSE multiplied by 1e6 (mm^2) | squared area-weighted fit normalized by target RMS squared | squared area-weighted fit normalized by target RMS squared |",
        "| Forward / adjoint rtol | 5e-4 / 5e-4 | 1e-5 / 1e-7 | 1e-5 / 1e-7 |",
        "| Endpoint acceptance | solver success only | inversion and active-tensor spectra are recorded diagnostics | SPD active tensor and det(F)>=0.2 checks |",
        "",
        f"The artificial-cut rule adds {fixed['added_fixed_vertices']:,} fixed vertices. None is an observed IsFace or IsLip vertex, but the June endpoint moved those vertices by {fixed['june_displacement_rms_on_added_vertices_mm']:.6f} mm RMS and {fixed['june_displacement_max_on_added_vertices_mm']:.6f} mm maximum. They touch {fixed['historical_active_tetrahedra_incident_to_added_vertices']:,} historically active tetrahedra. This proves the boundary changed degrees of freedom used by the June state; only a matched solve can measure how much it suppresses the smile.",
        "",
        "The current screen retains 35 of 103 historical muscle labels. The 68 excluded labels contain 58.6% of the historical active cells and 32.2% of the saved volume-weighted squared activation offset. Masseter superficial accounts for most of that excluded activation diagnostic, but this is an optimizer-use ranking, not a causal muscle attribution. A counterfactual equilibrium solve is required to decide which excluded labels improve the surface fit.",
        "",
        "## Minimal matched comparison",
        "",
        "Use the same rest volume, original 27,036-vertex fixed mask, all 288,235 historical active tetrahedra, no skin energy, the June target, and the same material and solver tolerances for both Raw6 and Raw6-S. Initialize both from rest and zero activation, give them the same fixed step budget, and report both the original uniform RMS and the current area-weighted RMS. Raw6-S should differ only by the within-muscle shared-face penalty. Keep inversion, det(F), and active-tensor eigenvalues as reported diagnostics so distorted trials remain inspectable.",
        "",
        "The running diagnosis Raw6 case switches to no skin but still changes the active mask, fixed boundary, muscle material, target weighting, optimizer, objective normalization, and solver tolerances. Its partial trajectory therefore cannot isolate per-tetrahedron capacity or be compared directly with the June best endpoint. If it stalls, the next control should be a matched Adam run before attributing the result to the active mask.",
        "",
        "The closest exact rerun command is:",
        "",
        "```bash",
        "cd $REPO_ROOT/exp/2026/06/17/human-face-smile-prestrain-v2",
        'DEBUG=1 CHERRIES_NAME="Human face Smile no-skin lr0.3 baseline loss-mm2" CHERRIES_TAGS="human-face,smile,inverse,no-skin,lr0.3,loss-mm2,baseline200,local" uv run python src/20-inverse-human-face.py --case-set no-skin --inverse-lr 0.3 --inverse-max-steps 200 --mandatory-baseline-steps 200 --segment-steps 8 --time-budget-hours 10 --reserve-minutes 5 --step-time-budget-s 180 --live-plot-dir figs/live --output-summary data/22-no-skin-lr03-summary.json --output-table data/22-no-skin-lr03-table.md',
        "```",
        "",
        "That command would target the historical output names. Add a unique `--case-label` and separate aggregate output paths before rerunning so the verified artifacts are not overwritten.",
        "",
        "## Complete region coverage",
        "",
        "The activation-energy share is computed from the saved offset H=A_inv-I with muscle-fraction-volume weights. It describes where the historical optimizer used controls and does not isolate a muscle's causal effect on the smile.",
        "",
        "| Current screen | ID | Muscle label | Cells | H Frobenius RMS | Saved energy share | Tets touching added fixed vertices |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    rows = sorted(
        summary["muscle_region_coverage"],
        key=lambda row: (row["current_screen"] != "retained", row["muscle_id"]),
    )
    for row in rows:
        lines.append(
            f"| {row['current_screen']} | {row['muscle_id']} | {row['name']} | "
            f"{row['cells']:,} | {row['historical_activation_frobenius_rms']:.6g} | "
            f"{100.0 * row['historical_activation_energy_share']:.4f}% | "
            f"{row['tetrahedra_incident_to_added_fixed_vertices']:,} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite evidence directory: {OUTPUT}")
    for path in (
        SOURCE,
        SOURCE_TARGET,
        SOURCE_SUMMARY,
        SOURCE_TRACE,
        FIXTURE,
        SKIN,
        CURRENT_RUNNER,
        CURRENT_RAW6_CONFIG,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    historical = pv.read(SOURCE)
    source_target = pv.read(SOURCE_TARGET)
    volume = pv.read(FIXTURE)
    skin = pv.read(SKIN)
    if not all(
        isinstance(mesh, pv.UnstructuredGrid)
        for mesh in (historical, source_target, volume)
    ) or not isinstance(skin, pv.PolyData):
        raise TypeError("unexpected artifact dataset type")
    if not (
        np.array_equal(historical.cells, volume.cells)
        and np.array_equal(historical.celltypes, volume.celltypes)
        and np.array_equal(historical.points, volume.points)
        and np.array_equal(source_target.cells, volume.cells)
        and np.array_equal(source_target.points, volume.points)
    ):
        raise ValueError("June and September volume topology/rest points differ")
    if not np.array_equal(
        historical.point_data["Smile"], volume.point_data["Smile"], equal_nan=True
    ):
        raise ValueError("June and September Smile target arrays differ")

    historical_active = np.asarray(historical.cell_data["ActivationMask"], dtype=bool)
    fixture_historical_active = np.asarray(
        volume.cell_data["HistoricalActivationMask"], dtype=bool
    )
    current_active = np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    historical_fixed = np.asarray(historical.point_data["IsFixed"], dtype=bool)
    fixture_historical_fixed = np.asarray(
        volume.point_data["HistoricalIsFixed"], dtype=bool
    )
    if not np.array_equal(historical_active, fixture_historical_active):
        raise ValueError("historical activation masks do not match")
    if not np.array_equal(historical_fixed, fixture_historical_fixed):
        raise ValueError("historical fixed masks do not match")

    rest = np.asarray(volume.points, dtype=np.float64)
    u = np.asarray(historical.point_data["Displacement"], dtype=np.float64)
    q_full = np.asarray(historical.cell_data["ActivationInv"], dtype=np.float64)
    target_raw = np.asarray(volume.point_data["Smile"], dtype=np.float64)
    if u.shape != rest.shape or not np.isfinite(u).all():
        raise ValueError("invalid historical displacement")
    if not np.allclose(u[historical_fixed], 0.0, rtol=0.0, atol=0.0):
        raise ValueError("historical fixed vertices moved")
    if not np.allclose(q_full[~historical_active], 0.0, rtol=0.0, atol=0.0):
        raise ValueError("inactive historical tetrahedra contain activation offsets")
    q = q_full[historical_active].copy()
    ainv = activation_matrix(q)
    tets = tetrahedra(volume)
    current = rest + u
    deformation_det = detf(rest, current, tets)
    activation_det = np.linalg.det(ainv)
    activation_eigenvalues = np.linalg.eigvalsh(ainv)

    top = np.asarray(volume.point_data["IsFace"], dtype=bool) & np.isfinite(
        target_raw
    ).all(axis=1)
    weights = surface_weights(volume, skin, top)
    target = np.zeros_like(rest)
    target[top] = target_raw[top]
    pred = u[top]
    target_top = target[top]
    target_rms = float(math.sqrt(np.sum(weights[:, None] * target_top**2)))
    fit_rms = float(math.sqrt(np.sum(weights[:, None] * (pred - target_top) ** 2)))
    motion_rms = float(math.sqrt(np.sum(weights[:, None] * pred**2)))
    projection = float(np.sum(weights[:, None] * pred * target_top) / target_rms**2)
    orthogonal = pred - projection * target_top

    added = np.asarray(volume.point_data["CutBoundaryAddedFixed"], dtype=bool)
    incident_added = np.any(added[tets], axis=1)
    coverage = region_coverage(volume, historical_active, current_active, q_full, tets)
    excluded_energy = sum(
        row["historical_activation_energy_share"]
        for row in coverage
        if row["current_screen"] == "excluded"
    )
    source_summary = json.loads(SOURCE_SUMMARY.read_text())
    common_metrics = {
        "weighting": "September rest-skin triangle lumped area weights normalized over finite IsFace target vertices",
        "target_vertices": int(np.count_nonzero(top)),
        "target_rms_mm": 1000.0 * target_rms,
        "fit_rms_mm": 1000.0 * fit_rms,
        "fit_rms_over_D": fit_rms / target_rms,
        "fit_objective": (fit_rms / target_rms) ** 2,
        "motion_rms_mm": 1000.0 * motion_rms,
        "target_projection_amplitude": projection,
        "target_projection_residual_over_D": float(
            math.sqrt(np.sum(weights[:, None] * orthogonal**2)) / target_rms
        ),
        "detF_min": float(deformation_det.min()),
        "detF_max": float(deformation_det.max()),
        "inverted_tetrahedra": int(np.count_nonzero(deformation_det <= 0.0)),
        "tetrahedra_below_detF_0_2": int(np.count_nonzero(deformation_det < 0.2)),
        "activation_eigen_min": float(activation_eigenvalues.min()),
        "activation_eigen_max": float(activation_eigenvalues.max()),
        "non_spd_active_tetrahedra": int(
            np.count_nonzero(activation_eigenvalues[:, 0] <= 0.0)
        ),
        "activation_det_min": float(activation_det.min()),
        "activation_det_max": float(activation_det.max()),
    }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "label": "historical saved no-skin endpoint",
        "status": "historical_saved_endpoint_no_new_solve",
        "interpretation": (
            "Exact geometry/control readback of the June best saved endpoint; metrics "
            "are recomputed on CPU. No forward solve, control replay, fresh-rest check, "
            "or optimization was performed."
        ),
        "recorded_june_metrics": {
            "best_step": int(source_summary["best/step"]),
            "requested_optimizer_steps": int(source_summary["inverse/max_steps"]),
            "evaluations": int(source_summary["inverse/evaluations"]),
            "best_fit_rms_mm": float(source_summary["best/error_rms_mm"]),
            "final_fit_rms_mm": float(source_summary["final/error_rms_mm"]),
            "forward_failures": int(source_summary["inverse/forward_fail_count"]),
            "adjoint_failures": int(source_summary["inverse/adjoint_fail_count"]),
            "inverse_converged": bool(source_summary["inverse/converged"]),
            "stop_reason": str(source_summary["inverse/stop_reason"]),
            "last_forward_success": bool(source_summary["last/forward/success"]),
            "last_adjoint_success": bool(source_summary["last/adjoint/success"]),
            "outer_optimizer": "torch.optim.Adam",
            "inverse_learning_rate": float(source_summary["inverse/lr_initial"]),
            "adam_eps": float(source_summary["optimizer/adam_eps"]),
            "loss_scale": float(source_summary["loss/scale"]),
            "loss_type": str(source_summary["loss/type"]),
        },
        "optimizer_comparison": {
            "june": {
                "algorithm": "torch.optim.Adam",
                "learning_rate": float(source_summary["inverse/lr_initial"]),
                "epsilon": float(source_summary["optimizer/adam_eps"]),
                "loss_scale": float(source_summary["loss/scale"]),
                "loss_type": str(source_summary["loss/type"]),
            },
            "current_diagnosis_runner": {
                "algorithm": "volume-preconditioned L-BFGS direction with Armijo backtracking",
                "initial_inverse_mass_scale": 0.01,
                "history_pairs": 10,
                "maximum_line_search_trials": 12,
                "per_tetrahedron_step_radius": 0.1,
                "objective_scale": "fit divided by target area-weighted RMS squared",
            },
            "interpretation": "Optimizer and objective normalization differ; a stalled current run does not isolate active-mask capacity.",
        },
        "common_area_weighted_metrics": common_metrics,
        "activation_domain": {
            "parameterization": "one unconstrained packed symmetric A_inv-I offset per active tetrahedron",
            "historical_active_tetrahedra": int(historical_active.sum()),
            "historical_scalar_dofs": int(6 * historical_active.sum()),
            "historical_regions": len(coverage),
            "current_screen_active_tetrahedra": int(current_active.sum()),
            "current_raw6_scalar_dofs": int(6 * current_active.sum()),
            "current_screen_regions": len(
                np.unique(np.asarray(volume.cell_data["MuscleId"])[current_active])
            ),
            "excluded_historical_active_tetrahedra": int(
                np.count_nonzero(historical_active & ~current_active)
            ),
            "excluded_cell_fraction": float(
                np.count_nonzero(historical_active & ~current_active)
                / historical_active.sum()
            ),
            "excluded_saved_activation_energy_share": excluded_energy,
        },
        "fixed_boundary_comparison": {
            "historical_fixed_vertices": int(historical_fixed.sum()),
            "current_fixed_vertices": int(
                np.asarray(volume.point_data["IsFixed"], dtype=bool).sum()
            ),
            "added_fixed_vertices": int(added.sum()),
            "added_vertices_overlapping_IsFace": int(
                np.count_nonzero(added & np.asarray(volume.point_data["IsFace"], bool))
            ),
            "added_vertices_overlapping_IsLip": int(
                np.count_nonzero(added & np.asarray(volume.point_data["IsLip"], bool))
            ),
            "june_displacement_rms_on_added_vertices_mm": float(
                1000.0 * np.linalg.norm(u[added]) / math.sqrt(added.sum())
            ),
            "june_displacement_max_on_added_vertices_mm": float(
                1000.0 * np.linalg.norm(u[added], axis=1).max()
            ),
            "all_tetrahedra_incident_to_added_vertices": int(incident_added.sum()),
            "historical_active_tetrahedra_incident_to_added_vertices": int(
                np.count_nonzero(incident_added & historical_active)
            ),
            "current_active_tetrahedra_incident_to_added_vertices": int(
                np.count_nonzero(incident_added & current_active)
            ),
        },
        "muscle_region_coverage": coverage,
        "source_hashes": {
            "historical_endpoint": digest(SOURCE),
            "historical_target": digest(SOURCE_TARGET),
            "historical_summary": digest(SOURCE_SUMMARY),
            "historical_trace": digest(SOURCE_TRACE),
            "current_fixture_volume": digest(FIXTURE),
            "current_fixture_skin": digest(SKIN),
            "current_diagnosis_runner": digest(CURRENT_RUNNER),
            "current_raw6_no_skin_config": digest(CURRENT_RAW6_CONFIG),
            "historical_runner_sources": {
                path.name: digest(path)
                for path in sorted((JUNE / "src").glob("_human_face*.py"))
            }
            | {
                "20-inverse-human-face.py": digest(
                    JUNE / "src/20-inverse-human-face.py"
                )
            },
        },
        "limitations": [
            "The exported displacement and controls were not solved under the September material, skin, fixed-boundary, active-mask, or solver settings.",
            "The recorded June endpoint is the best saved optimizer state, not a certified stationary point.",
            "The saved endpoint contains inverted tetrahedra and non-SPD active tensors; these are retained exactly and reported as diagnostics.",
            "Region activation magnitude ranks optimizer usage and cannot establish causal importance without matched counterfactual solves.",
        ],
    }

    OUTPUT.mkdir(parents=True)
    np.savez_compressed(
        OUTPUT / "final.npz",
        q=q,
        Ainv=ainv,
        u=u,
        step=np.asarray(source_summary["best/step"], dtype=np.int64),
        active_ids=np.flatnonzero(historical_active),
        rest_points=rest,
        fixed_mask=historical_fixed,
        activation_mask=historical_active,
    )

    exported = volume.copy(deep=True)
    for key in list(exported.point_data):
        if key not in {
            "GlobalPointId",
            "IsFace",
            "IsFixed",
            "IsLip",
            "FixedMask",
            "FixedValue",
            "HistoricalIsFixed",
            "ArtificialCutIncident",
            "CutBoundaryAddedFixed",
        }:
            del exported.point_data[key]
    for key in list(exported.cell_data):
        if key not in {
            "MuscleId",
            "MuscleFraction",
            "FatFraction",
            "AponeurosisFraction",
            "Volume",
        }:
            del exported.cell_data[key]
    exported.points = current
    exported.point_data["IsFixed"] = historical_fixed
    exported.point_data["FixedMask"] = np.repeat(historical_fixed[:, None], 3, axis=1)
    exported.point_data["RestPosition"] = rest
    exported.point_data["Displacement"] = u
    exported.point_data["TargetDisplacement"] = target
    exported.cell_data["ActivationMask"] = historical_active
    exported.cell_data["ActivationInv"] = q_full
    full_ainv = np.broadcast_to(np.eye(3), (volume.n_cells, 3, 3)).copy()
    full_ainv[historical_active] = ainv
    exported.cell_data["ActivationInverseMatrix"] = full_ainv.reshape(-1, 9)
    exported.cell_data["DetAinv"] = np.linalg.det(full_ainv)
    exported.cell_data["DetF"] = deformation_det
    exported.save(OUTPUT / "final.vtu")

    summary["export_hashes"] = {
        "final_npz": digest(OUTPUT / "final.npz"),
        "final_vtu": digest(OUTPUT / "final.vtu"),
        "export_script": digest(Path(__file__)),
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    (ROOT / "docs/11-historical-baseline.md").write_text(markdown(summary))


if __name__ == "__main__":
    main()
