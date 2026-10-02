# ruff: noqa: EM101, EM102, TRY003
"""Localize tetrahedron inversions in the prescribed-skin forward run."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from joint_common import ProfileJoint, archive_sources, sha256, write_json
from scipy.spatial import cKDTree

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    rebased_volume: Path
    parent_manifest: Path
    geometry: Path
    candidate_001: Path
    candidate_002: Path
    checkpoint_dir: Path
    guarded_checkpoint: Path
    output_dir: Path = cherries.output("simple-skin-forward-inversion-audit")


def deformation_matrices(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    return np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))


def distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if not len(values):
        return {
            "count": 0,
            "minimum": None,
            "q10": None,
            "median": None,
            "q90": None,
            "maximum": None,
        }
    q = np.quantile(values, (0.0, 0.1, 0.5, 0.9, 1.0))
    return {
        "count": len(values),
        "minimum": float(q[0]),
        "q10": float(q[1]),
        "median": float(q[2]),
        "q90": float(q[3]),
        "maximum": float(q[4]),
    }


def source_surfaces(path: Path) -> dict[str, pv.PolyData]:
    with np.load(path, allow_pickle=False) as archive:
        result = {}
        for bone in ("cranium", "mandible"):
            points = np.asarray(archive[f"{bone}_points_m"])
            faces = np.asarray(archive[f"{bone}_faces"], dtype=np.int64)
            result[bone] = pv.PolyData(
                points, np.column_stack((np.full(len(faces), 3), faces))
            )
    return result


def closest_vertex_distances_mm(
    positions: np.ndarray,
    tets: np.ndarray,
    cell_ids: np.ndarray,
    surfaces: dict[str, pv.PolyData],
) -> dict[str, np.ndarray]:
    if not len(cell_ids):
        return {name: np.empty(0, dtype=np.float64) for name in surfaces}
    query = positions[tets[cell_ids]].reshape(-1, 3)
    result = {}
    for name, surface in surfaces.items():
        _, closest = surface.find_closest_cell(query, return_closest_point=True)
        result[name] = (
            np.linalg.norm(query - closest, axis=1).reshape(-1, 4).min(axis=1) * 1e3
        )
    return result


def main(cfg: Config) -> None:  # noqa: C901, PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    rebased = pv.read(cfg.rebased_volume)
    parent_manifest = json.loads(cfg.parent_manifest.read_text())
    parent_path = Path(parent_manifest["fixture"]["volume_path"])
    parent = pv.read(parent_path)
    if rebased.n_points != parent.n_points or rebased.n_cells != parent.n_cells:
        raise ValueError("parent and rebased meshes differ in size")
    tets = np.asarray(rebased.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    parent_tets = np.asarray(parent.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    if not np.array_equal(tets, parent_tets):
        raise ValueError("parent and rebased tetrahedron connectivity differs")
    reference = np.asarray(rebased.points, dtype=np.float64)
    parent_reference = np.asarray(parent.points, dtype=np.float64)
    dm = deformation_matrices(reference, tets)
    dm_parent = deformation_matrices(parent_reference, tets)
    dm_inv = np.linalg.inv(dm)
    parent_inv = np.linalg.inv(dm_parent)
    rest_volume = np.linalg.det(dm) / 6.0
    parent_volume = np.linalg.det(dm_parent) / 6.0
    if np.any(rest_volume <= 0) or np.any(parent_volume <= 0):
        raise ValueError("reference mesh contains a nonpositive tetrahedron")
    repair_jacobian = np.linalg.det(dm @ parent_inv)
    repair_node_displacement = np.linalg.norm(reference - parent_reference, axis=1)
    repair_moved_cell = np.any(repair_node_displacement[tets] > 1e-9, axis=1)
    repair_jacobian_cell = np.abs(repair_jacobian - 1.0) > 1e-6
    with np.load(cfg.candidate_001, allow_pickle=False) as archive:
        candidate_001 = np.asarray(archive["initial_displacement_m"], dtype=np.float64)
    with np.load(cfg.candidate_002, allow_pickle=False) as archive:
        candidate_002 = np.asarray(archive["initial_displacement_m"], dtype=np.float64)
    if candidate_001.shape != reference.shape or candidate_002.shape != reference.shape:
        raise ValueError("repair candidates differ from FEM point layout")
    repair_delta = np.linalg.norm(candidate_002 - candidate_001, axis=1)
    repair_delta_nodes = np.flatnonzero(repair_delta > 1e-12)
    repair_delta_cell = np.any(np.isin(tets, repair_delta_nodes), axis=1)
    candidate_001_jacobian = np.linalg.det(
        deformation_matrices(parent_reference + candidate_001, tets) @ parent_inv
    )
    candidate_001_inverted = candidate_001_jacobian <= 0
    candidate_001_out_of_range = (candidate_001_jacobian < 0.25) | (
        candidate_001_jacobian > 2.0
    )
    repair_tree = cKDTree(reference[repair_delta_nodes])
    surfaces = source_surfaces(cfg.geometry)

    checkpoints = sorted(cfg.checkpoint_dir.glob("checkpoint-step-*.npz"))
    if not checkpoints:
        raise ValueError("no checkpoint-step-*.npz files found")
    checkpoint_rows: list[dict[str, Any]] = []
    failed_records: list[dict[str, Any]] = []
    failed_ids_by_step: dict[int, np.ndarray] = {}
    failed_j_by_step: dict[int, np.ndarray] = {}
    for path in checkpoints:
        step = int(path.stem.rsplit("-", 1)[1])
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"displacement_m"}:
                raise ValueError(f"unexpected checkpoint arrays in {path}")
            displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
        if displacement.shape != reference.shape or not np.isfinite(displacement).all():
            raise ValueError(f"invalid displacement in {path}")
        current = reference + displacement
        jacobian = np.linalg.det(deformation_matrices(current, tets) @ dm_inv)
        failed = np.flatnonzero(jacobian <= 0)
        failed_ids_by_step[step] = failed
        failed_j_by_step[step] = jacobian[failed]
        proximity = closest_vertex_distances_mm(current, tets, failed, surfaces)
        nearest_distance = (
            np.minimum(proximity["cranium"], proximity["mandible"])
            if len(failed)
            else np.empty(0)
        )
        nearest_bone = (
            np.where(
                proximity["cranium"] <= proximity["mandible"],
                "cranium",
                "mandible",
            )
            if len(failed)
            else np.empty(0, dtype="U8")
        )
        centroids = (
            current[tets[failed]].mean(axis=1) if len(failed) else np.empty((0, 3))
        )
        distance_to_repair_node_mm = (
            repair_tree.query(centroids)[0] * 1e3 if len(failed) else np.empty(0)
        )
        checkpoint_rows.append(
            {
                "step": step,
                "path": str(path.resolve()),
                "sha256": sha256(path),
                "inverted_tetrahedra": len(failed),
                "jacobian": {
                    **distribution(jacobian),
                    "q0001": float(np.quantile(jacobian, 1e-6)),
                    "q001": float(np.quantile(jacobian, 1e-5)),
                    "q01": float(np.quantile(jacobian, 1e-4)),
                    "q1": float(np.quantile(jacobian, 1e-3)),
                },
                "inverted_in_repair_moved_cell": int(
                    np.count_nonzero(repair_moved_cell[failed])
                ),
                "inverted_in_repair_jacobian_cell": int(
                    np.count_nonzero(repair_jacobian_cell[failed])
                ),
                "inverted_incident_to_candidate002_local_repair_node": int(
                    np.count_nonzero(repair_delta_cell[failed])
                ),
                "inverted_overlapping_candidate001_inverted": int(
                    np.count_nonzero(candidate_001_inverted[failed])
                ),
                "inverted_overlapping_candidate001_out_of_range": int(
                    np.count_nonzero(candidate_001_out_of_range[failed])
                ),
                "distance_to_candidate002_local_repair_node_mm": distribution(
                    distance_to_repair_node_mm
                ),
                "within_0p5_mm_of_candidate002_local_repair_node": int(
                    np.count_nonzero(distance_to_repair_node_mm <= 0.5)
                ),
                "nearest_bone": {
                    "cranium": int(np.count_nonzero(nearest_bone == "cranium")),
                    "mandible": int(np.count_nonzero(nearest_bone == "mandible")),
                    "within_0p1_mm": int(np.count_nonzero(nearest_distance <= 0.1)),
                    "within_1_mm": int(np.count_nonzero(nearest_distance <= 1.0)),
                    "minimum_vertex_distance_mm": distribution(nearest_distance),
                },
                "inverted_centroid_bbox_mm": (
                    {
                        "minimum": (centroids.min(axis=0) * 1e3).tolist(),
                        "maximum": (centroids.max(axis=0) * 1e3).tolist(),
                    }
                    if len(failed)
                    else None
                ),
            }
        )
        for local, cell_id in enumerate(failed):
            failed_records.append(
                {
                    "step": step,
                    "cell_id": int(cell_id),
                    "jacobian": float(jacobian[cell_id]),
                    "fat_fraction": float(rebased.cell_data["FatFraction"][cell_id]),
                    "aponeurosis_fraction": float(
                        rebased.cell_data["AponeurosisFraction"][cell_id]
                    ),
                    "muscle_fraction": float(
                        rebased.cell_data["MuscleFraction"][cell_id]
                    ),
                    "repair_jacobian": float(repair_jacobian[cell_id]),
                    "incident_to_candidate002_local_repair_node": bool(
                        repair_delta_cell[cell_id]
                    ),
                    "candidate001_inverted": bool(candidate_001_inverted[cell_id]),
                    "candidate001_out_of_range": bool(
                        candidate_001_out_of_range[cell_id]
                    ),
                    "distance_to_candidate002_local_repair_node_mm": float(
                        distance_to_repair_node_mm[local]
                    ),
                    "rest_volume_m3": float(rest_volume[cell_id]),
                    "nearest_bone": str(nearest_bone[local]),
                    "nearest_bone_vertex_distance_mm": float(nearest_distance[local]),
                    "centroid_x_mm": float(centroids[local, 0] * 1e3),
                    "centroid_y_mm": float(centroids[local, 1] * 1e3),
                    "centroid_z_mm": float(centroids[local, 2] * 1e3),
                }
            )

    union = np.unique(
        np.concatenate([ids for ids in failed_ids_by_step.values() if len(ids)])
    )
    dominant = np.column_stack(
        (
            np.asarray(rebased.cell_data["FatFraction"])[union],
            np.asarray(rebased.cell_data["AponeurosisFraction"])[union],
            np.asarray(rebased.cell_data["MuscleFraction"])[union],
        )
    )
    dominant_names = np.asarray(("fat", "aponeurosis", "muscle"))[dominant.argmax(1)]
    with np.load(cfg.guarded_checkpoint, allow_pickle=False) as archive:
        guarded_displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    guarded_current = reference + guarded_displacement
    guarded_jacobian = np.linalg.det(
        deformation_matrices(guarded_current, tets) @ dm_inv
    )
    limiting_cell = int(np.argmin(guarded_jacobian))
    limiting_ids = np.asarray([limiting_cell], dtype=np.int64)
    limiting_proximity = closest_vertex_distances_mm(
        guarded_current, tets, limiting_ids, surfaces
    )
    limiting_centroid = guarded_current[tets[limiting_cell]].mean(axis=0)
    limiting_history = {}
    for path in checkpoints:
        step = int(path.stem.rsplit("-", 1)[1])
        with np.load(path, allow_pickle=False) as archive:
            displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
        current_dm = deformation_matrices(reference + displacement, tets)
        limiting_history[str(step)] = float(
            np.linalg.det(current_dm[limiting_cell] @ dm_inv[limiting_cell])
        )
    failed_csv = cfg.output_dir / "failed-tetrahedra.csv"
    with failed_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failed_records[0]))
        writer.writeheader()
        writer.writerows(failed_records)
    arrays: dict[str, np.ndarray] = {"union_failed_cell_ids": union}
    for step, failed_ids in failed_ids_by_step.items():
        arrays[f"failed_cell_ids_step_{step:05d}"] = failed_ids
        arrays[f"failed_jacobian_step_{step:05d}"] = failed_j_by_step[step]
    np.savez_compressed(cfg.output_dir / "failed-tetrahedra.npz", **arrays)

    summary = {
        "schema": "joint-simple-forward-inversion-audit-v1",
        "success": True,
        "scope": "numerical geometry diagnosis only; mechanics unchanged",
        "inputs": {
            "rebased_volume": str(cfg.rebased_volume.resolve()),
            "rebased_volume_sha256": sha256(cfg.rebased_volume),
            "parent_volume": str(parent_path.resolve()),
            "parent_volume_sha256": sha256(parent_path),
            "geometry": str(cfg.geometry.resolve()),
            "geometry_sha256": sha256(cfg.geometry),
            "candidate_001": str(cfg.candidate_001.resolve()),
            "candidate_001_sha256": sha256(cfg.candidate_001),
            "candidate_002": str(cfg.candidate_002.resolve()),
            "candidate_002_sha256": sha256(cfg.candidate_002),
            "guarded_checkpoint": str(cfg.guarded_checkpoint.resolve()),
            "guarded_checkpoint_sha256": sha256(cfg.guarded_checkpoint),
        },
        "reference_comparison": {
            "parent_positive_tetrahedra": int(np.count_nonzero(parent_volume > 0)),
            "rebased_positive_tetrahedra": int(np.count_nonzero(rest_volume > 0)),
            "tetrahedra": len(tets),
            "parent_volume_m3": distribution(parent_volume),
            "rebased_volume_m3": distribution(rest_volume),
            "repair_jacobian": distribution(repair_jacobian),
            "repair_jacobian_minimum_expected": 0.2501,
            "repair_jacobian_maximum_expected": 1.9999,
            "repair_moved_cell_definition": "any vertex moved by more than 1e-9 m from parent reference",
            "repair_moved_cells": int(np.count_nonzero(repair_moved_cell)),
            "repair_jacobian_cell_definition": "abs(parent-to-rebased J - 1) > 1e-6",
            "repair_jacobian_cells": int(np.count_nonzero(repair_jacobian_cell)),
            "candidate001_inverted_tetrahedra": int(
                np.count_nonzero(candidate_001_inverted)
            ),
            "candidate001_out_of_range_tetrahedra": int(
                np.count_nonzero(candidate_001_out_of_range)
            ),
            "candidate002_local_repair_node_definition": "candidate002 displacement differs from candidate001 by more than 1e-12 m",
            "candidate002_local_repair_nodes": len(repair_delta_nodes),
            "candidate002_local_repair_incident_cells": int(
                np.count_nonzero(repair_delta_cell)
            ),
            "candidate002_local_repair_displacement_m": distribution(
                repair_delta[repair_delta_nodes]
            ),
        },
        "checkpoints": checkpoint_rows,
        "failure_union": {
            "tetrahedra": len(union),
            "dominant_tissue_counts": {
                name: int(np.count_nonzero(dominant_names == name))
                for name in ("fat", "aponeurosis", "muscle")
            },
            "pure_fat_tetrahedra": int(
                np.count_nonzero(
                    np.asarray(rebased.cell_data["FatFraction"])[union] == 1.0
                )
            ),
            "in_repair_moved_cells": int(np.count_nonzero(repair_moved_cell[union])),
            "in_repair_jacobian_cells": int(
                np.count_nonzero(repair_jacobian_cell[union])
            ),
            "incident_to_candidate002_local_repair_nodes": int(
                np.count_nonzero(repair_delta_cell[union])
            ),
            "overlapping_candidate001_inverted": int(
                np.count_nonzero(candidate_001_inverted[union])
            ),
            "overlapping_candidate001_out_of_range": int(
                np.count_nonzero(candidate_001_out_of_range[union])
            ),
            "repair_jacobian": distribution(repair_jacobian[union]),
            "rebased_rest_volume_m3": distribution(rest_volume[union]),
        },
        "guarded_run_terminal": {
            "inverted_tetrahedra": int(np.count_nonzero(guarded_jacobian <= 0)),
            "jacobian": {
                **distribution(guarded_jacobian),
                "q0001": float(np.quantile(guarded_jacobian, 1e-6)),
                "q001": float(np.quantile(guarded_jacobian, 1e-5)),
                "q01": float(np.quantile(guarded_jacobian, 1e-4)),
                "q1": float(np.quantile(guarded_jacobian, 1e-3)),
            },
            "counts_at_or_below_j": {
                str(threshold): int(np.count_nonzero(guarded_jacobian <= threshold))
                for threshold in (1e-5, 1e-4, 1e-3, 0.01, 0.1, 0.25, 0.5, 0.9)
            },
            "limiting_cell": {
                "cell_id": limiting_cell,
                "jacobian": float(guarded_jacobian[limiting_cell]),
                "tetrahedron_point_ids": tets[limiting_cell].tolist(),
                "fat_fraction": float(rebased.cell_data["FatFraction"][limiting_cell]),
                "aponeurosis_fraction": float(
                    rebased.cell_data["AponeurosisFraction"][limiting_cell]
                ),
                "muscle_fraction": float(
                    rebased.cell_data["MuscleFraction"][limiting_cell]
                ),
                "rebased_rest_volume_m3": float(rest_volume[limiting_cell]),
                "rebased_edge_matrix_condition_number": float(
                    np.linalg.cond(dm[limiting_cell])
                ),
                "candidate001_jacobian": float(candidate_001_jacobian[limiting_cell]),
                "parent_to_rebased_jacobian": float(repair_jacobian[limiting_cell]),
                "incident_to_candidate002_local_repair_node": bool(
                    repair_delta_cell[limiting_cell]
                ),
                "maximum_candidate002_local_repair_motion_on_tet_mm": float(
                    repair_delta[tets[limiting_cell]].max() * 1e3
                ),
                "in_unguarded_failure_union": bool(limiting_cell in set(union)),
                "unguarded_checkpoint_jacobian_history": limiting_history,
                "deformed_centroid_mm": (limiting_centroid * 1e3).tolist(),
                "minimum_vertex_distance_to_source_cranium_mm": float(
                    limiting_proximity["cranium"][0]
                ),
                "minimum_vertex_distance_to_source_mandible_mm": float(
                    limiting_proximity["mandible"][0]
                ),
            },
        },
        "artifacts": {
            "failed_csv": str(failed_csv.resolve()),
            "failed_csv_sha256": sha256(failed_csv),
            "failed_npz": str((cfg.output_dir / "failed-tetrahedra.npz").resolve()),
            "failed_npz_sha256": sha256(cfg.output_dir / "failed-tetrahedra.npz"),
        },
        "interpretation": [
            "Checkpoint 0 is exactly valid in the rebased reference, so these are solver-path inversions rather than inherited inverted elements.",
            "The repair preserved positive volume but many eventual failures lie in cells whose parent-to-rebased Jacobian or vertices changed materially.",
            "Bone proximity is measured as the minimum exact source-triangle distance among each tetrahedron's four current vertices; it localizes but does not prove contact caused inversion.",
        ],
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
