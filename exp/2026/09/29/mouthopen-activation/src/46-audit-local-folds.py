"""Audit rejected jaw carries with and without inverted-cell boundary faces."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import ipctk
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    fixture: Path = Path("30-pruned-fixture")
    first_source: Path = Path("35-forward-pruned-002")
    run: Path = Path("45-forward-allow-inversions")
    pose_source: Path = Path("10-mandible/prepared.npz")
    output: Path = Path("46-local-fold-audit")


def record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "sha256": digest, "bytes": path.stat().st_size}


def rigid_boundary(
    points: np.ndarray,
    current_u: np.ndarray,
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    pivot: np.ndarray,
    harmonic_weight: np.ndarray,
    fixed: np.ndarray,
    jaw: np.ndarray,
) -> np.ndarray:
    old_r = Rotation.from_rotvec(current_pose[:3]).as_matrix()
    new_r = Rotation.from_rotvec(target_pose[:3]).as_matrix()
    x = points + current_u
    carried = (x - pivot - current_pose[3:]) @ old_r @ new_r.T + pivot + target_pose[3:]
    candidate = current_u + harmonic_weight[:, None] * (carried - x)
    boundary = np.zeros_like(points)
    boundary[jaw] = (
        (points[jaw] - pivot) @ new_r.T + pivot + target_pose[3:] - points[jaw]
    )
    candidate[fixed] = boundary[fixed]
    return candidate


def detf(
    points: np.ndarray, tets: np.ndarray, dm_inv: np.ndarray, u: np.ndarray
) -> np.ndarray:
    x = points + u
    ds = np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(ds @ dm_inv)


def make_collision_mesh(rest: np.ndarray, faces: np.ndarray) -> ipctk.CollisionMesh:
    faces = np.asfortranarray(faces, dtype=np.int32)
    edges = np.asfortranarray(ipctk.edges(faces), dtype=np.int32)
    return ipctk.CollisionMesh(
        rest_positions=np.asfortranarray(rest, dtype=np.float64),
        edges=edges,
        faces=faces,
    )


def intersection(mesh: ipctk.CollisionMesh, vertices: np.ndarray) -> bool:
    return bool(
        ipctk.has_intersections(
            mesh, np.asfortranarray(vertices, dtype=np.float64), ipctk.LBVH()
        )
    )


def ccd_fraction(
    mesh: ipctk.CollisionMesh, vertices0: np.ndarray, vertices1: np.ndarray
) -> float:
    vertices0 = np.asfortranarray(vertices0, dtype=np.float64)
    vertices1 = np.asfortranarray(vertices1, dtype=np.float64)
    candidates = ipctk.Candidates()
    broad_phase = ipctk.LBVH()
    candidates.build(
        mesh=mesh,
        vertices_t0=vertices0,
        vertices_t1=vertices1,
        inflation_radius=0.0,
        broad_phase=broad_phase,
    )
    result = candidates.compute_collision_free_stepsize(
        mesh=mesh,
        vertices_t0=vertices0,
        vertices_t1=vertices1,
        min_distance=0.0,
        narrow_phase_ccd=ipctk.TightInclusionCCD(),
    )
    return float(result)


def case_audit(
    name: str,
    row: dict[str, Any],
    state_path: Path,
    points: np.ndarray,
    tets: np.ndarray,
    dm_inv: np.ndarray,
    surface_ids: np.ndarray,
    surface_faces: np.ndarray,
    surface_rest: np.ndarray,
    topology: dict[str, int],
    full_mesh: ipctk.CollisionMesh,
    fixed: np.ndarray,
    jaw: np.ndarray,
    pivot: np.ndarray,
    pose: np.ndarray,
    harmonic_weight: np.ndarray,
) -> dict[str, Any]:
    with np.load(state_path, allow_pickle=False) as data:
        current = np.asarray(data["displacement"], dtype=np.float64).copy()
        current_pose = np.asarray(data["pose"], dtype=np.float64).copy()
        current_fraction = float(data["fraction"])
    assert abs(current_fraction - float(row["from_fraction"])) < 1e-13
    target_fraction = float(row["target_fraction"])
    target_pose = target_fraction * pose
    candidate = rigid_boundary(
        points,
        current,
        current_pose,
        target_pose,
        pivot,
        harmonic_weight,
        fixed,
        jaw,
    )
    current_j = detf(points, tets, dm_inv, current)
    trial_j = detf(points, tets, dm_inv, candidate)
    current_bad = np.flatnonzero(current_j <= 0)
    trial_bad = np.flatnonzero(trial_j <= 0)
    current_bad_points = np.zeros(len(points), dtype=bool)
    trial_bad_points = np.zeros(len(points), dtype=bool)
    current_bad_points[np.unique(tets[current_bad])] = True
    trial_bad_points[np.unique(tets[trial_bad])] = True
    union_bad_points = current_bad_points | trial_bad_points

    current_surface = surface_rest + current[surface_ids]
    trial_surface = surface_rest + candidate[surface_ids]
    current_intersects = intersection(full_mesh, current_surface)
    trial_intersects = intersection(full_mesh, trial_surface)
    full_ccd = ccd_fraction(full_mesh, current_surface, trial_surface)

    local_bad = union_bad_points[surface_ids][surface_faces].any(axis=1)
    kept_faces = surface_faces[~local_bad]
    kept_mesh = make_collision_mesh(surface_rest, kept_faces)
    current_kept_intersects = intersection(kept_mesh, current_surface)
    trial_kept_intersects = intersection(kept_mesh, trial_surface)
    filtered_ccd = ccd_fraction(kept_mesh, current_surface, trial_surface)

    assert (
        row.get("carry_collision_fraction") is None
        or abs(float(row["carry_collision_fraction"]) - full_ccd) <= 1e-12
    )

    alpha = full_ccd
    near_ccd_alpha = max(0.0, alpha - max(1e-8, 1e-6 * alpha))
    near_surface = (
        surface_rest + (current + near_ccd_alpha * (candidate - current))[surface_ids]
    )
    return {
        "name": name,
        "source_state": record(state_path),
        "attempt_index": int(row["index"]),
        "from_fraction": current_fraction,
        "target_fraction": target_fraction,
        "pose_step_fraction": float(row["step"]),
        "recorded_ccd_fraction": row.get("carry_collision_fraction"),
        "recomputed_full_boundary_ccd_fraction": full_ccd,
        "recomputed_filtered_boundary_ccd_fraction": filtered_ccd,
        "current_deformation": {
            "min_J": float(current_j.min()),
            "inverted_cells": len(current_bad),
            "inverted_cell_ids": current_bad.tolist(),
        },
        "trial_deformation": {
            "min_J": float(trial_j.min()),
            "inverted_cells": len(trial_bad),
            "inverted_cell_ids": trial_bad.tolist(),
        },
        "surface_intersections": {
            "current_full_boundary": current_intersects,
            "trial_full_boundary": trial_intersects,
            "trial_at_last_safe_ccd_fraction": intersection(full_mesh, near_surface),
            "current_after_union_filter": current_kept_intersects,
            "trial_after_union_filter": trial_kept_intersects,
            "filter_removes_ccd_restriction": filtered_ccd >= 1.0 and full_ccd < 1.0,
        },
        "fold_vertex_face_filter": {
            "rule": "exclude every extracted boundary triangle incident to any vertex of a tet inverted in current OR trial state",
            "current_inverted_tet_count": len(current_bad),
            "trial_inverted_tet_count": len(trial_bad),
            "union_inverted_vertex_count": int(union_bad_points.sum()),
            "boundary_face_count_total": len(surface_faces),
            "boundary_faces_excluded": int(local_bad.sum()),
            "boundary_faces_retained": int((~local_bad).sum()),
            "excluded_fraction": float(local_bad.mean()),
        },
        "boundary_shape": {
            "vertices": len(surface_rest),
            "faces": len(surface_faces),
            "boundary_edges": topology["boundary_edges"],
            "nonmanifold_edges": topology["nonmanifold_edges"],
        },
        "interpretation_limit": "fold-incident boundary faces are masked only for this diagnostic; the mesh, constitutive model, and physical contact policy are unchanged",
    }


def surface_topology(faces: np.ndarray) -> dict[str, int]:
    edges = np.sort(
        np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(counts > 2)),
    }


def main(cfg: Config) -> None:
    fixture = cherries.input(GROUP / "data" / cfg.fixture)
    first_source = cherries.input(GROUP / "data" / cfg.first_source)
    run = cherries.input(GROUP / "data" / cfg.run)
    pose_source = cherries.input(GROUP / "data" / cfg.pose_source)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)

    summary_path = run / "summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] not in {"running", "initializing"}
    attempts = summary["attempts"]
    rejected = [row for row in attempts if row["status"] == "rejected_carry_collision"]
    first_rejected = rejected[0]
    last_rejected = rejected[-1]
    volume = pv.read(fixture / "volume.vtu")
    points = np.asarray(volume.points, dtype=np.float64)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    names = [
        str(value) for value in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    jaw = fixed & (np.asarray(volume.point_data["GroupId"]) == names.index("Mandible"))
    assert not np.any(fixed[tets].all(axis=1))
    dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    dm_inv = np.linalg.inv(dm)
    surface = volume.extract_surface(algorithm=None, pass_pointid=True)
    surface_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    quads = np.asarray(surface.faces).reshape(-1, 4)
    assert np.all(quads[:, 0] == 3)
    surface_faces = quads[:, 1:].astype(np.int32)
    surface_rest = np.asfortranarray(np.asarray(surface.points), dtype=np.float64)
    full_mesh = make_collision_mesh(surface_rest, surface_faces)
    topology = surface_topology(surface_faces)
    first_state_path = first_source / "final.npz"
    last_state_path = run / "final.npz"
    with np.load(first_source / "harmonic-weight.npz", allow_pickle=False) as data:
        harmonic_weight = np.asarray(data["weight"], dtype=np.float64).copy()
    with np.load(pose_source, allow_pickle=False) as data:
        pose = np.asarray(data["pose"], dtype=np.float64).copy()
        pivot = np.asarray(data["pivot"], dtype=np.float64).copy()
    assert harmonic_weight.shape == (len(points),)
    results = [
        case_audit(
            "first rejected carry",
            first_rejected,
            first_state_path,
            points,
            tets,
            dm_inv,
            surface_ids,
            surface_faces,
            surface_rest,
            topology,
            full_mesh,
            fixed,
            jaw,
            pivot,
            pose,
            harmonic_weight,
        ),
        case_audit(
            "last rejected carry",
            last_rejected,
            last_state_path,
            points,
            tets,
            dm_inv,
            surface_ids,
            surface_faces,
            surface_rest,
            topology,
            full_mesh,
            fixed,
            jaw,
            pivot,
            pose,
            harmonic_weight,
        ),
    ]
    result = {
        "schema": "mouthopen-local-fold-boundary-audit-v1",
        "scope": "CPU geometry diagnostics only; no forward solve, no contact forces, no mesh edits",
        "run_status": summary["status"],
        "completed_pose_fraction": summary["completed_pose_fraction"],
        "source_receipts": {
            "volume": record(fixture / "volume.vtu"),
            "run_summary": record(summary_path),
            "run_final": record(run / "final.npz"),
            "first_final": record(first_source / "final.npz"),
            "harmonic_weight": record(first_source / "harmonic-weight.npz"),
            "pose": record(pose_source),
            "script": record(Path(__file__)),
        },
        "results": results,
        "conclusion": "A reduced, fold-incident face mask is a diagnostic sensitivity check only; any CCD relaxation remains exploratory and does not make the deformed volume physically valid.",
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "source.py").write_text(Path(__file__).read_text())
    print(json.dumps({"output": str(output), "results": results}, indent=2))


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
