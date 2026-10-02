"""Audit self-collision of boundary faces with only prescribed vertices."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from itertools import pairwise
from pathlib import Path

import ipctk
import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402

LOG = logging.getLogger(__name__)
FRACTIONS = (0.0, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class Config(cherries.BaseConfig):
    output: Path = Path("18-prescribed-boundary")
    fixture: Path = (
        ROOT / "exp/2026/09/29/mouthopen-activation/data/30-pruned-fixture/volume.vtu"
    )
    pose_source: Path = (
        ROOT / "exp/2026/09/29/mouthopen-activation/data/10-mandible/prepared.npz"
    )
    min_distance_m: float = 1e-8


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def chord_ccd(
    mesh: ipctk.CollisionMesh,
    broad_phase: ipctk.LBVH,
    narrow_phase: ipctk.TightInclusionCCD,
    start: np.ndarray,
    end: np.ndarray,
    min_distance: float,
) -> dict:
    candidates = ipctk.Candidates()
    candidates.build(
        mesh=mesh,
        vertices_t0=start,
        vertices_t1=end,
        inflation_radius=0.0,
        broad_phase=broad_phase,
    )
    fraction = float(
        candidates.compute_collision_free_stepsize(
            mesh=mesh,
            vertices_t0=start,
            vertices_t1=end,
            min_distance=min_distance,
            narrow_phase_ccd=narrow_phase,
        )
    )
    return {
        "collision_free_fraction": fraction,
        "candidate_counts": {
            "vv": len(candidates.vv_candidates),
            "ev": len(candidates.ev_candidates),
            "ee": len(candidates.ee_candidates),
            "fv": len(candidates.fv_candidates),
        },
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    out = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not out.exists()
    volume = pv.read(cfg.fixture)
    points = np.asarray(volume.points, dtype=np.float64)
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    names = [str(name) for name in np.asarray(volume.field_data["GroupName"]).ravel()]
    group = np.asarray(volume.point_data["GroupId"], dtype=np.int64)
    jaw = fixed & (group == names.index("Mandible"))
    assert int(jaw.sum()) == 5989
    np.testing.assert_array_equal(
        np.asarray(volume.point_data["GlobalPointId"]), np.arange(volume.n_points)
    )
    with np.load(cfg.pose_source) as z:
        pose = z["pose"].copy()
        pivot = z["pivot"].copy()
    assert pose.shape == (6,)
    assert pivot.shape == (3,)
    surface = volume.extract_surface(algorithm=None, pass_pointid=True)
    original_ids = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    all_faces = original_ids[np.asarray(surface.faces).reshape(-1, 4)[:, 1:]]
    chosen = all_faces[fixed[all_faces].all(axis=1)]
    ids, inverse = np.unique(chosen, return_inverse=True)
    faces = np.asfortranarray(inverse.reshape(-1, 3), dtype=np.int32)
    local_rest = np.asfortranarray(points[ids], dtype=np.float64)
    mesh = ipctk.CollisionMesh(local_rest, ipctk.edges(faces), faces)
    mesh.init_adjacencies()
    broad_phase = ipctk.LBVH()
    narrow_phase = ipctk.TightInclusionCCD()
    local_jaw = jaw[ids]
    face_groups = group[chosen]

    def positions(alpha: float) -> np.ndarray:
        angle = alpha * pose[:3]
        translation = alpha * pose[3:]
        result = points[ids].copy()
        result[local_jaw] = (
            (points[ids[local_jaw]] - pivot) @ Rotation.from_rotvec(angle).as_matrix().T
            + pivot
            + translation
        )
        return np.asfortranarray(result, dtype=np.float64)

    def intersects(alpha: float) -> bool:
        return bool(ipctk.has_intersections(mesh, positions(alpha), broad_phase))

    states = []
    for alpha in FRACTIONS:
        hit = intersects(alpha)
        states.append({"pose_fraction": alpha, "has_intersections": hit})
        LOG.info("Prescribed boundary pose %.6g: intersections %s", alpha, hit)
    chords = []
    for a, b in pairwise(FRACTIONS):
        start_intersects = next(
            row["has_intersections"] for row in states if row["pose_fraction"] == a
        )
        entry = {
            "from_fraction": a,
            "to_fraction": b,
            "start_intersects": start_intersects,
        }
        if not start_intersects:
            entry.update(
                chord_ccd(
                    mesh,
                    broad_phase,
                    narrow_phase,
                    positions(a),
                    positions(b),
                    cfg.min_distance_m,
                )
            )
        else:
            entry["ccd_status"] = "not_applicable_intersecting_start"
        chords.append(entry)
    first_endpoint_bracket = next(
        (
            (a["pose_fraction"], b["pose_fraction"])
            for a, b in pairwise(states)
            if not a["has_intersections"] and b["has_intersections"]
        ),
        None,
    )
    refined = None
    if first_endpoint_bracket is not None:
        low, high = first_endpoint_bracket
        for _ in range(16):
            mid = (low + high) / 2
            if intersects(mid):
                high = mid
            else:
                low = mid
        refined = {
            "last_clear_fraction": low,
            "first_detected_fraction": high,
            "resolution_fraction": high - low,
            "iterations": 16,
        }
    counts = {
        "cranium_pure": int(
            np.all(face_groups == names.index("Cranium"), axis=1).sum()
        ),
        "mandible_pure": int(
            np.all(face_groups == names.index("Mandible"), axis=1).sum()
        ),
        "mixed_or_other": int(
            (
                ~np.all(face_groups == names.index("Cranium"), axis=1)
                & ~np.all(face_groups == names.index("Mandible"), axis=1)
            ).sum()
        ),
    }
    result = {
        "schema": "prescribed-fem-boundary-collision-audit-v1",
        "sources_sha256": {
            str(path.resolve()): sha256(path)
            for path in [Path(__file__), cfg.fixture, cfg.pose_source]
        },
        "units": "metres",
        "pose": pose.tolist(),
        "pivot": pivot.tolist(),
        "policy": "Only IsFixed vertices prescribed; IsFixed intersection Mandible receives rigid pose, other fixed vertices remain stationary.",
        "boundary": {
            "selected_faces": len(chosen),
            "selected_vertices": len(ids),
            "selected_jaw_vertices": int(local_jaw.sum()),
            "face_groups": counts,
        },
        "endpoints": states,
        "chord_ccd": chords,
        "first_endpoint_intersection_bracket": refined,
        "limits": [
            "No free-tissue faces are included.",
            "CCD follows straight vertex chords; the rigid rotation path between sampled poses is curved.",
            "Endpoint intersection tests do not establish containment or validity of separate bone meshes.",
        ],
    }
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    LOG.info("Wrote %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
