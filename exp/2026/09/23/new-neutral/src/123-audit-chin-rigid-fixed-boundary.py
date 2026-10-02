"""Audit intended IsFixed FEM constraints for an archived chin pose."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    estimate: Path = GROUP / "data/chin-rigid-pose-001/estimate.json"
    reference: Path = GROUP / "data/reference-clearance-002/reference-clearance.npz"
    volume: Path = GROUP / "data/reference-clearance-002/repaired-reference-volume.vtu"
    frozen_state: Path = JOINT / "data/frozen-neutral-004/state.npz"
    output: Path = GROUP / "data/isfixed-fixed-boundary-audit-002/receipt.json"


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def signed_ratio(
    points: np.ndarray, tetrahedra: np.ndarray, rest: np.ndarray
) -> np.ndarray:
    edges = np.transpose(
        points[tetrahedra[:, 1:]] - points[tetrahedra[:, :1]], (0, 2, 1)
    )
    return np.linalg.det(edges) / rest


def rigid_points(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    return (points - pivot) @ rotation.T + pivot + pose[3:]


def _tetra_receipt(
    tetrahedra: np.ndarray,
    ratio: np.ndarray,
    fixed: np.ndarray,
    cranium: np.ndarray,
    jaw: np.ndarray,
) -> dict[str, object]:
    all_fixed = np.flatnonzero(np.isin(tetrahedra, fixed).all(axis=1))
    inverted = all_fixed[ratio[all_fixed] <= 0]
    return {
        "all_fixed": len(all_fixed),
        "all_fixed_inverted": len(inverted),
        "all_fixed_min_J": float(ratio[all_fixed].min()) if len(all_fixed) else None,
        "all_fixed_inversion_ids_and_composition": [
            {
                "cell_id": int(cell),
                "J": float(ratio[cell]),
                "cranium_label_vertices": int(np.isin(tetrahedra[cell], cranium).sum()),
                "jaw_routed_vertices": int(np.isin(tetrahedra[cell], jaw).sum()),
            }
            for cell in inverted
        ],
    }


def main(cfg: Config) -> None:
    assert not cfg.output.exists(), cfg.output
    estimate = json.loads(cfg.estimate.read_text())
    assert estimate["schema"] == "chin-rigid-pose-estimate-v1"
    pose = np.asarray(estimate["pose_rad_m"], dtype=np.float64)
    pivot = np.asarray(estimate["mandible_pivot_m"], dtype=np.float64)
    assert pose.shape == (6,)
    assert pivot.shape == (3,)
    with np.load(cfg.reference, allow_pickle=False) as archive:
        reference = np.asarray(archive["reference_points_m"], dtype=np.float64)
        repaired = np.asarray(archive["repaired_points_m"], dtype=np.float64)
        repair_displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    mesh = pv.read(cfg.volume)
    tetrahedra = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    mesh_isfixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    assert mesh_isfixed.shape == (mesh.n_points,)
    intended_fixed = np.flatnonzero(mesh_isfixed).astype(np.int64)
    saved_mask = np.asarray(mesh.point_data["FixedMask"], dtype=bool).any(axis=1)
    with np.load(cfg.frozen_state, allow_pickle=False) as state:
        isfixed = np.asarray(state["historical_fixed_node_ids"], dtype=np.int64)
        cranium = np.asarray(state["cranium_node_ids"], dtype=np.int64)
        mandible = np.asarray(state["mandible_node_ids"], dtype=np.int64)
        state_pivot = np.asarray(state["mandible_pivot_m"], dtype=np.float64)
    assert np.array_equal(pivot, state_pivot)
    assert tetrahedra.max() < len(repaired)
    assert np.array_equal(repaired - reference, repair_displacement)
    assert not np.intersect1d(cranium, mandible).size
    assert np.array_equal(intended_fixed, isfixed)
    jaw = np.intersect1d(intended_fixed, mandible, assume_unique=True)
    cranium_fixed = np.intersect1d(isfixed, cranium, assume_unique=True)
    old_union = np.union1d(isfixed, np.union1d(cranium, mandible))
    assert np.array_equal(np.flatnonzero(saved_mask), old_union)
    assert np.array_equal(repaired[intended_fixed], reference[intended_fixed])
    assert float(np.abs(repair_displacement[intended_fixed]).max()) == 0.0

    # Intended boundary semantics: IsFixed is the sole FEM clamp.  Group labels
    # route the jaw only after intersection with that mask.
    intended_posed = repaired.copy()
    intended_posed[jaw] = rigid_points(repaired[jaw], pivot, pose)
    saved_old_policy_posed = repaired.copy()
    saved_old_policy_posed[mandible] = rigid_points(repaired[mandible], pivot, pose)
    rest_edges = np.transpose(
        repaired[tetrahedra[:, 1:]] - repaired[tetrahedra[:, :1]], (0, 2, 1)
    )
    rest = np.linalg.det(rest_edges)
    assert np.all(rest > 0)
    intended_ratio = signed_ratio(intended_posed, tetrahedra, rest)
    saved_old_policy_ratio = signed_ratio(saved_old_policy_posed, tetrahedra, rest)
    intended = _tetra_receipt(
        tetrahedra, intended_ratio, intended_fixed, cranium_fixed, jaw
    )
    saved = _tetra_receipt(
        tetrahedra, saved_old_policy_ratio, old_union, cranium, mandible
    )
    receipt = {
        "schema": "isfixed-chin-rigid-fixed-boundary-audit-v2",
        "scope": "CPU kinematic diagnostic only; no forward physics was evaluated.",
        "boundary_policy": {
            "intended": "IsFixed is the sole FEM clamp; jaw motion is IsFixed intersect Mandible.",
            "saved_archived_mask": "union(IsFixed, GroupId=Cranium, GroupId=Mandible), which is invalidated old policy.",
        },
        "pose_rad_m": pose.tolist(),
        "pivot_m": pivot.tolist(),
        "inputs": {
            "estimate": record(cfg.estimate),
            "reference": record(cfg.reference),
            "volume": record(cfg.volume),
            "frozen_state": record(cfg.frozen_state),
            "boundary_source": record(JOINT / "src/joint_full_skull_contact.py"),
        },
        "fixed_nodes": {
            "intended_isfixed": len(intended_fixed),
            "intended_cranium_label_and_isfixed": len(cranium_fixed),
            "intended_jaw_isfixed_intersection": len(jaw),
            "saved_archived_fixedmask": int(saved_mask.sum()),
            "saved_extra_group_labelled": int(saved_mask.sum() - len(intended_fixed)),
            "repaired_intended_fixed_coordinate_displacement_max_m": float(
                np.abs(repair_displacement[intended_fixed]).max()
            ),
        },
        "tetrahedra": {
            "intended_isfixed_policy": intended,
            "saved_old_policy_comparison_only": saved,
            "intended_overall_min_J": float(intended_ratio.min()),
            "intended_overall_inverted": int(np.count_nonzero(intended_ratio <= 0)),
            "saved_old_policy_overall_min_J": float(saved_old_policy_ratio.min()),
            "saved_old_policy_overall_inverted": int(
                np.count_nonzero(saved_old_policy_ratio <= 0)
            ),
        },
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(cfg.output, receipt)
    cherries.log_output(cfg.output)
    cherries.log_metrics(
        {
            "fixed_boundary/intended_all_fixed_tetrahedra": intended["all_fixed"],
            "fixed_boundary/intended_all_fixed_inverted": intended[
                "all_fixed_inverted"
            ],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
