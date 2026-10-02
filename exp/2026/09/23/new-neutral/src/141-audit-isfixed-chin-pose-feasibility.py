"""Measure the immutable IsFixed-tetrahedron pose limit for MouthOpen."""

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
sys.path[:0] = [str(GROUP / "src"), str(JOINT / "src")]

from chin_rigid_pose import estimate_rigid_chin_pose  # noqa: E402
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    """All inputs are saved artifacts; this audit never constructs a solver."""

    target_bundle: Path = GROUP / "data/blendshapes-isfixed-001/blendshapes.npz"
    neutral_run: Path = GROUP / "data/forward-isfixed-001"
    reference_dir: Path = GROUP / "data/reference-clearance-002"
    chin_patch: Path = GROUP / "data/chin-rigid-pose-001/estimate.json"
    frozen_state: Path = JOINT / "data/frozen-neutral-004/state.npz"
    output: Path = GROUP / "data/isfixed-chin-pose-feasibility-001/receipt.json"
    positive_margin_j: float = 0.1


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def posed(
    points: np.ndarray, jaw: np.ndarray, pivot: np.ndarray, pose: np.ndarray
) -> np.ndarray:
    """Apply the solver's rigid pose to the only original jaw fixed DOFs."""
    result = points.copy()
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    result[jaw] = (points[jaw] - pivot) @ rotation.T + pivot + pose[3:]
    return result


def min_fixed_j(
    base: np.ndarray,
    tetrahedra: np.ndarray,
    rest_volume: np.ndarray,
    fixed_tetrahedra: np.ndarray,
    jaw: np.ndarray,
    pivot: np.ndarray,
    pose: np.ndarray,
) -> tuple[float, int]:
    trial = posed(base, jaw, pivot, pose)
    tet = tetrahedra[fixed_tetrahedra]
    edges = np.transpose(trial[tet[:, 1:]] - trial[tet[:, :1]], (0, 2, 1))
    ratio = np.linalg.det(edges) / rest_volume[fixed_tetrahedra]
    return float(ratio.min()), int(np.count_nonzero(ratio <= 0))


def maximum_fraction(
    *,
    threshold: float,
    pose_full: np.ndarray,
    evaluate: callable,
) -> float:
    """Largest linear SE(3)-coordinate fraction whose immutable cells clear a gate."""
    assert threshold >= 0
    if evaluate(pose_full)[0] >= threshold:
        return 1.0
    assert evaluate(np.zeros(6))[0] >= threshold
    lo, hi = 0.0, 1.0
    for _ in range(60):
        middle = (lo + hi) / 2
        if evaluate(middle * pose_full)[0] >= threshold:
            lo = middle
        else:
            hi = middle
    return lo


def main(cfg: Config) -> None:
    assert not cfg.output.exists(), cfg.output
    assert cfg.positive_margin_j > 0
    with np.load(cfg.target_bundle, allow_pickle=False) as archive:
        names = list(np.asarray(archive["expression_names"], dtype=str))
        index = names.index("MouthOpen")
        global_ids = np.asarray(archive["skin_global_ids"], dtype=np.int64)
        triangles = np.asarray(archive["skin_triangles"], dtype=np.int64)
        neutral_skin = np.asarray(archive["new_neutral_points_m"], dtype=np.float64)
        target_skin = np.asarray(archive["target_points_m"][index], dtype=np.float64)
    with np.load(cfg.neutral_run / "endpoint.npz", allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    with np.load(
        cfg.reference_dir / "reference-clearance.npz", allow_pickle=False
    ) as archive:
        reference = np.asarray(archive["repaired_points_m"], dtype=np.float64)
    volume = pv.read(cfg.reference_dir / "repaired-reference-volume.vtu")
    tetrahedra = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    isfixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    with np.load(cfg.frozen_state, allow_pickle=False) as archive:
        historical_fixed = np.asarray(
            archive["historical_fixed_node_ids"], dtype=np.int64
        )
        mandible = np.asarray(archive["mandible_node_ids"], dtype=np.int64)
        pivot = np.asarray(archive["mandible_pivot_m"], dtype=np.float64)
    assert np.array_equal(np.flatnonzero(isfixed), historical_fixed)
    assert neutral_skin.shape == target_skin.shape == (len(global_ids), 3)
    assert np.array_equal(
        neutral_skin, reference[global_ids] + displacement[global_ids]
    )
    assert np.abs(displacement[isfixed]).max() <= 5e-16
    patch = np.asarray(
        json.loads(cfg.chin_patch.read_text())["patch_local_ids"], dtype=np.int64
    )
    chin = estimate_rigid_chin_pose(neutral_skin, target_skin, triangles, patch, pivot)
    pose_full = np.asarray(chin["pose_rad_m"], dtype=np.float64)
    jaw = np.intersect1d(historical_fixed, mandible, assume_unique=True)
    fixed_tetrahedra = np.isin(tetrahedra, historical_fixed).all(axis=1)
    rest_edges = np.transpose(
        reference[tetrahedra[:, 1:]] - reference[tetrahedra[:, :1]], (0, 2, 1)
    )
    rest_volume = np.linalg.det(rest_edges)
    assert np.all(rest_volume > 0)
    base = reference + displacement

    def evaluate(pose: np.ndarray) -> tuple[float, int]:
        return min_fixed_j(
            base, tetrahedra, rest_volume, fixed_tetrahedra, jaw, pivot, pose
        )

    fractions = {}
    for label, fraction in {
        "zero": 0.0,
        "one_percent": 0.01,
        "full_chin_fit": 1.0,
    }.items():
        minimum, inverted = evaluate(fraction * pose_full)
        fractions[label] = {
            "fraction": fraction,
            "minimum_J": minimum,
            "inverted_fixed_tetrahedra": inverted,
        }
    strict_fraction = maximum_fraction(
        threshold=0.0, pose_full=pose_full, evaluate=evaluate
    )
    margin_fraction = maximum_fraction(
        threshold=cfg.positive_margin_j, pose_full=pose_full, evaluate=evaluate
    )
    rotation_deg = chin["fit_rotation_degrees"]
    translation_mm = 1000 * chin["translation_norm_m"]
    receipt = {
        "schema": "isfixed-chin-pose-feasibility-v1",
        "scope": "CPU kinematic gate only; no collision query, harmonic carry, forward solve, or inverse step was run.",
        "boundary_contract": "The only original FEM jaw motion is IsFixed intersect Mandible. Tetrahedra whose four vertices are IsFixed cannot be repaired by a free-DOF solve.",
        "inputs": {
            "target_bundle": record(cfg.target_bundle),
            "neutral_endpoint": record(cfg.neutral_run / "endpoint.npz"),
            "reference": record(cfg.reference_dir / "reference-clearance.npz"),
            "volume": record(cfg.reference_dir / "repaired-reference-volume.vtu"),
            "chin_patch": record(cfg.chin_patch),
            "frozen_state": record(cfg.frozen_state),
        },
        "chin_fit": chin,
        "fixed_boundary": {
            "isfixed_vertices": int(isfixed.sum()),
            "jaw_isfixed_vertices": len(jaw),
            "all_isfixed_tetrahedra": int(fixed_tetrahedra.sum()),
        },
        "samples": fractions,
        "pose_limits": {
            "strictly_positive_J_fraction": strict_fraction,
            "strictly_positive_J_rotation_deg": strict_fraction * rotation_deg,
            "strictly_positive_J_translation_mm": strict_fraction * translation_mm,
            "margin_J": cfg.positive_margin_j,
            "margin_fraction": margin_fraction,
            "margin_rotation_deg": margin_fraction * rotation_deg,
            "margin_translation_mm": margin_fraction * translation_mm,
        },
        "recommendation": {
            "initial_fraction": 0.01,
            "candidate_gate": f"all-IsFixed tetrahedra min J >= {cfg.positive_margin_j}",
            "reason": "The full chin fit crosses J=0 in immutable all-IsFixed tetrahedra; a carry or equilibrium solve cannot recover that violation.",
        },
    }
    cfg.output.parent.mkdir(parents=True)
    write_json(cfg.output, receipt)
    cherries.log_output(cfg.output)
    cherries.log_metrics(
        {
            "pose/full_rotation_deg": rotation_deg,
            "pose/full_translation_mm": translation_mm,
            "gate/strict_fraction": strict_fraction,
            "gate/margin_fraction": margin_fraction,
            "gate/full_minimum_J": fractions["full_chin_fit"]["minimum_J"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
