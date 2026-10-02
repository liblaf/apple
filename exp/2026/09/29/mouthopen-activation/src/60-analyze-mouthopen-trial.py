"""Independently audit the terminal contact-off MouthOpen trial on CPU."""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))

from experiment import Profile  # noqa: E402

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("60-mouthopen-analysis")
    forward: Path = GROUP / "data/49-forward-contact-off"
    fit: Path | None = None
    fixture: Path = GROUP / "data/30-pruned-fixture"
    prepared: Path = GROUP / "data/10-mandible/prepared.npz"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def check_record(record: dict[str, Any]) -> dict[str, Any]:
    path = Path(record["path"])
    actual = sha256(path)
    return {
        "path": str(path),
        "exists": path.is_file(),
        "declared_sha256": record["sha256"],
        "actual_sha256": actual,
        "matches": actual == record["sha256"],
    }


def check_snapshot(manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(manifest_path)
    rows = []
    for module, record in manifest.items():
        path = Path(record["path"])
        actual = sha256(path)
        rows.append(
            {
                "module": module,
                "path": str(path),
                "declared_sha256": record["sha256"],
                "actual_sha256": actual,
                "matches": actual == record["sha256"],
            }
        )
    assert all(row["matches"] for row in rows)
    return {"path": str(manifest_path), "modules": len(rows), "all_match": True}


def tet_jacobians(
    points: np.ndarray, tets: np.ndarray, displacement: np.ndarray
) -> np.ndarray:
    current = points + displacement
    rest_dm = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    current_dm = np.transpose(current[tets[:, 1:]] - current[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(current_dm) / np.linalg.det(rest_dm)


def skin_metrics(
    points: np.ndarray,
    skin_ids: np.ndarray,
    triangles: np.ndarray,
    target_displacement: np.ndarray,
    displacement: np.ndarray,
    patch_ids: np.ndarray,
) -> dict[str, float]:
    reference = points[skin_ids]
    actual = reference + displacement[skin_ids]
    target = reference + target_displacement
    tri_ref = reference[triangles]
    area = (
        np.linalg.norm(
            np.cross(tri_ref[:, 1] - tri_ref[:, 0], tri_ref[:, 2] - tri_ref[:, 0]),
            axis=1,
        )
        / 2
    )
    mass = np.zeros(len(skin_ids))
    np.add.at(mass, triangles.ravel(), np.repeat(area / 3, 3))
    mass /= mass.sum()
    error = actual - target
    fit = 1000 * np.sqrt(np.sum(mass[:, None] * error**2))

    def normals(x: np.ndarray) -> np.ndarray:
        tri = x[triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(cross, axis=1)
        assert np.all(length > 1e-14)
        return cross / length[:, None]

    dot = np.clip(np.einsum("ij,ij->i", normals(actual), normals(target)), -1, 1)
    normal = np.rad2deg(np.sqrt(np.dot(area, np.arccos(dot) ** 2) / area.sum()))
    chin_mass = mass[patch_ids]
    chin = 1000 * np.sqrt(
        np.sum(chin_mass * np.sum(error[patch_ids] ** 2, axis=1)) / chin_mass.sum()
    )
    return {
        "fit_rms_mm": float(fit),
        "normal_angle_rms_deg": float(normal),
        "chin_rms_mm": float(chin),
    }


def audit_forward(cfg: Config) -> dict[str, Any]:  # noqa: PLR0915
    summary_path = cfg.forward / "summary.json"
    final_path = cfg.forward / "final.npz"
    summary = load_json(summary_path)
    assert summary["status"] != "running", "forward trial is still running"
    assert final_path.is_file(), "terminal forward checkpoint is absent"
    assert summary["final_checkpoint"]["sha256"] == sha256(final_path)
    declared_inputs = {
        name: check_record(record) for name, record in summary["inputs"].items()
    }
    assert all(row["matches"] for row in declared_inputs.values())
    snapshots = check_snapshot(cfg.forward / "source-manifest.json")
    with np.load(final_path, allow_pickle=False) as source:
        displacement = source["displacement"].copy()
        pose = source["pose"].copy()
        fraction = float(source["fraction"])
    with np.load(cfg.prepared, allow_pickle=False) as source:
        full_pose = source["pose"].copy()
        pivot = source["pivot"].copy()
        patch_ids = source["patch_ids"].copy()
    np.testing.assert_allclose(pose, fraction * full_pose, rtol=0, atol=1e-14)
    assert fraction == float(summary["completed_pose_fraction"])
    mesh = pv.read(cfg.fixture / "volume.vtu")
    skin = pv.read(cfg.fixture / "skin.vtp")
    points = np.asarray(mesh.points)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:]
    assert displacement.shape == points.shape
    fixed = np.asarray(mesh.point_data["IsFixed"], dtype=bool)
    group_names = [str(name) for name in mesh.field_data["GroupName"]]
    jaw = fixed & (
        np.asarray(mesh.point_data["GroupId"]) == group_names.index("Mandible")
    )
    expected = np.zeros_like(points)
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    expected[jaw] = (points[jaw] - pivot) @ rotation.T + pivot + pose[3:] - points[jaw]
    np.testing.assert_allclose(displacement[fixed], expected[fixed], rtol=0, atol=1e-12)
    assert not np.any(fixed[tets].all(axis=1))
    jacobians = tet_jacobians(points, tets, displacement)
    assert np.isfinite(jacobians).all()
    rest_determinants = np.linalg.det(
        np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    )
    assert np.all(rest_determinants > 0)
    rest_volumes = rest_determinants / 6
    inverted = jacobians <= 0
    volumes = (
        np.asarray(mesh.cell_data["Volume"], dtype=float)
        if "Volume" in mesh.cell_data
        else rest_volumes
    )
    assert np.allclose(volumes, rest_volumes, rtol=2e-12, atol=1e-15)
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    triangles = np.asarray(skin.faces).reshape(-1, 4)[:, 1:]
    skin_stats = skin_metrics(
        points,
        skin_ids,
        triangles,
        np.asarray(mesh.point_data["MouthOpen"])[skin_ids],
        displacement,
        patch_ids,
    )
    final = summary["final"]
    recomputed = {
        "minimum_J": float(jacobians.min()),
        "inverted_cells": int(inverted.sum()),
        "inverted_cell_fraction": float(inverted.mean()),
        "inverted_rest_volume_fraction": float(
            rest_volumes[inverted].sum() / rest_volumes.sum()
        ),
        **skin_stats,
    }
    for key in (
        "minimum_J",
        "inverted_cells",
        "inverted_cell_fraction",
        "inverted_rest_volume_fraction",
        "fit_rms_mm",
    ):
        np.testing.assert_allclose(recomputed[key], final[key], rtol=2e-8, atol=2e-11)
    force_atol = float(summary["config"]["force_atol"])
    accepted = [row for row in summary["attempts"] if row["status"] == "accepted"]
    for row in accepted:
        checkpoint = check_record(row["checkpoint"])
        assert checkpoint["matches"]
        assert row["metrics"]["force_norm"] <= force_atol
        with np.load(checkpoint["path"], allow_pickle=False) as source:
            assert float(source["fraction"]) == float(row["target_fraction"])
    assert final["force_norm"] <= force_atol
    return {
        "status": summary["status"],
        "full_pose": bool(summary["numerically_converged_full_pose"] and fraction == 1),
        "checkpoint": {"path": str(final_path), "sha256": sha256(final_path)},
        "declared_inputs": declared_inputs,
        "source_snapshot": snapshots,
        "pose_fraction": fraction,
        "boundary": {
            "fixed_vertices": int(fixed.sum()),
            "jaw_fixed_vertices": int(jaw.sum()),
            "allfixed_cells": 0,
            "prescribed_displacements_verified": True,
        },
        "cells": {"points": int(mesh.n_points), "tetrahedra": int(mesh.n_cells)},
        "accepted_checkpoints": len(accepted),
        "force_receipts_within_declared_gate": True,
        "recomputed": recomputed,
        "summary_final": final,
        "solver_claim": bool(summary["numerically_converged_full_pose"]),
        "physical_validity_claim": bool(summary["physical_validity_claim"]),
    }


def dual_norm(gradient: np.ndarray, mass: np.ndarray) -> float:
    symmetric = (gradient + np.swapaxes(gradient, -1, -2)) / 2
    return float(np.sqrt(np.sum(np.sum(symmetric**2, axis=(-2, -1)) / mass)))


def checkpoint_metrics(mesh: Any, displacement: np.ndarray) -> dict[str, float | int]:
    points, skin_ids, triangles = (
        mesh["rest_points"],
        mesh["skin_ids"],
        mesh["triangles"],
    )
    target, weights = mesh["target_displacement_skin"], mesh["skin_vertex_weights"]
    error = displacement[skin_ids] - target
    fit = 1000 * np.sqrt(np.sum(weights[:, None] * error**2))
    reference, actual = points[skin_ids], points[skin_ids] + displacement[skin_ids]
    desired = reference + target

    def normals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tri = x[triangles]
        cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(cross, axis=1)
        assert np.all(length > 1e-14)
        return cross / length[:, None], length / 2

    _, area = normals(reference)
    actual_n, _ = normals(actual)
    desired_n, _ = normals(desired)
    dot = np.clip(np.einsum("ij,ij->i", actual_n, desired_n), -1, 1)
    normal = np.rad2deg(np.sqrt(np.dot(area, np.arccos(dot) ** 2) / area.sum()))
    detf = tet_jacobians(points, mesh["tets"], displacement)
    inverted = detf <= 0
    active = np.zeros(len(detf), dtype=bool)
    active[mesh["active_ids"]] = True
    return {
        "fit_rms_mm": float(fit),
        "normal_angle_rms_deg": float(normal),
        "detF_min": float(detf.min()),
        "inverted_all_cells": int(np.count_nonzero(inverted)),
        "inverted_active_cells": int(np.count_nonzero(inverted & active)),
        "inverted_inactive_cells": int(np.count_nonzero(inverted & ~active)),
    }


def audit_fit(fit: Path) -> dict[str, Any]:  # noqa: PLR0915
    summary = load_json(fit / "summary.json")
    assert summary["status"] != "running", "activation fit is still running"
    assert summary["status"] == "completed_attempt_budget"
    last = fit / "last.npz"
    assert last.is_file()
    assert summary["final_checkpoint"]["sha256"] == sha256(last)
    assert summary["initialization"] == "zero S, full prescribed jaw, fresh Adam"
    assert summary["mode"] == "psd6"
    assert float(summary["config"]["smooth_weight"]) == 7.2e-6
    assert int(summary["config"]["steps"]) == 200
    assert int(summary["attempted_updates"]) == 200
    assert int(summary["optimizer_updates"]) + int(summary["skipped_updates"]) == 200
    declared_inputs = {
        name: check_record(record) for name, record in summary["inputs"].items()
    }
    assert all(row["matches"] for row in declared_inputs.values())
    snapshots = check_snapshot(fit / "source-manifest.json")
    mesh = np.load(fit / "mesh.npz", allow_pickle=False)
    state = np.load(last, allow_pickle=False)
    initial = np.load(fit / "step-0000.npz", allow_pickle=False)
    forward_checkpoint = Path(summary["inputs"]["final.npz"]["path"])
    with np.load(forward_checkpoint, allow_pickle=False) as source:
        parent_u = source["displacement"].copy()
    assert int(initial["step"]) == 0
    np.testing.assert_allclose(initial["q"], 0, rtol=0, atol=0)
    np.testing.assert_allclose(initial["S"], 0, rtol=0, atol=0)
    np.testing.assert_allclose(initial["u"], parent_u, rtol=0, atol=0)
    q = state["q"].copy()
    assert q.shape == (len(mesh["active_ids"]), 6)
    # psd6 stores the six Mandel coordinates of S; rebuild and test its eigenvalues.
    matrices = np.zeros((len(q), 3, 3))
    matrices[:, 0, 0], matrices[:, 1, 1], matrices[:, 2, 2] = q[:, :3].T
    matrices[:, 0, 1] = matrices[:, 1, 0] = q[:, 3] / np.sqrt(2)
    matrices[:, 1, 2] = matrices[:, 2, 1] = q[:, 4] / np.sqrt(2)
    matrices[:, 0, 2] = matrices[:, 2, 0] = q[:, 5] / np.sqrt(2)
    matrix_error = np.abs(matrices - state["S"])
    np.testing.assert_allclose(
        matrices, state["S"], rtol=4 * np.finfo(float).eps, atol=0
    )
    np.testing.assert_allclose(state["B"], np.eye(3) + state["S"], rtol=0, atol=0)
    assert np.linalg.eigvalsh(matrices).min() >= -2e-12
    i, j, w = mesh["edge_i"], mesh["edge_j"], mesh["edge_weight"]
    eigval = np.linalg.eigvalsh(matrices)
    factor = float(mesh["regularizer_factor"])
    amplitude = factor * np.sum(w[:, None] * (eigval[i] - eigval[j]) ** 2)
    trace = np.einsum("eab,eab->e", matrices[i], matrices[j])
    orientation = (
        factor * 2 * np.sum(w * (np.sum(eigval[i] * eigval[j], axis=1) - trace))
    )
    direct = factor * np.sum(w * np.sum((matrices[i] - matrices[j]) ** 2, axis=(1, 2)))
    np.testing.assert_allclose(amplitude + orientation, direct, rtol=2e-10, atol=2e-12)
    gradients = np.load(fit / "gradient-components.npz", allow_pickle=False)
    l2 = dual_norm(gradients["l2_tensor_gradient"], gradients["active_volume_weights"])
    smooth = dual_norm(
        gradients["regularizer_tensor_gradient"], gradients["active_volume_weights"]
    )
    weighted = float(gradients["smooth_weight"]) * smooth
    ratio = weighted / l2
    balance = load_json(fit / "gradient-balance.json")
    assert balance["checkpoint"]["sha256"] == sha256(last)
    assert balance["components"]["sha256"] == sha256(fit / "gradient-components.npz")
    np.testing.assert_allclose(
        ratio, balance["smoothness_to_l2_gradient_ratio"], rtol=2e-10
    )
    recomputed = checkpoint_metrics(mesh, state["u"])
    for key in (
        "fit_rms_mm",
        "normal_angle_rms_deg",
        "detF_min",
        "inverted_all_cells",
    ):
        np.testing.assert_allclose(
            recomputed[key], summary["last_metrics"][key], rtol=2e-8
        )
    np.testing.assert_allclose(
        direct, summary["last_metrics"]["activation_smoothness"], rtol=2e-8
    )
    trace = load_jsonl(fit / "trace.jsonl")
    receipts = load_jsonl(fit / "solver-receipts.jsonl")
    proposals_path = fit / "proposals.jsonl"
    proposals = load_jsonl(proposals_path) if proposals_path.is_file() else []
    assert len(trace) == int(summary["optimizer_updates"]) + 1
    assert len(receipts) == len(trace)
    assert len(proposals) == int(summary["skipped_updates"])
    assert int(trace[-1]["attempt"]) == int(state["step"])
    assert int(receipts[-1]["attempt"]) == int(state["step"])
    force_atol = float(summary["config"]["force_atol"])
    for receipt in receipts:
        forward, adjoint = receipt["forward"], receipt["adjoint"]
        assert forward["success"]
        assert forward["solver_valid"]
        assert forward["accepted_force_norm"] <= force_atol
        assert adjoint["success"]
        assert float(adjoint["accepted_absolute_residual"]) <= float(
            adjoint["accepted_threshold"]
        )
    rest_dm = np.transpose(
        mesh["rest_points"][mesh["tets"][:, 1:]]
        - mesh["rest_points"][mesh["tets"][:, :1]],
        (0, 2, 1),
    )
    rest_volume = np.linalg.det(rest_dm) / 6
    assert np.all(rest_volume > 0)
    final_j = tet_jacobians(mesh["rest_points"], mesh["tets"], state["u"])
    final_inverted = final_j <= 0
    final_inverted_rest_volume_fraction = float(
        rest_volume[final_inverted].sum() / rest_volume.sum()
    )
    last_geometry = receipts[-1]["forward"]["geometry"]
    assert bool(last_geometry["has_intersections"]) == (
        not bool(last_geometry["no_intersections"])
    )
    return {
        "status": summary["status"],
        "checkpoint": summary["final_checkpoint"],
        "declared_inputs": declared_inputs,
        "source_snapshot": snapshots,
        "budget": {
            "attempted_updates": summary["attempted_updates"],
            "optimizer_updates": summary["optimizer_updates"],
            "skipped_updates": summary["skipped_updates"],
        },
        "initial_zero_activation_and_parent_seed_verified": True,
        "solver_receipts_within_declared_gates": True,
        "geometry": {
            "inverted_rest_volume_fraction": final_inverted_rest_volume_fraction,
            "has_intersections": bool(last_geometry["has_intersections"]),
            "intersection_scope": last_geometry["intersection_scope"],
            "receipt_attempt": int(receipts[-1]["attempt"]),
        },
        "eta": float(gradients["smooth_weight"]),
        "final_metrics": recomputed,
        "psd_verified": True,
        "mandel_reconstruction": {
            "max_abs_error": float(matrix_error.max()),
            "max_relative_error": float(
                matrix_error.max() / max(float(np.abs(state["S"]).max()), 1.0)
            ),
        },
        "roughness": {
            "direct_frobenius": float(direct),
            "eigenvalue_amplitude": float(amplitude),
            "orientation": float(orientation),
            "identity_verified": True,
            "note": "This spectral decomposition is exact for full PSD tensors; it is not the rank-one amplitude/direction split.",
        },
        "gradient_ratio": {
            "l2_dual_norm": l2,
            "regularizer_dual_norm": smooth,
            "weighted_regularizer_dual_norm": weighted,
            "smoothness_to_l2": ratio,
            "normal_loss_excluded_from_denominator": True,
        },
        "solver_converged": bool(summary.get("solver_converged", False)),
        "orientation_valid": bool(summary.get("orientation_valid", False)),
    }


def main(cfg: Config) -> None:
    out = cherries.output(cfg.output / "analysis.json", mkdir=True).parent
    assert not (out / "analysis.json").exists(), out
    forward = audit_forward(cfg)
    fit = None
    if cfg.fit is not None and (cfg.fit / "summary.json").is_file():
        fit = audit_fit(cfg.fit)
    result = {
        "schema": "mouthopen-cpu-analysis-v1",
        "forward": forward,
        "fit": fit,
        "interpretation": {
            "solver_strictness": "Forward force receipts are checked against the declared 1e-10 gate. Activation adjoint receipts are checked only when a terminal fit exists.",
            "geometry_validity": "Finite inverted cells and any boundary intersections remain physical-invalidity diagnostics even if the numerical force solve converges.",
            "activation_availability": "Gradient ratio and graph roughness are unavailable until a terminal activation fit is supplied.",
        },
    }
    (out / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            "forward_pose_fraction": forward["pose_fraction"],
            "fit_available": int(fit is not None),
        }
    )
    LOG.info("Wrote independent audit to %s", out / "analysis.json")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
