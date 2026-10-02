"""Independently audit a terminal four-stage MouthOpen activation chain on CPU."""

# ruff: noqa: C901, PLR0915

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402

STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")


class Config(cherries.BaseConfig):
    output: Path = Path("80-four-stage-analysis")
    chain: Path = GROUP / "data/70-mouthopen-four-stage"
    fixture: Path = GROUP / "data/30-pruned-fixture"
    prepared: Path = GROUP / "data/10-mandible/prepared.npz"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(row) for row in path.read_text().splitlines() if row]


def matrix(q: np.ndarray) -> np.ndarray:
    out = np.zeros((*q.shape[:-1], 3, 3))
    out[..., 0, 0], out[..., 1, 1], out[..., 2, 2] = q[..., :3].T
    out[..., 0, 1] = out[..., 1, 0] = q[..., 3] / np.sqrt(2)
    out[..., 1, 2] = out[..., 2, 1] = q[..., 4] / np.sqrt(2)
    out[..., 0, 2] = out[..., 2, 0] = q[..., 5] / np.sqrt(2)
    return out


def detf(points: np.ndarray, tets: np.ndarray, u: np.ndarray) -> np.ndarray:
    rest = np.transpose(points[tets[:, 1:]] - points[tets[:, :1]], (0, 2, 1))
    current = points + u
    deformed = np.transpose(current[tets[:, 1:]] - current[tets[:, :1]], (0, 2, 1))
    return np.linalg.det(deformed) / np.linalg.det(rest)


def metrics(mesh: Any, u: np.ndarray) -> dict[str, float | int]:
    points, ids, tri, target, mass = (
        mesh["rest_points"],
        mesh["skin_ids"],
        mesh["triangles"],
        mesh["target_displacement_skin"],
        mesh["skin_vertex_weights"],
    )
    error = u[ids] - target
    fit = 1000 * np.sqrt(np.sum(mass[:, None] * error**2))
    ref, actual, expected = points[ids], points[ids] + u[ids], points[ids] + target

    def normals(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        face = x[tri]
        cross = np.cross(face[:, 1] - face[:, 0], face[:, 2] - face[:, 0])
        length = np.linalg.norm(cross, axis=1)
        assert np.all(length > 1e-14)
        return cross / length[:, None], length / 2

    _, area = normals(ref)
    na, _ = normals(actual)
    ne, _ = normals(expected)
    angle = np.arccos(np.clip(np.einsum("ij,ij->i", na, ne), -1, 1))
    j = detf(points, mesh["tets"], u)
    active = np.zeros(len(j), bool)
    active[mesh["active_ids"]] = True
    inverse = j <= 0
    rest = np.transpose(
        points[mesh["tets"][:, 1:]] - points[mesh["tets"][:, :1]], (0, 2, 1)
    )
    volume = np.linalg.det(rest) / 6
    assert np.all(volume > 0)
    return {
        "fit_rms_mm": float(fit),
        "normal_angle_rms_deg": float(
            np.rad2deg(np.sqrt(np.dot(area, angle**2) / area.sum()))
        ),
        "detF_min": float(j.min()),
        "inverted_all_cells": int(inverse.sum()),
        "inverted_active_cells": int(np.count_nonzero(inverse & active)),
        "inverted_inactive_cells": int(np.count_nonzero(inverse & ~active)),
        "inverted_rest_volume_fraction": float(volume[inverse].sum() / volume.sum()),
    }


def verify_manifest(path: Path) -> int:
    records = load(path)
    for record in records.values():
        assert sha256(Path(record["path"])) == record["sha256"]
    return len(records)


def verify_records(records: dict[str, dict[str, Any]]) -> int:
    for record in records.values():
        assert sha256(Path(record["path"])) == record["sha256"]
    return len(records)


def dual_norm(g: np.ndarray, mass: np.ndarray) -> float:
    symmetric = (g + np.swapaxes(g, -1, -2)) / 2
    return float(np.sqrt(np.sum(np.sum(symmetric**2, axis=(-2, -1)) / mass)))


def roughness(state: Any, mesh: Any, mode: str) -> dict[str, Any]:
    s = state["S"]
    i, j, w = mesh["edge_i"], mesh["edge_j"], mesh["edge_weight"]
    factor = float(mesh["regularizer_factor"])
    direct = factor * np.sum(w * np.sum((s[i] - s[j]) ** 2, axis=(1, 2)))
    eigenvalue = np.linalg.eigvalsh(s)
    amplitude = factor * np.sum(w[:, None] * (eigenvalue[i] - eigenvalue[j]) ** 2)
    orientation = (
        factor
        * 2
        * np.sum(
            w
            * (
                np.sum(eigenvalue[i] * eigenvalue[j], axis=1)
                - np.einsum("eab,eab->e", s[i], s[j])
            )
        )
    )
    np.testing.assert_allclose(direct, amplitude + orientation, rtol=2e-10, atol=2e-12)
    label = (
        "signed_eigenvalue_amplitude"
        if mode == "symmetric6"
        else "eigenvalue_amplitude"
    )
    result = {
        "direct": float(direct),
        label: float(amplitude),
        "orientation": float(orientation),
        "identity_verified": True,
    }
    if mode.startswith("rankone"):
        q = state["q"]
        amplitude_q = q[:, 0]
        axis = state["fixed_axes"] if mode == "rankone_fixed" else q[:, 1:]
        axis = axis / np.linalg.norm(axis, axis=1, keepdims=True)
        rank_amplitude = factor * np.sum(w * (amplitude_q[i] - amplitude_q[j]) ** 2)
        rank_direction = (
            factor
            * 2
            * np.sum(
                w
                * amplitude_q[i]
                * amplitude_q[j]
                * (1 - np.einsum("ij,ij->i", axis[i], axis[j]) ** 2)
            )
        )
        np.testing.assert_allclose(
            direct, rank_amplitude + rank_direction, rtol=2e-10, atol=2e-12
        )
        result["rankone_amplitude"] = float(rank_amplitude)
        result["rankone_direction"] = float(rank_direction)
    return result


def audit_stage(
    folder: Path,
    mode: str,
    mesh: Any,
    expected_parent: Path,
    boundary: np.ndarray,
    fixed: np.ndarray,
) -> dict[str, Any]:
    summary = load(folder / "summary.json")
    assert summary["mode"] == mode
    assert summary["status"] == "completed_attempt_budget"
    assert summary["parent_checkpoint"]["sha256"] == sha256(expected_parent)
    assert summary["final_checkpoint"]["sha256"] == sha256(folder / "last.npz")
    assert summary["initialization"]["controls"]["sha256"] == sha256(
        folder / "initialization.npz"
    )
    declared_inputs = verify_records(summary["inputs"])
    sources = verify_manifest(folder / "source-manifest.json")
    initial = np.load(folder / "initialization.npz", allow_pickle=False)
    last = np.load(folder / "last.npz", allow_pickle=False)
    assert str(initial["mode"]) == mode
    assert str(last["mode"]) == mode
    np.testing.assert_allclose(initial["B"], initial["S"] + np.eye(3), rtol=0, atol=0)
    np.testing.assert_allclose(last["B"], last["S"] + np.eye(3), rtol=0, atol=0)
    if mode in {"symmetric6", "psd6"}:
        np.testing.assert_allclose(
            matrix(initial["q"]), initial["S"], rtol=4 * np.finfo(float).eps, atol=0
        )
        np.testing.assert_allclose(
            matrix(last["q"]), last["S"], rtol=4 * np.finfo(float).eps, atol=0
        )
    if mode.startswith("rankone"):
        q = initial["q"]
        axes = initial["fixed_axes"] if mode == "rankone_fixed" else q[:, 1:]
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        reconstructed = q[:, :1, None] * axes[:, :, None] * axes[:, None, :]
        np.testing.assert_allclose(initial["S"], reconstructed, rtol=4e-13, atol=4e-13)
        q_last = last["q"]
        axes_last = last["fixed_axes"] if mode == "rankone_fixed" else q_last[:, 1:]
        axes_last = axes_last / np.linalg.norm(axes_last, axis=1, keepdims=True)
        reconstructed_last = (
            q_last[:, :1, None] * axes_last[:, :, None] * axes_last[:, None, :]
        )
        np.testing.assert_allclose(
            last["S"], reconstructed_last, rtol=4e-13, atol=4e-13
        )
    if mode == "psd6":
        assert np.linalg.eigvalsh(initial["S"]).min() >= -2e-12
        assert np.linalg.eigvalsh(last["S"]).min() >= -2e-12
    if mode.startswith("rankone"):
        assert np.linalg.matrix_rank(last["S"], tol=1e-10).max() <= 1
    np.testing.assert_allclose(
        initial["u_seed"],
        np.load(expected_parent, allow_pickle=False)["u"]
        if mode != "symmetric6"
        else np.load(expected_parent, allow_pickle=False)["displacement"],
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(
        last["u"][fixed],
        boundary[fixed],
        rtol=0,
        atol=1e-12,
    )
    trace, receipts = (
        jsonl(folder / "trace.jsonl"),
        jsonl(folder / "solver-receipts.jsonl"),
    )
    proposal = folder / "proposals.jsonl"
    proposals = jsonl(proposal) if proposal.is_file() else []
    assert len(trace) == int(summary["optimizer_updates"]) + 1
    assert len(receipts) == len(trace)
    assert len(proposals) == int(summary["skipped_updates"])
    trace_attempts = {int(row["attempt"]) for row in trace}
    proposal_attempts = {int(row["attempt"]) for row in proposals}
    assert trace_attempts.isdisjoint(proposal_attempts)
    assert trace_attempts | proposal_attempts == set(range(201))
    assert (
        int(last["step"]) == int(trace[-1]["attempt"]) == int(receipts[-1]["attempt"])
    )
    for point in (0, 50, 100, 150, 200):
        checkpoint_path = folder / f"step-{point:04d}.npz"
        assert checkpoint_path.is_file() == (point in trace_attempts)
        if checkpoint_path.is_file():
            checkpoint = np.load(checkpoint_path, allow_pickle=False)
            assert int(checkpoint["step"]) == point
    atol = float(summary["config"]["force_atol"])
    for receipt in receipts:
        forward, adjoint = receipt["forward"], receipt["adjoint"]
        assert forward["success"]
        assert forward["solver_valid"]
        assert forward["accepted_force_norm"] <= atol
        assert adjoint["success"]
        assert float(adjoint["accepted_absolute_residual"]) <= float(
            adjoint["accepted_threshold"]
        )
    measured = metrics(mesh, last["u"])
    for key in ("fit_rms_mm", "normal_angle_rms_deg", "detF_min", "inverted_all_cells"):
        np.testing.assert_allclose(
            measured[key], summary["last_metrics"][key], rtol=2e-8
        )
    gradient = load(folder / "gradient-balance.json")
    if gradient["status"] == "available":
        component = np.load(folder / "gradient-components.npz", allow_pickle=False)
        assert gradient["checkpoint"]["sha256"] == sha256(folder / "last.npz")
        l2, reg = (
            dual_norm(
                component["l2_tensor_gradient"], component["active_volume_weights"]
            ),
            dual_norm(
                component["regularizer_tensor_gradient"],
                component["active_volume_weights"],
            ),
        )
        ratio = float(component["smooth_weight"]) * reg / l2
        assert gradient["components"]["sha256"] == sha256(
            folder / "gradient-components.npz"
        )
        np.testing.assert_allclose(
            component["active_volume_weights"],
            mesh["active_volume_weights"],
            rtol=0,
            atol=0,
        )
        assert float(component["smooth_weight"]) == float(
            summary["config"]["smooth_weight"]
        )
        np.testing.assert_allclose(
            ratio, gradient["smoothness_to_l2_gradient_ratio"], rtol=2e-10
        )
        for receipt in (gradient["l2_adjoint"], gradient["adjoint"]):
            assert receipt["success"]
            assert float(receipt["accepted_absolute_residual"]) <= float(
                receipt["accepted_threshold"]
            )
    else:
        ratio = None
    optimizer_initial = torch.load(folder / "optimizer-initial.pt", weights_only=False)
    assert optimizer_initial["mode"] == mode
    assert optimizer_initial["optimizer"]["state"] == {}
    np.testing.assert_allclose(
        optimizer_initial["q"].numpy(), initial["q"], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        optimizer_initial["u"], initial["u_seed"], rtol=0, atol=0
    )
    latest = torch.load(folder / "optimizer-latest.pt", weights_only=False)
    assert latest["mode"] == mode
    assert int(latest["accepted_attempt"]) == int(last["step"])
    assert 0 <= int(latest["accepted_attempt"]) <= int(latest["attempt"]) <= 200
    np.testing.assert_allclose(latest["q"].numpy(), last["q"], rtol=0, atol=0)
    np.testing.assert_allclose(latest["u"], last["u"], rtol=0, atol=0)
    rough = roughness(last, mesh, mode)
    np.testing.assert_allclose(
        rough["direct"], summary["last_metrics"]["activation_smoothness"], rtol=2e-8
    )
    return {
        "mode": mode,
        "sources": sources,
        "declared_inputs": declared_inputs,
        "budget": {
            key: summary[key]
            for key in ("attempted_updates", "optimizer_updates", "skipped_updates")
        },
        "metrics": measured,
        "roughness": rough,
        "gradient_ratio": ratio,
        "endpoint_diagnostic": gradient["status"],
        "solver_converged": bool(summary["solver_converged"]),
        "orientation_valid": bool(summary["orientation_valid"]),
        "checkpoint": summary["final_checkpoint"],
        "fresh_adam_verified": True,
    }


def main(cfg: Config) -> None:
    out = cherries.output(cfg.output / "analysis.json", mkdir=True).parent
    assert not (out / "analysis.json").exists(), out
    chain = load(cfg.chain / "chain-status.json")
    assert chain["status"] == "completed_attempt_budgets"
    assert tuple(chain["stage_sequence"]) == STAGES
    protocol = load(cfg.chain / "protocol.json")
    assert protocol["schema"] == chain["schema"]
    verify_records(chain["inputs"])
    verify_manifest(cfg.chain / "source-manifest.json")
    mesh = np.load(cfg.chain / "mesh.npz", allow_pickle=False)
    parent = Path(chain["inputs"]["final.npz"]["path"])
    with (
        np.load(parent, allow_pickle=False) as source,
        np.load(cfg.prepared, allow_pickle=False) as prepared,
    ):
        pose, pivot = source["pose"], prepared["pivot"]
    volume = pv.read(cfg.fixture / "volume.vtu")
    fixed = np.asarray(volume.point_data["IsFixed"], bool)
    names = [str(name) for name in volume.field_data["GroupName"]]
    jaw = fixed & (np.asarray(volume.point_data["GroupId"]) == names.index("Mandible"))
    boundary = np.zeros_like(volume.points)
    boundary[jaw] = (
        (volume.points[jaw] - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
        - volume.points[jaw]
    )
    results, predecessor = {}, parent
    for mode in STAGES:
        folder = cfg.chain / mode
        results[mode] = audit_stage(folder, mode, mesh, predecessor, boundary, fixed)
        predecessor = folder / "last.npz"
    result = {
        "schema": "mouthopen-four-stage-cpu-audit-v1",
        "chain": str(cfg.chain),
        "stages": results,
        "interpretation": "Strict solver receipts establish numerical convergence. Inversions and complete-boundary intersections remain physical-invalidity diagnostics.",
    }
    (out / "analysis.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    cherries.log_metrics(
        {
            f"{mode}/fit_rms_mm": row["metrics"]["fit_rms_mm"]
            for mode, row in results.items()
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
