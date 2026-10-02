"""Prepare immutable Stage 3 activation endpoints for fresh transition solves."""

# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
PREVIOUS = ROOT / "exp/2026/09/30/mouthopen-smile-collisions"
MOUTH = ROOT / "exp/2026/09/29/mouthopen-activation"
SMILE = ROOT / "exp/2026/09/21/stress-activation-loss"
sys.path.insert(0, str(SMILE / "src"))
from experiment import Profile  # noqa: E402


class Config(cherries.BaseConfig):
    output: Path = Path("10-stage3")
    previous_reference: Path = PREVIOUS / "data/50-fixed-reference"
    previous_run: Path = PREVIOUS / "data/75-fixed-activation-contact-002"
    previous_audit: Path = PREVIOUS / "data/54-fixed-contact-audit-003/summary.json"
    smile_checkpoint: Path = (
        SMILE / "data/51-visualization-checkpoints-002/l2-normal/"
        "l2-normal-rankone_fixed/last.npz"
    )
    smile_mesh: Path = (
        SMILE / "data/51-visualization-checkpoints-002/l2-normal/mesh.npz"
    )
    mouth_checkpoint: Path = (
        MOUTH / "data/70-mouthopen-four-stage/rankone_fixed/last.npz"
    )
    mouth_mesh: Path = MOUTH / "data/70-mouthopen-four-stage/mesh.npz"
    pruned_mapping: Path = MOUTH / "data/30-pruned-fixture/mapping.npz"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def check_record(row: dict[str, Any]) -> Path:
    path = Path(row["path"])
    assert path.is_file(), path
    assert sha256(path) == row["sha256"], path
    return path


def check_fixed_state(state: dict[str, np.ndarray], label: str) -> None:
    assert str(state["mode"]) == "rankone_fixed", label
    assert str(state["activation_model"]) == "strain", label
    q, axes, tensor = state["q"], state["fixed_axes"], state["S"]
    assert q.shape == (len(tensor), 1)
    assert axes.shape == (len(tensor), 3)
    assert tensor.shape == (len(tensor), 3, 3)
    assert np.isfinite(q).all()
    assert np.isfinite(axes).all()
    assert np.isfinite(tensor).all()
    np.testing.assert_allclose(np.linalg.norm(axes, axis=1), 1, rtol=0, atol=3e-14)
    np.testing.assert_allclose(
        tensor,
        q[..., None] * axes[:, :, None] * axes[:, None, :],
        rtol=3e-14,
        atol=3e-14,
    )
    np.testing.assert_allclose(tensor, tensor.transpose(0, 2, 1), rtol=0, atol=3e-14)
    assert np.linalg.eigvalsh(tensor).min() >= -3e-14


def main(cfg: Config) -> None:
    output = cherries.output(cfg.output / "summary.json", mkdir=True)
    assert not output.exists(), output
    previous_endpoints = cfg.previous_reference / "endpoints.npz"
    previous_mesh = cfg.previous_reference / "mesh.npz"
    fixture_volume = cfg.previous_reference / "fixture/volume.vtu"
    fixture_skin = cfg.previous_reference / "fixture/skin.vtp"
    seed_source = cfg.previous_run / "frames/frame-000.npz"
    input_paths = {
        "previous_reference_summary": cfg.previous_reference / "summary.json",
        "previous_mesh": previous_mesh,
        "previous_endpoints": previous_endpoints,
        "fixture_volume": fixture_volume,
        "fixture_skin": fixture_skin,
        "smile_stage3_checkpoint": cfg.smile_checkpoint,
        "smile_checkpoint_mesh": cfg.smile_mesh,
        "mouthopen_stage3_checkpoint": cfg.mouth_checkpoint,
        "mouthopen_checkpoint_mesh": cfg.mouth_mesh,
        "pruned_mapping": cfg.pruned_mapping,
        "warm_seed_frame": seed_source,
        "warm_seed_run_summary": cfg.previous_run / "summary.json",
        "warm_seed_audit": cfg.previous_audit,
    }
    for path in input_paths.values():
        cherries.input(path)
    inputs = {name: record(path) for name, path in input_paths.items()}
    previous_summary = json.loads((cfg.previous_reference / "summary.json").read_text())
    assert previous_summary["status"] == "prepared"
    assert (
        previous_summary["outputs"]["mesh.npz"]["sha256"]
        == inputs["previous_mesh"]["sha256"]
    )
    assert (
        previous_summary["outputs"]["endpoints.npz"]["sha256"]
        == inputs["previous_endpoints"]["sha256"]
    )
    previous_run = json.loads((cfg.previous_run / "summary.json").read_text())
    assert previous_run["status"] == "completed"
    assert previous_run["provenance_verified"] is True
    frame = previous_run["frames"][0]
    assert frame["index"] == 0
    assert frame["beta"] == 0.0
    assert frame["alpha"] == 1.0
    assert frame["checkpoint"]["sha256"] == inputs["warm_seed_frame"]["sha256"]
    audit = json.loads(cfg.previous_audit.read_text())
    assert audit["status"] == "verified_completed"
    assert (
        audit["numerical_summary_sha256"] == inputs["warm_seed_run_summary"]["sha256"]
    )
    assert audit["activation_endpoints_sha256"]["Smile"] == digest(
        np.load(previous_endpoints, allow_pickle=False)["S_smile"]
    )

    with np.load(previous_mesh, allow_pickle=False) as archive:
        reference_mesh = {name: archive[name].copy() for name in archive.files}
    with np.load(previous_endpoints, allow_pickle=False) as archive:
        old = {name: archive[name].copy() for name in archive.files}
    with np.load(cfg.pruned_mapping, allow_pickle=False) as archive:
        point_map = archive["new_to_original_point"].copy()
        cell_map = archive["new_to_original_cell"].copy()
    np.testing.assert_array_equal(old["new_to_original_point"], point_map)
    np.testing.assert_array_equal(old["new_to_original_cell"], cell_map)
    with np.load(cfg.smile_mesh, allow_pickle=False) as archive:
        smile_mesh = {name: archive[name].copy() for name in archive.files}
    np.testing.assert_array_equal(
        point_map[reference_mesh["tets"]], smile_mesh["tets"][cell_map]
    )
    source_rows = np.searchsorted(
        smile_mesh["active_ids"], cell_map[reference_mesh["active_ids"]]
    )
    np.testing.assert_array_equal(
        smile_mesh["active_ids"][source_rows], cell_map[reference_mesh["active_ids"]]
    )
    with np.load(cfg.mouth_mesh, allow_pickle=False) as archive:
        np.testing.assert_array_equal(reference_mesh["tets"], archive["tets"])
        np.testing.assert_array_equal(
            reference_mesh["active_ids"], archive["active_ids"]
        )

    with np.load(cfg.smile_checkpoint, allow_pickle=False) as archive:
        smile = {name: archive[name].copy() for name in archive.files}
    with np.load(cfg.mouth_checkpoint, allow_pickle=False) as archive:
        mouth = {name: archive[name].copy() for name in archive.files}
    check_fixed_state(smile, "Smile Stage 3")
    check_fixed_state(mouth, "MouthOpen Stage 3")
    assert smile["S"].shape[0] == len(smile_mesh["active_ids"])
    assert mouth["S"].shape[0] == len(reference_mesh["active_ids"])
    assert bool(mouth["solver_valid"])
    assert not bool(smile["solver_valid"])
    smile_s, smile_q, smile_axes = (
        smile["S"][source_rows],
        smile["q"][source_rows],
        smile["fixed_axes"][source_rows],
    )
    mouth_s, mouth_q, mouth_axes = mouth["S"], mouth["q"], mouth["fixed_axes"]
    for tensor in (smile_s, mouth_s):
        assert tensor.shape == (len(reference_mesh["active_ids"]), 3, 3)
    np.testing.assert_allclose(
        smile_s,
        smile_q[..., None] * smile_axes[:, :, None] * smile_axes[:, None, :],
        rtol=3e-14,
        atol=3e-14,
    )

    mesh_out = cherries.output(cfg.output / "mesh.npz")
    endpoints_out = cherries.output(cfg.output / "endpoints.npz")
    seed_out = cherries.output(cfg.output / "seed.npz")
    shutil.copy2(previous_mesh, mesh_out)
    assert sha256(mesh_out) == inputs["previous_mesh"]["sha256"]
    np.savez_compressed(
        endpoints_out,
        S_smile=smile_s,
        S_mouthopen=mouth_s,
        q_smile=smile_q,
        q_mouthopen=mouth_q,
        fixed_axes_smile=smile_axes,
        fixed_axes_mouthopen=mouth_axes,
        pose_mouthopen=old["pose_mouthopen"],
        pivot=old["pivot"],
        smile_source_rows=source_rows,
        new_to_original_point=old["new_to_original_point"],
        new_to_original_cell=old["new_to_original_cell"],
    )
    with np.load(seed_source, allow_pickle=False) as archive:
        seed_u, seed_full, seed_pose = (
            archive["u"].copy(),
            archive["u_full"].copy(),
            archive["pose"].copy(),
        )
    assert seed_u.shape == reference_mesh["rest_points"].shape
    assert np.isfinite(seed_u).all()
    assert np.isfinite(seed_full).all()
    np.testing.assert_array_equal(seed_full[: len(seed_u)], seed_u)
    np.testing.assert_allclose(seed_pose, old["pose_mouthopen"], rtol=0, atol=1e-12)
    np.savez_compressed(seed_out, u=seed_u, u_full=seed_full, pose=seed_pose)

    fields = {
        "S_smile": digest(smile_s),
        "S_mouthopen": digest(mouth_s),
        "q_smile": digest(smile_q),
        "q_mouthopen": digest(mouth_q),
        "fixed_axes_smile": digest(smile_axes),
        "fixed_axes_mouthopen": digest(mouth_axes),
        "pose_mouthopen": digest(old["pose_mouthopen"]),
        "pivot": digest(old["pivot"]),
        "smile_source_rows": digest(source_rows),
        "new_to_original_point": digest(old["new_to_original_point"]),
        "new_to_original_cell": digest(old["new_to_original_cell"]),
        "seed_u": digest(seed_u),
        "seed_u_full": digest(seed_full),
        "seed_pose": digest(seed_pose),
    }
    summary = {
        "schema": "stage3-mouthopen-smile-preparation-v1",
        "status": "prepared",
        "inputs": inputs,
        "outputs": {
            name: record(path)
            for name, path in {
                "mesh.npz": mesh_out,
                "endpoints.npz": endpoints_out,
                "seed.npz": seed_out,
            }.items()
        },
        "fixture": "external repaired fixed-reference fixture; no fixture copy is made",
        "activation_policy": "exact Stage 3 rankone_fixed active-strain S tensors; B=I+S; no inverse fit",
        "smile_historical_source": {
            "solver_valid": False,
            "meaning": "saved Stage 3 tensor accepted only as the control for a fresh forward re-equilibration",
        },
        "mouthopen_historical_source": {"solver_valid": True},
        "mapping": {
            "original_smile_active_cells": len(smile_mesh["active_ids"]),
            "pruned_active_cells": len(source_rows),
            "removed_fully_fixed_active_cells": len(smile_mesh["active_ids"])
            - len(source_rows),
            "exact_point_and_cell_identity_verified": True,
        },
        "warm_seed": {
            "source": inputs["warm_seed_frame"],
            "audit": inputs["warm_seed_audit"],
            "meaning": "audited Stage 4 contact geometry warm seed only; no Stage 4 activation is used in the new controls; all Stage 3 states require fresh re-equilibration",
        },
        "field_sha256": fields,
    }
    output.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report = cherries.output("../docs/10-stage3-preparation.md", mkdir=True)
    report.write_text(
        "# Stage 3 transition preparation\n\n"
        "This preparation reuses the repaired fixed-reference mesh and external fixture, then writes Stage 3 fixed-axis endpoint tensors in the pruned active-cell order. "
        "The Smile source is a historical finite-budget checkpoint with `solver_valid=false`; it is a control source only and needs a fresh forward re-equilibration. "
        "`seed.npz` preserves the audited Stage 4 contact geometry solely as a displacement warm seed.\n"
    )
    cherries.log_metrics(
        {
            "active_cells": len(source_rows),
            "removed_active_cells": len(smile_mesh["active_ids"]) - len(source_rows),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
