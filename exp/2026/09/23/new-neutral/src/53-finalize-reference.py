"""Package the alternated IPC-clearance and positive-volume reference."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from liblaf import cherries

HERE = Path(__file__).resolve().parent
GROUP = HERE.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
NEUTRAL = ROOT / "exp/2026/09/22/neutral-newton"
SOLVERS = ROOT / "exp/2026/09/22/solver-performance/src"
sys.path[:0] = [str(SOLVERS), str(JOINT / "src"), str(NEUTRAL / "src")]

from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_equilibrium import configure_cuda  # noqa: E402
from joint_rigid_eye_contact import build_eye_collision_physics  # noqa: E402
from profile_input_binding import bind_frozen_neutral_load  # noqa: E402


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/reference-clearance-002"
    candidate: Path = output_dir / "selected-candidate.npz"
    alternation_receipt: Path = output_dir / "alternation-receipt.json"
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    dhat_m: float = 1e-4
    projection_target_m: float = 1.001e-4
    screen_radius_m: float = 2.002e-4
    screen_stiffness_mpa: float = 0.01


def _repair_module() -> Any:
    path = HERE / "50-repair-reference.py"
    spec = importlib.util.spec_from_file_location("repair_reference", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def main(cfg: Config) -> None:
    assert cfg.output_dir.is_dir(), cfg.output_dir
    assert cfg.candidate.is_file(), cfg.candidate
    assert cfg.alternation_receipt.is_file(), cfg.alternation_receipt
    assert not (cfg.output_dir / "receipt.json").exists()
    assert cfg.dhat_m > 0
    assert cfg.projection_target_m >= cfg.dhat_m
    alternation = json.loads(cfg.alternation_receipt.read_text())
    assert alternation["schema"] == "reference-clearance-alternation-v1"
    assert alternation["success"] is True
    assert alternation["selected_candidate"]["sha256"] == sha256(cfg.candidate)
    repair = _repair_module()
    configure_cuda()
    with bind_frozen_neutral_load(
        cfg.neutral_dir,
        cfg.output_dir,
        allow_pncg_curvature_clamps=True,
        allow_isfixed_boundary=True,
        unused_inverse_sha256="0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4",
    ) as neutral:
        physics, baseline = build_eye_collision_physics(neutral, cfg.eyes_dir)
    del baseline
    archive = np.load(cfg.candidate)
    reference = np.asarray(archive["reference_points_m"], dtype=np.float64)
    repaired = np.asarray(archive["repaired_points_m"], dtype=np.float64)
    displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    assert np.array_equal(repaired - reference, displacement)
    assert np.array_equal(reference, np.asarray(physics.points, dtype=np.float64))
    fixed = np.asarray(physics.full_skull.geometry.fixed_global_ids, dtype=np.int64)
    assert not np.any(displacement[fixed])
    tetrahedra = repair._tet_metrics(  # noqa: SLF001
        reference, repaired, np.asarray(physics.tets)
    )
    assert tetrahedra["inverted_tetrahedra"] == 0
    contact = repair._strict_ipc_audit(  # noqa: SLF001
        physics,
        displacement,
        cfg.dhat_m,
        cfg.screen_radius_m,
        cfg.screen_stiffness_mpa,
    )
    assert contact["meets_target"], contact
    final = cfg.output_dir / "reference-clearance.npz"
    np.savez_compressed(
        final,
        reference_points_m=reference,
        repaired_points_m=repaired,
        displacement_m=displacement,
    )
    meshes = repair._write_meshes(physics, repaired, cfg.output_dir)  # noqa: SLF001
    receipt = {
        "schema": "reference-clearance-repair-v2",
        "success": True,
        "scope": "geometric constitutive-reference repair only; no forward equilibrium solve",
        "requested_dhat_m": cfg.dhat_m,
        "projection_target_m": cfg.projection_target_m,
        "achieved_active_minimum_distance_m": contact["active_minimum_distance_m"],
        "achieved_margin_above_requested_m": contact["active_minimum_distance_m"]
        - cfg.dhat_m,
        "strict_ipc": contact,
        "tetrahedra": tetrahedra,
        "fixed_coordinate_displacement_max_m": float(np.abs(displacement[fixed]).max()),
        "rigid_coordinates_changed": False,
        "candidate": _record(cfg.candidate),
        "alternation_receipt": _record(cfg.alternation_receipt),
        "archive": _record(final),
        "meshes": meshes,
        "inputs": {
            "neutral_manifest": _record(cfg.neutral_dir / "manifest.json"),
            "eyes_manifest": _record(cfg.eyes_dir / "manifest.json"),
        },
        "source": _record(Path(__file__)),
    }
    write_json(cfg.output_dir / "receipt.json", receipt)
    cherries.log_output(final)
    cherries.log_output(cfg.output_dir / "receipt.json")
    cherries.log_metrics(
        {
            "reference/minimum_distance_m": contact["active_minimum_distance_m"],
            "reference/margin_above_dhat_m": receipt[
                "achieved_margin_above_requested_m"
            ],
            "reference/detF_min": tetrahedra["detF_min"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
