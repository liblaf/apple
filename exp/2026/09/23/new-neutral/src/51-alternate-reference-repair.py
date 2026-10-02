"""Alternate exact IPC clearance projection and fixed-node volume repair."""

from __future__ import annotations

import importlib.util
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
from reference_volume_repair import untangle_tetrahedra  # noqa: E402


class Config(cherries.BaseConfig):
    output_dir: Path = GROUP / "data/reference-clearance-002"
    initial_candidate: Path = (
        GROUP / "data/reference-clearance-001/attempt-01/surface-projected.npz"
    )
    neutral_dir: Path = JOINT / "data/frozen-neutral-004"
    eyes_dir: Path = JOINT / "data/rigid-eyes-001"
    dhat_m: float = 1e-4
    projection_target_m: float = 1.001e-4
    screen_radius_m: float = 2.002e-4
    screen_stiffness_mpa: float = 0.01
    maximum_rounds: int = 16


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
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.initial_candidate.is_file(), cfg.initial_candidate
    assert cfg.dhat_m > 0
    assert cfg.projection_target_m >= cfg.dhat_m
    assert cfg.screen_radius_m > cfg.projection_target_m
    assert cfg.maximum_rounds > 0
    cfg.output_dir.mkdir(parents=True)
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
    initial = np.load(cfg.initial_candidate)
    reference = np.asarray(initial["reference_points_m"], dtype=np.float64)
    displacement = np.asarray(initial["displacement_m"], dtype=np.float64)
    assert np.array_equal(reference, np.asarray(physics.points, dtype=np.float64))
    assert np.array_equal(reference + displacement, initial["repaired_points_m"])
    tets = np.asarray(physics.tets, dtype=np.int64)
    fixed = np.asarray(physics.full_skull.geometry.fixed_global_ids, dtype=np.int64)
    rounds: list[dict[str, Any]] = []
    for round_index in range(1, cfg.maximum_rounds + 1):
        trace = cfg.output_dir / f"ipc-stencil-round-{round_index:02d}.json"
        displacement = repair._project_ipc_stencils(  # noqa: SLF001
            physics,
            displacement,
            cfg.projection_target_m,
            cfg.screen_radius_m,
            cfg.screen_stiffness_mpa,
            64,
            trace,
        )
        repaired, volume = untangle_tetrahedra(
            reference, reference + displacement, tets, fixed
        )
        displacement = repaired - reference
        contact = repair._strict_ipc_audit(  # noqa: SLF001
            physics,
            displacement,
            cfg.dhat_m,
            cfg.screen_radius_m,
            cfg.screen_stiffness_mpa,
        )
        tetrahedra = repair._tet_metrics(reference, repaired, tets)  # noqa: SLF001
        candidate = cfg.output_dir / f"alternation-{round_index:02d}.npz"
        np.savez_compressed(
            candidate,
            reference_points_m=reference,
            repaired_points_m=repaired,
            displacement_m=displacement,
        )
        row = {
            "round": round_index,
            "candidate": _record(candidate),
            "ipc_stencil_trace": _record(trace),
            "volume": volume,
            "tetrahedra": tetrahedra,
            "strict_ipc": contact,
        }
        rounds.append(row)
        write_json(cfg.output_dir / "alternation.json", rounds)
        if (
            volume["success"]
            and tetrahedra["inverted_tetrahedra"] == 0
            and contact["meets_target"]
        ):
            selected = cfg.output_dir / "selected-candidate.npz"
            np.savez_compressed(
                selected,
                reference_points_m=reference,
                repaired_points_m=repaired,
                displacement_m=displacement,
            )
            receipt = {
                "schema": "reference-clearance-alternation-v1",
                "success": True,
                "requested_dhat_m": cfg.dhat_m,
                "projection_target_m": cfg.projection_target_m,
                "initial_candidate": _record(cfg.initial_candidate),
                "selected_candidate": _record(selected),
                "rounds": rounds,
            }
            write_json(cfg.output_dir / "alternation-receipt.json", receipt)
            cherries.log_output(selected)
            cherries.log_output(cfg.output_dir / "alternation-receipt.json")
            cherries.log_metrics(
                {
                    "reference/rounds": round_index,
                    "reference/minimum_distance_m": contact[
                        "active_minimum_distance_m"
                    ],
                    "reference/detF_min": tetrahedra["detF_min"],
                }
            )
            return
    msg = "Alternation exhausted its round budget before both geometry gates passed"
    raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
