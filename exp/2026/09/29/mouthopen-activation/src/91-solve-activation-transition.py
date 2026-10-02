# Copyright (c) 2026 liblaf
# ruff: noqa: C901, E402, PLR0915
"""Re-equilibrate a tensor activation and prescribed jaw transition."""

from __future__ import annotations

import importlib.util
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
spec = importlib.util.spec_from_file_location(
    "transition_reuse", GROUP / "src/56-fit-mouthopen-reuse.py"
)
assert spec is not None
assert spec.loader is not None
ADAPTER = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ADAPTER
spec.loader.exec_module(ADAPTER)
FIT = ADAPTER.FIT
BASE = FIT.base
from experiment import Profile
from stress_physics import ForwardConvergenceError, configure

LOG = logging.getLogger(__name__)


class Config(FIT.Config):
    output: Path = Path("91-smile-mouthopen-transition")
    frames: int = 121
    initial_max_newton_steps: int = 1000
    initial_seed: Path | None = None
    reuse_shift_force_ratio: float = 0.0
    minimum_alpha_step: float = 1e-5
    max_subdivisions: int = 6
    smile: Path = (
        ROOT / "exp/2026/09/21/stress-activation-loss/data/"
        "51-visualization-checkpoints-002/l2-normal/l2-normal-rankone_learned/last.npz"
    )
    smile_mesh: Path = (
        ROOT
        / "exp/2026/09/28/activation-smoothness-continuation/data/10-sweep/mesh.npz"
    )
    mouthopen: Path = GROUP / "data/70-mouthopen-four-stage/rankone_learned/last.npz"


@torch.no_grad()
def main(cfg: Config) -> None:
    assert cfg.frames >= 3
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    (out / "frames").mkdir()
    inputs = {
        "smile": cfg.smile,
        "smile_mesh": cfg.smile_mesh,
        "smile_checkpoint_mesh": cfg.smile.parents[1] / "mesh.npz",
        "mouthopen": cfg.mouthopen,
        "mouth_mesh": GROUP / "data/70-mouthopen-four-stage/mesh.npz",
        "pose": GROUP / "data/10-mandible/prepared.npz",
        "volume": cfg.fixture / "volume.vtu",
        "skin": cfg.fixture / "skin.vtp",
        "mapping": cfg.fixture / "mapping.npz",
        "harmonic_weight": GROUP / "data/35-forward-pruned-002/harmonic-weight.npz",
    }
    receipts = {name: BASE.record(path) for name, path in inputs.items()}
    BASE.write(out / "inputs.json", receipts)
    cherries.log_input(out / "inputs.json")
    with np.load(cfg.smile) as z:
        assert str(z["mode"]) == "rankone_learned"
        assert str(z["activation_model"]) == "strain"
        smile_s, smile_u = z["S"].copy(), z["u"].copy()
        smile_valid = bool(z["solver_valid"])
    with np.load(cfg.mouthopen) as z:
        assert str(z["mode"]) == "rankone_learned"
        assert str(z["activation_model"]) == "strain"
        assert bool(z["solver_valid"])
        mouth_s, mouth_u = z["S"].copy(), z["u"].copy()
    with np.load(cfg.smile_mesh) as z:
        original_active = z["active_ids"].copy()
        original_points = z["rest_points"].copy()
        original_tets = z["tets"].copy()
    with np.load(inputs["smile_checkpoint_mesh"]) as z:
        np.testing.assert_array_equal(z["rest_points"], original_points)
        np.testing.assert_array_equal(z["tets"], original_tets)
        np.testing.assert_array_equal(z["active_ids"], original_active)
    with np.load(inputs["pose"]) as z:
        full_pose, pivot = z["pose"].copy(), z["pivot"].copy()
    with np.load(inputs["mapping"]) as z:
        point_map = z["new_to_original_point"].copy()
    with np.load(inputs["harmonic_weight"]) as z:
        weight = z["weight"].copy()
    configure()
    ADAPTER.install_reuse_policy()
    original_policy = ADAPTER.stress_physics._newton_policy  # noqa: SLF001

    def policy() -> tuple:
        cached, error_type, solve_reuse, mean_edge = original_policy()

        def solve_through_convergence(*args: object, **kwargs: object) -> object:
            kwargs["reuse_shift_force_ratio"] = cfg.reuse_shift_force_ratio
            return solve_reuse(*args, **kwargs)

        return cached, error_type, solve_through_convergence, mean_edge

    ADAPTER.stress_physics._newton_policy = policy  # noqa: SLF001
    physics = FIT.make_physics(cfg, np.zeros(6), pivot)(
        cfg.fixture,
        activation_model="strain",
        atol=cfg.force_atol,
        max_newton_steps=cfg.max_newton_steps,
        newton_linear_max_steps=cfg.linear_max_steps,
    )
    np.testing.assert_array_equal(physics.points, original_points[point_map])
    cell_map = np.asarray(physics.mesh.cell_data["OriginalCellId"], dtype=np.int64)
    np.testing.assert_array_equal(point_map[physics.tets], original_tets[cell_map])
    new_original_active = cell_map[physics.ids]
    source_rows = np.searchsorted(original_active, new_original_active)
    np.testing.assert_array_equal(original_active[source_rows], new_original_active)
    assert smile_s.shape == (len(original_active), 3, 3)
    smile_s = smile_s[source_rows]
    assert mouth_s.shape == smile_s.shape == (len(physics.ids), 3, 3)
    for tensor in (smile_s, mouth_s):
        np.testing.assert_allclose(tensor, tensor.transpose(0, 2, 1), atol=1e-12)
        assert np.linalg.eigvalsh(tensor).min() > -1e-12
    with np.load(inputs["mouth_mesh"]) as z:
        np.testing.assert_array_equal(z["rest_points"], physics.points)
        np.testing.assert_array_equal(z["tets"], physics.tets)
        np.testing.assert_array_equal(z["active_ids"], physics.ids)
        skin_ids = z["skin_ids"].copy()
        skin_weights = z["skin_vertex_weights"].copy()
    shutil.copy2(inputs["mouth_mesh"], out / "mesh.npz")
    np.savez_compressed(
        out / "endpoints.npz",
        S_smile=smile_s,
        S_mouthopen=mouth_s,
        pose_mouthopen=full_pose,
        pivot=pivot,
        smile_source_rows=source_rows,
        new_to_original_point=point_map,
        new_to_original_cell=cell_map,
    )
    fixed = np.asarray(physics.mesh.point_data["IsFixed"], dtype=bool)
    group = list(physics.mesh.field_data["GroupName"]).index("Mandible")
    jaw = fixed & (np.asarray(physics.mesh.point_data["GroupId"]) == group)
    np.testing.assert_allclose(weight[jaw], 1, atol=1e-8, rtol=0)
    np.testing.assert_allclose(weight[fixed & ~jaw], 0, atol=1e-8, rtol=0)
    smile_seed = smile_u[point_map].copy()
    np.testing.assert_allclose(smile_seed[fixed], 0, atol=1e-12, rtol=0)
    seed_provenance = None
    initialization_seed = smile_seed
    if cfg.initial_seed is not None:
        previous_folder = cfg.initial_seed.parent
        previous_summary = json.loads((previous_folder / "summary.json").read_text())
        assert previous_summary["status"] == "failed"
        assert not previous_summary["frames"]
        assert "smile_reequilibration" not in previous_summary
        assert previous_summary["failure"]["type"] == "ForwardConvergenceError"
        for name, receipt in receipts.items():
            assert previous_summary["inputs"][name] == receipt
        previous_sources = json.loads(
            (previous_folder / "source-manifest.json").read_text()
        )
        for item in previous_sources.values():
            assert BASE.record(Path(item["path"]))["sha256"] == item["sha256"]
            if Path(item["source"]) != Path(__file__).resolve():
                assert BASE.record(Path(item["source"]))["sha256"] == item["sha256"]
        with np.load(cfg.initial_seed) as z:
            initialization_seed = z["u"].copy()
        assert initialization_seed.shape == smile_seed.shape
        assert np.isfinite(initialization_seed).all()
        np.testing.assert_allclose(initialization_seed[fixed], 0, atol=1e-12, rtol=0)
        seed_provenance = {
            "seed": BASE.record(cfg.initial_seed),
            "parent_summary": BASE.record(previous_folder / "summary.json"),
            "meaning": "last finite initial Smile solver iterate from failed force-gate run; warm start only",
        }
    summary = {
        "schema": "activation-expression-transition-v1",
        "status": "initializing",
        "config": cfg.model_dump(mode="json"),
        "inputs": receipts,
        "mapping": {
            "original_active_cells": len(original_active),
            "remaining_active_cells": len(physics.ids),
            "removed_active_fully_fixed_cells": len(original_active) - len(physics.ids),
            "exact_point_and_cell_identity_verified": True,
        },
        "interpolation": "S(alpha)=(1-alpha) S_smile+alpha S_mouthopen; B=I+S; full symmetric tensors; no rank-one projection",
        "jaw": "rotation exp(alpha*rotvec) about the saved pivot, linear translation alpha*t; prescribed on mandible fixed vertices",
        "timing": "alpha=(1-cos(pi*frame/(frames-1)))/2; one re-equilibrated state per rendered transition frame",
        "solver": "same strict MouthOpen force gate, corrected physical det(F), no skin, contact off, Newton shift reuse; no inverse fitting",
        "smile_source_solver_valid": smile_valid,
        "initial_seed": seed_provenance,
        "reuse_shift_force_ratio": cfg.reuse_shift_force_ratio,
        "physical_validity_claim": False,
        "frames": [],
        "substeps": [],
        "failures": [],
        "material_spec": physics.material_spec,
    }
    BASE.write(out / "summary.json", summary)
    BASE.freeze(out)
    s0, s1 = torch.as_tensor(smile_s), torch.as_tensor(mouth_s)
    started = time.perf_counter()

    def solve(alpha: float, seed: np.ndarray) -> tuple[np.ndarray, dict]:
        pose = alpha * full_pose
        boundary = np.zeros_like(physics.points)
        boundary[jaw] = (
            (physics.points[jaw] - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
            + pivot
            + pose[3:]
            - physics.points[jaw]
        )
        model = physics.forward.model
        physics.boundary_u = boundary
        model.dof_map.fixed_values = (
            torch.as_tensor(boundary).flatten()[model.dof_map.fixed_indices].clone()
        )
        physics.expected_fixed = model.dof_map.fixed_values.clone()
        seed = seed.copy()
        seed[fixed] = boundary[fixed]
        value = (
            physics.solve((1 - alpha) * s0 + alpha * s1, seed).numpy(force=True).copy()
        )
        receipt = dict(physics.last_forward)
        assert receipt["solver_valid"]
        assert receipt["accepted_force_norm"] <= cfg.force_atol
        receipt["search_shift_policy"] = "reuse"
        receipt["reuse_shift_force_ratio"] = cfg.reuse_shift_force_ratio
        return value, receipt

    def carry(seed: np.ndarray, a: float, b: float) -> np.ndarray:
        old_pose, new_pose = a * full_pose, b * full_pose
        old_r = Rotation.from_rotvec(old_pose[:3]).as_matrix()
        new_r = Rotation.from_rotvec(new_pose[:3]).as_matrix()
        x = physics.points + seed
        carried = (x - pivot - old_pose[3:]) @ old_r @ new_r.T + pivot + new_pose[3:]
        return seed + weight[:, None] * (carried - x)

    def advance(
        seed: np.ndarray, a: float, b: float, depth: int = 0
    ) -> tuple[np.ndarray, dict]:
        try:
            return solve(b, carry(seed, a, b))
        except ForwardConvergenceError as error:
            summary["failures"].append(
                {
                    "from_alpha": a,
                    "target_alpha": b,
                    "depth": depth,
                    "message": str(error),
                    "receipt": getattr(error, "receipt", None),
                }
            )
            BASE.write(out / "summary.json", summary)
            if depth >= cfg.max_subdivisions or b - a < 2 * cfg.minimum_alpha_step:
                raise
            middle = (a + b) / 2
            LOG.warning("Subdividing failed alpha interval %.8f -> %.8f", a, b)
            u_mid, receipt = advance(seed, a, middle, depth + 1)
            path = out / f"substep-{len(summary['substeps']):03d}.npz"
            np.savez_compressed(path, u=u_mid, alpha=middle, pose=middle * full_pose)
            summary["substeps"].append(
                {
                    "alpha": middle,
                    "checkpoint": BASE.record(path),
                    "diagnostics": receipt,
                }
            )
            return advance(u_mid, middle, b, depth + 1)

    def change_metrics(u: np.ndarray, ref: np.ndarray) -> dict:
        delta = u - ref
        return {
            "maximum_vertex_change_mm": float(
                1000 * np.linalg.norm(delta, axis=1).max()
            ),
            "skin_rms_change_mm": float(
                1000 * np.sqrt(np.sum(skin_weights[:, None] * delta[skin_ids] ** 2))
            ),
        }

    try:
        LOG.info("Re-equilibrating historical Smile on the pruned mesh")
        physics.forward.optimizer.max_steps = cfg.initial_max_newton_steps
        physics.forward_tolerance["max_newton_steps"] = cfg.initial_max_newton_steps
        current, initial_receipt = solve(0.0, initialization_seed)
        physics.forward.optimizer.max_steps = cfg.max_newton_steps
        physics.forward_tolerance["max_newton_steps"] = cfg.max_newton_steps
        summary["smile_reequilibration"] = {
            **change_metrics(current, smile_seed),
            "diagnostics": initial_receipt,
        }
        BASE.write(out / "summary.json", summary)
        LOG.info("Verifying strict MouthOpen endpoint replay")
        mouth_replayed, replay_receipt = solve(1.0, mouth_u)
        summary["mouthopen_replay"] = {
            **change_metrics(mouth_replayed, mouth_u),
            "diagnostics": replay_receipt,
        }
        summary["status"] = "running"
        last_alpha = 0.0
        for index in range(cfg.frames):
            alpha = float((1 - np.cos(np.pi * index / (cfg.frames - 1))) / 2)
            if index == 0:
                receipt = initial_receipt
            else:
                current, receipt = advance(current, last_alpha, alpha)
            path = out / "frames" / f"frame-{index:03d}.npz"
            np.savez_compressed(path, u=current, alpha=alpha, pose=alpha * full_pose)
            summary["frames"].append(
                {
                    "index": index,
                    "alpha": alpha,
                    "checkpoint": BASE.record(path),
                    "diagnostics": receipt,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            summary["elapsed_seconds"] = time.perf_counter() - started
            BASE.write(out / "summary.json", summary)
            cherries.set_step(index)
            cherries.log_metrics(
                {
                    "alpha": alpha,
                    "force_norm": receipt["accepted_force_norm"],
                    "inverted_cells": receipt["inverted_cells"],
                }
            )
            LOG.info(
                "Frame %d/%d alpha %.6f force %.3g inverted %d",
                index + 1,
                cfg.frames,
                alpha,
                receipt["accepted_force_norm"],
                receipt["inverted_cells"],
            )
            last_alpha = alpha
        summary["mouthopen_endpoint_difference"] = change_metrics(current, mouth_u)
        summary["status"] = "completed"
    except Exception as error:
        np.savez_compressed(
            out / "failed-solver-state.npz",
            u=physics.forward.state.u.numpy(force=True),
        )
        summary["status"] = "failed"
        summary["failure"] = {
            "type": type(error).__name__,
            "message": str(error),
            "receipt": getattr(error, "receipt", None),
        }
        BASE.write(out / "summary.json", summary)
        raise
    summary["elapsed_seconds"] = time.perf_counter() - started
    for item in receipts.values():
        assert BASE.record(Path(item["path"]))["sha256"] == item["sha256"]
    for item in json.loads((out / "source-manifest.json").read_text()).values():
        assert BASE.record(Path(item["source"]))["sha256"] == item["sha256"]
    summary["outputs"] = {
        name: BASE.record(out / name)
        for name in ("mesh.npz", "endpoints.npz", "source-manifest.json")
    }
    BASE.write(out / "summary.json", summary)
    LOG.info("Completed all %d equilibrium frames", cfg.frames)


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
