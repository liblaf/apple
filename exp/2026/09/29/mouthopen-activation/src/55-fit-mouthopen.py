"""Fit MouthOpen contraction tensors after a full prescribed-jaw continuation."""

# ruff: noqa: C901, E402, PLR0912, PLR0915
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation

GROUP = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "mouthopen_forward_base", GROUP / "src/35-forward-pruned-mouthopen.py"
)
assert spec is not None
assert spec.loader is not None
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

import stress_study
from activation_models import project_
from experiment import Profile
from mouthopen_geometry import MouthOpenGeometry
from run_support import NUMERICAL_FAILURES, append_jsonl, metrics, save_state
from stress_physics import (
    FacePhysics,
    ForwardConvergenceError,
    NewtonCgForwardOptimizer,
)
from stress_study import StressStudy

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    output: Path = Path("55-mouthopen-fit")
    fixture: Path = GROUP / "data/30-pruned-fixture"
    forward: Path = GROUP / "data/49-forward-contact-off"
    steps: int = 200
    learning_rate: float = 0.05
    smooth_weight: float = 7.2e-6
    normal_weight: float = 1.0
    force_atol: float = 1e-10
    adjoint_rtol: float = 1e-7
    max_newton_steps: int = 100
    linear_max_steps: int = 3000
    inversion_fraction_limit: float = 0.001


def make_physics(cfg: Config, pose: np.ndarray, pivot: np.ndarray) -> type:
    class MouthOpenPhysics(FacePhysics):
        def __init__(self, fixture: Path, **options: Any) -> None:
            super().__init__(fixture, **options)
            ids = np.asarray(self.skin.point_data["GlobalPointId"], dtype=np.int64)
            self.top = ids.copy()
            self.top_t = torch.as_tensor(ids)
            self.target = np.zeros_like(self.points)
            self.target[ids] = np.asarray(self.mesh.point_data["MouthOpen"])[ids]
            assert np.isfinite(self.target).all()
            self.weights = self.surface_weights()
            self.weights_t = torch.as_tensor(self.weights)
            self.D = float(
                np.sqrt(np.sum(self.weights[:, None] * self.target[ids] ** 2))
            )
            fixed = np.asarray(self.mesh.point_data["IsFixed"], bool)
            group = list(self.mesh.field_data["GroupName"]).index("Mandible")
            jaw = fixed & (np.asarray(self.mesh.point_data["GroupId"]) == group)
            self.boundary_u = np.zeros_like(self.points)
            self.boundary_u[jaw] = (
                (self.points[jaw] - pivot)
                @ Rotation.from_rotvec(pose[:3]).as_matrix().T
                + pivot
                + pose[3:]
                - self.points[jaw]
            )
            model = self.forward.model
            assert model.collision is None
            np.testing.assert_array_equal(
                model.dof_map.fixed_indices.numpy(force=True),
                np.flatnonzero(np.repeat(fixed, 3)),
            )
            assert not np.any(fixed & np.asarray(self.mesh.point_data["IsLip"], bool))
            model.dof_map.fixed_values = (
                torch.as_tensor(self.boundary_u)
                .flatten()[model.dof_map.fixed_indices]
                .clone()
            )
            self.expected_fixed = model.dof_map.fixed_values.clone()
            self.geometry = MouthOpenGeometry(self.mesh)
            prior = self.forward.optimizer
            self.forward.optimizer = NewtonCgForwardOptimizer(
                force_atol=prior.force_atol,
                force_rtol=prior.force_rtol,
                max_steps=prior.max_steps,
                linear_rtol=prior.linear_rtol,
                linear_max_steps=prior.linear_max_steps,
                max_step_norm=prior.max_step_norm,
                initial_shift_scale=0,
                require_convergence=True,
            )
            self.forward_tolerance["newton_initial_shift_scale"] = 0
            self.material_spec["jaw_motion"] = (
                "prescribed full chin-derived MouthOpen pose; not optimized"
            )
            self.material_spec["jaw_enabled"] = True
            self.material_spec["collision_policy"] = (
                "historical contact-off model; complete FEM boundary intersections "
                "reported diagnostically, no CCD rejection or contact energy"
            )

        def solve(self, activation: torch.Tensor, seed: np.ndarray) -> torch.Tensor:
            u = super().solve(activation, seed)
            torch.testing.assert_close(
                u.flatten()[self.forward.model.dof_map.fixed_indices],
                self.expected_fixed,
                rtol=0,
                atol=1e-12,
            )
            j = self.detf(u.numpy(force=True))
            if np.count_nonzero(j <= 0) > cfg.inversion_fraction_limit * len(j):
                message = "inverted-cell count exceeds declared few-cell limit"
                raise ForwardConvergenceError(message)
            audit = self.geometry.audit(u.numpy(force=True))
            self.last_forward["geometry"] = audit
            self.last_forward["inverted_cells"] = int(np.count_nonzero(j <= 0))
            self.last_forward["minimum_J"] = float(j.min())
            self.last_forward["orientation_valid"] = bool(np.all(j > 0))
            return u

    return MouthOpenPhysics


def main(cfg: Config) -> None:
    out = cherries.output(cfg.output / "summary.json", mkdir=True).parent
    assert not (out / "summary.json").exists(), out
    forward_summary = cherries.input(cfg.forward / "summary.json")
    forward_checkpoint = cherries.input(cfg.forward / "final.npz")
    prepared = cherries.input(GROUP / "data/10-mandible/prepared.npz")
    parent = json.loads(forward_summary.read_text())
    assert (
        parent["final_checkpoint"]["sha256"]
        == base.record(forward_checkpoint)["sha256"]
    )
    assert parent["completed_pose_fraction"] == 1.0, (
        "full jaw pose required before activation fitting"
    )
    with np.load(forward_checkpoint) as z:
        seed, pose = z["displacement"].copy(), z["pose"].copy()
        assert float(z["fraction"]) == 1
    with np.load(prepared) as z:
        pivot = z["pivot"].copy()
        np.testing.assert_array_equal(pose, z["pose"])
    stress_study.FIXTURE = cfg.fixture
    stress_study.FacePhysics = make_physics(cfg, pose, pivot)
    study = StressStudy(
        activation_model="strain",
        atol=cfg.force_atol,
        adjoint_rtol=cfg.adjoint_rtol,
        max_newton_steps=cfg.max_newton_steps,
        newton_linear_max_steps=cfg.linear_max_steps,
    )
    p = study.physics
    assert len(p.ids) == 288172
    assert len(p.graph[0]) == 501313
    assert p.diff.require_convergence
    assert p.forward.optimizer.require_convergence
    study.save_geometry(out / "mesh.npz")
    base.freeze(out)
    q = torch.nn.Parameter(torch.zeros((len(p.ids), 6)))
    adam = torch.optim.Adam([q], lr=cfg.learning_rate, eps=1e-8)
    summary = {
        "schema": "mouthopen-contraction-fit-v1",
        "status": "running",
        "mode": "psd6",
        "activation_model": "strain",
        "config": cfg.model_dump(mode="json"),
        "initialization": "zero S, full prescribed jaw, fresh Adam",
        "solver_policy": "converged primal and adjoint required; finite limited inversions allowed",
        "inputs": {
            str(path.name): base.record(path)
            for path in (
                forward_summary,
                forward_checkpoint,
                prepared,
                cfg.fixture / "volume.vtu",
                cfg.fixture / "skin.vtp",
            )
        },
        "optimizer_updates": 0,
        "attempted_updates": 0,
        "skipped_updates": 0,
        "material_spec": p.material_spec,
        "inversion_limit_cells": int(cfg.inversion_fraction_limit * p.mesh.n_cells),
    }
    base.write(out / "summary.json", summary)
    result = None
    best = float("inf")
    started = time.perf_counter()
    for attempt in range(cfg.steps + 1):
        previous_q = q.detach().clone()
        previous_adam = copy.deepcopy(adam.state_dict())
        if result is not None:
            q.grad = result["gradient"].clone()
            adam.step()
            q.grad = None
            project_(q, "psd6")
        try:
            candidate = study.evaluate(
                q,
                "psd6",
                None,
                seed,
                cfg.normal_weight,
                cfg.smooth_weight,
                component_gradients=False,
            )
            assert candidate["solver_valid"]
        except NUMERICAL_FAILURES as error:
            with torch.no_grad():
                q.copy_(previous_q)
            adam.load_state_dict(previous_adam)
            for group in adam.param_groups:
                group["lr"] *= 0.5
            summary["skipped_updates"] += 1
            summary["last_failure"] = {
                "attempt": attempt,
                "type": type(error).__name__,
                "message": str(error),
                "receipt": getattr(error, "receipt", None),
            }
            append_jsonl(
                out / "proposals.jsonl", {**summary["last_failure"], "accepted": False}
            )
            if result is None:
                summary["status"] = "initial_gradient_failed"
                base.write(out / "summary.json", summary)
                break
        else:
            if result is not None:
                summary["optimizer_updates"] += 1
            result = candidate
            seed = result["u"].copy()
            row = {
                "attempt": attempt,
                "optimizer_updates": summary["optimizer_updates"],
                "elapsed_seconds": time.perf_counter() - started,
                **metrics(result),
            }
            row["orientation_valid"] = row["inverted_all_cells"] == 0
            cherries.set_step(attempt)
            cherries.log_metrics(
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (int, float))
                }
            )
            append_jsonl(out / "trace.jsonl", row)
            append_jsonl(
                out / "solver-receipts.jsonl",
                {
                    "attempt": attempt,
                    "forward": result["forward"],
                    "adjoint": result["adjoint"],
                },
            )
            save_state(out / "last.npz", q, None, result, "psd6", attempt, "strain")
            if attempt == 0 or attempt % 50 == 0 or attempt == cfg.steps:
                save_state(
                    out / f"step-{attempt:04d}.npz",
                    q,
                    None,
                    result,
                    "psd6",
                    attempt,
                    "strain",
                )
            if row["objective"] < best:
                best = row["objective"]
                save_state(
                    out / "best-objective.npz",
                    q,
                    None,
                    result,
                    "psd6",
                    attempt,
                    "strain",
                )
                summary["best_attempt"] = attempt
            summary["last_metrics"] = row
            if attempt == 0:
                summary["initial_metrics"] = row
            LOG.info(
                "MouthOpen attempt %d/%d: fit %.4f mm, normal %.4f deg, inverted %d",
                attempt,
                cfg.steps,
                row["fit_rms_mm"],
                row["normal_angle_rms_deg"],
                row["inverted_all_cells"],
            )
            torch.save(
                {
                    "q": q.detach().cpu(),
                    "optimizer": adam.state_dict(),
                    "attempt": attempt,
                    "u": seed,
                },
                out / "optimizer-latest.pt",
            )
        summary["attempted_updates"] = attempt
        summary["elapsed_seconds"] = time.perf_counter() - started
        base.write(out / "summary.json", summary)
    else:
        summary["status"] = "completed_attempt_budget"
    if result is not None:
        summary["final_checkpoint"] = base.record(out / "last.npz")
        summary["orientation_valid"] = result["inverted_all_cells"] == 0
        summary["solver_converged"] = result["solver_valid"]
        try:
            diagnostic = study.evaluate(
                q,
                "psd6",
                None,
                seed,
                cfg.normal_weight,
                cfg.smooth_weight,
                component_gradients=True,
            )
        except NUMERICAL_FAILURES as error:
            summary["gradient_diagnostic_failure"] = str(error)
        else:
            np.savez_compressed(
                out / "gradient-components.npz",
                l2_tensor_gradient=diagnostic["l2_tensor_gradient"].numpy(force=True),
                regularizer_tensor_gradient=diagnostic[
                    "regularizer_tensor_gradient"
                ].numpy(force=True),
                active_volume_weights=study.active_weights,
                smooth_weight=cfg.smooth_weight,
            )
            base.write(
                out / "gradient-balance.json",
                {
                    **metrics(diagnostic),
                    "checkpoint": summary["final_checkpoint"],
                    "components": base.record(out / "gradient-components.npz"),
                    "metric": "full symmetric S covectors; dual normalized effective-active-volume norm",
                    "ratio": "norm(eta dR/dS) / norm(dL2/dS); normal term excluded from denominator",
                    "displacement_change_from_saved_m": float(
                        np.max(np.abs(diagnostic["u"] - seed))
                    ),
                    "forward": diagnostic["forward"],
                    "l2_adjoint": diagnostic["l2_adjoint"],
                    "adjoint": diagnostic["adjoint"],
                },
            )
    base.write(out / "summary.json", summary)
    base.freeze(out)
    cherries.log_metrics(
        {
            "optimizer_updates": summary["optimizer_updates"],
            "skipped_updates": summary["skipped_updates"],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
