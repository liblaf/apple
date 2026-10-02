"""Shared objective and saved-state evidence for the activation study."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from activation_controls import (
    baseline_z,
    control_c,
    learned_axis_raw6,
    learned_axis_z,
    smoothness,
    tensile_z,
)
from study_physics import FacePhysics

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
MU = 0.03 / (2 * 1.49)
LENGTH_M = 0.005


class StudySolveError(RuntimeError):
    """A declared inner solve failed; its optimizer trial is invalid."""


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, Path):
            return str(item)
        raise TypeError(type(item))

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=convert) + "\n"
    )
    temporary.replace(path)


def write_trace(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def archive_runtime(out: Path, fixture: Path) -> dict:
    sources = {}
    for name, module in tuple(sys.modules.items()):
        source = getattr(module, "__file__", None)
        if not source or not source.endswith(".py"):
            continue
        p = Path(source).resolve()
        local = p.parent == GROUP / "src"
        if not (local or name.startswith(("liblaf.apple", "liblaf.peach"))):
            continue
        relative = (
            Path("experiment") / p.name
            if local
            else Path("runtime") / Path(*name.split(".")).with_suffix(".py")
        )
        target = out / "sources" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, target)
        sources[name] = {"path": str(p), "sha256": digest(p), "snapshot": str(target)}
    return {
        "sources": sources,
        "inputs": {
            name: {"path": str(fixture / name), "sha256": digest(fixture / name)}
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=ROOT, text=True
        ).strip(),
        "commit_enabled": False,
    }


def control_z(q: torch.Tensor, model: str) -> torch.Tensor:
    assert model in {"raw6", "tensor", "learned-axis"}
    if model == "raw6":
        return baseline_z(q)
    if model == "learned-axis":
        return learned_axis_z(q)
    return tensile_z(q)


class Objective:
    def __init__(
        self,
        physics: FacePhysics,
        *,
        smoothness_weight: float = 0.0,
        smoothness_field: str = "Z",
    ):
        self.physics = physics
        self.model = physics.activation_model
        self.smoothness_weight = smoothness_weight
        self.smoothness_field = smoothness_field
        assert smoothness_field in {"C", "Z"}
        assert smoothness_field != "C" or self.model in {"raw6", "learned-axis"}
        assert smoothness_weight >= 0
        self.i = torch.as_tensor(physics.graph[0], dtype=torch.long)
        self.j = torch.as_tensor(physics.graph[1], dtype=torch.long)
        self.edge_weight = torch.as_tensor(physics.graph[2])
        self.volume = float(physics.volumes.sum())
        self.target = torch.as_tensor(physics.target[physics.top])
        assert abs(physics.material_spec["muscle_mu_code_MPa"] - MU) < 1e-15
        assert physics.material_spec["skin_E_MPa"] == 0

    def penalty(self, z: torch.Tensor) -> torch.Tensor:
        return smoothness(z, self.i, self.j, self.edge_weight, LENGTH_M, self.volume)

    def __call__(
        self, q: torch.Tensor, seed: np.ndarray, *, component_gradients=False
    ) -> dict:
        tick = time.perf_counter()
        q.grad = None
        if self.model == "raw6":
            active = q
        elif self.model == "learned-axis":
            active = learned_axis_raw6(q)
        else:
            active = MU * tensile_z(q)
        u_tensor = self.physics.solve(active, seed)
        forward = dict(self.physics.last_forward)
        if not forward["success"]:
            raise StudySolveError(f"Forward equilibrium failed: {forward}")
        data = (u_tensor[self.physics.top_t] - self.target).square().mean() * 1e6
        assert torch.isfinite(data)
        data.backward()
        adjoint_solution = self.physics.diff.last_adjoint_solution
        assert adjoint_solution is not None
        if not adjoint_solution.success:
            raise StudySolveError(f"Adjoint solve failed: {adjoint_solution}")
        adjoint = self.physics.check_adjoint()
        assert q.grad is not None and torch.isfinite(q.grad).all()
        fit_gradient = q.grad.detach().clone()
        # This fresh coordinate graph avoids a second mechanics adjoint.
        z = control_z(q, self.model)
        c = control_c(q, self.model) if self.model != "tensor" else None
        smooth_z = self.penalty(z)
        smooth_c = self.penalty(c) if c is not None else None
        smooth = smooth_c if self.smoothness_field == "C" else smooth_z
        smooth_gradient = None
        if component_gradients:
            smooth_gradient = torch.autograd.grad(smooth, q, retain_graph=True)[
                0
            ].detach()
        if self.smoothness_weight:
            (self.smoothness_weight * smooth).backward()
        assert torch.isfinite(q.grad).all()
        result = {
            "objective_mm2": float(data.detach()),
            "objective_total": float(
                data.detach() + self.smoothness_weight * smooth.detach()
            ),
            "smoothness": float(smooth.detach()),
            "smoothness_weight": self.smoothness_weight,
            "smoothness_Z": float(smooth_z.detach()),
            "fit_gradient_rms": float(fit_gradient.square().mean().sqrt()),
            "gradient_rms": float(q.grad.square().mean().sqrt()),
            "regularizer_gradient_rms": float(
                (q.grad - fit_gradient).square().mean().sqrt()
            ),
            "u": u_tensor.detach().cpu().numpy().copy(),
            "Z": z.detach().cpu().numpy().copy(),
            "forward": forward,
            "adjoint": adjoint,
            "objective_s": time.perf_counter() - tick,
        }
        if c is not None:
            result["C"] = c.detach().cpu().numpy().copy()
            result["smoothness_C"] = float(smooth_c.detach())
        if component_gradients:
            result["fit_gradient"] = fit_gradient
            result["smooth_gradient"] = smooth_gradient
        return result


def save_state(path: Path, physics: FacePhysics, state: dict) -> None:
    extra = {"C": state["C"]} if "C" in state else {}
    np.savez_compressed(
        path,
        step=np.array(state["step"]),
        q=state["q"],
        Z=state["Z"],
        u=state["u"],
        active_ids=physics.ids,
        rest_points=physics.points,
        solver_valid=np.array(True),
        model=np.array(physics.activation_model),
        **extra,
    )


def make_adam(
    q: torch.Tensor, settings: dict, learning_rate: float
) -> torch.optim.Adam:
    return torch.optim.Adam(
        [q],
        lr=learning_rate,
        eps=settings["adam_eps"],
        betas=tuple(settings["betas"]),
        weight_decay=0,
        amsgrad=False,
        maximize=False,
        foreach=None,
        fused=None,
    )


def archived_initial_state(
    physics: FacePhysics, record: dict
) -> tuple[torch.Tensor, np.ndarray]:
    path = Path(record["path"])
    assert digest(path) == record["sha256"]
    with np.load(path) as state:
        assert bool(state["solver_valid"])
        assert np.array_equal(state["active_ids"], physics.ids)
        assert np.array_equal(state["rest_points"], physics.points)
        return torch.as_tensor(state["q"].copy()), state["u"].copy()


def control_metrics(q: np.ndarray, c: np.ndarray, model: str) -> dict:
    if model == "learned-axis":
        strength = (q * q).sum(axis=1)
        shortening = strength / (1 + strength)
        return {
            f"shortening_fraction_p{percentile}": float(
                np.percentile(shortening, percentile)
            )
            for percentile in (0, 50, 90, 99, 100)
        }
    assert model == "raw6"
    values = np.linalg.eigvalsh(c + np.eye(3))
    return {
        "B_eigenvalue_min": float(values.min()),
        "B_nonpositive_eigenvalue_cells": int((values[:, 0] <= 0).sum()),
    }


def volume_metrics(physics: FacePhysics, u: np.ndarray) -> dict:
    j = physics.detf(u)
    active = j[physics.ids]
    return {
        "detF_min": float(j.min()),
        "detF_max": float(j.max()),
        "inverted_all_cells": int((j <= 0).sum()),
        "inverted_active_cells": int((active <= 0).sum()),
        "active_volume_weighted_rms_detF_minus_1": float(
            np.sqrt(np.average((active - 1) ** 2, weights=physics.volumes))
        ),
    }
