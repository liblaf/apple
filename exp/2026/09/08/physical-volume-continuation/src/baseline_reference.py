# ruff: noqa: FBT003, PLR0915, PT018
"""Rerun the 200-update Raw6 baseline with a physical determinant volume term."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import torch
from baseline_physics import FacePhysics, configure
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]

GROUP = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT
FIXTURE = (
    ORIGINAL / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
)
LOG = logging.getLogger(__name__)


class ProfileCometNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = FIXTURE
    output_dir: Path = GROUP / "data/20-baseline"
    steps: int = 200
    checkpoint_interval: int = 10
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    resume: Path | None = None


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")


def activation(q: np.ndarray) -> np.ndarray:
    A = np.broadcast_to(np.eye(3), (len(q), 3, 3)).copy()
    A[:, 0, 0] += q[:, 0]
    A[:, 1, 1] += q[:, 1]
    A[:, 2, 2] += q[:, 2]
    A[:, 0, 1] = A[:, 1, 0] = q[:, 3]
    A[:, 1, 2] = A[:, 2, 1] = q[:, 4]
    A[:, 0, 2] = A[:, 2, 0] = q[:, 5]
    return A


def save_state(path: Path, physics: FacePhysics, state: dict[str, Any]) -> None:
    np.savez_compressed(
        path,
        q=state["q"],
        Ainv=activation(state["q"]),
        u=state["u"],
        rest_points=physics.points,
        active_ids=physics.ids,
        step=np.array(state["step"]),
        solver_valid=np.array(state["solver_valid"]),
        physical_volume_energy=np.array(True),
    )


def archive_runtime(out: Path) -> dict[str, Any]:
    records = {}
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
        records[name] = {"path": str(p), "sha256": digest(p), "snapshot": str(target)}
    return records


def row_metrics(physics: FacePhysics, u: np.ndarray, q: np.ndarray) -> dict[str, Any]:
    pred, target = u[physics.top], physics.target[physics.top]
    error = pred - target
    area = physics.weights
    J = physics.detf(u)
    actJ = J[physics.ids]
    volume_weights = physics.volumes
    eig = np.linalg.eigvalsh(activation(q))
    tet = physics.tets[27306]
    rest = physics.points[tet]
    current = rest + u[tet]
    F = (current[1:] - current[:1]).T @ np.linalg.inv((rest[1:] - rest[:1]).T)
    stretches = np.linalg.svd(F, compute_uv=False)
    return {
        "fit_rms_mm": float(1000 * np.linalg.norm(error) / math.sqrt(len(error))),
        "motion_rms_mm": float(1000 * np.linalg.norm(pred) / math.sqrt(len(pred))),
        "area_weighted_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(area * np.sum(error**2, axis=1)))
        ),
        "area_weighted_motion_rms_mm": float(
            1000 * np.sqrt(np.sum(area * np.sum(pred**2, axis=1)))
        ),
        "target_projection": float(np.sum(pred * target) / np.sum(target**2)),
        "detF_min": float(J.min()),
        "detF_max": float(J.max()),
        "inverted_all_cells": int((J <= 0).sum()),
        "inverted_active_cells": int((actJ <= 0).sum()),
        "active_volume_weighted_RMS_detF_minus_1": float(
            np.sqrt(np.average((actJ - 1) ** 2, weights=volume_weights))
        ),
        "A_eigen_min": float(eig.min()),
        "A_eigen_max": float(eig.max()),
        "non_spd_active_cells": int((eig[:, 0] <= 0).sum()),
        "cell27306_detF": float(np.linalg.det(F)),
        "cell27306_stretch_max": float(stretches[0]),
        "cell27306_stretch_middle": float(stretches[1]),
        "cell27306_stretch_min": float(stretches[2]),
    }


def main(cfg: Config) -> None:
    assert cfg.steps == 200 and cfg.learning_rate == 0.3 and cfg.adam_eps == 0.01
    assert cfg.checkpoint_interval == 10
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Output directory must be empty: {out}"
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    validation = GROUP / "data/10-physical-volume-validation.json"
    assert validation.is_file(), (
        "The physical-volume material must pass validation first."
    )
    validation_result = json.loads(validation.read_text())
    assert validation_result["status"] == "passed"
    assert validation_result["material_source"]["sha256"] == digest(
        GROUP / "src/volume_preserving_active.py"
    )
    configure()
    physics = FacePhysics(
        cfg.fixture,
        skin_factor=0.0,
        fat_factor=1.0,
        muscle_factor=1.0,
        rtol=5e-4,
        atol=1e-10,
        adjoint_rtol=5e-4,
        soft_nu=0.49,
        fat_nu=0.49,
        fat_model="stable",
    )
    assert len(physics.ids) == 288235 and physics.n_regions == 103
    assert len(physics.points) == 228660 and len(physics.tets) == 1146517
    assert len(physics.top) == 15302
    q = torch.nn.Parameter(torch.zeros((len(physics.ids), 6)))
    optimizer = torch.optim.Adam(
        [q], lr=cfg.learning_rate, eps=cfg.adam_eps, betas=(0.9, 0.999)
    )
    seed = np.zeros_like(physics.points)
    start_step = 0
    best: dict[str, Any] | None = None
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, weights_only=False)
        assert checkpoint["energy_law"] == "physical-volume-active-strain-v1"
        assert checkpoint["material_sha256"] == digest(
            GROUP / "src/volume_preserving_active.py"
        )
        with torch.no_grad():
            q.copy_(checkpoint["q"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        seed = checkpoint["u"]
        best = checkpoint["best"]
        # Checkpoint q is an evaluated state; consume its cached exact gradient.
        q.grad = checkpoint["gradient"].to(q.device)
        optimizer.step()
        start_step = checkpoint["step"] + 1
    sources = archive_runtime(out)
    provenance = {
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "sources": sources,
        "inputs": {
            name: {
                "path": str(cfg.fixture / name),
                "sha256": digest(cfg.fixture / name),
            }
            for name in ("volume.vtu", "skin.vtp", "summary.json")
        },
        "validation": {"path": str(validation), "sha256": digest(validation)},
        "historical_reference": str(
            ORIGINAL
            / "exp/2026/09/07/face-actuation-diagnosis/data/11-historical-no-skin/final.npz"
        ),
        "energy_law": "mu/2*(||F Ainv||^2-3)-mu*(detF-1)+lambda0/2*(detF-1)^2",
        "changed_physics": "Only the determinant argument of active muscle changes from F Ainv to F.",
        "optimizer_protocol": "200 Adam updates; 201 evaluations; zero q and rest initialization; lr .3, eps .01, betas .9/.999; uniform coordinate MSE times 1e6; no regularizers or activation clamps.",
        "failure_policy": "Stop visibly on a nonfinite value or unsuccessful forward/adjoint solve; no learning-rate changes or invalid-gradient updates.",
        "inverse_stationarity_claimed": False,
        "commit_enabled": False,
        "resume": None
        if cfg.resume is None
        else {"path": str(cfg.resume), "sha256": digest(cfg.resume)},
    }
    write_json(out / "provenance.json", provenance)
    target = torch.as_tensor(physics.target[physics.top])
    trace = []
    start = time.perf_counter()
    status = "running"
    failure = None
    LOG.info(
        "Starting physical-volume baseline: 200 updates, %d controls, float64, %s.",
        q.numel(),
        torch.cuda.get_device_name(),
    )
    try:
        for step in range(start_step, cfg.steps + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            u_tensor = physics.solve(q, seed)
            forward_receipt = dict(physics.last_forward)
            u = u_tensor.detach().cpu().numpy()
            assert forward_receipt["success"], (
                f"Forward solve failed at step {step}: {forward_receipt}"
            )
            loss = (u_tensor[physics.top_t] - target).square().mean() * 1e6
            assert torch.isfinite(loss)
            loss.backward()
            adjoint = physics.diff.last_adjoint_solution
            assert adjoint is not None and adjoint.success, (
                f"Adjoint solve failed at step {step}: {adjoint}"
            )
            assert q.grad is not None and torch.isfinite(q.grad).all()
            q_numpy = q.detach().cpu().numpy().copy()
            row = {
                "step": step,
                "objective_mm2": float(loss.detach()),
                "gradient_rms": float(
                    torch.linalg.vector_norm(q.grad).detach() / math.sqrt(q.numel())
                ),
                "forward_steps": forward_receipt["steps"],
                "forward_grad_norm": forward_receipt["grad_norm"],
                "forward_success": True,
                "adjoint_success": True,
                "elapsed_s": time.perf_counter() - start,
                **row_metrics(physics, u, q_numpy),
            }
            state = {
                "step": step,
                "q": q_numpy,
                "u": u.copy(),
                "objective_mm2": row["objective_mm2"],
                "solver_valid": True,
                "metrics": row,
            }
            if best is None or state["objective_mm2"] < best["objective_mm2"]:
                best = state
            row["best_step"] = best["step"]
            row["step_s"] = time.perf_counter() - step_start
            trace.append(row)
            with (out / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(trace)
            receipt = {
                "step": step,
                "forward": forward_receipt,
                "adjoint": {"success": True, "result": str(adjoint.result)},
            }
            with (out / "solver-receipts.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt, default=str) + "\n")
            cherries.log_metrics(
                {
                    k: row[k]
                    for k in (
                        "objective_mm2",
                        "fit_rms_mm",
                        "motion_rms_mm",
                        "detF_min",
                        "inverted_all_cells",
                        "cell27306_detF",
                        "gradient_rms",
                    )
                },
                step=step,
            )
            LOG.info(
                "Step %03d / 200: fit %.4f mm, motion %.4f mm, min J %.5f, inversions %d, selected J %.5f; %.1f s.",
                step,
                row["fit_rms_mm"],
                row["motion_rms_mm"],
                row["detF_min"],
                row["inverted_all_cells"],
                row["cell27306_detF"],
                row["step_s"],
            )
            if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                save_state(out / f"step-{step:04d}.npz", physics, state)
                save_state(out / "best.npz", physics, best)
                torch.save(
                    {
                        "energy_law": "physical-volume-active-strain-v1",
                        "material_sha256": digest(
                            GROUP / "src/volume_preserving_active.py"
                        ),
                        "step": step,
                        "q": q.detach().clone(),
                        "u": u.copy(),
                        "gradient": q.grad.detach().clone(),
                        "optimizer": optimizer.state_dict(),
                        "best": best,
                    },
                    out / "optimizer-latest.pt",
                )
            if step == cfg.steps:
                save_state(out / "last.npz", physics, state)
                status = "completed_200_updates_not_stationarity_certified"
                break
            seed = u.copy()
            optimizer.step()
    except BaseException as error:
        status = "failed_before_budget_completed"
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "step": step,
            "forward_receipt": physics.last_forward,
        }
        write_json(out / "failure.json", failure)
        raise
    finally:
        if best is not None:
            save_state(out / "final.npz", physics, best)
            physics.save_mesh(out / "final.vtu", best["u"], activation(best["q"]))
        summary = {
            "status": status,
            "last_evaluated_step": None if not trace else trace[-1]["step"],
            "best_step": None if best is None else best["step"],
            "best_metrics": None if best is None else best["metrics"],
            "last_metrics": None if not trace else trace[-1],
            "failure": failure,
            "elapsed_s": time.perf_counter() - start,
            "materials": physics.material_spec,
            "forward_tolerances": physics.forward_tolerance,
            "provenance": str(out / "provenance.json"),
        }
        write_json(out / "summary.json", summary)
        for name in (
            "config.json",
            "provenance.json",
            "summary.json",
            "trace.csv",
            "solver-receipts.jsonl",
        ):
            path = out / name
            if path.exists():
                cherries.log_output(path)
        LOG.info(
            "Baseline status: %s; best step %s. Outputs: %s",
            status,
            summary["best_step"],
            out,
        )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
