"""Matched fixed-budget face inversions with bounded, fiber-free active stress."""

# ruff: noqa: EM102, PLR0915, TRY003

from __future__ import annotations

import csv
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from face_physics import FacePhysics, configure
from tensor_controls import eigenvalues, matrices, project, raw6_matrices

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
REPO = HERE.parents[4]
OLD = HERE.parent / "face-actuation-diagnosis"
QREF = 3.0 * 0.03 / (2.0 * (1.0 + 0.49))
COMPLETED = False
logger = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = OLD / "data/12-historical-fixture"
    output_dir: Path = cherries.output("20-psd", mkdir=True)
    model: Literal["tensor", "raw6"] = "tensor"
    steps: int = 64
    checkpoint_interval: int = 16
    learning_rate: float = 0.3
    adam_eps: float = 0.01
    stress_reference_mpa: float = QREF
    stress_cap_mpa: float = 10.0 * QREF
    smoothness_weight: float = 0.0
    magnitude_weight: float = 0.0
    rank_weight: float = 0.0
    smooth_length_m: float = 0.005
    record_component_gradients: bool = False


def write_json(path: Path, value: Any) -> None:
    def convert(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item))

    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False, default=convert)
        + "\n"
    )
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def archive(out: Path, cfg: Config) -> dict[str, Any]:
    sources = out / "sources"
    sources.mkdir()
    for directory, target in (
        (HERE / "src", sources / "experiment"),
        (REPO / "src/liblaf/apple", sources / "liblaf/apple"),
    ):
        shutil.copytree(
            directory, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    inputs = {
        name: {
            "path": str((cfg.fixture / name).resolve()),
            "sha256": sha256(cfg.fixture / name),
        }
        for name in ("volume.vtu", "skin.vtp", "summary.json")
    }
    source_hashes = {
        str(p.relative_to(sources)): sha256(p) for p in sorted(sources.rglob("*.py"))
    }
    receipt = {
        "sources": source_hashes,
        "inputs": inputs,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "command": [sys.executable, *sys.argv],
        "cwd": str(Path.cwd()),
        "commit_enabled": False,
    }
    write_json(out / "provenance.json", receipt)
    return receipt


class Objective:
    def __init__(self, physics: FacePhysics, cfg: Config) -> None:
        self.physics, self.cfg = physics, cfg
        self.i = torch.as_tensor(physics.graph[0], dtype=torch.long)
        self.j = torch.as_tensor(physics.graph[1], dtype=torch.long)
        self.edge_weight = torch.as_tensor(physics.graph[2])
        self.mass = torch.as_tensor(physics.volumes / physics.volumes.sum())
        self.volume = float(physics.volumes.sum())
        self.target = torch.as_tensor(physics.target[physics.top])

    def regularizers(
        self, q: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cfg.model == "raw6":
            weights = q.new_tensor((1, 1, 1, 2, 2, 2))
        else:
            weights = torch.ones(6, device=q.device, dtype=q.dtype)
        delta = q[self.i] - q[self.j]
        smooth = (
            self.cfg.smooth_length_m**2
            * (self.edge_weight[:, None] * weights * delta.square()).sum()
            / self.volume
        )
        magnitude = (self.mass * (weights * q.square()).sum(-1)).sum()
        rank = (
            self.mass * (q[:, :3].sum(-1).square() - (weights * q.square()).sum(-1))
        ).sum()
        return smooth, magnitude, rank

    def __call__(self, q: torch.Tensor, seed: np.ndarray) -> dict[str, Any]:
        q.grad = None
        active = (
            self.cfg.stress_reference_mpa * matrices(q)
            if self.cfg.model == "tensor"
            else q
        )
        u = self.physics.solve(active, seed)
        if not self.physics.last_forward["success"]:
            raise RuntimeError(
                f"forward equilibrium failed: {self.physics.last_forward}"
            )
        data = (u[self.physics.top_t] - self.target).square().mean() * 1e6
        # The adjoint is evaluated once. Explicit regularizer gradients are added afterward.
        data.backward()
        adjoint = self.physics.diff.last_adjoint_solution
        if (
            adjoint is None
            or not adjoint.success
            or q.grad is None
            or not torch.isfinite(q.grad).all()
        ):
            raise RuntimeError(
                f"adjoint failed or returned a nonfinite gradient: {adjoint}"
            )
        fit_gradient = q.grad.detach().clone()
        smooth, magnitude, rank = self.regularizers(q)
        regularizer = (
            self.cfg.smoothness_weight * smooth
            + self.cfg.magnitude_weight * magnitude
            + self.cfg.rank_weight * rank
        )
        gradients: dict[str, torch.Tensor] = {}
        if self.cfg.record_component_gradients:
            for name, value in (
                ("smoothness", smooth),
                ("magnitude", magnitude),
                ("rank", rank),
            ):
                gradients[name] = torch.autograd.grad(value, q, retain_graph=True)[
                    0
                ].detach()
        regularizer.backward()
        assert q.grad is not None
        assert torch.isfinite(q.grad).all()
        receipt = {
            "data_objective_mm2": float(data.detach()),
            "smoothness": float(smooth.detach()),
            "magnitude": float(magnitude.detach()),
            "rank_penalty": float(rank.detach()),
            "objective": float(data.detach() + regularizer.detach()),
            "fit_gradient_rms": float(fit_gradient.square().mean().sqrt()),
            "gradient_rms": float(q.grad.square().mean().sqrt()),
            "regularizer_gradient_rms": float(
                (q.grad - fit_gradient).square().mean().sqrt()
            ),
            "u": u.detach().cpu().numpy(),
            "forward": dict(self.physics.last_forward),
            "adjoint": {
                "success": bool(adjoint.success),
                "result": str(adjoint.result),
            },
        }
        if self.cfg.record_component_gradients:
            receipt["component_gradients"] = {
                "fit": fit_gradient.cpu().numpy(),
                **{name: grad.cpu().numpy() for name, grad in gradients.items()},
            }
        return receipt


def metrics(
    physics: FacePhysics, q: torch.Tensor, result: dict[str, Any], cfg: Config
) -> dict[str, Any]:
    u = result["u"]
    pred, target, weight = u[physics.top], physics.target[physics.top], physics.weights
    residual = pred - target
    det = physics.detf(u)
    row: dict[str, Any] = {
        "fit_rms_mm": float(1000 * np.sqrt(np.mean(np.sum(residual**2, axis=1)))),
        "area_fit_rms_mm": float(1000 * np.sqrt(np.sum(weight[:, None] * residual**2))),
        "area_motion_rms_mm": float(1000 * np.sqrt(np.sum(weight[:, None] * pred**2))),
        "target_projection": float(
            np.sum(weight[:, None] * pred * target)
            / np.sum(weight[:, None] * target**2)
        ),
        "detF_min": float(det.min()),
        "detF_max": float(det.max()),
        "inverted_tetrahedra": int(np.count_nonzero(det <= 0)),
    }
    if cfg.model == "tensor":
        eig = eigenvalues(matrices(q)).cpu().numpy() * cfg.stress_reference_mpa
        mass = physics.volumes / physics.volumes.sum()
        assert eig.min() >= -1e-10
        assert eig.max() <= cfg.stress_cap_mpa + 1e-10
        total = eig.sum(axis=1)
        denominator = float(np.sum(mass * total**2))
        strength = float(np.sum(mass * total))
        row.update(
            Q_eigen_min_mpa=float(eig.min()),
            Q_eigen_max_mpa=float(eig.max()),
            Q_trace_mean_mpa=float(np.sum(mass * total)),
            Q_rms_mpa=float(np.sqrt(np.sum(mass * np.sum(eig**2, axis=1)))),
            upper_cap_cell_fraction=float(
                np.mean(eig[:, -1] >= cfg.stress_cap_mpa * (1 - 1e-7))
            ),
            principal_tension_fraction=float(np.sum(mass * eig[:, -1]) / strength)
            if strength > 0
            else None,
            rank_mixing_fraction=float(
                np.sum(mass * (total**2 - np.sum(eig**2, axis=1))) / denominator
            )
            if denominator > 0
            else None,
            zero_stress_cell_fraction=float(
                np.mean(eig[:, -1] <= cfg.stress_reference_mpa * 1e-8)
            ),
        )
    else:
        eig = eigenvalues(raw6_matrices(q)).cpu().numpy()
        row.update(
            Ainv_eigen_min=float(eig.min()),
            Ainv_eigen_max=float(eig.max()),
            non_spd_activation_tets=int(np.count_nonzero(eig[:, 0] <= 0)),
        )
    return row


def snapshot(
    path: Path,
    physics: FacePhysics,
    q: torch.Tensor,
    result: dict[str, Any],
    step: int,
    cfg: Config,
    *,
    mesh: bool,
) -> None:
    arrays = {
        "q": q.detach().cpu().numpy(),
        "u": result["u"],
        "step": np.asarray(step),
        "solver_valid": np.asarray(True),  # noqa: FBT003
        "active_ids": physics.ids,
    }
    key = "Q" if cfg.model == "tensor" else "Ainv"
    values = (
        cfg.stress_reference_mpa * matrices(q)
        if cfg.model == "tensor"
        else raw6_matrices(q)
    )
    arrays[key] = values.detach().cpu().numpy()
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)
    if mesh:
        kwargs = {"active_stress" if cfg.model == "tensor" else "ainv": arrays[key]}
        physics.save_mesh(path.with_suffix(".vtu"), result["u"], **kwargs)


def run(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    assert cfg.steps > 0
    assert cfg.checkpoint_interval > 0
    assert cfg.learning_rate > 0
    assert cfg.adam_eps > 0
    assert cfg.stress_reference_mpa == QREF
    assert cfg.stress_cap_mpa > 0
    assert min(cfg.smoothness_weight, cfg.magnitude_weight, cfg.rank_weight) >= 0
    if cfg.model == "raw6":
        assert cfg.rank_weight == cfg.smoothness_weight == cfg.magnitude_weight == 0
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"output directory must be empty: {out}"
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    provenance = archive(out, cfg)
    configure()
    physics = FacePhysics(cfg.fixture, activation_model=cfg.model)
    assert len(physics.ids) == 288235
    assert physics.n_regions == 103
    assert len(physics.graph[0]) == 501409
    objective = Objective(physics, cfg)
    q = torch.nn.Parameter(torch.zeros((len(physics.ids), 6)))
    optimizer = torch.optim.Adam([q], lr=cfg.learning_rate, eps=cfg.adam_eps)
    initial_hash = hashlib.sha256(q.detach().cpu().numpy().tobytes()).hexdigest()
    trace: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    started = time.perf_counter()
    last = None
    seed = np.zeros_like(physics.points)
    projection = {
        "projection_rms": 0.0,
        "projected_negative_eigenvalue_fraction": 0.0,
        "projected_upper_eigenvalue_fraction": 0.0,
        "unprojected_update_rms": 0.0,
        "actual_update_rms": 0.0,
    }
    try:
        for step in range(cfg.steps + 1):
            cherries.set_step(step)
            current = objective(q, seed)
            scalar = {
                key: value
                for key, value in current.items()
                if key not in {"u", "forward", "adjoint", "component_gradients"}
            }
            row = {
                "step": step,
                **scalar,
                **metrics(physics, q, current, cfg),
                **projection,
                "forward_steps": current["forward"]["steps"],
                "forward_grad_norm": current["forward"]["grad_norm"],
                "solver_valid": True,
                "elapsed_s": time.perf_counter() - started,
            }
            trace.append(row)
            last = current
            with (out / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(trace)
            with (out / "solver-receipts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "forward": current["forward"],
                            "adjoint": current["adjoint"],
                        },
                        allow_nan=False,
                    )
                    + "\n"
                )
            if best is None or row["objective"] < best["objective"]:
                best = dict(row)
                snapshot(out / "best.npz", physics, q, current, step, cfg, mesh=False)
            snapshot(out / "latest.npz", physics, q, current, step, cfg, mesh=False)
            if step % cfg.checkpoint_interval == 0 or step == cfg.steps:
                snapshot(
                    out / f"step-{step:04d}.npz",
                    physics,
                    q,
                    current,
                    step,
                    cfg,
                    mesh=True,
                )
                state = {
                    "step": step,
                    "q": q.detach().cpu(),
                    "u": current["u"],
                    "optimizer": optimizer.state_dict(),
                    "config": cfg.model_dump(mode="json"),
                }
                torch.save(state, out / "optimizer-latest.pt")
            if cfg.record_component_gradients:
                np.savez_compressed(
                    out / f"gradients-{step:04d}.npz", **current["component_gradients"]
                )
            write_json(out / "progress.json", row)
            cherries.log_metrics(
                {
                    "face": {
                        key: value
                        for key, value in row.items()
                        if isinstance(value, (int, float))
                    }
                }
            )
            logger.info(
                "step=%d fit=%.6f mm motion=%.6f mm smooth=%.6g rank=%.6g forward=%d elapsed=%.1fs",
                step,
                row["area_fit_rms_mm"],
                row["area_motion_rms_mm"],
                row["smoothness"],
                row["rank_penalty"],
                row["forward_steps"],
                row["elapsed_s"],
            )
            print(
                json.dumps(
                    {
                        key: row[key]
                        for key in (
                            "step",
                            "area_fit_rms_mm",
                            "area_motion_rms_mm",
                            "smoothness",
                            "rank_penalty",
                            "forward_steps",
                            "elapsed_s",
                        )
                    }
                ),
                flush=True,
            )
            if step == cfg.steps:
                break
            seed = current["u"]
            q_before = q.detach().clone()
            optimizer.step()
            unprojected_update_rms = float(
                (q.detach() - q_before).square().mean().sqrt()
            )
            if cfg.model == "tensor":
                projection = project(q, cfg.stress_cap_mpa / cfg.stress_reference_mpa)
            projection["unprojected_update_rms"] = unprojected_update_rms
            projection["actual_update_rms"] = float(
                (q.detach() - q_before).square().mean().sqrt()
            )
    except BaseException as error:
        write_json(
            out / "failure.json",
            {
                "type": type(error).__name__,
                "message": str(error),
                "attempted_step": len(trace),
                "last_valid_step": trace[-1]["step"] if trace else None,
                "last_forward": getattr(physics, "last_forward", None),
            },
        )
        with (out / "failed-trial.npz").open("wb") as stream:
            np.savez_compressed(
                stream,
                q=q.detach().cpu().numpy(),
                u=physics.forward.state.u.detach().cpu().numpy(),
            )
        raise
    assert last is not None
    assert best is not None
    snapshot(out / "final.npz", physics, q, last, cfg.steps, cfg, mesh=True)
    saved_best = np.load(out / "best.npz")
    best_q = torch.as_tensor(saved_best["q"])
    snapshot(
        out / "best.npz",
        physics,
        best_q,
        {"u": saved_best["u"]},
        int(saved_best["step"]),
        cfg,
        mesh=True,
    )
    summary = {
        "status": "completed_fixed_budget",
        "inverse_convergence_claimed": False,
        "config": cfg.model_dump(mode="json"),
        "provenance": provenance,
        "initialization": {
            "control_sha256": initial_hash,
            "controls": "all zero",
            "displacement": "rest",
            "adam_moments": "zero",
        },
        "materials": physics.material_spec,
        "forward_solver": physics.forward_tolerance,
        "adjoint_solver": {
            "relative_tolerance": 5e-4,
            "max_steps": 10000,
            "implementations": ["CupyCG", "CupyMinRes"],
        },
        "mesh": {
            "active_tetrahedra": len(physics.ids),
            "scalar_controls": q.numel(),
            "graph_edges": len(physics.graph[0]),
            "regions": physics.n_regions,
        },
        "constraint": {
            "kind": "spectral PSD box"
            if cfg.model == "tensor"
            else "unbounded direct active-strain offset",
            "reference_second_piola_stress_cap_mpa": cfg.stress_cap_mpa
            if cfg.model == "tensor"
            else None,
        },
        "geometry_rejection_enabled": False,
        "fiber_directions_used": False,
        "primary_endpoint": trace[-1],
        "best_objective_endpoint": best,
        "endpoint_policy": "actual common final step is primary; best total-objective state is secondary",
        "wall_s": time.perf_counter() - started,
    }
    write_json(out / "summary.json", summary)
    for name in (
        "summary.json",
        "trace.csv",
        "config.json",
        "provenance.json",
        "solver-receipts.jsonl",
    ):
        cherries.log_output(out / name)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(run, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
