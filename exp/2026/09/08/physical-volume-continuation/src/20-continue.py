# ruff: noqa: C901, CPY001, EM102, FBT001, PLR0915, PLW0603, PT018, TRY003, TRY301
"""Continue an evaluated physical-volume checkpoint without resetting Adam."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import baseline_reference as b
import comet_ml
import numpy as np
import pydantic_settings as ps
import torch
from baseline_physics import FacePhysics, configure
from continuation_metrics import ContinuationMetrics
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
CANON = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline"
)
LOG = logging.getLogger(__name__)
DONE = False
KNOWN_CELL = 573586
KNOWN_J_FLOOR = -0.3046486160938099


class Comet(plugins.Comet):
    @core.impl
    def start(self) -> None:
        exp = comet_ml.start(
            project_name=self.run.project_name,
            experiment_config=comet_ml.ExperimentConfig(
                disabled=False,
                name=self.run.run_name,
                tags=self.run.tags,
                log_env_details=False,
                log_git_patch=False,
                auto_log_co2=False,
            ),
        )
        self.run.log_other("cherries/comet/url", exp.url)


class Profile(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(Comet(run=run))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = b.FIXTURE
    resume: Path = CANON / "data/20-baseline/optimizer-latest.pt"
    output_dir: Path = GROUP / "data/20-fit300"
    end_step: int = 300
    checkpoint_interval: int = 10


def save_optimizer(
    path: Path,
    step: int,
    q: torch.nn.Parameter,
    u: np.ndarray,
    optimizer: torch.optim.Adam,
    best: dict[str, Any],
    recovered: bool,
) -> None:
    assert int(optimizer.state[q]["step"]) == step
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "energy_law": "physical-volume-active-strain-v1",
            "material_sha256": b.digest(GROUP / "src/volume_preserving_active.py"),
            "step": step,
            "q": q.detach().clone(),
            "u": u.copy(),
            "gradient": q.grad.detach().clone(),
            "optimizer": optimizer.state_dict(),
            "best": best,
            "known_cell_recovered": recovered,
        },
        temporary,
    )
    temporary.replace(path)


def main(cfg: Config) -> None:
    global DONE
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Output directory must be empty: {out}"
    assert cfg.checkpoint_interval == 10 and cfg.end_step in (300, 400)
    validation = CANON / "data/10-physical-volume-validation.json"
    receipt = json.loads(validation.read_text())
    material_hash = b.digest(GROUP / "src/volume_preserving_active.py")
    assert receipt["status"] == "passed"
    assert receipt["material_source"]["sha256"] == material_hash
    canonical_provenance = json.loads(
        (CANON / "data/20-baseline/provenance.json").read_text()
    )
    for name, record in canonical_provenance["inputs"].items():
        assert b.digest(cfg.fixture / name) == record["sha256"]
    for record in canonical_provenance["sources"].values():
        assert b.digest(Path(record["path"])) == record["sha256"], record["path"]
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
    diagnostics = ContinuationMetrics(fixture=cfg.fixture)
    assert len(physics.ids) == 288235 and physics.n_regions == 103
    assert len(physics.points) == 228660 and len(physics.tets) == 1146517
    assert len(physics.top) == 15302
    q = torch.nn.Parameter(torch.zeros((len(physics.ids), 6)))
    optimizer = torch.optim.Adam([q], lr=0.3, eps=0.01, betas=(0.9, 0.999))
    checkpoint = torch.load(cfg.resume, weights_only=False)
    assert checkpoint["q"].shape == q.shape == (288235, 6)
    assert checkpoint["gradient"].shape == q.shape
    assert checkpoint["energy_law"] == "physical-volume-active-strain-v1"
    assert checkpoint["material_sha256"] == material_hash
    start_step = int(checkpoint["step"])
    assert start_step == cfg.end_step - 100
    with torch.no_grad():
        q.copy_(checkpoint["q"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    group = optimizer.param_groups[0]
    assert group["lr"] == 0.3 and group["eps"] == 0.01
    assert group["betas"] == (0.9, 0.999) and group["weight_decay"] == 0
    assert not group["amsgrad"] and int(optimizer.state[q]["step"]) == start_step
    q.grad = checkpoint["gradient"].to(q.device)
    assert torch.isfinite(q).all() and torch.isfinite(q.grad).all()
    seed = np.asarray(checkpoint["u"]).copy()
    assert seed.shape == physics.points.shape and np.isfinite(seed).all()
    best = checkpoint["best"]
    previous_q = q.detach().cpu().numpy().copy()
    previous_u = seed.copy()
    recovered = checkpoint.get("known_cell_recovered", False)
    origin_j = physics.detf(seed)
    assert np.isfinite(origin_j).all()
    assert set(np.flatnonzero(origin_j <= 0)) <= {KNOWN_CELL}
    assert origin_j[KNOWN_CELL] >= KNOWN_J_FLOOR
    origin = {
        "step": start_step,
        **b.row_metrics(physics, seed, previous_q),
        **diagnostics.evaluate(seed, previous_q),
    }
    b.write_json(out / "origin.json", origin)
    b.write_json(out / "config.json", cfg.model_dump(mode="json"))
    b.write_json(
        out / "provenance.json",
        {
            "command": sys.argv,
            "cwd": str(Path.cwd()),
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "sources": b.archive_runtime(out),
            "diagnostics": diagnostics.provenance,
            "resume": {"path": str(cfg.resume), "sha256": b.digest(cfg.resume)},
            "canonical_provenance": {
                "path": str(CANON / "data/20-baseline/provenance.json"),
                "sha256": b.digest(CANON / "data/20-baseline/provenance.json"),
            },
            "protocol": {
                "path": str(GROUP / "docs/10-protocol.md"),
                "sha256": b.digest(GROUP / "docs/10-protocol.md"),
            },
            "resume_semantics": "Evaluated q_k/u_k/g_k and Adam step k; consume cached g_k once, then evaluate q_(k+1). Save before consuming each evaluated gradient.",
            "geometry_guard": {
                "grandfathered_cell": KNOWN_CELL,
                "minimum_allowed_known_J": KNOWN_J_FLOOR,
                "stop_on_new_inverted_id": True,
                "recovery_latched": True,
            },
            "physically_valid_claimed": False,
            "inverse_stationarity_claimed": False,
            "commit_enabled": False,
        },
    )
    # Exactly the original runner's first resume update; no duplicated solve or reset.
    optimizer.step()
    del checkpoint
    target = torch.as_tensor(physics.target[physics.top])
    trace = []
    start = time.perf_counter()
    status = "running"
    failure = None
    last = None
    try:
        for step in range(start_step + 1, cfg.end_step + 1):
            step_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            u_tensor = physics.solve(q, seed)
            forward = dict(physics.last_forward)
            assert forward["success"], forward
            u = u_tensor.detach().cpu().numpy().copy()
            q_numpy = q.detach().cpu().numpy().copy()
            J = physics.detf(u)
            inverted = np.flatnonzero(J <= 0)
            new_ids = sorted(set(inverted) - ({KNOWN_CELL} if not recovered else set()))
            geometry_ok = (
                np.isfinite(J).all() and not new_ids and J[KNOWN_CELL] >= KNOWN_J_FLOOR
            )
            if not geometry_ok:
                rejected = {"step": step, "q": q_numpy, "u": u, "solver_valid": True}
                b.save_state(out / f"rejected-step-{step:04d}.npz", physics, rejected)
                b.write_json(
                    out / "rejected-geometry.json",
                    {
                        "step": step,
                        "inverted_ids": inverted.tolist(),
                        "new_ids": new_ids,
                        "known_cell_J": float(J[KNOWN_CELL]),
                        "nonfinite_count": int((~np.isfinite(J)).sum()),
                        "forward": forward,
                        "gradient_evaluated": False,
                    },
                )
                raise RuntimeError(
                    f"Geometry guard rejected step {step}; see rejected-geometry.json"
                )
            recovered = recovered or J[KNOWN_CELL] >= 0
            loss = (u_tensor[physics.top_t] - target).square().mean() * 1e6
            assert torch.isfinite(loss)
            loss.backward()
            adjoint = physics.diff.last_adjoint_solution
            assert adjoint is not None and adjoint.success
            assert q.grad is not None and torch.isfinite(q.grad).all()
            row = {
                "step": step,
                "objective_mm2": float(loss.detach()),
                "gradient_rms": float(
                    torch.linalg.vector_norm(q.grad).detach() / math.sqrt(q.numel())
                ),
                "forward_steps": forward["steps"],
                "forward_grad_norm": forward["grad_norm"],
                "forward_success": True,
                "adjoint_success": True,
                **b.row_metrics(physics, u, q_numpy),
                **diagnostics.evaluate(
                    u, q_numpy, previous_u=previous_u, previous_q=previous_q
                ),
                "known_cell_J": float(J[KNOWN_CELL]),
                "known_cell_recovered": bool(recovered),
                "new_inverted_cells": len(new_ids),
                "pure_muscle_inverted_cells": int(
                    (
                        (J <= 0) & (physics.mesh.cell_data["MuscleFraction"] >= 0.999)
                    ).sum()
                ),
                "elapsed_s": time.perf_counter() - start,
            }
            state = {
                "step": step,
                "q": q_numpy,
                "u": u,
                "objective_mm2": row["objective_mm2"],
                "solver_valid": True,
                "metrics": row,
            }
            if state["objective_mm2"] < best["objective_mm2"]:
                best = state
            last = state
            row["best_step"] = best["step"]
            row["step_s"] = time.perf_counter() - step_start
            trace.append(row)
            with (out / "trace.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(trace)
            with (out / "solver-receipts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "step": step,
                            "forward": forward,
                            "adjoint": {"success": True, "result": str(adjoint.result)},
                        },
                        default=str,
                    )
                    + "\n"
                )
            cherries.log_metrics(
                {
                    k: row[k]
                    for k in (
                        "fit_rms_mm",
                        "objective_mm2",
                        "known_cell_J",
                        "gradient_rms",
                    )
                },
                step=step,
            )
            LOG.info(
                "Step %d / %d: fit %.5f mm, known J %.6f; %.1f s",
                step,
                cfg.end_step,
                row["fit_rms_mm"],
                row["known_cell_J"],
                row["step_s"],
            )
            if step % cfg.checkpoint_interval == 0 or step == cfg.end_step:
                b.save_state(out / f"step-{step:04d}.npz", physics, state)
                save_optimizer(
                    out / "optimizer-latest.pt", step, q, u, optimizer, best, recovered
                )
            if step == cfg.end_step:
                status = "completed_block_not_stationarity_certified"
                break
            previous_u, previous_q = u, q_numpy
            seed = u.copy()
            optimizer.step()
    except BaseException as error:
        status = "stopped_before_budget_completed"
        failure = {"type": type(error).__name__, "message": str(error), "step": step}
        b.write_json(out / "failure.json", failure)
        raise
    finally:
        if last is not None:
            b.save_state(out / "last.npz", physics, last)
        b.save_state(out / "final.npz", physics, best)
        physics.save_mesh(out / "final.vtu", best["u"], b.activation(best["q"]))
        b.write_json(
            out / "summary.json",
            {
                "status": status,
                "origin_step": start_step,
                "requested_end_step": cfg.end_step,
                "last_accepted_step": None if last is None else last["step"],
                "best_step": best["step"],
                "best_metrics": best["metrics"],
                "last_metrics": None if not trace else trace[-1],
                "failure": failure,
                "elapsed_s": time.perf_counter() - start,
                "materials": physics.material_spec,
                "forward_tolerances": physics.forward_tolerance,
            },
        )
        for name in (
            "config.json",
            "provenance.json",
            "origin.json",
            "summary.json",
            "trace.csv",
            "solver-receipts.jsonl",
        ):
            path = out / name
            if path.exists():
                cherries.log_output(path)
        LOG.info("Continuation status: %s; best step %s; %s", status, best["step"], out)
    DONE = True


if __name__ == "__main__":
    cherries.main(main, profile=None if os.getenv("DEBUG") == "1" else Profile)
    if not DONE:
        raise SystemExit(1)
