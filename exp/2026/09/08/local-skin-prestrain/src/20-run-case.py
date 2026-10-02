"""Fixed-activation and equally budgeted re-fit ablations of local skin prestrain."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

import comet_ml
import experiment_io as b
import numpy as np
import pydantic_settings as ps
import torch
from continuation_metrics import ContinuationMetrics, _array_sha256
from liblaf.cherries import core, plugins, profiles
from local_physics import FacePhysics, configure

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
CANON = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline"
)
CHECKPOINT = CANON / "data/20-baseline/optimizer-latest.pt"
CHECKPOINT_SHA = "c2f77e4bdb09e8e0be50d0f8ec9a9926ea7a4bc08db5a368ec83dcb768707b58"
LOG = logging.getLogger(__name__)
DONE = False


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
    case: Literal["no-skin", "skin-zero", "skin-local-1pct"]
    stage: Literal["forward", "refit"]
    fixture: Path = b.FIXTURE
    field: Path = GROUP / "data/10-prestrain-field/skin-prestrain.npz"
    steps: int = 200
    checkpoint_interval: int = 10
    resume: Path | None = None


def write_trace(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(cfg: Config) -> None:
    global DONE
    assert cfg.steps == 200 and cfg.checkpoint_interval == 10
    out = (
        GROUP
        / "data"
        / f"{'20' if cfg.stage == 'forward' else '30'}-{cfg.stage}-{cfg.case}"
    )
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), f"Refuse to overwrite a run: {out}"
    assert cfg.resume is None, (
        "Resume is not implemented for this frozen first-run protocol"
    )
    assert b.digest(CHECKPOINT) == CHECKPOINT_SHA
    material_hash = b.digest(GROUP / "src/volume_preserving_active.py")
    validation_path = CANON / "data/10-physical-volume-validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "passed"
    assert validation["material_source"]["sha256"] == material_hash
    skin_validation_path = GROUP / "data/11-skin-validation/summary.json"
    if cfg.case != "no-skin":
        skin_validation = json.loads(skin_validation_path.read_text())
        assert skin_validation["status"] == "passed"
    field_validation_path = cfg.field.with_name("validation.json")
    field_validation = json.loads(field_validation_path.read_text())
    assert field_validation["status"] == "passed_cpu_prestrain_field_validation"
    canonical = json.loads((CANON / "data/20-baseline/provenance.json").read_text())
    for name, record in canonical["inputs"].items():
        assert b.digest(cfg.fixture / name) == record["sha256"]
    field = np.load(cfg.field)
    for name, expected in field_validation["hashes"].items():
        assert _array_sha256(field[name]) == expected, name
    skin_ainv = field["activation_inv"] if cfg.case == "skin-local-1pct" else None
    configure()
    physics = FacePhysics(
        cfg.fixture,
        skin_factor=float(cfg.case != "no-skin"),
        skin_activation_inv=skin_ainv,
    )
    assert len(physics.ids) == 288235 and len(physics.top) == 15302
    assert np.array_equal(field["point_ids"], physics.skin.point_data["GlobalPointId"])
    assert np.array_equal(
        field["triangles"], np.asarray(physics.skin.faces).reshape(-1, 4)[:, 1:]
    )
    diagnostics = ContinuationMetrics(fixture=cfg.fixture)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    assert checkpoint["step"] == 200
    assert checkpoint["energy_law"] == "physical-volume-active-strain-v1"
    q = torch.nn.Parameter(checkpoint["q"].to(device="cuda", dtype=torch.float64))
    seed = np.asarray(checkpoint["u"]).copy()
    if cfg.stage == "refit":
        forward_path = GROUP / "data" / f"20-forward-{cfg.case}" / "final.npz"
        forward = np.load(forward_path)
        assert np.array_equal(forward["q"], q.detach().cpu().numpy())
        seed = forward["u"].copy()
    optimizer = torch.optim.Adam([q], lr=0.3, eps=0.01, betas=(0.9, 0.999))
    assert len(optimizer.state) == 0
    target = torch.as_tensor(physics.target[physics.top])
    sources = b.archive_runtime(out)
    provenance = {
        "case": cfg.case,
        "stage": cfg.stage,
        "command": sys.argv,
        "cwd": str(Path.cwd()),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(),
        "source_checkpoint_step": 200,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], text=True
        ).strip(),
        "canonical_checkpoint": {"path": str(CHECKPOINT), "sha256": CHECKPOINT_SHA},
        "field": {"path": str(cfg.field), "sha256": b.digest(cfg.field)},
        "field_validation": {
            "path": str(field_validation_path),
            "sha256": b.digest(field_validation_path),
        },
        "skin_validation": None
        if cfg.case == "no-skin"
        else {
            "path": str(skin_validation_path),
            "sha256": b.digest(skin_validation_path),
        },
        "volume_validation": {
            "path": str(validation_path),
            "sha256": b.digest(validation_path),
        },
        "sources": sources,
        "inputs": canonical["inputs"],
        "materials": physics.material_spec,
        "forward_tolerances": physics.forward_tolerance,
        "diagnostics": diagnostics.provenance,
        "objective": "uniform finite-IsFace coordinate MSE * 1e6; no regularization",
        "optimizer": {
            "name": "Adam",
            "lr": 0.3,
            "eps": 0.01,
            "betas": [0.9, 0.999],
            "updates": 0 if cfg.stage == "forward" else cfg.steps,
            "moments": "fresh zero moments for every case; no cached gradient from old physics",
        },
        "failure_policy": "Record inversions and non-SPD activation as diagnostics. Stop on nonfinite values or failed forward/adjoint solves. Never update with a failed solve.",
        "commit_enabled": False,
        "inverse_stationarity_claimed": False,
    }
    b.write_json(out / "config.json", cfg.model_dump(mode="json"))
    b.write_json(out / "provenance.json", provenance)
    for path in (CHECKPOINT, cfg.field):
        cherries.log_input(path)
    trace = []
    best = None
    failure = None
    status = "running"
    start = time.perf_counter()
    previous_u, previous_q = seed.copy(), q.detach().cpu().numpy().copy()
    end_step = cfg.steps if cfg.stage == "refit" else 0
    try:
        for step in range(end_step + 1):
            tick = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            u_tensor = physics.solve(q, seed)
            forward_receipt = dict(physics.last_forward)
            assert forward_receipt["success"], (
                f"Forward failed at {step}: {forward_receipt}"
            )
            u = u_tensor.detach().cpu().numpy().copy()
            loss = (u_tensor[physics.top_t] - target).square().mean() * 1e6
            assert torch.isfinite(loss)
            adjoint_receipt = None
            if cfg.stage == "refit":
                loss.backward()
                adjoint_receipt = physics.check_adjoint()
                assert q.grad is not None and torch.isfinite(q.grad).all()
            q_numpy = q.detach().cpu().numpy().copy()
            row = {
                "step": step,
                "source_checkpoint_step": 200,
                "objective_mm2": float(loss.detach()),
                "gradient_rms": None
                if q.grad is None
                else float(q.grad.norm().detach() / math.sqrt(q.numel())),
                "forward_steps": forward_receipt["steps"],
                "forward_grad_norm": forward_receipt["grad_norm"],
                "forward_success": True,
                "adjoint_success": cfg.stage == "refit",
                **b.row_metrics(physics, u, q_numpy),
                **diagnostics.evaluate(u, q_numpy, previous_u, previous_q),
                "elapsed_s": time.perf_counter() - start,
                "step_s": time.perf_counter() - tick,
            }
            assert math.isclose(
                row["objective_mm2"] * 3, row["fit_rms_mm"] ** 2, rel_tol=1e-12
            )
            state = dict(
                step=step,
                q=q_numpy,
                u=u,
                objective_mm2=row["objective_mm2"],
                solver_valid=True,
                metrics=row.copy(),
            )
            if best is None or row["objective_mm2"] < best["objective_mm2"]:
                best = state
            row["best_step"] = best["step"]
            trace.append(row)
            write_trace(out / "trace.csv", trace)
            with (out / "solver-receipts.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            step=step, forward=forward_receipt, adjoint=adjoint_receipt
                        )
                    )
                    + "\n"
                )
            # Save every displayed surface so matched-fit comparisons use actual
            # evaluated states without interpolating geometry or running physics.
            np.savez_compressed(
                out / f"surface-{step:04d}.npz",
                step=np.array(step),
                point_ids=diagnostics.skin_ids,
                u=u[diagnostics.skin_ids],
            )
            if step % cfg.checkpoint_interval == 0 or step == end_step:
                b.save_state(out / f"step-{step:04d}.npz", physics, state)
                b.save_state(out / "best.npz", physics, best)
            if cfg.stage == "refit":
                tmp = out / "optimizer-latest.tmp"
                torch.save(
                    dict(
                        case=cfg.case,
                        energy_law="physical-volume-active-strain-v1",
                        material_sha256=material_hash,
                        field_sha256=b.digest(cfg.field),
                        step=step,
                        q=q.detach().clone(),
                        u=u,
                        gradient=q.grad.detach().clone(),
                        optimizer=optimizer.state_dict(),
                        best=best,
                    ),
                    tmp,
                )
                tmp.replace(out / "optimizer-latest.pt")
            cherries.log_metrics(
                {
                    key: value
                    for key, value in row.items()
                    if isinstance(value, (int, float))
                },
                step=step,
            )
            LOG.info(
                "%s %s %03d/%d: fit %.4f mm, motion %.4f mm, NLF %.4f mm; %.1f s.",
                cfg.case,
                cfg.stage,
                step,
                end_step,
                row["fit_rms_mm"],
                row["motion_rms_mm"],
                row["roi_right_nose_to_mouth_fit_vector_rms_mm"],
                row["step_s"],
            )
            if step == end_step:
                b.save_state(out / "last.npz", physics, state)
                status = (
                    "completed_fixed_activation_forward"
                    if cfg.stage == "forward"
                    else "completed_200_updates_not_stationarity_certified"
                )
                break
            previous_u, previous_q = u, q_numpy
            seed = u
            optimizer.step()
    except BaseException as error:
        status = "failed_before_completion"
        failure = {
            "type": type(error).__name__,
            "message": str(error),
            "step": step,
            "forward": getattr(physics, "last_forward", None),
        }
        b.write_json(out / "failure.json", failure)
        raise
    finally:
        if best is not None:
            b.save_state(out / "final.npz", physics, best)
        summary = {
            "case": cfg.case,
            "stage": cfg.stage,
            "status": status,
            "last_evaluated_step": None if not trace else trace[-1]["step"],
            "best_step": None if best is None else best["step"],
            "best_metrics": None if best is None else best["metrics"],
            "last_metrics": None if not trace else trace[-1],
            "failure": failure,
            "elapsed_s": time.perf_counter() - start,
            "materials": physics.material_spec,
        }
        b.write_json(out / "summary.json", summary)
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
        LOG.info("Final status: %s. Outputs: %s", status, out)
    DONE = True


if __name__ == "__main__":
    cherries.main(main, profile=None if os.getenv("DEBUG") == "1" else Profile)
    if not DONE:
        raise SystemExit(1)
