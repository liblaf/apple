# ruff: noqa: E402, PLR0915
"""Sequential fixed-proposal comparisons of shift reuse and CUDA contact HVPs."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
JOINT = GROUP.parent.parent / "21/joint-activation-material-mandible"
sys.path[:0] = [str(GROUP / "src"), str(JOINT / "src")]
from accelerated_solvers import accelerate_runtime
from joint_common import sha256, write_json
from joint_equilibrium import configure_cuda
from joint_fields import project_activation_
from remote_paths import install_loader_path_relocation
from smile_collision import audit_collision_state, audit_required_collision


class Config(cherries.BaseConfig):
    checkpoint: Path
    output_dir: Path
    source_root: Path = GROUP.parents[4]
    inputs_dir: Path = JOINT / "data/expression-inputs-002"
    variants: str = "reset_cpu,reuse_cpu,reset_gpu,reuse_gpu"
    method: str = "hybrid_diag"
    forward_atol: float = 1e-8
    forward_wall_seconds: float | None = None
    linear_rtol: float = 1e-3
    newton_switch_atol: float = 1e-7
    ipc_threads: int = 8
    max_newton_steps: int = 100
    rms_agreement_mm: float = 0.001
    maximum_agreement_mm: float = 0.01


def runner_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "forward_replay_fitter", JOINT / "src/93-fit-expressions.py"
    )
    assert spec is not None
    assert spec.loader is not None
    runner = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = runner
    spec.loader.exec_module(runner)
    return runner


def proposal(state: dict, runner: Any) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.nn.Parameter(state["activation"].clone())
    jaw = torch.nn.Parameter(state["jaw_normalized"].clone())
    optimizer = torch.optim.Adam((q, jaw), lr=0.3)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    assert all(group["lr"] == 0.3 for group in optimizer.param_groups)
    q.grad = state["gradient_q"].clone()
    jaw.grad = state["gradient_jaw"].clone()
    optimizer.step()
    project_activation_(q, runner.CAP)
    with torch.no_grad():
        jaw.clamp_(runner.HINGE_MIN, runner.HINGE_MAX)
    return q.detach(), jaw.detach()


def main(cfg: Config) -> None:
    assert cfg.checkpoint.is_file()
    assert not cfg.output_dir.exists(), cfg.output_dir
    variants = cfg.variants.split(",")
    assert variants[0] == "reset_cpu"
    assert len(set(variants)) == len(variants)
    assert set(variants) <= {"reset_cpu", "reuse_cpu", "reset_gpu", "reuse_gpu"}
    assert cfg.method in {"hybrid_diag", "hybrid_block"}
    cfg.output_dir.mkdir(parents=True)
    install_loader_path_relocation(source_root=cfg.source_root)
    configure_cuda()
    ipctk.set_num_threads(cfg.ipc_threads)
    state = torch.load(cfg.checkpoint, map_location="cuda", weights_only=False)
    assert state["metrics"]["shape"]["inverted_tetrahedra"] == 0
    runner = runner_module()
    q, jaw = proposal(state, runner)
    torch.save(
        {"activation": q.cpu(), "jaw": jaw.cpu()}, cfg.output_dir / "fixed-proposal.pt"
    )
    sources = [
        Path(__file__),
        GROUP / "src/accelerated_solvers.py",
        JOINT / "src/93-fit-expressions.py",
    ]
    if any(v.endswith("gpu") for v in variants):
        sources.append(GROUP / "src/gpu_contact.py")
    source_dir = cfg.output_dir / "sources"
    source_dir.mkdir()
    for path in sources:
        (source_dir / path.name).write_bytes(path.read_bytes())
    write_json(
        cfg.output_dir / "protocol.json",
        {
            "schema": "fixed-adam-proposal-forward-optimizations-v1",
            "checkpoint": {
                "path": str(cfg.checkpoint),
                "sha256": sha256(cfg.checkpoint),
                "accepted_steps": state["accepted_steps"],
            },
            "proposal_sha256": sha256(cfg.output_dir / "fixed-proposal.pt"),
            "inputs": {
                n: sha256(cfg.inputs_dir / n) for n in ("manifest.json", "state.npz")
            },
            "sources": {p.name: sha256(p) for p in sources},
            "config": cfg.model_dump(mode="json"),
            "scope": "Identical next full projected Adam proposal; fresh runtimes, same seed, sequential same GPU; forward solves only.",
            "gradient_validation": "Separate fixed-state adjoint sweep; this forward replay reports physical and displacement agreement, not gradient equivalence.",
        },
    )
    rows = []
    reference = None
    for index, variant in enumerate(variants):
        cherries.set_step(index)
        shift_policy, contact = variant.split("_")
        fit_cfg = runner.Config(
            _cli_parse_args=False,
            output_dir=cfg.output_dir / "scratch" / variant,
            inputs_dir=cfg.inputs_dir,
            calibration_source=None,
            forward_atol=cfg.forward_atol,
            adjoint_rtol=1e-7,
            ipc_threads=cfg.ipc_threads,
            learning_rate=0.3,
            magnitude_weight=0.0,
            jaw_weight=0.0,
            outer_step_policy="full_adam",
            trial_prescreen=False,
            pose_first=False,
            pose_collision=True,
        )
        fitter = runner.Fitter(fit_cfg)
        runtime = accelerate_runtime(
            fitter.runtime,
            cfg.method,
            rest_points=fitter.physics.points,
            wall_seconds=cfg.forward_wall_seconds,
            linear_rtol=cfg.linear_rtol,
            max_newton_steps=cfg.max_newton_steps,
            newton_switch_atol=cfg.newton_switch_atol,
            shift_policy=shift_policy,
        )
        fitter.runtime = runtime
        fitter.physics.runtime = runtime
        gpu = None
        if contact == "gpu":
            from gpu_contact import install_gpu_contact

            gpu = install_gpu_contact(fitter.physics)
        coverage = audit_required_collision(fitter.physics)
        torch.cuda.synchronize()
        started = time.perf_counter()
        try:
            u = fitter.solve(
                q,
                jaw,
                state["displacement_m"],
                state["jaw_normalized"],
                "forward-probe",
            ).detach()
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            forward = copy.deepcopy(runtime.last_forward)
            shape = fitter.physics.metrics(u, target_index=12)
            collision = audit_collision_state(
                fitter.physics, u, runner.hinge_pose(jaw, fitter.hinge_axis)
            )
            valid = (
                forward["success"]
                and forward["grad_norm"] <= cfg.forward_atol
                and shape["inverted_tetrahedra"] == 0
                and collision["state_feasible"]
            )
            row = {
                "variant": variant,
                "success": bool(valid),
                "seconds": seconds,
                "forward": forward,
                "shape": shape,
                "collision": collision,
                "coverage": coverage,
            }
            if gpu is not None:
                row["gpu_contact"] = {
                    "uploads": gpu.adapter.uploads,
                    "products": gpu.adapter.products,
                }
            if reference is None:
                assert valid, "Control failed physical checks"
                reference = u.clone()
                row["agreement"] = {
                    "skin_rms_mm": 0.0,
                    "maximum_node_mm": 0.0,
                    "pass": True,
                }
            else:
                delta_mm = 1000 * (u - reference)
                rms = float(
                    (fitter.weights * delta_mm[fitter.obs].square().sum(-1))
                    .sum()
                    .sqrt()
                )
                maximum = float(torch.linalg.vector_norm(delta_mm, dim=-1).max())
                row["agreement"] = {
                    "skin_rms_mm": rms,
                    "maximum_node_mm": maximum,
                    "pass": rms <= cfg.rms_agreement_mm
                    and maximum <= cfg.maximum_agreement_mm,
                }
            torch.save(
                {
                    "displacement_m": u.cpu(),
                    "activation": q.cpu(),
                    "jaw_normalized": jaw.cpu(),
                },
                cfg.output_dir / f"{variant}.pt",
            )
        except Exception as error:
            row = {
                "variant": variant,
                "success": False,
                "error_type": type(error).__name__,
                "failure": str(error),
                "seconds": time.perf_counter() - started,
                "forward": copy.deepcopy(runtime.last_forward),
            }
            if variant == variants[0]:
                write_json(cfg.output_dir / "control-failure.json", row)
                raise
        finally:
            if gpu is not None:
                gpu.uninstall()
        rows.append(row)
        write_json(cfg.output_dir / "results.json", rows)
        print(
            json.dumps(
                {
                    "variant": variant,
                    "success": row["success"],
                    "seconds": row["seconds"],
                    "agreement": row.get("agreement"),
                    "failure": row.get("failure"),
                }
            ),
            flush=True,
        )
        del fitter, runtime
        torch.cuda.empty_cache()
    write_json(
        cfg.output_dir / "summary.json",
        {
            "schema": "fixed-adam-proposal-forward-optimizations-v1",
            "all_physical_checks_pass": all(r["success"] for r in rows),
            "all_agreement_checks_pass": all(
                r.get("agreement", {}).get("pass", False) for r in rows
            ),
            "results": rows,
        },
    )
    cherries.log_metrics(
        {
            r["variant"]: {
                "seconds": r["seconds"],
                "physical_success": float(r["success"]),
            }
            for r in rows
        }
    )


if __name__ == "__main__":
    cherries.main(main)
