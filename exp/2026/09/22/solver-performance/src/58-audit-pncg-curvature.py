# ruff: noqa: E402, PLR0915
"""Replay only the recorded PNCG phase and split its negative curvature."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries
from liblaf.apple.warp.model._adapter import WarpModelAdapter
from liblaf.apple.warp.model._model import WarpModel

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "curvature_profile", HERE / "56-profile-hybrid.py"
)
assert spec is not None
assert spec.loader is not None
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)
cold, benchmark = profile.cold, profile.benchmark

import hybrid_first_solver
from pncg_first import run_pncg_phase
from profile_input_binding import bind_frozen_neutral_load


class Config(profile.Config):
    output_dir: Path = cold.EXPERIMENT / "data/pncg-curvature-audit-001"
    reference_dir: Path = cold.EXPERIMENT / "data/hybrid-first-profile-002"


class AuditCompleteError(Exception):
    """Stop the diagnostic before any Newton step or convergence claim."""


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    old_sources = json.loads((cfg.reference_dir / "provenance.json").read_text())
    for name in (
        "joint-experiment/joint_materials.py",
        "apple/warp/fem/_base.py",
        "apple/collision/_collision.py",
        "solver-performance/pncg_first.py",
        "solver-performance/accelerated_solvers.py",
    ):
        assert provenance["sources"][name] == old_sources["sources"][name], name
    reference = json.loads((cfg.reference_dir / "protocol.json").read_text())
    for name in ("checkpoint", "neutral_checkpoint"):
        assert benchmark.sha256(getattr(cfg, name)) == reference[f"{name}_sha256"]
    assert benchmark.sha256(cfg.source_root / "uv.lock") == reference["uv_lock_sha256"]
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    cold.configure_cuda()
    runner = cold.load_runner()
    inputs = json.loads((cfg.inputs_dir / "manifest.json").read_text())
    neutral_dir = Path(inputs["parent_frozen_neutral"]["directory"])
    with bind_frozen_neutral_load(neutral_dir, cfg.output_dir):
        fitter = cold.build_fitter(
            cfg, runner, "hybrid_first", cfg.output_dir / "scratch"
        )
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    neutral = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    q = checkpoint["activation"].to(device="cuda", dtype=torch.float64)
    jaw = checkpoint["jaw_normalized"].to(device="cuda", dtype=torch.float64)
    seed = neutral["displacement_m"].to(device="cuda", dtype=torch.float64)
    initial_force = cold.warm_operators(
        fitter=fitter, runner=runner, q=q, jaw=jaw, seed=seed
    )
    assert math.isclose(initial_force, reference["initial_force"], rel_tol=1e-12)
    result: dict[str, Any] = {}

    def audit_phase(problem: Any, state: Any, **kwargs: Any) -> Any:
        original_quad = problem.hess_quad

        def split_quad(current: Any, direction: torch.Tensor) -> torch.Tensor:
            value = original_quad(current, direction)
            if float(value) <= 0:
                model = problem.model
                full = model.dof_map.to_full_grad(direction)
                components = {}
                for name, potential in model.warp_model.__wrapped__.potentials.items():
                    adapter = WarpModelAdapter(WarpModel({name: potential}))
                    components[name] = float(adapter.hess_quad(current.u, full))
                components["collision_gauss_newton"] = float(
                    model.collision.hess_quad(current.collision, current.u, full)
                )
                assert math.isclose(
                    sum(components.values()), float(value), rel_tol=1e-10
                )
                result.update(components=components, total=float(value))
                torch.save(
                    {
                        "displacement_m": current.u.detach().cpu(),
                        "direction_full": full.detach().cpu(),
                        "activation": q.cpu(),
                        "jaw_normalized": jaw.cpu(),
                    },
                    cfg.output_dir / "negative-curvature-state.pt",
                )
            return value

        def observe(row: dict) -> None:
            assert row["step"] <= 12, "PNCG replay diverged from recorded short phase"

        problem.hess_quad = split_quad
        try:
            _, receipt = run_pncg_phase(
                problem,
                state,
                atol=kwargs["atol"],
                max_step_norm=kwargs["max_step_norm"],
                callback=observe,
            )
        finally:
            problem.hess_quad = original_quad
        result["pncg"] = receipt
        assert receipt["reason"] == "nonpositive_curvature"
        assert receipt["steps"] == 11
        raise AuditCompleteError

    original_solver = hybrid_first_solver.hybrid_first
    hybrid_first_solver.hybrid_first = audit_phase
    try:
        with torch.no_grad():
            fitter.solve(q, jaw, seed, torch.zeros_like(jaw), "audit/pncg_curvature")
    except AuditCompleteError:
        pass
    else:
        message = "PNCG audit did not reach the recorded handoff"
        raise AssertionError(message)
    finally:
        hybrid_first_solver.hybrid_first = original_solver
    previous = json.loads((cfg.reference_dir / "solver-result.json").read_text())
    old_pncg = previous["failure_receipt"]["pncg"]
    assert math.isclose(result["total"], old_pncg["curvature"], rel_tol=1e-6)
    result.update(
        scope="PNCG-only diagnostic; no Newton, no inverse update, no timing comparison",
        reference_dir=str(cfg.reference_dir),
        initial_force=initial_force,
        reference_curvature=old_pncg["curvature"],
        ipctk_version=ipctk.__version__,
    )
    output = cfg.output_dir / "curvature-audit.json"
    benchmark.write_json(output, result)
    cherries.log_output(output)
    cherries.log_metrics(
        {f"curvature/{key}": value for key, value in result["components"].items()}
    )
    print(
        json.dumps(
            {key: result[key] for key in ("components", "total", "reference_curvature")}
        ),
        flush=True,
    )


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
