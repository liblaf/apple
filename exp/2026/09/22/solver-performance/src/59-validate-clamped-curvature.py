# ruff: noqa: E402, PLR0915
"""Validate per-contribution PNCG curvature clamps at the saved Smile handoff."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries
from liblaf.apple.warp.model._adapter import WarpModelAdapter
from liblaf.apple.warp.model._model import WarpModel

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "clamped_curvature_profile", HERE / "56-profile-hybrid.py"
)
assert spec is not None
assert spec.loader is not None
profile = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = profile
spec.loader.exec_module(profile)
cold, benchmark = profile.cold, profile.benchmark

from profile_input_binding import bind_frozen_neutral_load


class Config(profile.Config):
    output_dir: Path = cold.EXPERIMENT / "data/pncg-clamped-curvature-002"
    audit_dir: Path = cold.EXPERIMENT / "data/pncg-curvature-audit-001"
    reference_dir: Path = cold.EXPERIMENT / "data/hybrid-first-profile-002"


def synchronized(operation: Any) -> tuple[Any, float]:
    torch.cuda.synchronize()
    started = time.perf_counter()
    value = operation()
    torch.cuda.synchronize()
    return value, time.perf_counter() - started


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.audit_dir.is_dir(), cfg.audit_dir
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    saved = torch.load(
        cfg.audit_dir / "negative-curvature-state.pt",
        map_location="cpu",
        weights_only=False,
    )
    prior = json.loads((cfg.audit_dir / "curvature-audit.json").read_text())
    reference = json.loads((cfg.reference_dir / "protocol.json").read_text())
    assert (
        benchmark.tensor_sha256(saved["activation"]) == reference["activation_sha256"]
    )
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    assert torch.equal(saved["activation"], checkpoint["activation"])
    assert torch.equal(saved["jaw_normalized"], checkpoint["jaw_normalized"])
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    cold.configure_cuda()
    runner = cold.load_runner()
    inputs = json.loads((cfg.inputs_dir / "manifest.json").read_text())
    neutral_dir = Path(inputs["parent_frozen_neutral"]["directory"])
    with bind_frozen_neutral_load(
        neutral_dir,
        cfg.output_dir,
        allow_pncg_curvature_clamps=True,
    ):
        fitter = cold.build_fitter(
            cfg, runner, "hybrid_first", cfg.output_dir / "scratch"
        )

    q = saved["activation"].to(device="cuda", dtype=torch.float64)
    jaw = saved["jaw_normalized"].to(device="cuda", dtype=torch.float64)
    displacement = saved["displacement_m"].to(device="cuda", dtype=torch.float64)
    direction = saved["direction_full"].to(device="cuda", dtype=torch.float64)
    physics, model = fitter.physics, fitter.runtime.forward.model
    assert displacement.shape == direction.shape == (model.n_points, 3)
    pose = runner.hinge_pose(jaw, fitter.hinge_axis)
    model.set_materials(
        physics.expression_materials(
            skin_multiplier=torch.ones((), device="cuda", dtype=torch.float64),
            active_stress=runner.activation_stresses_mpa(q, runner.REFERENCE_MPA),
        )
    )
    model.dof_map.fixed_values = physics.boundary(pose).detach().clone()
    state = model.State(u=displacement)
    state.collision = model.collision.state_at(state.u)

    tissue: dict[str, float] = {}
    tissue_seconds: dict[str, float] = {}
    for name, potential in model.warp_model.__wrapped__.potentials.items():
        adapter = WarpModelAdapter(WarpModel({name: potential}))
        value, seconds = synchronized(
            lambda adapter=adapter: adapter.hess_quad(state.u, direction)
        )
        tissue[name] = float(value)
        tissue_seconds[name] = seconds

    raw_terms, raw_seconds = synchronized(
        lambda: model.collision.raw_hess_quad_terms(state.collision, state.u, direction)
    )
    contact, clamped_seconds = synchronized(
        lambda: model.collision.hess_quad(state.collision, state.u, direction)
    )
    raw_contact = float(sum(raw_terms))
    clamped_terms = tuple(max(term, 0.0) for term in raw_terms)
    clamped_contact = float(sum(clamped_terms))
    assert math.isclose(float(contact), clamped_contact, rel_tol=1e-12, abs_tol=1e-18)
    assert math.isclose(
        raw_contact,
        prior["components"]["collision_gauss_newton"],
        rel_tol=1e-10,
        abs_tol=1e-18,
    )
    assert raw_contact < 0.0
    assert clamped_contact > 0.0
    assert any(term < 0.0 for term in raw_terms)
    assert any(term > 0.0 for term in raw_terms)

    tissue_total = float(sum(tissue.values()))
    total = tissue_total + clamped_contact
    assert tissue_total >= 0.0
    assert total > 0.0
    expected_non_membrane = {
        name: value
        for name, value in prior["components"].items()
        if name not in {"skin", "collision_gauss_newton"}
    }
    for name, old_value in expected_non_membrane.items():
        assert math.isclose(tissue[name], old_value, rel_tol=1e-10, abs_tol=1e-18)

    result = {
        "schema": "pncg-clamped-curvature-validation-v1",
        "scope": (
            "Saved-state curvature diagnostic only: no forward iteration, Newton "
            "step, inverse update, or full benchmark."
        ),
        "input": {
            "saved_handoff": str(
                (cfg.audit_dir / "negative-curvature-state.pt").resolve()
            ),
            "prior_audit": str((cfg.audit_dir / "curvature-audit.json").resolve()),
            "activation_sha256": benchmark.tensor_sha256(saved["activation"]),
            "jaw_sha256": benchmark.tensor_sha256(saved["jaw_normalized"]),
        },
        "curvature": {
            "tissue_terms": tissue,
            "tissue_total": tissue_total,
            "contact": {
                "term_count": len(raw_terms),
                "negative_term_count": sum(term < 0.0 for term in raw_terms),
                "positive_term_count": sum(term > 0.0 for term in raw_terms),
                "zero_term_count": sum(term == 0.0 for term in raw_terms),
                "raw_sum": raw_contact,
                "sum_of_individual_clamps": clamped_contact,
                "wrapper_value": float(contact),
            },
            "clamped_total": total,
        },
        "timing_seconds": {
            "tissue_terms": tissue_seconds,
            "contact_raw_term_enumeration": raw_seconds,
            "contact_clamped_wrapper": clamped_seconds,
        },
        "source_provenance": provenance,
        "ipctk_version": ipctk.__version__,
    }
    output = cfg.output_dir / "clamped-curvature.json"
    benchmark.write_json(output, result)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "curvature/tissue_total": tissue_total,
            "curvature/contact_raw_sum": raw_contact,
            "curvature/contact_individual_clamps": clamped_contact,
            "curvature/total": total,
            "contact/term_count": len(raw_terms),
            "contact/negative_term_count": result["curvature"]["contact"][
                "negative_term_count"
            ],
        }
    )
    print(json.dumps(result["curvature"]), flush=True)


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
