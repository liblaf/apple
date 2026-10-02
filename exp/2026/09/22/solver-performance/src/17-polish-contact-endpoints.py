"""Test whether stricter contact equilibrium resolves endpoint differences.

This is a new accuracy diagnostic.  It does not revise the original benchmark
or retroactively accept its failed equivalence gates.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import ipctk

EXPERIMENT = Path(__file__).resolve().parent.parent
REPLAY_PATH = Path(__file__).with_name("16-replay-candidates.py")
CONTACT_CASE = "contact_expression"
TARGET_FORCE_NORM = 1e-12


def load_replay() -> Any:
    spec = importlib.util.spec_from_file_location(
        "solver_performance_replay", REPLAY_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


replay = load_replay()
benchmark = replay.benchmark


class Config(benchmark.Config):
    output_dir: Path = EXPERIMENT / "data/contact-endpoint-polish-001"
    reference_dir: Path = EXPERIMENT / "data/paratera-contact-002"
    hybrid_dir: Path = EXPERIMENT / "data/paratera-contact-hybrid-003"
    wall_seconds: float = 120.0
    max_newton_steps: int = 20
    ipc_threads: int | None = 8


def bind_output(row: dict[str, Any], directory: Path, method: str) -> dict[str, Any]:
    """Require a successful saved endpoint and bind its tensor artifact."""
    assert row["case"] == CONTACT_CASE
    assert row["method"] == method
    assert row["success"] is True
    assert row["geometry"]["inverted_tetrahedra"] == 0
    assert row["ipc_threads"] == 8
    output_path = directory / "outputs" / Path(row["output_path"]).name
    assert output_path.is_file(), output_path
    output = benchmark.torch.load(output_path, map_location="cpu", weights_only=False)
    for name in ("displacement", "gradient_q", "gradient_jaw"):
        assert benchmark.tensor_sha256(output[name]) == row[f"{name}_sha256"]
    bound = copy.deepcopy(row)
    bound["output_path"] = str(output_path)
    bound["output_sha256"] = benchmark.sha256(output_path)
    return bound


def load_hybrid_endpoints(
    cfg: Config, *, original_sources: dict[str, dict[str, str]], target_degrees: float
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    directory = cfg.hybrid_dir.resolve()
    assert directory.is_dir(), directory
    provenance_path = directory / "provenance.json"
    results_path = directory / "results.json"
    provenance = json.loads(provenance_path.read_text())
    sources = replay.validate_physical_sources(
        reference_dir=directory, provenance=provenance, source_root=cfg.source_root
    )
    assert sources == original_sources
    rows = json.loads(results_path.read_text())
    endpoints = {}
    for method in ("hybrid_diag", "hybrid_block"):
        matches = [
            row
            for row in rows
            if row["case"] == CONTACT_CASE and row["method"] == method
        ]
        assert len(matches) == 1, (method, len(matches))
        endpoint = bind_output(matches[0], directory, method)
        assert endpoint["proposal"]["target_jaw_degrees"] == target_degrees
        endpoints[method] = endpoint
    return endpoints, {
        "directory": str(directory),
        "files": {
            "provenance": replay.record(provenance_path),
            "results": replay.record(results_path),
        },
        "physical_source_hashes": sources,
        "endpoints": endpoints,
    }


def polish_fixture(original_make_fixture: Any, endpoint: dict[str, Any]) -> Any:
    """Use an endpoint displacement and its target jaw only as the new seed."""

    def make_fixture(cfg: Config, case: str, frozen: dict[str, dict[str, str]]) -> Any:
        inputs, physics, checkpoint, q, jaw = original_make_fixture(cfg, case, frozen)
        assert case == CONTACT_CASE
        assert benchmark.tensor_sha256(q) == endpoint["proposal"]["activation_sha256"]
        assert float(jaw[0]) * 10 == endpoint["proposal"]["target_jaw_degrees"]
        saved = benchmark.torch.load(
            endpoint["output_path"], map_location="cpu", weights_only=False
        )
        checkpoint = copy.deepcopy(checkpoint)
        checkpoint["displacement_m"] = saved["displacement"].detach().clone()
        # The saved displacement already belongs to the target jaw.  Keeping
        # this seed jaw prevents applying the proposal increment a second time.
        checkpoint["jaw_normalized"] = jaw.detach().cpu().clone()
        physics.runtime.tolerances["atol"] = TARGET_FORCE_NORM
        return inputs, physics, checkpoint, q, jaw

    return make_fixture


def run_polish(
    cfg: Config,
    *,
    endpoint_name: str,
    endpoint: dict[str, Any],
    frozen: dict[str, dict[str, str]],
    runner: Any,
) -> dict[str, Any]:
    original_make_fixture = benchmark.make_fixture
    endpoint_dir = cfg.output_dir / "endpoints" / endpoint_name
    endpoint_dir.mkdir(parents=True)
    original_output_dir = cfg.output_dir
    try:
        benchmark.make_fixture = polish_fixture(original_make_fixture, endpoint)
        cfg.output_dir = endpoint_dir
        polished = benchmark.run_one(
            cfg, case=CONTACT_CASE, method="newton_diag", frozen=frozen, runner=runner
        )
    finally:
        benchmark.make_fixture = original_make_fixture
        cfg.output_dir = original_output_dir
    polished["endpoint"] = endpoint_name
    polished["seed_endpoint"] = endpoint
    polished["prior_forward_wall_seconds"] = endpoint["forward_wall_seconds"]
    polished["total_forward_wall_seconds"] = (
        endpoint["forward_wall_seconds"] + polished["forward_wall_seconds"]
        if polished["success"]
        else None
    )
    return polished


def main(cfg: Config) -> None:
    assert cfg.wall_seconds == 120.0
    assert cfg.max_newton_steps == 20
    assert cfg.ipc_threads == 8
    assert cfg.source_root.is_dir(), cfg.source_root
    frozen, original_reference, controls = replay.validate_reference(
        cfg, cases=(CONTACT_CASE,)
    )
    original = controls[CONTACT_CASE]
    assert original["ipc_threads"] == cfg.ipc_threads
    target_degrees = original["proposal"]["target_jaw_degrees"]
    hybrids, hybrid_reference = load_hybrid_endpoints(
        cfg,
        original_sources=original_reference["physical_source_hashes"],
        target_degrees=target_degrees,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    benchmark.archive_benchmark_sources(cfg)
    benchmark.write_json(
        cfg.output_dir / "reference.json",
        {"original": original_reference, "hybrids": hybrid_reference},
    )
    from remote_paths import install_loader_path_relocation

    install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    assert int(ipctk.get_num_threads()) == cfg.ipc_threads
    benchmark.configure_cuda()
    runner = benchmark.load_runner()
    polished_original = run_polish(
        cfg,
        endpoint_name="original",
        endpoint=original,
        frozen=frozen,
        runner=runner,
    )
    rows = [polished_original]
    benchmark.write_json(cfg.output_dir / "results.json", rows)
    for method, endpoint in hybrids.items():
        row = run_polish(
            cfg,
            endpoint_name=method,
            endpoint=endpoint,
            frozen=frozen,
            runner=runner,
        )
        row["comparison_to_polished_original"] = benchmark.compare(
            polished_original, row
        )
        rows.append(row)
        benchmark.write_json(cfg.output_dir / "results.json", rows)
    summary = {
        "schema": "contact-endpoint-stricter-accuracy-diagnostic-v1",
        "success": all(
            row["success"]
            and row["geometry"]["inverted_tetrahedra"] == 0
            and row.get("comparison_to_polished_original", {"comparable": True})[
                "comparable"
            ]
            and all(
                row.get("comparison_to_polished_original", {})
                .get("equivalence_gate", {})
                .values()
            )
            for row in rows
        ),
        "scope": {
            "claim": "test whether remaining equilibrium residual explains prior endpoint differences",
            "not_retroactive": "this stricter diagnostic does not revise original comparison acceptance",
            "old_runs_mutated": False,
        },
        "protocol": {
            "case": CONTACT_CASE,
            "target_jaw_degrees": target_degrees,
            "primal_force_tolerance": TARGET_FORCE_NORM,
            "solver": "exact diagonal safeguarded Newton from each saved endpoint",
            "max_newton_steps": cfg.max_newton_steps,
            "wall_seconds_per_endpoint": cfg.wall_seconds,
            "ipc_threads": int(ipctk.get_num_threads()),
            "seed": "saved endpoint displacement and its target jaw; no additional jaw increment",
        },
        "reference": {"original": original_reference, "hybrids": hybrid_reference},
        "frozen_inputs": frozen,
        "results": rows,
        "source_sha256": benchmark.sha256(Path(__file__)),
    }
    benchmark.write_json(cfg.output_dir / "summary.json", summary)
    benchmark.cherries.log_metrics(
        {
            "contact_polish/success": float(summary["success"]),
            "contact_polish/arms": len(rows),
        }
    )
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    benchmark.cherries.main(main, profile=benchmark.ProfilePerformance)
