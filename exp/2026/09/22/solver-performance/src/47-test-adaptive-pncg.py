# ruff: noqa: PLR0915, PT018
"""One cold-start Smile solve with stall-triggered single Newton corrections."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import ipctk
import torch

from liblaf import cherries

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "cold_forward", HERE / "43-compare-cold-forward.py"
)
assert spec is not None and spec.loader is not None
cold = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = cold
spec.loader.exec_module(cold)
benchmark = cold.benchmark


class Config(cold.Config):
    output_dir: Path = cold.EXPERIMENT / "data/adaptive-pncg-cold-001"
    baseline: Path = cold.EXPERIMENT / "data/cold-forward-comparison-001/summary.json"
    window_steps: int = 100
    minimum_reduction: float = 0.1
    required_poor_windows: int = 2


def main(cfg: Config) -> None:
    assert not cfg.output_dir.exists(), cfg.output_dir
    assert cfg.forward_atol == 1e-8 and cfg.linear_rtol == 1e-3
    assert cfg.max_newton_steps == 100
    assert cfg.wall_seconds is None or cfg.wall_seconds > 0
    assert cfg.ipc_threads == 8
    assert cfg.window_steps > 0 and cfg.minimum_reduction == 0.1
    assert cfg.required_poor_windows == 2
    baseline = json.loads(cfg.baseline.read_text())
    assert (
        benchmark.sha256(cfg.checkpoint) == baseline["protocol"]["checkpoint"]["sha256"]
    )
    assert (
        benchmark.sha256(cfg.neutral_checkpoint)
        == baseline["protocol"]["neutral_checkpoint"]["sha256"]
    )
    for name, digest in baseline["protocol"]["inputs"].items():
        assert benchmark.sha256(cfg.inputs_dir / name) == digest
    cfg.output_dir.mkdir(parents=True)
    provenance = benchmark.archive_benchmark_sources(cfg)
    old_sources = json.loads((cfg.baseline.parent / "provenance.json").read_text())[
        "sources"
    ]
    physical_source_changes = [
        name
        for name, digest in old_sources.items()
        if name.startswith(("apple/", "joint-experiment/", "tensor-reference/"))
        and provenance["sources"][name] != digest
    ]
    assert not physical_source_changes, physical_source_changes
    cold.install_loader_path_relocation(source_root=cfg.source_root)
    ipctk.set_num_threads(cfg.ipc_threads)
    assert int(ipctk.get_num_threads()) == cfg.ipc_threads
    cold.configure_cuda()
    checkpoint = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)
    neutral = torch.load(cfg.neutral_checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["expression"] == "Smile" and checkpoint["accepted_steps"] == 16
    assert neutral["accepted_steps"] == 0
    q, jaw, seed = (
        checkpoint["activation"],
        checkpoint["jaw_normalized"],
        neutral["displacement_m"],
    )
    initial = {
        "activation_sha256": benchmark.tensor_sha256(q),
        "jaw_sha256": benchmark.tensor_sha256(jaw),
        "seed_displacement_sha256": benchmark.tensor_sha256(seed),
        "seed_jaw_sha256": benchmark.tensor_sha256(torch.zeros_like(jaw)),
        "target_jaw_normalized": float(jaw[0]),
    }
    assert initial == baseline["protocol"]["initial"]
    protocol = {
        "schema": "cold-smile-adaptive-pncg-v1",
        "initial": initial,
        "checkpoint_sha256": benchmark.sha256(cfg.checkpoint),
        "neutral_checkpoint_sha256": benchmark.sha256(cfg.neutral_checkpoint),
        "baseline_sha256": benchmark.sha256(cfg.baseline),
        "baseline_protocol": baseline["protocol"],
        "config": cfg.model_dump(mode="json"),
        "physical_source_changes": physical_source_changes,
        "switch_rule": f"Two consecutive {cfg.window_steps}-accepted-PNCG-update windows each reduce segment-best force by less than 10%; one safeguarded Newton correction; fresh PNCG optimizer and window history; repeat.",
        "scope": "One cold displacement start with fixed active stress, fixed prestress, jaw zero, full skull and eye contact; no outer optimization, adjoint, or load continuation. JIT/operator prewarm excluded. Historical baselines are reused, not rerun.",
        "logging": "Accepted-state force, mechanical energy and elapsed time every update. PNCG energy is reused from its accepted line search; Newton post-update energy is explicitly evaluated. Logging costs are included.",
    }
    benchmark.write_json(cfg.output_dir / "protocol.json", protocol)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": True, "method": "adaptive_diag"}
    )
    original_build = cold.build_fitter
    with (cfg.output_dir / "accepted-trace.jsonl").open("x") as stream:

        def record(row: dict) -> None:
            stream.write(json.dumps(row) + "\n")
            stream.flush()
            if row["kind"] != "pncg" or row["pncg_steps"] % 100 == 0:
                print(json.dumps(row), flush=True)

        def build_fitter(
            config: Config, runner: Any, method: str, directory: Path
        ) -> Any:
            fitter = original_build(config, runner, method, directory)
            fitter.runtime.adaptive_options = {
                "window_steps": cfg.window_steps,
                "minimum_reduction": cfg.minimum_reduction,
                "required_poor_windows": cfg.required_poor_windows,
            }
            fitter.runtime.trace_callback = record
            return fitter

        cold.build_fitter = build_fitter
        try:
            result = cold.run_one(
                cfg,
                runner=cold.load_runner(),
                method="adaptive_diag",
                q_cpu=q,
                jaw_cpu=jaw,
                seed_cpu=seed,
                target_displacement_cpu=checkpoint["displacement_m"],
            )
        finally:
            cold.build_fitter = original_build
    assert result["initial"] == initial
    assert (
        abs(
            result["prewarm_force_norm"] / baseline["results"][0]["prewarm_force_norm"]
            - 1
        )
        < 1e-10
    )
    summary = {
        "schema": protocol["schema"],
        "protocol": protocol,
        "result": result,
        "historical_baselines": copy.deepcopy(baseline["results"]),
    }
    benchmark.write_json(cfg.output_dir / "summary.json", summary)
    benchmark.write_json(
        cfg.output_dir / "status.json", {"running": False, "success": result["success"]}
    )
    cherries.log_metrics(
        {
            "adaptive/success": float(result["success"]),
            "adaptive/seconds": result["forward_wall_seconds"],
        }
    )
    print(
        json.dumps(
            {"success": result["success"], "seconds": result["forward_wall_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
