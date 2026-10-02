"""Replay only accelerated candidates against a completed matched control."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import ipctk

EXPERIMENT = Path(__file__).resolve().parent.parent
BENCHMARK_PATH = Path(__file__).with_name("10-benchmark.py")


def load_benchmark() -> Any:
    spec = importlib.util.spec_from_file_location(
        "solver_performance_benchmark", BENCHMARK_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


benchmark = load_benchmark()


class Config(benchmark.Config):
    output_dir: Path = EXPERIMENT / "data/solver-performance-replay-001"
    reference_dir: Path
    methods: str = "hybrid_diag,hybrid_block"
    cases: str = "contact_expression"


CHECKPOINTS = {
    "collision_off_pose": ("run007", "run007_initial"),
    "contact_expression": ("run006", "run006_initial"),
}
INPUTS = {
    "inputs_manifest": "manifest.json",
    "inputs_state": "state.npz",
}
PHYSICAL_SOURCES = {
    "apple": Path("src/liblaf/apple"),
    "joint-experiment": Path("exp/2026/09/21/joint-activation-material-mandible/src"),
    "tensor-reference": Path("exp/2026/09/07/tensor-active-stress/src"),
}


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": benchmark.sha256(path)}


def reference_copy(reference_dir: Path, relative: Path, expected_sha256: str) -> Path:
    path = reference_dir / "frozen-inputs" / relative
    assert path.is_file(), path
    assert benchmark.sha256(path) == expected_sha256, path
    return path


def validate_physical_sources(
    *, reference_dir: Path, provenance: dict[str, Any], source_root: Path
) -> dict[str, dict[str, str]]:
    """Require the runtime physics source trees to equal the control archive."""
    expected = provenance["sources"]
    result: dict[str, dict[str, str]] = {}
    for label, relative_root in PHYSICAL_SOURCES.items():
        prefix = f"{label}/"
        records = {
            key: digest for key, digest in expected.items() if key.startswith(prefix)
        }
        assert records, label
        for key, digest in records.items():
            relative = Path(key).relative_to(label)
            reference = reference_dir / "sources" / key
            current = source_root / relative_root / relative
            assert benchmark.sha256(reference) == digest, reference
            assert benchmark.sha256(current) == digest, current
        result[label] = records
    return result


def control_for_case(
    results: list[dict[str, Any],],
    reference_dir: Path,
    case: str,
    target_jaw_degrees: float,
) -> dict[str, Any]:
    matches = [
        row for row in results if row["case"] == case and row["method"] == "original"
    ]
    assert len(matches) == 1, (case, len(matches))
    control = copy.deepcopy(matches[0])
    assert control["success"] is True
    assert control["geometry"]["inverted_tetrahedra"] == 0
    assert control["proposal"]["target_jaw_degrees"] == target_jaw_degrees
    assert "ipc_threads" in control
    assert Path(control["output_path"]).name == f"{case}-original.pt"
    output_path = reference_dir / "outputs" / Path(control["output_path"]).name
    assert output_path.is_file(), output_path
    output = benchmark.torch.load(output_path, map_location="cpu", weights_only=False)
    for name in ("displacement", "gradient_q", "gradient_jaw"):
        assert benchmark.tensor_sha256(output[name]) == control[f"{name}_sha256"]
    control["output_path"] = str(output_path)
    control["output_sha256"] = benchmark.sha256(output_path)
    return control


def validate_reference(
    cfg: Config, *, cases: tuple[str, ...]
) -> tuple[dict[str, dict[str, str]], dict[str, Any], dict[str, dict[str, Any]]]:
    """Bind a candidate replay to the exact valid original control evidence."""
    reference_dir = cfg.reference_dir.resolve()
    assert reference_dir.is_dir(), reference_dir
    frozen_path = reference_dir / "frozen-inputs.json"
    provenance_path = reference_dir / "provenance.json"
    results_path = reference_dir / "results.json"
    summary_path = reference_dir / "summary.json"
    frozen = json.loads(frozen_path.read_text())
    provenance = json.loads(provenance_path.read_text())
    results = json.loads(results_path.read_text())
    assert provenance["sources"]
    source_hashes = validate_physical_sources(
        reference_dir=reference_dir, provenance=provenance, source_root=cfg.source_root
    )
    replay_frozen: dict[str, dict[str, str]] = {}
    for case, (directory, config_name) in CHECKPOINTS.items():
        reference = frozen[case]
        source_path = getattr(cfg, config_name)
        assert benchmark.sha256(source_path) == reference["sha256"], source_path
        copied = reference_copy(
            reference_dir, Path(directory) / "initial.pt", reference["sha256"]
        )
        replay_frozen[case] = {
            "source": str(copied),
            "copy": str(copied),
            "sha256": reference["sha256"],
        }
    for key, filename in INPUTS.items():
        reference = frozen[key]
        current = cfg.inputs_dir / filename
        assert benchmark.sha256(current) == reference["sha256"], current
        reference_copy(reference_dir, Path("inputs") / filename, reference["sha256"])
    proposals = {
        "collision_off_pose": cfg.collision_off_jaw_degrees,
        "contact_expression": cfg.contact_jaw_degrees,
    }
    controls = {
        case: control_for_case(results, reference_dir, case, proposals[case])
        for case in cases
    }
    reference_receipt = {
        "directory": str(reference_dir),
        "files": {
            "frozen_inputs": record(frozen_path),
            "provenance": record(provenance_path),
            "results": record(results_path),
            "summary": record(summary_path) if summary_path.exists() else None,
        },
        "batch_complete": summary_path.exists(),
        "controls": controls,
        "physical_source_hashes": source_hashes,
    }
    return replay_frozen, reference_receipt, controls


def main(cfg: Config) -> None:
    methods = benchmark.parse_names(
        cfg.methods, allowed={"exact_pncg", "hybrid_diag", "hybrid_block"}
    )
    cases = benchmark.parse_names(cfg.cases, allowed=set(CHECKPOINTS))
    assert cfg.wall_seconds > 0
    assert 0 < cfg.linear_rtol < 1
    assert cfg.max_newton_steps > 0
    assert cfg.ipc_threads is None or cfg.ipc_threads > 0
    assert cfg.source_root.is_dir(), cfg.source_root
    frozen, reference, controls = validate_reference(cfg, cases=cases)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    benchmark.archive_benchmark_sources(cfg)
    benchmark.write_json(cfg.output_dir / "reference.json", reference)
    from remote_paths import install_loader_path_relocation

    install_loader_path_relocation(source_root=cfg.source_root)
    if cfg.ipc_threads is not None:
        ipctk.set_num_threads(cfg.ipc_threads)
    actual_threads = int(ipctk.get_num_threads())
    for control in controls.values():
        assert control["ipc_threads"] == actual_threads
        assert Path(control["output_path"]).is_file(), control["output_path"]
        control["validity_gate"] = {
            "zero_inverted_tetrahedra": control["geometry"]["inverted_tetrahedra"] == 0
        }
    benchmark.configure_cuda()
    runner = benchmark.load_runner()
    rows: list[dict[str, Any]] = []
    for case in cases:
        control = controls[case]
        for method in methods:
            row = benchmark.run_one(
                cfg, case=case, method=method, frozen=frozen, runner=runner
            )
            row["comparison_to_reference_original"] = benchmark.compare(control, row)
            rows.append(row)
            benchmark.write_json(cfg.output_dir / "results.json", rows)
    summary = {
        "schema": "saved-state-solver-performance-candidate-replay-v1",
        "success": all(
            row["success"]
            and row["comparison_to_reference_original"]["comparable"]
            and all(
                row["comparison_to_reference_original"]["equivalence_gate"].values()
            )
            for row in rows
        ),
        "scope": {
            "control": "completed valid original arm reused without rerunning it",
            "candidates": "requested exact or hybrid accelerated arms only",
            "old_runs_mutated": False,
        },
        "protocol": {
            "reference_dir": str(cfg.reference_dir.resolve()),
            "methods": list(methods),
            "cases": list(cases),
            "proposal_degrees": {
                "collision_off_pose": cfg.collision_off_jaw_degrees,
                "contact_expression": cfg.contact_jaw_degrees,
            },
            "ipc_threads_requested": cfg.ipc_threads,
            "ipc_threads_actual": actual_threads,
        },
        "reference": reference,
        "frozen_inputs": frozen,
        "results": rows,
        "source_sha256": benchmark.sha256(Path(__file__)),
    }
    benchmark.write_json(cfg.output_dir / "summary.json", summary)
    benchmark.cherries.log_metrics(
        {
            "solver_replay/success": float(summary["success"]),
            "solver_replay/arms": len(rows),
        }
    )
    if not summary["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    benchmark.cherries.main(main, profile=benchmark.ProfilePerformance)
