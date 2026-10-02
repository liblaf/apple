"""Finalize the terminal partial outcome of the active-tension face diagnostic."""

# ruff: noqa: EM101, EM102, TRY003

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = ROOT.parents[4]
PRODUCER = ROOT / "data/88-active-tension-face"
PRIOR_FAILED = ROOT / "data/88-active-tension-face-failed-duplicate-npz-key"
SERVICE_LOG = ROOT / "logs/88-active-tension-face-service.log"
PRIOR_FAILED_LOG = ROOT / "logs/88-active-tension-face-failed-duplicate-npz-key.log"
OUTPUT = ROOT / "data/93-active-tension-face-outcome.json"
ACCEPTED_LABELS = ("gain-0000", "gain-0100", "gain-0300")
FAILED_LABEL = "gain-1000"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def records_under(root: Path) -> dict[str, dict[str, object]]:
    return {
        str(path.relative_to(root)): record(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"expected an object in {path}")
    return value


def accepted_case(label: str) -> dict[str, object]:
    path = PRODUCER / "cases" / label
    diagnostics = read_json(path / "diagnostics.json")
    forward = diagnostics["forward"]
    physical = diagnostics["physical_deformation"]
    surface = diagnostics["surface_motion"]
    muscle = diagnostics["selected_muscle_motion"]
    lip = diagnostics["lip_motion"]
    if not forward["success"]:
        raise AssertionError(f"accepted case {label} has a failed solver receipt")
    if physical["inverted_tets_all"] != 0:
        raise AssertionError(f"accepted case {label} has inverted tetrahedra")
    artifacts = {
        name: record(path / name)
        for name in ("diagnostics.json", "state.npz", "state.vtu")
    }
    return {
        "label": label,
        "status": "accepted_equilibrium",
        "gain": diagnostics["gain"],
        "tension_MPa": diagnostics["tension_MPa"],
        "seed": diagnostics["seed"],
        "target_role": diagnostics["target_role"],
        "solver": forward,
        "geometry": {
            "detF_all": physical["detF_all"],
            "detF_selected": physical["detF_selected"],
            "inverted_tets_all": physical["inverted_tets_all"],
            "inverted_tets_selected": physical["inverted_tets_selected"],
            "fiber_stretch_selected": physical["fiber_stretch_F_selected"],
            "fiber_stretch_fraction_volume_weighted_mean": physical[
                "fiber_stretch_F_selected_fraction_volume_weighted_mean"
            ],
        },
        "motion": {
            "muscle_centroid_rms_mm": muscle["centroid_displacement_rms_mm"],
            "surface_rms_mm": surface["weighted_rms_mm"],
            "surface_max_mm": surface["weighted_max_mm"],
            "smile_projection": surface["smile_target_projection_amplitude"],
            "lip_rms_mm": lip["rms_mm"],
            "lip_radial_outward_mean_mm": lip["mean_outward_radial_xy_mm"],
        },
        "stress": diagnostics["stress"],
        "artifacts": artifacts,
    }


def failed_case() -> dict[str, object]:
    path = PRODUCER / "cases" / FAILED_LABEL
    failure = read_json(path / "failure.json")
    forward = failure["forward"]
    if forward["success"]:
        raise AssertionError("gain 10 unexpectedly has a successful solver receipt")
    if forward["result"] != "max_steps_reached" or forward["steps"] != 10000:
        raise AssertionError("gain 10 failure is not the declared fixed-budget outcome")
    forbidden = [
        name
        for name in ("diagnostics.json", "state.npz", "state.vtu")
        if (path / name).exists()
    ]
    if forbidden:
        raise AssertionError(
            f"failed gain 10 contains invented state artifacts: {forbidden}"
        )
    return {
        "label": FAILED_LABEL,
        "status": "failed_forward_no_state",
        "gain": failure["gain"],
        "tension_MPa": failure["tension_MPa"],
        "seed": failure["seed"],
        "failure_type": "ForwardConvergenceError",
        "failure_reason": "The unchanged 10,000-step forward budget was exhausted before the unchanged convergence tolerance was met.",
        "solver": forward,
        "artifacts": {"failure.json": record(path / "failure.json")},
        "absent_state_artifacts": forbidden == [],
        "geometry_claim": None,
    }


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    config = read_json(PRODUCER / "config.json")
    provenance = read_json(PRODUCER / "provenance.json")
    warp_audit = read_json(PRODUCER / "warp-audit.json")
    if config["gains"] != [0.0, 1.0, 3.0, 10.0]:
        raise AssertionError("producer gain list changed")
    if config["forward_rtol"] != 1.0e-5 or config["forward_atol"] != 1.0e-12:
        raise AssertionError("producer tolerance changed")
    if config["muscle_factor"] != 0.8 or config["soft_nu"] != 0.46:
        raise AssertionError("producer muscle material changed")
    frozen_source = PRODUCER / "sources/88-active-tension-face.py"
    current_source = ROOT / "src/88-active-tension-face.py"
    if sha256(frozen_source) != sha256(current_source):
        raise AssertionError("current source88 differs from the executed frozen source")
    missing_terminals = [
        name
        for name in ("summary.json", "manifest.json")
        if not (PRODUCER / name).exists()
    ]
    if missing_terminals != ["summary.json", "manifest.json"]:
        raise AssertionError("unexpected producer terminal-file state")
    log_text = SERVICE_LOG.read_text()
    required_log_evidence = (
        "https://www.comet.com/liblaf/apple/76c20908ab484430a5c6db268fde038d",
        "cherries/exception  : face_physics.ForwardConvergenceError",
        "'result': 'max_steps_reached'",
        "'steps': 10000",
    )
    for evidence in required_log_evidence:
        if evidence not in log_text:
            raise AssertionError(f"service log lacks terminal evidence: {evidence}")
    accepted = [accepted_case(label) for label in ACCEPTED_LABELS]
    failed = failed_case()
    producer_files = records_under(PRODUCER)
    receipt = {
        "schema_version": 1,
        "status": "partial_forward_failure",
        "generated_at": datetime.now().astimezone().isoformat(),
        "scope": "Terminal receipt for the fixed-budget, target-independent, no-skin active-tension face diagnostic.",
        "interpretation": "Gains 0, 1, and 3 are accepted equilibria. Gain 10 is a recorded forward failure and has no accepted state or geometry.",
        "producer": {
            "directory": str(PRODUCER.resolve()),
            "git_sha": "d56fa1b553b287b22b2cf7bb82d46117e34ed6bb",
            "executed_source": record(frozen_source),
            "current_source_matches_executed": True,
            "config": {"record": record(PRODUCER / "config.json"), "value": config},
            "provenance": {
                "record": record(PRODUCER / "provenance.json"),
                "value": provenance,
            },
            "warp_audit": {
                "record": record(PRODUCER / "warp-audit.json"),
                "value": warp_audit,
            },
            "recursive_files": producer_files,
            "recursive_file_count": len(producer_files),
            "terminal_files_absent_due_failure": missing_terminals,
        },
        "run_contract": {
            "gains": [0.0, 1.0, 3.0, 10.0],
            "same_zero_displacement_seed_each_gain": True,
            "continuation": False,
            "inverse_optimization": False,
            "target_in_equilibrium": False,
            "skin_enabled": False,
            "geometry_rejection_gate": False,
            "activation_inverse": "identity for every cell and every gain",
            "muscle_E_MPa": 0.024,
            "muscle_nu": 0.46,
            "reference_tension_MPa": 0.024657534246575345,
            "forward_max_steps": 10000,
            "forward_rtol": 1.0e-5,
            "forward_atol": 1.0e-12,
        },
        "service": {
            "unit": "apple-face-active-tension-88.service",
            "invocation_id": "51e829e8c21a4810933543e624cdf499",
            "main_pid_at_launch": 793581,
            "unit_launch_timestamp": "2026-09-07T16:51:34+08:00",
            "cherries_start_timestamp": "2026-09-07T16:51:40.062692+08:00",
            "cherries_end_timestamp": "2026-09-07T16:55:03.694184+08:00",
            "comet_url": "https://www.comet.com/liblaf/apple/76c20908ab484430a5c6db268fde038d",
            "service_log": record(SERVICE_LOG),
            "observed_systemd_result": "success",
            "observed_exec_main_status": 0,
            "experiment_result": "partial_forward_failure",
            "status_discrepancy": "The Cherries exception hook recorded the ForwardConvergenceError but the process exited 0, so systemd success is not experiment success.",
        },
        "accepted_cases": accepted,
        "failed_cases": [failed],
        "renderable_case_labels": list(ACCEPTED_LABELS),
        "non_renderable_case_labels": [FAILED_LABEL],
        "prior_runner_failure": {
            "status": "superseded_runner_serialization_failure",
            "directory": str(PRIOR_FAILED.resolve()),
            "service_invocation_id": "60cc57bb773a4bb4a986a90a171bde8b",
            "reason": "The first attempt solved gain 0, then passed selected_active_local_ids twice to numpy.savez_compressed.",
            "recursive_files": records_under(PRIOR_FAILED),
            "service_log": record(PRIOR_FAILED_LOG),
            "used_for_physics_results": False,
        },
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(OUTPUT), **record(OUTPUT)}, indent=2))


if __name__ == "__main__":
    main()
