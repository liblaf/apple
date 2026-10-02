"""CPU-only verification of accepted active-tension forward states and failure receipt."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "data/88-active-tension-face"
OUTCOME = ROOT / "data/93-active-tension-face-outcome.json"
OUTPUT = ROOT / "data/92-active-tension-face-validation.json"
GAINS = (0.0, 1.0, 3.0, 10.0)
ACCEPTED = GAINS[:-1]
SMILE_IDS = (57, 58, 63, 64, 142, 143, 218, 219, 283, 284)


def require(condition: object, message: str) -> None:
    """Stop before writing a receipt that could be mistaken for validation."""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    """Hash a fixture, saved state, or provenance source."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Make a local content record independently from source88's receipts."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load(path: Path) -> dict[str, np.ndarray]:
    """Copy NPZ arrays while their archive is open."""
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def check_recorded_hashes(document: dict[str, Any], key: str) -> int:
    """Rehash each source88 provenance source/input mapping entry."""
    count = 0
    for name, receipt in document[key].items():
        path = Path(receipt["path"])
        require(path.is_file(), f"missing {key} {name}: {path}")
        require(path.stat().st_size == receipt["bytes"], f"size changed: {path}")
        require(sha256(path) == receipt["sha256"], f"hash changed: {path}")
        count += 1
    return count


def expected_tension(
    n_cells: int, selected: np.ndarray, gain: float, reference: float
) -> np.ndarray:
    """Construct the sole permitted spatial active-tension field."""
    field = np.zeros(n_cells)
    field[selected] = gain * reference
    return field


def verify_mesh(
    fixture: pv.UnstructuredGrid,
    mesh: pv.UnstructuredGrid,
    state: dict[str, np.ndarray],
    selected: np.ndarray,
    gain: float,
    reference: float,
) -> dict[str, bool]:
    """Check topology, saved displacement, identity activation, and exact tension field."""
    identity = np.broadcast_to(np.eye(3), (fixture.n_cells, 3, 3))
    pattern = np.zeros(fixture.n_cells, dtype=np.int8)
    pattern[selected] = 1
    gain_field = np.zeros(fixture.n_cells)
    gain_field[selected] = gain
    checks = {
        "point_count": bool(mesh.n_points == fixture.n_points),
        "cell_count": bool(mesh.n_cells == fixture.n_cells),
        "cell_types_exact": bool(np.array_equal(mesh.celltypes, fixture.celltypes)),
        "cells_exact": bool(np.array_equal(mesh.cells, fixture.cells)),
        "points_exact": bool(np.array_equal(mesh.points, fixture.points + state["u"])),
        "rest_position_exact": bool(
            np.array_equal(mesh.point_data["RestPosition"], fixture.points)
        ),
        "displacement_exact": bool(
            np.array_equal(mesh.point_data["Displacement"], state["u"])
        ),
        "activation_inverse_identity": bool(
            np.array_equal(
                mesh.cell_data["ActivationInverseMatrix"].reshape(-1, 3, 3), identity
            )
        ),
        "tension_pattern_exact": bool(
            np.array_equal(mesh.cell_data["ActiveTensionPattern"], pattern)
        ),
        "gain_field_exact": bool(
            np.array_equal(mesh.cell_data["ActiveTensionGain"], gain_field)
        ),
        "tension_field_exact": bool(
            np.array_equal(
                mesh.cell_data["ActiveTensionMPa"],
                expected_tension(fixture.n_cells, selected, gain, reference),
            )
        ),
    }
    require(all(checks.values()), f"saved VTK contract mismatch: {checks}")
    return checks


def expected_row(
    row: dict[str, str], diagnostics: dict[str, Any], gain: float, tension: float
) -> dict[str, bool]:
    """Require the CSV trace to agree exactly with each accepted case receipt."""
    checks = {
        "case": row["case"] == diagnostics["case"],
        "gain": float(row["gain"]) == gain == diagnostics["gain"],
        "tension": float(row["tension_MPa"]) == tension == diagnostics["tension_MPa"],
        "surface_rms": float(row["surface_rms_mm"])
        == diagnostics["surface_motion"]["weighted_rms_mm"],
        "projection": float(row["smile_projection"])
        == diagnostics["surface_motion"]["smile_target_projection_amplitude"],
        "detf": float(row["detF_min_all"])
        == diagnostics["physical_deformation"]["detF_all"]["min"],
        "forward_steps": int(row["forward_steps"]) == diagnostics["forward"]["steps"],
    }
    require(all(checks.values()), f"trace/diagnostic mismatch: {checks}")
    return checks


def verify_outcome_record(outcome: dict[str, Any], reference: float) -> dict[str, bool]:
    """Require data93 to describe this partial result, rather than merely name it."""
    producer = outcome["producer"]
    contract = outcome["run_contract"]
    service = outcome["service"]
    accepted = outcome["accepted_cases"]
    failed = outcome["failed_cases"]
    recursive = producer["recursive_files"]
    expected_labels = [f"gain-{round(100 * gain):04d}" for gain in ACCEPTED]
    checks = {
        "schema": outcome["schema_version"] == 1,
        "producer_directory": Path(producer["directory"]).resolve() == RUN.resolve(),
        "terminal_files_absent": producer["terminal_files_absent_due_failure"]
        == ["summary.json", "manifest.json"],
        "contract": contract["gains"] == list(GAINS)
        and contract["same_zero_displacement_seed_each_gain"]
        and not contract["continuation"]
        and not contract["inverse_optimization"]
        and not contract["target_in_equilibrium"]
        and not contract["skin_enabled"]
        and contract["activation_inverse"] == "identity for every cell and every gain"
        and contract["muscle_E_MPa"] == 0.024
        and contract["muscle_nu"] == 0.46
        and contract["reference_tension_MPa"] == reference
        and contract["forward_max_steps"] == 10000
        and contract["forward_rtol"] == 1e-5
        and contract["forward_atol"] == 1e-12,
        "service_discrepancy": service["observed_systemd_result"] == "success"
        and service["observed_exec_main_status"] == 0
        and service["experiment_result"] == "partial_forward_failure"
        and "exited 0" in service["status_discrepancy"]
        and "not experiment success" in service["status_discrepancy"],
        "accepted_labels": [case["label"] for case in accepted] == expected_labels
        and outcome["renderable_case_labels"] == expected_labels,
        "accepted_artifacts": all(
            case["status"] == "accepted_equilibrium"
            and all(
                recursive[f"cases/{case['label']}/{name}"] == receipt
                for name, receipt in case["artifacts"].items()
            )
            for case in accepted
        ),
        "failed_case": len(failed) == 1
        and failed[0]["label"] == "gain-1000"
        and failed[0]["status"] == "failed_forward_no_state"
        and failed[0]["gain"] == 10.0
        and failed[0]["tension_MPa"] == 10.0 * reference
        and failed[0]["seed"] == "zero displacement"
        and not failed[0]["solver"]["success"]
        and failed[0]["solver"]["steps"] == 10000
        and failed[0]["solver"]["result"] == "max_steps_reached"
        and outcome["non_renderable_case_labels"] == ["gain-1000"]
        and failed[0]["absent_state_artifacts"] is True,
    }
    require(all(checks.values()), f"data93 contract mismatch: {checks}")
    return checks


def main() -> None:
    """Validate partial accepted output without invoking Warp or a face solve."""
    require(not OUTPUT.exists(), f"refusing to overwrite {OUTPUT}")
    require(OUTCOME.is_file(), "wait for data93 terminal outcome receipt")
    outcome = json.loads(OUTCOME.read_text())
    require(
        outcome["status"] == "partial_forward_failure",
        f"unexpected terminal status: {outcome['status']}",
    )
    config = json.loads((RUN / "config.json").read_text())
    fixture_dir = Path(config["fixture"])
    fixture_path = fixture_dir / "volume.vtu"
    fixture = pv.read(fixture_path)
    require(
        isinstance(fixture, pv.UnstructuredGrid), "fixture is not an unstructured grid"
    )
    active = np.asarray(fixture.cell_data["ActivationMask"], dtype=bool)
    muscle = np.asarray(fixture.cell_data["MuscleId"], dtype=int)
    selected = active & np.isin(muscle, SMILE_IDS)
    fibers = np.asarray(fixture.cell_data["ActivationFiber"], dtype=np.float64)
    norms = np.linalg.norm(fibers[selected], axis=1)
    require(
        np.allclose(norms, 1.0, rtol=1e-12, atol=1e-12), "selected fibers are not unit"
    )
    require(tuple(config["gains"]) == GAINS, "gains differ from approved contract")
    require(
        config["muscle_factor"] == 0.8 and config["soft_nu"] == 0.46,
        "passive muscle contract changed",
    )
    require(
        config["fat_factor"] == 1.0
        and config["fat_nu"] == 0.49
        and config["fat_model"] == "stable",
        "passive fat contract changed",
    )
    require(
        config["forward_rtol"] == 1e-5 and config["forward_atol"] == 1e-12,
        "forward tolerance changed",
    )
    provenance = json.loads((RUN / "provenance.json").read_text())
    provenance_count = check_recorded_hashes(
        provenance, "sources"
    ) + check_recorded_hashes(provenance, "inputs")
    trace_path = RUN / "trace.csv"
    with trace_path.open(newline="") as stream:
        trace = {row["case"]: row for row in csv.DictReader(stream)}
    require(
        set(trace) == {f"gain-{round(100 * gain):04d}" for gain in ACCEPTED},
        "trace contains unaccepted or missing case",
    )
    reference = 3.0 * (0.024 / (2.0 * (1.0 + 0.46)))
    outcome_checks = verify_outcome_record(outcome, reference)
    cases = []
    for gain in ACCEPTED:
        name = f"gain-{round(100 * gain):04d}"
        directory = RUN / "cases" / name
        state_path, vtu_path, diagnostics_path = (
            directory / "state.npz",
            directory / "state.vtu",
            directory / "diagnostics.json",
        )
        state, mesh, diagnostics = (
            load(state_path),
            pv.read(vtu_path),
            json.loads(diagnostics_path.read_text()),
        )
        require(
            isinstance(mesh, pv.UnstructuredGrid), f"not an unstructured VTK: {name}"
        )
        tension = gain * reference
        checks = {
            "gain_exact": bool(
                np.asarray(state["gain"]).item() == gain == diagnostics["gain"]
            ),
            "tension_exact": bool(
                np.asarray(state["tension_MPa"]).item()
                == tension
                == diagnostics["tension_MPa"]
            ),
            "zero_seed_recorded": diagnostics["seed"]
            == "zero displacement; no continuation from another gain",
            "activation_identity_recorded": diagnostics["activation_inverse"][
                "encoding_max_abs"
            ]
            == 0.0,
            "forward_success": bool(diagnostics["forward"]["success"]),
            "selected_ids_exact": bool(
                np.array_equal(
                    np.sort(state["selected_cell_ids"]), np.flatnonzero(selected)
                )
            ),
        }
        require(all(checks.values()), f"NPZ/diagnostic mismatch {name}: {checks}")
        mesh_checks = verify_mesh(fixture, mesh, state, selected, gain, reference)
        trace_checks = expected_row(trace[name], diagnostics, gain, tension)
        cases.append(
            {
                "id": name,
                "gain": gain,
                "tension_MPa": tension,
                "npz_checks": checks,
                "mesh_checks": mesh_checks,
                "trace_checks": trace_checks,
                "state": record(state_path),
                "mesh": record(vtu_path),
                "diagnostics": record(diagnostics_path),
            }
        )
    failure_path = RUN / "cases/gain-1000/failure.json"
    failure = json.loads(failure_path.read_text())
    failure_checks = {
        "root_failure_exact_copy": (RUN / "failure.json").read_bytes()
        == failure_path.read_bytes(),
        "gain_is_ten": failure["gain"] == 10.0,
        "tension_is_gain_times_3mu": failure["tension_MPa"] == 10.0 * reference,
        "zero_seed": failure["seed"] == "zero displacement",
        "forward_is_failure": failure["forward"]["success"] is False,
        "max_steps_recorded": failure["forward"]["steps"] == 10000
        and failure["forward"]["result"] == "max_steps_reached",
        "no_converged_gain10_state": not (RUN / "cases/gain-1000/state.npz").exists()
        and not (RUN / "cases/gain-1000/state.vtu").exists(),
    }
    require(
        all(failure_checks.values()),
        f"gain10 failure receipt mismatch: {failure_checks}",
    )
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scope": "CPU-only validation of accepted gains 0/1/3 and explicit fixed-budget gain10 failure",
        "inputs": {
            "outcome": record(OUTCOME),
            "config": record(RUN / "config.json"),
            "provenance": record(RUN / "provenance.json"),
            "trace": record(trace_path),
            "fixture_volume": record(fixture_path),
            "gain10_failure": record(failure_path),
        },
        "provenance_records_rehashed": provenance_count,
        "contract": {
            "gains": GAINS,
            "accepted_gains": ACCEPTED,
            "smile_ids": SMILE_IDS,
            "selected_cell_count": int(selected.sum()),
            "selected_fiber_norm_min": float(norms.min()),
            "selected_fiber_norm_max": float(norms.max()),
            "reference_tension_MPa": reference,
        },
        "accepted_cases": cases,
        "gain10_failure_checks": failure_checks,
        "outcome_record_checks": outcome_checks,
        "process_exit_is_not_forward_success": True,
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
