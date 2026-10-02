"""CPU-only exact-state validation for the two saved skin-transmission forwards."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv

HERE = Path(__file__).resolve().parent.parent
RUN = HERE / "data/85-forward-field-skin-transmission"
FIELD_RUN = HERE / "data/80-forward-field-diffusion"
SOURCE = HERE / "data/38-historical-adam-raw6-continuation/final.npz"
FIXTURE = HERE / "data/12-historical-fixture/volume.vtu"
OUTPUT = HERE / "data/86-skin-transmission-validation.json"
SPEC = importlib.util.spec_from_file_location(
    "field_helpers_for_skin_validation", HERE / "src/80-forward-field-diffusion.py"
)
assert SPEC is not None
assert SPEC.loader is not None
FIELD = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = FIELD
SPEC.loader.exec_module(FIELD)


def require(condition: object, message: str) -> None:
    """Fail without a success-looking receipt when a saved-state contract breaks."""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    """Hash an exact read input or generated snapshot."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe one artifact independently of the producer's receipt."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load(path: Path) -> dict[str, np.ndarray]:
    """Copy the arrays out of a saved NPZ file."""
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def verify_mesh(
    fixture: pv.UnstructuredGrid,
    state: dict[str, np.ndarray],
    mesh_path: Path,
    active: np.ndarray,
) -> dict[str, bool]:
    """Verify literal NPZ displacement/activation and immutable fixture topology."""
    mesh = pv.read(mesh_path)
    expected_activation = np.broadcast_to(np.eye(3), (fixture.n_cells, 3, 3)).copy()
    expected_activation[active] = state["Ainv"]
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
        "activation_inverse_exact": bool(
            np.array_equal(
                mesh.cell_data["ActivationInverseMatrix"].reshape(-1, 3, 3),
                expected_activation,
            )
        ),
    }
    require(all(checks.values()), f"NPZ/VTU mismatch: {mesh_path}")
    return checks


def main() -> None:
    """Write one independent receipt for the completed skin-on states."""
    require(not OUTPUT.exists(), f"refusing to overwrite {OUTPUT}")
    summary = json.loads((RUN / "summary.json").read_text())
    exit_receipt = json.loads((RUN / "service-exit-receipt.json").read_text())
    require(summary["status"] == "completed", "skin run did not complete")
    require(
        exit_receipt["service"]["result"] == "success",
        "service did not exit successfully",
    )
    require(
        exit_receipt["service"]["exec_main_status"] == 0, "main exit status is nonzero"
    )
    fixture = pv.read(FIXTURE)
    require(isinstance(fixture, pv.UnstructuredGrid), "fixture topology type changed")
    active = FIELD.BASE.graph_arrays(fixture)["active"]
    source = load(SOURCE)
    q0, seed = source["q"], source["u"]
    require(bool(source["solver_valid"]), "source checkpoint invalid")
    require(
        np.array_equal(source["Ainv"], FIELD.BASE.activation_matrix(q0)),
        "source q/Ainv mismatch",
    )
    expected_materials = {
        "skin_E_MPa": 0.024,
        "skin_nu": 0.46,
        "skin_thickness_m": 0.001,
        "skin_prestrain": 0.0,
        "contact_enabled": False,
        "fat_E_MPa": 0.003,
        "muscle_E_MPa": 0.03,
        "fat_nu": 0.49,
        "muscle_nu": 0.49,
    }
    material_checks = {
        key: bool(summary["materials"][key] == value)
        for key, value in expected_materials.items()
    }
    require(
        all(material_checks.values()), f"material contract mismatch: {material_checks}"
    )
    expected_solver = {
        "max_steps": 5000,
        "rtol": 5e-4,
        "atol": 1e-10,
        "line_search_max_steps": 10,
    }
    require(summary["solver"] == expected_solver, "solver contract mismatch")
    seed_hash = FIELD.array_hash(seed)
    cases = []
    for name, no_skin in (
        ("baseline_skin_0.12", "baseline"),
        ("strong_skin_0.12", "strong"),
    ):
        state_path, mesh_path = RUN / name / "state.npz", RUN / name / "final.vtu"
        state = load(state_path)
        reference = load(FIELD_RUN / no_skin / "state.npz")
        diagnostics = json.loads((RUN / name / "diagnostics.json").read_text())
        checks = {
            "solver_valid": bool(state["solver_valid"]),
            "forward_success": bool(diagnostics["forward"]["success"]),
            "q_exact_no_skin_field": bool(np.array_equal(state["q"], reference["q"])),
            "baseline_q_exact_source": bool(
                name != "baseline_skin_0.12" or np.array_equal(state["q"], q0)
            ),
            "q_ainv_exact": bool(
                np.array_equal(state["Ainv"], FIELD.BASE.activation_matrix(state["q"]))
            ),
            "source_seed_hash_exact": bool(
                diagnostics["seed_sha256"] == seed_hash == summary["seed_sha256"]
            ),
            "same_seed_contract": bool(diagnostics["seed_sha256"] == seed_hash),
        }
        require(all(checks.values()), f"{name}: validation check failed: {checks}")
        mesh_checks = verify_mesh(fixture, state, mesh_path, active)
        cases.append(
            {
                "id": name,
                "checks": checks,
                "mesh_checks": mesh_checks,
                "state": record(state_path),
                "mesh": record(mesh_path),
                "diagnostics": record(RUN / name / "diagnostics.json"),
                "q_sha256": FIELD.array_hash(state["q"]),
                "u_sha256": FIELD.array_hash(state["u"]),
            }
        )
    local, upstream = (
        HERE / "src/85-historical-adam-skin-physics.py",
        HERE / "src/historical_adam_physics.py",
    )
    guard = '        if skin_factor != 0.0:\n            raise ValueError("matched historical comparison requires zero skin energy")\n'
    require(
        local.read_text() == upstream.read_text().replace(guard, "", 1),
        "local skin physics delta is not the approved one-guard deletion",
    )
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scope": "CPU-only independent validation of two completed skin-on saved states",
        "inputs": {
            "run_summary": record(RUN / "summary.json"),
            "service_exit": record(RUN / "service-exit-receipt.json"),
            "solver_stdout": record(RUN / "solver-stdout.log"),
            "source_checkpoint": record(SOURCE),
            "fixture_volume": record(FIXTURE),
            "data80_summary": record(FIELD_RUN / "summary.json"),
        },
        "physics_delta": {
            "upstream": record(upstream),
            "local": record(local),
            "approved_only_delta": "deleted nonzero skin_factor rejection guard",
        },
        "materials": {"checks": material_checks, "actual": summary["materials"]},
        "solver": expected_solver,
        "source_seed_sha256": seed_hash,
        "cases": cases,
    }
    OUTPUT.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
