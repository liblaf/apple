"""Read-only validation of the extended conservative activation-diffusion run."""

# ruff: noqa: PLR0915

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

HERE = Path(__file__).resolve().parent.parent
DEFAULT_RUN = HERE / "data/94-extended-field-diffusion"
DEFAULT_OUTPUT = HERE / "data/97-extended-diffusion-validation.json"
BASE_SPEC = importlib.util.spec_from_file_location(
    "historical_base_for_extended_diffusion_validation",
    HERE / "src/30-run-historical-adam.py",
)
assert BASE_SPEC is not None
assert BASE_SPEC.loader is not None
BASE = importlib.util.module_from_spec(BASE_SPEC)
sys.modules[BASE_SPEC.name] = BASE
BASE_SPEC.loader.exec_module(BASE)

FINITE_CASES = {"r010": 0.1, "r001": 0.01}
CONSTANT_CASE = "component_constant"
PACKED_WEIGHTS = np.asarray((1, 1, 1, 2, 2, 2), dtype=np.float64)


def require(condition: object, message: str) -> None:
    """Fail before producing a receipt that might look like a successful audit."""
    if not condition:
        raise ValueError(message)


def sha256(path: Path) -> str:
    """Hash literal file content."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash an array with dtype and shape, independent of archive compression."""
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def runner_array_hash(array: np.ndarray) -> str:
    """Match data94's archived byte-only array-hash convention exactly."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Make an independently computed artifact record."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Copy state arrays before closing the archive."""
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def component_means(
    components: np.ndarray, mass: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Compute volume-times-fraction weighted six-vector means by component."""
    component_mass = np.bincount(components, weights=mass)
    return np.stack(
        [
            np.bincount(components, weights=mass * q[:, column]) / component_mass
            for column in range(q.shape[1])
        ],
        axis=1,
    )


def relative_residuals(
    matrix: sp.spmatrix, mass: np.ndarray, q: np.ndarray, q0: np.ndarray
) -> list[float]:
    """Evaluate the finite-tau diffusion equations after saved mean correction."""
    values = []
    for column in range(q.shape[1]):
        rhs = mass * q0[:, column]
        numerator = float(np.linalg.norm(matrix @ q[:, column] - rhs))
        denominator = float(np.linalg.norm(rhs))
        values.append(numerator if denominator == 0.0 else numerator / denominator)
    return values


def verify_record(receipt: dict[str, Any]) -> None:
    """Verify an archived runner record still names the current artifact bytes."""
    path = Path(receipt["path"])
    require(path.is_file(), f"recorded input missing: {path}")
    require(path.stat().st_size == receipt["bytes"], f"recorded size changed: {path}")
    require(sha256(path) == receipt["sha256"], f"recorded hash changed: {path}")


def verify_mesh(
    fixture: pv.UnstructuredGrid,
    active: np.ndarray,
    state: dict[str, np.ndarray],
    path: Path,
) -> dict[str, bool]:
    """Ensure a VTK endpoint is exactly the saved NPZ on fixture topology."""
    mesh = pv.read(path)
    require(isinstance(mesh, pv.UnstructuredGrid), f"not an unstructured grid: {path}")
    full_ainv = np.broadcast_to(np.eye(3), (fixture.n_cells, 3, 3)).copy()
    full_ainv[active] = state["Ainv"]
    checks = {
        "point_count": mesh.n_points == fixture.n_points,
        "cell_count": mesh.n_cells == fixture.n_cells,
        "cell_types_exact": np.array_equal(mesh.celltypes, fixture.celltypes),
        "cells_exact": np.array_equal(mesh.cells, fixture.cells),
        "points_exact": np.array_equal(mesh.points, fixture.points + state["u"]),
        "rest_position_exact": np.array_equal(
            mesh.point_data["RestPosition"], fixture.points
        ),
        "displacement_exact": np.array_equal(
            mesh.point_data["Displacement"], state["u"]
        ),
        "activation_inverse_exact": np.array_equal(
            mesh.cell_data["ActivationInverseMatrix"].reshape(-1, 3, 3), full_ainv
        ),
    }
    require(all(checks.values()), f"NPZ/VTU mismatch for {path}: {checks}")
    return checks


def quadratic_roughness(
    q: np.ndarray,
    edges: tuple[np.ndarray, np.ndarray, np.ndarray],
    mass: np.ndarray,
    length: float,
) -> float:
    """Use the published R normalization over a supplied edge set."""
    i, j, weight = edges
    difference = q[i] - q[j]
    return float(
        length**2
        * np.sum(weight[:, None] * PACKED_WEIGHTS[None, :] * difference**2 / 1.5)
        / mass.sum()
        / BASE.AREF**2
    )


def cross_muscle_audit(
    fixture: pv.UnstructuredGrid,
    graph: dict[str, np.ndarray],
    fields: dict[str, np.ndarray],
    length: float,
) -> dict[str, Any]:
    """Describe cross-muscle discontinuities without changing the diffusion operator."""
    active = graph["active"]
    tets = np.asarray(fixture.cells, dtype=np.int64).reshape(-1, 5)[:, 1:]
    fraction = np.asarray(fixture.cell_data["MuscleFraction"], dtype=np.float64)
    all_i, all_j, all_weight = BASE.active_graph(
        np.asarray(fixture.points, dtype=np.float64),
        tets,
        active,
        np.zeros(len(active), dtype=np.int64),
        fraction,
    )
    muscle = np.asarray(fixture.cell_data["MuscleId"], dtype=np.int64)[active]
    same = muscle[all_i] == muscle[all_j]
    within = (all_i[same], all_j[same], all_weight[same])
    cross = (all_i[~same], all_j[~same], all_weight[~same])
    require(
        np.array_equal(within[0], graph["i"])
        and np.array_equal(within[1], graph["j"])
        and np.allclose(within[2], graph["weight"], rtol=0.0, atol=0.0),
        "all-active adjacency does not recover the original within-muscle graph",
    )
    measures = {}
    for name, q in fields.items():
        within_r = quadratic_roughness(q, within, graph["volume"], length)
        cross_r = quadratic_roughness(q, cross, graph["volume"], length)
        all_r = quadratic_roughness(
            q, (all_i, all_j, all_weight), graph["volume"], length
        )
        require(
            np.isclose(all_r, within_r + cross_r, rtol=1e-12, atol=1e-20),
            f"{name}: all-edge roughness does not decompose",
        )
        measures[name] = {
            "R_within_same_MuscleId": within_r,
            "R_cross_MuscleId": cross_r,
            "R_all_active_shared_faces": all_r,
            "R_all_equals_within_plus_cross": True,
        }
    return {
        "scope": "descriptive only; the executed diffusion still used only same-MuscleId shared-face edges",
        "active_inactive_interfaces": "excluded: this audit contains only pairs of active tetrahedra sharing a face",
        "all_active_shared_face_edge_count": len(all_i),
        "within_same_MuscleId_edge_count": int(same.sum()),
        "cross_MuscleId_edge_count": int((~same).sum()),
        "measures": measures,
    }


def main(run_dir: Path, output: Path) -> None:
    """Audit a completed data94 run without invoking Cherries or a forward solve."""
    require(run_dir.is_dir(), f"run directory missing: {run_dir}")
    require(not output.exists(), f"refusing to overwrite receipt: {output}")
    summary_path = run_dir / "summary.json"
    require(summary_path.is_file(), "wait for data94 summary before validation")
    summary = json.loads(summary_path.read_text())
    require(
        summary["status"] == "completed", f"run not fully valid: {summary['status']}"
    )
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text())
    fixture_path = Path(config["fixture"]) / "volume.vtu"
    checkpoint_path = Path(config["source_checkpoint"])
    require(
        fixture_path.is_file() and checkpoint_path.is_file(), "recorded input missing"
    )
    require(
        tuple(config)
        == ("fixture", "source_checkpoint", "output_dir", "smooth_length"),
        "unexpected data94 config contract",
    )
    require(float(config["smooth_length"]) == 0.005, "smooth length changed")
    expected_materials = {
        "fat_E_MPa": 0.003,
        "fat_model": "stable",
        "fat_nu": 0.49,
        "muscle_E_MPa": 0.03,
        "muscle_model": "stable-active",
        "muscle_nu": 0.49,
        "skin_E_MPa": 0.0,
        "skin_prestrain": 0.0,
        "contact_enabled": False,
    }
    material_checks = {
        key: summary["materials"].get(key) == value
        for key, value in expected_materials.items()
    }
    require(
        all(material_checks.values()), f"material contract changed: {material_checks}"
    )
    for receipt in summary["provenance"]["inputs"]:
        verify_record(receipt)
    for receipt in summary["provenance"]["sources"]:
        verify_record(receipt)
    fixture = pv.read(fixture_path)
    require(
        isinstance(fixture, pv.UnstructuredGrid), "fixture is not an unstructured grid"
    )
    graph = BASE.graph_arrays(fixture)
    active, mass = graph["active"], graph["volume"]
    adjacency = sp.coo_matrix(
        (
            np.r_[graph["weight"], graph["weight"]],
            (np.r_[graph["i"], graph["j"]], np.r_[graph["j"], graph["i"]]),
        ),
        shape=(len(active), len(active)),
    ).tocsr()
    laplacian = sp.diags(np.asarray(adjacency.sum(axis=1)).ravel()) - adjacency
    n_components, components = connected_components(adjacency)
    source = load_npz(checkpoint_path)
    q0, seed = source["q"], source["u"]
    require(bool(source["solver_valid"]), "source checkpoint is not solver-valid")
    require(q0.shape == (len(active), 6), "source q shape differs from active graph")
    require(
        np.array_equal(source["Ainv"], BASE.activation_matrix(q0)),
        "source q/Ainv mismatch",
    )
    means0 = component_means(components, mass, q0)
    r0 = BASE.smoothness_numpy(q0, graph, float(config["smooth_length"]))
    source_hash, seed_hash = array_hash(q0), runner_array_hash(seed)
    prep = json.loads((run_dir / "field-preparation.json").read_text())
    require(prep["source_roughness"] == r0, "preflight source roughness differs")
    require(prep["seed_sha256"] == seed_hash, "preflight source seed differs")
    cases_by_id = {case["id"]: case for case in summary["cases"]}
    require(
        set(cases_by_id) == {*FINITE_CASES, CONSTANT_CASE}, "unexpected data94 cases"
    )
    fields: dict[str, np.ndarray] = {"baseline80": q0}
    validation_cases = []
    tolerance = {"max_steps": 5000, "rtol": 5e-4, "atol": 1e-10}
    for name, case in cases_by_id.items():
        case_dir = run_dir / name
        state_path, mesh_path, diagnostics_path = (
            case_dir / "state.npz",
            case_dir / "final.vtu",
            case_dir / "diagnostics.json",
        )
        state, diagnostics = (
            load_npz(state_path),
            json.loads(diagnostics_path.read_text()),
        )
        q = state["q"]
        require(q.shape == q0.shape, f"{name}: q shape mismatch")
        require(bool(state["solver_valid"]), f"{name}: non-valid state was saved")
        require(
            np.array_equal(state["Ainv"], BASE.activation_matrix(q)),
            f"{name}: q/Ainv mismatch",
        )
        require(
            case["status"] == "equilibrium_valid" and case["forward"]["success"],
            f"{name}: invalid forward",
        )
        require(
            diagnostics["q_sha256"] == runner_array_hash(q) == case["q_sha256"],
            f"{name}: q receipt mismatch",
        )
        require(
            diagnostics["seed_sha256"] == seed_hash == case["seed_sha256"],
            f"{name}: seed mismatch",
        )
        actual_tolerance = diagnostics["forward"]["tolerance"]
        tolerance_checks = {
            key: actual_tolerance[key] == value for key, value in tolerance.items()
        }
        require(all(tolerance_checks.values()), f"{name}: forward tolerances changed")
        roughness = BASE.smoothness_numpy(q, graph, float(config["smooth_length"]))
        ratio = roughness / r0
        field = diagnostics["field"]
        mean_error = float(
            np.max(np.abs(component_means(components, mass, q) - means0))
        )
        checks: dict[str, Any] = {
            "same_original_source_q": case["seed_sha256"] == seed_hash,
            "component_means_conserved": mean_error < 1e-11,
            "roughness_matches_diagnostics": np.isclose(
                roughness, diagnostics["roughness"], rtol=1e-12, atol=0.0
            ),
            "ratio_matches_diagnostics": np.isclose(
                ratio, field["roughness_ratio"], rtol=1e-12, atol=0.0
            ),
        }
        if name in FINITE_CASES:
            target = FINITE_CASES[name]
            tau = float(field["tau_m2"])
            require(tau > 0.0, f"{name}: finite diffusion needs positive tau")
            residuals = relative_residuals(
                sp.diags(mass) + tau * laplacian, mass, q, q0
            )
            checks["requested_ratio_1e-4"] = abs(ratio - target) < 1e-4
            checks["post_correction_linear_residual_max_1e-10"] = max(residuals) < 1e-10
        else:
            expected = means0[components]
            residuals = None
            checks["limit_label"] = field["limit"].startswith("tau tends to infinity")
            checks["zero_roughness"] = roughness == 0.0 and ratio == 0.0
            checks["literal_weighted_component_means"] = np.array_equal(q, expected)
            difference = q[graph["i"]] - q[graph["j"]]
            checks["within_component_edges_near_zero"] = (
                float(np.max(np.abs(difference))) < 1e-13
            )
        require(all(checks.values()), f"{name}: validation failed: {checks}")
        checks = {key: bool(value) for key, value in checks.items()}
        mesh_checks = verify_mesh(fixture, active, state, mesh_path)
        fields[f"{name}94"] = q
        validation_cases.append(
            {
                "id": name,
                "field": {
                    "tau_m2": field["tau_m2"],
                    "recomputed_roughness": roughness,
                    "recomputed_roughness_ratio": ratio,
                    "post_correction_relative_residuals": residuals,
                    "post_correction_component_mean_max_abs_error": mean_error,
                },
                "checks": {**checks, "forward_tolerances": tolerance_checks},
                "state": record(state_path),
                "diagnostics": record(diagnostics_path),
                "mesh": {"record": record(mesh_path), "checks": mesh_checks},
            }
        )
    baseline80 = HERE / "data/80-forward-field-diffusion"
    for name in ("baseline", "mild", "strong"):
        path = baseline80 / name / "state.npz"
        require(path.is_file(), f"missing data80 comparison state: {path}")
        fields[f"{name}80"] = load_npz(path)["q"]
    cross_audit = cross_muscle_audit(
        fixture, graph, fields, float(config["smooth_length"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scope": "read-only saved-field validation of data94 extended same-source diffusion",
        "inputs": {
            "summary": record(summary_path),
            "config": record(config_path),
            "provenance": record(run_dir / "provenance.json"),
            "field_preparation": record(run_dir / "field-preparation.json"),
            "fixture_volume": record(fixture_path),
            "source_checkpoint": record(checkpoint_path),
            "baseline80_summary": record(baseline80 / "summary.json"),
        },
        "source": {
            "q_hash": source_hash,
            "seed_hash": seed_hash,
            "roughness": r0,
            "active_tetrahedra": len(active),
            "connected_same_MuscleId_components": int(n_components),
        },
        "material_contract_checks": material_checks,
        "forward_tolerance": tolerance,
        "cases": validation_cases,
        "cross_muscle_boundary_audit": cross_audit,
    }
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    main(arguments.run_dir, arguments.output)
