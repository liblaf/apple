"""Independently validate saved conservative activation-field diffusion outputs."""

# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.sparse as sp
from experiment_profile import ProfileCometNoCommit
from scipy.sparse.csgraph import connected_components

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "historical_base_for_field_validation", HERE / "src/30-run-historical-adam.py"
)
assert SPEC is not None
assert SPEC.loader is not None
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


class Config(cherries.BaseConfig):
    """Read-only verifier configuration for one completed field-diffusion run."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = HERE / "data/12-historical-fixture"
    run_dir: Path = HERE / "data/80-forward-field-diffusion"
    output: Path = HERE / "data/84-field-diffusion-validation.json"


def sha256(path: Path) -> str:
    """Hash a file without relying on records made by the evaluated run."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash the literal saved array bytes and shape/dtype contract."""
    hasher = hashlib.sha256()
    hasher.update(str(array.dtype).encode())
    hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    """Return a self-contained record for a read input or saved snapshot."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def require(condition: object, message: str) -> None:
    """Fail before emitting a success-looking validation receipt."""
    if not condition:
        raise ValueError(message)


def loaded(path: Path) -> dict[str, np.ndarray]:
    """Copy NPZ arrays before the archive is closed."""
    with np.load(path) as archive:
        return {name: archive[name].copy() for name in archive.files}


def relative_residuals(
    matrix: sp.spmatrix, mass: np.ndarray, q: np.ndarray, q0: np.ndarray
) -> list[float]:
    """Compute residuals for the post-correction field actually saved to disk."""
    values = []
    for component in range(q.shape[1]):
        rhs = mass * q0[:, component]
        numerator = float(np.linalg.norm(matrix @ q[:, component] - rhs))
        denominator = float(np.linalg.norm(rhs))
        values.append(numerator if denominator == 0.0 else numerator / denominator)
    return values


def component_means(
    components: np.ndarray, mass: np.ndarray, q: np.ndarray
) -> np.ndarray:
    """Return all six volume-weighted means for every disconnected muscle piece."""
    component_mass = np.bincount(components, weights=mass)
    return np.stack(
        [
            np.bincount(components, weights=mass * q[:, component]) / component_mass
            for component in range(q.shape[1])
        ],
        axis=1,
    )


def verify_mesh(
    fixture: pv.UnstructuredGrid,
    state: dict[str, np.ndarray],
    path: Path,
    active: np.ndarray,
) -> dict[str, Any]:
    """Check the emitted VTK snapshot is the literal NPZ state on fixture topology."""
    mesh = pv.read(path)
    require(isinstance(mesh, pv.UnstructuredGrid), f"not an unstructured grid: {path}")
    q, u, ainv = state["q"], state["u"], state["Ainv"]
    expected_points = np.asarray(fixture.points) + u
    expected_full = np.broadcast_to(np.eye(3), (fixture.n_cells, 3, 3)).copy()
    expected_full[active] = ainv
    checks = {
        "point_count": mesh.n_points == fixture.n_points,
        "cell_count": mesh.n_cells == fixture.n_cells,
        "cell_types_exact": np.array_equal(mesh.celltypes, fixture.celltypes),
        "cells_exact": np.array_equal(mesh.cells, fixture.cells),
        "points_exact": np.array_equal(mesh.points, expected_points),
        "rest_position_exact": np.array_equal(
            mesh.point_data["RestPosition"], fixture.points
        ),
        "displacement_exact": np.array_equal(mesh.point_data["Displacement"], u),
        "activation_inverse_exact": np.array_equal(
            mesh.cell_data["ActivationInverseMatrix"].reshape(-1, 3, 3), expected_full
        ),
    }
    require(all(checks.values()), f"NPZ/VTU geometry or topology mismatch: {path}")
    return {
        "record": file_record(path),
        "checks": checks,
        "q_hash": array_hash(q),
        "u_hash": array_hash(u),
    }


def main(cfg: Config) -> None:
    """Validate a completed run and write one independent JSON receipt."""
    require(cfg.run_dir.is_dir(), f"run directory missing: {cfg.run_dir}")
    summary_path = cfg.run_dir / "summary.json"
    require(summary_path.is_file(), "wait for the run summary before validating")
    require(not cfg.output.exists(), f"refusing to overwrite receipt: {cfg.output}")
    summary = json.loads(summary_path.read_text())
    require(
        summary["status"] == "completed", f"run is not fully valid: {summary['status']}"
    )
    config = json.loads((cfg.run_dir / "config.json").read_text())
    fixture_path = Path(config["fixture"])
    checkpoint_path = Path(config["source_checkpoint"])
    require(
        fixture_path.resolve() == cfg.fixture.resolve(),
        "fixture override disagrees with run config",
    )
    require(checkpoint_path.is_file(), f"source checkpoint missing: {checkpoint_path}")
    fixture = pv.read(cfg.fixture / "volume.vtu")
    require(
        isinstance(fixture, pv.UnstructuredGrid), "fixture is not an unstructured grid"
    )
    graph = BASE.graph_arrays(fixture)
    active = graph["active"]
    mass = graph["volume"]
    adjacency = sp.coo_matrix(
        (
            np.r_[graph["weight"], graph["weight"]],
            (np.r_[graph["i"], graph["j"]], np.r_[graph["j"], graph["i"]]),
        ),
        shape=(len(active), len(active)),
    ).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    laplacian = sp.diags(degree) - adjacency
    n_components, components = connected_components(adjacency)

    source = loaded(checkpoint_path)
    q0, seed = source["q"], source["u"]
    require(bool(source["solver_valid"]), "source checkpoint is not solver-valid")
    require(q0.shape == (len(active), 6), "source q shape disagrees with fixture graph")
    require(
        np.array_equal(source["Ainv"], BASE.activation_matrix(q0)),
        "source q/Ainv mismatch",
    )
    mean0 = component_means(components, mass, q0)
    r0 = BASE.smoothness_numpy(q0, graph, float(config["smooth_length"]))
    require(r0 > 0.0, "source roughness is zero")

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
        all(material_checks.values()),
        f"matched material contract changed: {material_checks}",
    )
    expected_forward_tolerance = {"max_steps": 5000, "rtol": 5e-4, "atol": 1e-10}
    cases: list[dict[str, Any]] = []
    source_seed_hash = hashlib.sha256(seed.tobytes()).hexdigest()
    for row in summary["cases"]:
        name = row["id"]
        case_dir = cfg.run_dir / name
        state_path = case_dir / "state.npz"
        diagnostics_path = case_dir / "diagnostics.json"
        diagnostics = json.loads(diagnostics_path.read_text())
        state = loaded(state_path)
        q = state["q"]
        require(q.shape == q0.shape, f"{name}: q shape mismatch")
        require(
            np.array_equal(state["Ainv"], BASE.activation_matrix(q)),
            f"{name}: q/Ainv mismatch",
        )
        require(bool(state["solver_valid"]), f"{name}: saved state is not solver-valid")
        require(
            bool(diagnostics["forward"]["success"]),
            f"{name}: diagnostics forward is not valid",
        )
        require(
            row["status"] == "equilibrium_valid",
            f"{name}: summary calls invalid state valid",
        )
        require(
            row["seed_sha256"] == source_seed_hash, f"{name}: summary seed hash differs"
        )
        require(
            diagnostics["seed_sha256"] == source_seed_hash,
            f"{name}: diagnostic seed hash differs",
        )
        actual_tolerance = diagnostics["forward"]["tolerance"]
        solver_checks = {
            key: bool(np.isclose(actual_tolerance[key], value, rtol=0.0, atol=0.0))
            for key, value in expected_forward_tolerance.items()
        }
        require(
            all(solver_checks.values()),
            f"{name}: forward solver contract changed: {solver_checks}",
        )
        field = diagnostics["field"]
        tau = float(field["tau_m2"])
        matrix = sp.diags(mass) + tau * laplacian
        post_residuals = relative_residuals(matrix, mass, q, q0)
        mean_error = float(np.max(np.abs(component_means(components, mass, q) - mean0)))
        roughness = BASE.smoothness_numpy(q, graph, float(config["smooth_length"]))
        roughness_ratio = roughness / r0
        target = float(field.get("requested_roughness_ratio", 1.0))
        checks = {
            "baseline_q_exact_source": name != "baseline" or np.array_equal(q, q0),
            "saved_roughness_matches_diagnostics": bool(
                np.isclose(roughness, diagnostics["roughness"], rtol=1e-12, atol=0.0)
            ),
            "saved_ratio_matches_logged": bool(
                np.isclose(
                    roughness_ratio, field["roughness_ratio"], rtol=1e-12, atol=0.0
                )
            ),
            "roughness_target_tolerance_1e-4": abs(roughness_ratio - target) < 1e-4,
            "all_cases_same_source_seed": True,
            "component_means_conserved_post_correction": mean_error < 1e-11,
        }
        require(all(checks.values()), f"{name}: validation check failed: {checks}")
        mesh = verify_mesh(fixture, state, case_dir / "final.vtu", active)
        logged_pre = field.get("cg_relative_residuals")
        cases.append(
            {
                "id": name,
                "field": {
                    "tau_m2": tau,
                    "requested_roughness_ratio": target,
                    "recomputed_roughness": roughness,
                    "recomputed_roughness_ratio": roughness_ratio,
                    "logged_pre_correction_cg_relative_residuals": logged_pre,
                    "actual_post_correction_relative_residuals": post_residuals,
                    "actual_post_correction_max_relative_residual": max(post_residuals),
                    "post_correction_component_mean_max_abs_error": mean_error,
                },
                "checks": {**checks, "forward_solver_contract": solver_checks},
                "state": file_record(state_path),
                "diagnostics": file_record(diagnostics_path),
                "mesh": mesh,
            }
        )

    receipt = {
        "schema_version": 1,
        "status": "passed",
        "scope": "independent post-correction validation of the completed saved field-diffusion run",
        "source_run": file_record(summary_path),
        "inputs": {
            "run_config": file_record(cfg.run_dir / "config.json"),
            "run_provenance": file_record(cfg.run_dir / "provenance.json"),
            "field_preparation": file_record(cfg.run_dir / "field-preparation.json"),
            "fixture_volume": file_record(cfg.fixture / "volume.vtu"),
            "source_checkpoint": file_record(checkpoint_path),
        },
        "material_contract": {
            "checks": material_checks,
            "actual": summary["materials"],
        },
        "forward_solver_contract": expected_forward_tolerance,
        "source": {
            "q_hash": array_hash(q0),
            "seed_hash": source_seed_hash,
            "roughness": r0,
            "active_tetrahedra": len(active),
            "connected_components": int(n_components),
        },
        "checks": {
            "all_cases_same_seed": all(
                case["checks"]["all_cases_same_source_seed"] for case in cases
            ),
            "all_q_ainv_and_vtu_exact": True,
            "logged_pre_correction_residuals_are_kept_separate": True,
        },
        "cases": cases,
    }
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    cherries.log_output(cfg.output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
