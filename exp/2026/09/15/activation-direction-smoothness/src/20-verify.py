"""Independent checks for the matched 2-D activation/smoothness study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import activation_models as am
import numpy as np
import pydantic_settings as ps
import scipy.linalg as la
import study
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
OLD = study.ROOT / "exp/2026/09/14/fiber-contraction-parabola/data/100-adam-contraction"
HASH_BASENAMES = {
    "study.py",
    "activation_models.py",
    "physics2d.py",
    "10-run-pork-2d.py",
}


class ProfileVerification(profiles.Profile):
    def init(self):
        run = core.run
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_dirs: str = "tune-w0,tune-w001,tune-w01,tune-w1"
    output: Path = Path("20-checks")
    derivatives_only: bool = False
    residual_tolerance: float = 1e-10
    fixed_tolerance: float = 1e-14


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pack(mesh: Any, full_u: np.ndarray) -> np.ndarray:
    packed_u = np.empty(mesh.nfree)
    lookup = mesh.lookup
    free = lookup >= 0
    packed_u[lookup[free]] = full_u.reshape(-1)[free]
    return packed_u


def _smooth_controls(mesh: Any, mode: str) -> np.ndarray:
    """Make a strictly feasible, spatially varying point for a FD check."""
    centers = mesh.p[mesh.tri[mesh.muscle]].mean(axis=1)
    x, y = centers.T
    wave = 0.02 * np.sin(2 * np.pi * x) + 0.01 * np.cos(6 * np.pi * y)
    if mode == "unconstrained":
        return np.column_stack((0.08 + wave, 0.05 - wave, 0.02 + 0.5 * wave)).ravel()
    if mode == "contraction_only":
        # Positive determinant margin is deliberate: finite differences must
        # stay off the spectral projection boundary.
        return np.column_stack((0.12 + wave, 0.09 - wave, 0.015 + 0.25 * wave)).ravel()
    if mode == "learned_direction":
        return np.column_stack((0.12 + wave, 0.37 + 0.2 * wave)).ravel()
    assert mode == "x_contraction", mode
    return 0.12 + wave


def derivative_checks() -> dict[str, Any]:
    mesh = study.ph.build_mesh(20, 10)
    rng = np.random.default_rng(20260915)
    result: dict[str, Any] = {}
    for mode in sorted(am.MODES):
        q = _smooth_controls(mesh, mode)
        seed = np.zeros(mesh.nfree)
        _, _, gradient, values = study.evaluate(mesh, q, mode, 0.05, 0.1, seed)
        errors = []
        for _ in range(3):
            direction = rng.normal(size=q.size)
            direction /= np.linalg.norm(direction)
            step = 1e-6
            plus = study.evaluate(mesh, q + step * direction, mode, 0.05, 0.1, seed)[3][
                "objective"
            ]
            minus = study.evaluate(mesh, q - step * direction, mode, 0.05, 0.1, seed)[
                3
            ]["objective"]
            numeric = (plus - minus) / (2 * step)
            analytic = float(gradient @ direction)
            errors.append(
                {
                    "analytic": analytic,
                    "centered_difference": numeric,
                    "abs_error": abs(analytic - numeric),
                    "relative_error": abs(analytic - numeric)
                    / max(abs(analytic), abs(numeric), 1e-12),
                }
            )
        max_relative = max(item["relative_error"] for item in errors)
        # The forward/adjoint implementation was previously validated to
        # around 1e-9; this looser end-to-end gate includes Newton tolerance.
        assert max_relative < 2e-5, (mode, errors)
        result[mode] = {
            "control_size": int(q.size),
            "weight": 0.1,
            "height": 0.05,
            "objective": values["objective"],
            "directions": errors,
            "max_relative_error": max_relative,
        }
    return result


def source_hashes(folder: Path) -> dict[str, str]:
    protocol = json.loads((folder / "protocol.json").read_text())
    selected = {
        Path(key).name: value
        for key, value in protocol["source_sha256"].items()
        if Path(key).name in HASH_BASENAMES
    }
    assert set(selected) == HASH_BASENAMES, (folder, selected)
    for basename, expected in selected.items():
        snapshot = folder / "source" / basename
        assert snapshot.is_file(), snapshot
        actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert actual == expected, (snapshot, actual, expected)
    return selected


def verify_saved_controls(history: Any, mode: str) -> dict[str, float | int]:
    controls = history["controls"]
    worst_eigenvalue = np.inf
    worst_secondary = 0.0
    for q in controls:
        B = am.matrices(q, mode)
        eigenvalues = np.linalg.eigvalsh(B)
        worst_eigenvalue = min(worst_eigenvalue, float(eigenvalues.min()))
        if mode == "learned_direction":
            offset = B - np.eye(2)
            worst_secondary = max(
                worst_secondary,
                float(np.linalg.svd(offset, compute_uv=False)[:, 1].max()),
            )
        elif mode == "x_contraction":
            offset = B - np.eye(2)
            worst_secondary = max(
                worst_secondary,
                float(np.max(np.abs(offset[:, [0, 1, 1], [1, 0, 1]]))),
            )
    if mode != "unconstrained":
        assert worst_eigenvalue >= 1 - 1e-10, (mode, worst_eigenvalue)
    if mode in {"learned_direction", "x_contraction"}:
        assert worst_secondary <= 1e-10, (mode, worst_secondary)
    return {
        "saved_states": len(controls),
        "minimum_eigenvalue_B_all_saved_states": worst_eigenvalue,
        "rank_one_or_fixed_secondary_max_abs": worst_secondary,
    }


def endpoint_check(cfg: Config, mesh: Any, folder: Path) -> dict[str, Any]:
    summary = json.loads((folder / "summary.json").read_text())
    history = np.load(folder / "history.npz", allow_pickle=False)
    checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
    mode, height, weight = (
        summary["mode"],
        float(summary["height"]),
        float(summary["smooth_weight"]),
    )
    assert np.array_equal(history["controls"][-1], checkpoint["controls"])
    assert np.array_equal(history["u"][-1], checkpoint["u"])
    saved = verify_saved_controls(history, mode)
    q, full_u = checkpoint["controls"], checkpoint["u"]
    B = study.matrices(mesh, q, mode)
    assert np.allclose(checkpoint["B"], B, rtol=0, atol=1e-14)
    _, residual, hessian, J = study.ph.assemble(mesh, pack(mesh, full_u), B)
    assert hessian is not None
    symmetric_hessian = ((hessian + hessian.T) * 0.5).toarray()
    eigenvalue, vector = la.eigh(
        symmetric_hessian,
        subset_by_index=[0, 0],
        driver="evr",
        check_finite=True,
    )
    eigenvalue = float(eigenvalue[0])
    vector = vector[:, 0]
    eigen_residual = float(
        np.linalg.norm(symmetric_hessian @ vector - eigenvalue * vector)
        / max(np.linalg.norm(symmetric_hessian @ vector), 1.0)
    )
    fixed = mesh.lookup.reshape(-1, 2) < 0
    fixed_error = float(np.max(np.abs(full_u[fixed])))
    seed = pack(mesh, full_u)
    _, _, _, values = study.evaluate(mesh, q, mode, height, weight, seed)
    assert np.isclose(
        values["raw_loss"], summary["final"]["raw_loss"], rtol=1e-10, atol=1e-12
    )
    assert np.isclose(
        values["roughness"], summary["final"]["roughness"], rtol=1e-10, atol=1e-12
    )
    result = {
        "case": folder.name,
        "height": height,
        "mode": mode,
        "weight": weight,
        **saved,
        "equilibrium_residual_inf": float(np.linalg.norm(residual, np.inf)),
        "recomputed_raw_L2": values["raw_loss"],
        "recomputed_roughness": values["roughness"],
        "fixed_displacement_inf": fixed_error,
        "minimum_physical_J": float(J.min()),
        "maximum_physical_J": float(J.max()),
        "smallest_algebraic_hessian_eigenvalue": eigenvalue,
        "smallest_algebraic_hessian_eigen_residual": eigen_residual,
    }
    assert result["equilibrium_residual_inf"] <= cfg.residual_tolerance, result
    assert result["fixed_displacement_inf"] <= cfg.fixed_tolerance, result
    assert result["minimum_physical_J"] > 0.0, result
    assert eigen_residual <= 1e-5, result
    return result


def compare_old(folder: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    comparisons = []
    for case in cases:
        if case["weight"] != 0.0 or case["mode"] == "learned_direction":
            continue
        old_folder = OLD / f"h{round(case['height'] * 1000):03d}-{case['mode']}"
        assert old_folder.is_dir(), old_folder
        new_folder = folder / case["case"]
        old = np.load(old_folder / "checkpoint.npz", allow_pickle=False)
        new = np.load(new_folder / "checkpoint.npz", allow_pickle=False)
        q_error = float(np.max(np.abs(new["controls"] - old["controls"])))
        u_error = float(np.max(np.abs(new["u"] - old["u"])))
        old_summary = json.loads((old_folder / "summary.json").read_text())
        l2_error = abs(case["recomputed_raw_L2"] - old_summary["final"]["raw_loss"])
        assert q_error <= 5e-9, (case["case"], q_error)
        assert u_error <= 5e-9, (case["case"], u_error)
        assert l2_error <= 5e-11, (case["case"], l2_error)
        comparisons.append(
            {
                "case": case["case"],
                "old_case": old_folder.name,
                "endpoint_control_max_abs_error": q_error,
                "endpoint_displacement_max_abs_error": u_error,
                "raw_L2_abs_error": l2_error,
            }
        )
    return comparisons


def expected_case_names(protocol: dict[str, Any]) -> set[str]:
    config = protocol["config"]
    heights = [float(value) for value in config["heights"].split(",")]
    modes = config["modes"].split(",")
    weights = [float(value) for value in config["weights"].split(",")]
    return {
        f"h{round(height * 1000):03d}-{mode}-w{weight:g}"
        for height in heights
        for weight in weights
        for mode in modes
    }


def main(cfg: Config) -> None:
    derivatives = derivative_checks()
    if cfg.derivatives_only:
        print(json.dumps({"derivatives": derivatives}, indent=2, sort_keys=True))
        return
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    folders = [
        GROUP / "data" / name.strip()
        for name in cfg.input_dirs.split(",")
        if name.strip()
    ]
    assert folders, cfg.input_dirs
    for folder in folders:
        assert folder.is_dir(), folder
    hashes = {str(folder): source_hashes(folder) for folder in folders}
    first = next(iter(hashes.values()))
    assert all(value == first for value in hashes.values()), hashes
    cases = []
    comparisons = []
    for folder in folders:
        protocol = json.loads((folder / "protocol.json").read_text())
        expected = expected_case_names(protocol)
        root_summary = json.loads((folder / "summary.json").read_text())
        assert len(root_summary) == len(expected), (
            folder,
            len(root_summary),
            len(expected),
        )
        children = sorted(folder.glob("h*-*/summary.json"))
        assert {child.parent.name for child in children} == expected, (folder, expected)
        mesh = study.ph.build_mesh(protocol["config"]["nx"], protocol["config"]["ny"])
        folder_cases = [endpoint_check(cfg, mesh, child.parent) for child in children]
        cases.extend(folder_cases)
        folder_comparisons = compare_old(folder, folder_cases)
        comparisons.extend(folder_comparisons)
        if folder.name == "tune-w0":
            config = protocol["config"]
            if config["heights"] == "0.05,0.20" and config["weights"] == "0":
                assert len(folder_comparisons) == 6, folder_comparisons
    report = {
        "hessian_diagnostic": "dense symmetric scipy.linalg.eigh(subset_by_index=[0,0], driver='evr'); diagnostic only, no simulation change",
        "derivatives": derivatives,
        "requested_input_dirs": [str(folder) for folder in folders],
        "source_hashes": hashes,
        "cases": cases,
        "old_unsmoothed_comparisons": comparisons,
    }
    write_json(output / "checks.json", report)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileVerification)
