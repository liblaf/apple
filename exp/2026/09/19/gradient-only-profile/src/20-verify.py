"""Independent checks for the pure 2-D profile-gradient objective."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import gradient_study as gs
import numpy as np
import pydantic_settings as ps
import scipy.sparse.linalg as spla
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
LEGACY = gs.ROOT / "exp/2026/09/15/activation-direction-smoothness/data/tune-w0"
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
    input_dir: str = "10-gradient"
    output: Path = Path("20-checks")
    derivatives_only: bool = False
    residual_tolerance: float = 1e-10
    fixed_tolerance: float = 1e-14


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def pack(mesh: Any, full_u: np.ndarray) -> np.ndarray:
    """Pack a full nodal field without depending on the tested loss."""
    lookup = mesh.lookup
    result = np.empty(mesh.nfree)
    free = lookup >= 0
    result[lookup[free]] = np.asarray(full_u).reshape(-1)[free]
    return result


def _smooth_controls(mesh: Any, mode: str) -> np.ndarray:
    """A spatially varying, strictly feasible control for central differences."""
    centers = mesh.p[mesh.tri[mesh.muscle]].mean(axis=1)
    x, y = centers.T
    wave = 0.02 * np.sin(2 * np.pi * x) + 0.01 * np.cos(6 * np.pi * y)
    if mode == "unconstrained":
        return np.column_stack((0.08 + wave, 0.05 - wave, 0.02 + 0.5 * wave)).ravel()
    if mode == "contraction_only":
        return np.column_stack((0.12 + wave, 0.09 - wave, 0.015 + 0.25 * wave)).ravel()
    if mode == "learned_direction":
        return np.column_stack((0.12 + wave, 0.37 + 0.2 * wave)).ravel()
    assert mode == "x_contraction", mode
    return 0.12 + wave


def _direction_errors(
    function: Any, point: np.ndarray, analytic: np.ndarray, rng: np.random.Generator
) -> list[dict[str, float]]:
    values = []
    for _ in range(3):
        direction = rng.normal(size=point.size)
        direction /= np.linalg.norm(direction)
        step = 1e-6
        numeric = (
            function(point + step * direction) - function(point - step * direction)
        ) / (2 * step)
        exact = float(analytic @ direction)
        error = abs(exact - numeric)
        values.append(
            {
                "analytic": exact,
                "centered_difference": float(numeric),
                "abs_error": float(error),
                "relative_error": float(error / max(abs(exact), abs(numeric), 1e-12)),
            }
        )
    return values


def loss_derivative_check() -> dict[str, Any]:
    """Check ``d L_grad / d u`` before testing any physics or adjoint code."""
    mesh = gs.ph.build_mesh(20, 10)
    rng = np.random.default_rng(20260919)
    u = rng.normal(scale=0.01, size=mesh.nfree)
    height = 0.05
    value, gradient, diagnostics = gs.profile_loss(mesh, u, height)
    directions = _direction_errors(
        lambda trial: gs.profile_loss(mesh, trial, height)[0], u, gradient, rng
    )
    max_relative = max(item["relative_error"] for item in directions)
    assert max_relative < 2e-8, directions
    return {
        "objective": value,
        "slope_rms": diagnostics["slope_rms"],
        "directions": directions,
        "max_relative_error": max_relative,
    }


def target_and_fixed_checks() -> dict[str, Any]:
    """The sampled target is feasible and both fixed corners participate in it."""
    mesh = gs.ph.build_mesh(20, 10)
    height = 0.05
    top = gs.top_nodes(mesh)
    x = mesh.p[top, 0]
    target = np.zeros_like(mesh.p)
    target[top, 1] = 4 * height * x * (1 - x)
    target_u = pack(mesh, target)
    value, gradient, _ = gs.profile_loss(mesh, target_u, height)
    corners = top[[0, -1]]
    fixed = mesh.lookup.reshape(-1, 2)[corners]
    assert np.all(fixed < 0), fixed
    assert np.array_equal(target[corners], np.zeros((2, 2))), target[corners]
    assert value <= 1e-28, value
    assert np.linalg.norm(gradient, np.inf) <= 1e-13, gradient
    zero_value, zero_gradient, _ = gs.profile_loss(mesh, np.zeros(mesh.nfree), 0.0)
    assert zero_value == 0.0
    assert np.array_equal(zero_gradient, np.zeros(mesh.nfree))
    return {
        "target_gradient_loss": value,
        "target_gradient_inf": float(np.linalg.norm(gradient, np.inf)),
        "zero_height_zero_state_loss": zero_value,
        "fixed_corner_displacement_inf": float(np.linalg.norm(target[corners], np.inf)),
        "top_nodes_including_fixed_corners": len(top),
    }


def control_derivative_checks() -> dict[str, Any]:
    """Check the full solve-plus-adjoint derivative at feasible interior points."""
    mesh = gs.ph.build_mesh(20, 10)
    rng = np.random.default_rng(20260920)
    results: dict[str, Any] = {}
    for mode in sorted(gs.am.MODES):
        q = _smooth_controls(mesh, mode)
        seed = np.zeros(mesh.nfree)
        _, _, gradient, diagnostics = gs.evaluate(mesh, q, mode, 0.05, seed)
        directions = _direction_errors(
            lambda trial, mode=mode, seed=seed: gs.evaluate(
                mesh, trial, mode, 0.05, seed
            )[3]["objective"],
            q,
            gradient,
            rng,
        )
        max_relative = max(item["relative_error"] for item in directions)
        # Newton and the sparse adjoint add solver error to this end-to-end test.
        assert max_relative < 2e-5, (mode, directions)
        results[mode] = {
            "control_size": int(q.size),
            "objective": diagnostics["objective"],
            "directions": directions,
            "max_relative_error": max_relative,
        }
    return results


def derivative_checks() -> dict[str, Any]:
    return {
        "profile_loss": loss_derivative_check(),
        "target_and_fixed_endpoints": target_and_fixed_checks(),
        "end_to_end_control": control_derivative_checks(),
    }


def source_hashes(folder: Path) -> dict[str, dict[str, str]]:
    """Prove that shared physics and L2-study adapters are unchanged."""
    legacy_protocol = json.loads((LEGACY / "protocol.json").read_text())
    current_protocol = json.loads((folder / "protocol.json").read_text())
    legacy = {
        Path(path).name: digest
        for path, digest in legacy_protocol["source_sha256"].items()
        if Path(path).name in HASH_BASENAMES
    }
    current = {
        Path(path).name: digest
        for path, digest in current_protocol["source_sha256"].items()
        if Path(path).name in HASH_BASENAMES
    }
    assert set(legacy) == HASH_BASENAMES, legacy
    assert set(current) == HASH_BASENAMES, current
    assert current == legacy, (current, legacy)
    for basename, expected in current.items():
        snapshot = folder / "source" / basename
        assert snapshot.is_file(), snapshot
        actual = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        assert actual == expected, (snapshot, actual, expected)
    return {"legacy_l2": legacy, "gradient_only": current}


def smallest_hessian_eigenvalue(hessian: Any) -> dict[str, float | int | str | None]:
    """Request the algebraically smallest sparse Hessian eigenpair honestly.

    ARPACK can fail on an indefinite, poorly separated spectrum.  In that case
    this returns the failure rather than relabeling a near-zero eigenvalue as a
    stability result.
    """
    symmetric = (hessian + hessian.T) * 0.5
    try:
        eigenvalue, eigenvector = spla.eigsh(
            symmetric, k=1, which="SA", tol=1e-8, maxiter=2_000, ncv=80
        )
    except spla.ArpackNoConvergence as exc:
        return {
            "smallest_algebraic_hessian_eigenvalue": None,
            "smallest_algebraic_hessian_eigen_residual": None,
            "hessian_eigen_status": "ARPACK SA did not converge; no minimum-eigenvalue claim",
            "hessian_eigen_iterations": 2_000,
            "hessian_eigen_converged_pairs": len(exc.eigenvalues),
        }
    eig = float(eigenvalue[0])
    vec = eigenvector[:, 0]
    residual = float(
        np.linalg.norm(symmetric @ vec - eig * vec)
        / max(np.linalg.norm(symmetric @ vec), 1.0)
    )
    assert residual <= 1e-7, residual
    return {
        "smallest_algebraic_hessian_eigenvalue": eig,
        "smallest_algebraic_hessian_eigen_residual": residual,
        "hessian_eigen_status": "ARPACK smallest algebraic eigenpair",
        "hessian_eigen_iterations": None,
        "hessian_eigen_converged_pairs": 1,
    }


def endpoint_check(cfg: Config, mesh: Any, folder: Path) -> dict[str, Any]:
    summary = json.loads((folder / "summary.json").read_text())
    checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
    history = np.load(folder / "history.npz", allow_pickle=False)
    assert np.array_equal(history["controls"][-1], checkpoint["controls"])
    assert np.array_equal(history["u"][-1], checkpoint["u"])
    q, full_u = checkpoint["controls"], checkpoint["u"]
    mode, height = summary["mode"], float(summary["height"])
    B = gs.study.matrices(mesh, q, mode)
    assert np.allclose(checkpoint["B"], B, rtol=0, atol=1e-14)
    packed_u = pack(mesh, full_u)
    energy, residual, hessian, physical_J = gs.ph.assemble(mesh, packed_u, B)
    assert hessian is not None
    state, _, _, diagnostics = gs.evaluate(mesh, q, mode, height, packed_u)
    assert np.isclose(
        diagnostics["objective"], summary["final"]["objective"], rtol=1e-10, atol=1e-12
    )
    fixed = mesh.lookup.reshape(-1, 2) < 0
    fixed_error = float(np.max(np.abs(full_u[fixed])))
    eigen = smallest_hessian_eigenvalue(hessian)
    result = {
        "case": folder.name,
        "mode": mode,
        "height": height,
        "energy": energy,
        "equilibrium_residual_inf": float(np.linalg.norm(residual, np.inf)),
        "replayed_objective": diagnostics["objective"],
        "replayed_position_loss": diagnostics["position_loss"],
        "replayed_gradient_loss": diagnostics["gradient_loss"],
        "fixed_displacement_inf": fixed_error,
        "minimum_physical_J": float(physical_J.min()),
        "maximum_physical_J": float(physical_J.max()),
        **eigen,
        "replay_forward_iterations": state.iterations,
    }
    assert result["equilibrium_residual_inf"] <= cfg.residual_tolerance, result
    assert result["fixed_displacement_inf"] <= cfg.fixed_tolerance, result
    assert result["minimum_physical_J"] > 0.0, result
    return result


def l2_endpoint_check(cfg: Config, mesh: Any, folder: Path) -> dict[str, Any]:
    """Replay the matched L2 endpoint with the same physical stability test."""
    summary = json.loads((folder / "summary.json").read_text())
    checkpoint = np.load(folder / "checkpoint.npz", allow_pickle=False)
    q, full_u = checkpoint["controls"], checkpoint["u"]
    mode, height = summary["mode"], float(summary["height"])
    B = gs.study.matrices(mesh, q, mode)
    assert np.allclose(checkpoint["B"], B, rtol=0, atol=1e-14)
    packed_u = pack(mesh, full_u)
    _, residual, hessian, physical_J = gs.ph.assemble(mesh, packed_u, B)
    assert hessian is not None
    _, _, _, diagnostics = gs.study.evaluate(mesh, q, mode, height, 0.0, packed_u)
    assert np.isclose(
        diagnostics["raw_loss"], summary["final"]["raw_loss"], rtol=1e-10, atol=1e-12
    )
    eigen = smallest_hessian_eigenvalue(hessian)
    result = {
        "case": folder.name,
        "mode": mode,
        "height": height,
        "equilibrium_residual_inf": float(np.linalg.norm(residual, np.inf)),
        "replayed_raw_L2": diagnostics["raw_loss"],
        "fixed_displacement_inf": float(
            np.max(np.abs(full_u[mesh.lookup.reshape(-1, 2) < 0]))
        ),
        "minimum_physical_J": float(physical_J.min()),
        "maximum_physical_J": float(physical_J.max()),
        **eigen,
    }
    assert result["equilibrium_residual_inf"] <= cfg.residual_tolerance, result
    assert result["fixed_displacement_inf"] <= cfg.fixed_tolerance, result
    assert result["minimum_physical_J"] > 0.0, result
    return result


def main(cfg: Config) -> None:
    derivatives = derivative_checks()
    if cfg.derivatives_only:
        print(json.dumps({"derivatives": derivatives}, indent=2, sort_keys=True))
        return
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    source = GROUP / "data" / cfg.input_dir
    assert source.is_dir(), source
    protocol = json.loads((source / "protocol.json").read_text())
    mesh = gs.ph.build_mesh(protocol["config"]["nx"], protocol["config"]["ny"])
    children = sorted(source.glob("h*-*/summary.json"))
    expected = {
        f"h{round(float(height) * 1000):03d}-{mode}"
        for height in protocol["config"]["heights"].split(",")
        for mode in protocol["config"]["modes"].split(",")
    }
    assert {child.parent.name for child in children} == expected, (children, expected)
    legacy_protocol = json.loads((LEGACY / "protocol.json").read_text())
    legacy_mesh = gs.ph.build_mesh(
        legacy_protocol["config"]["nx"], legacy_protocol["config"]["ny"]
    )
    legacy_children = sorted(LEGACY.glob("h*-w0/summary.json"))
    assert len(legacy_children) == 8, legacy_children
    report = {
        "hessian_diagnostic": "smallest algebraic eigenvalue from scipy.sparse.linalg.eigsh(which='SA'); a negative value identifies a saddle direction",
        "derivatives": derivatives,
        "source_hashes": source_hashes(source),
        "cases": [endpoint_check(cfg, mesh, child.parent) for child in children],
        "legacy_l2_cases": [
            l2_endpoint_check(cfg, legacy_mesh, child.parent)
            for child in legacy_children
        ],
    }
    write_json(output / "checks.json", report)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileVerification)
