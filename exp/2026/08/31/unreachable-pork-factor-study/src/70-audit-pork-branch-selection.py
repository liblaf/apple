"""Audit 2-D endpoint branch selection without changing the folding study."""

from __future__ import annotations

# ruff: noqa: EM101, EM102, FBT003, SLF001, TRY003
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_root: Path = Path("data/60-pork-folding-2d")
    output_dir: Path = cherries.output("70-branch-selection-audit", mkdir=True)


CASES = ("l010-band-per_cell-nu49", "l010-band-shared-nu49")
EPSILONS = (1e-3, 1e-5, 1e-7)


def runner() -> Any:
    path = Path(__file__).with_name("60-run-pork-folding-2d.py")
    spec = importlib.util.spec_from_file_location("pork_branch_audit", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def strict_cfg(module: Any) -> Any:
    return module.Config(
        validate_derivatives=False, forward_tolerance=1e-10, forward_max_iterations=1000
    )


def extract(module: Any, directory: Path):
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    case_data = summary["case"]
    case = module.Case(
        case_data["name"],
        float(case_data["length"]),
        case_data["muscle_layout"],
        case_data["activation_mode"],
        float(case_data["poisson"]),
    )
    shared = case.activation_mode == "shared"
    module._SHARED = shared
    mesh = module.build_mesh(case)
    module._GROUPS = module.group_map(mesh, shared)
    grid = pv.read(directory / "final.vtu")
    values = np.column_stack(
        [
            grid.cell_data[key]
            for key in ("ActivationXX", "ActivationYY", "ActivationXY")
        ]
    )
    active = values[mesh.muscle]
    controls = active[0].copy() if shared else active.ravel().copy()
    refinement = summary.get("inverse", {}).get("refinement", {})
    start_step = refinement.get("start_step") if isinstance(refinement, dict) else None
    seed_path = (
        directory / "frames" / f"step-{int(start_step) - 1:04d}.vtu"
        if isinstance(start_step, int) and start_step > 0
        else directory / "final.vtu"
    )
    if not seed_path.is_file():
        raise FileNotFoundError(seed_path)
    displacement = np.asarray(pv.read(seed_path).point_data["Displacement"], float)[
        :, :2
    ]
    return (
        summary,
        case,
        mesh,
        controls,
        displacement.ravel()[mesh.free].copy(),
        seed_path,
    )


def evaluate(
    module: Any, mesh: Any, controls: np.ndarray, initial: np.ndarray, cfg: Any
):
    state = module.core.solve(mesh, controls, "stable", initial, cfg)
    if not state.converged:
        raise RuntimeError(f"forward failed: {state.failure!r}")
    value, du, _ = module.loss(mesh, state.u, module.HEIGHT, "l2", mesh.p[:, 0].max())
    _, _, _, mixed, *_ = module.assembly(mesh, state.u, controls, "stable", False, True)
    gradient = np.asarray(mixed.T @ module.spla.spsolve(state.h, -du)).ravel()
    residual = float(np.linalg.norm(state.r) / np.sqrt(mesh.nfree))
    if not (
        np.isfinite(value) and np.isfinite(gradient).all() and np.isfinite(residual)
    ):
        raise FloatingPointError("non-finite strict evaluation")
    return state, float(value), gradient, residual


def inertia(hessian: Any) -> dict[str, Any]:
    eig = np.linalg.eigvalsh(((hessian + hessian.T) * 0.5).toarray())
    threshold = max(float(np.max(np.abs(eig))), 1.0) * 1e-12
    return {
        "dimension": int(eig.size),
        "negative": int((eig < -threshold).sum()),
        "near_zero": int((np.abs(eig) <= threshold).sum()),
        "positive": int((eig > threshold).sum()),
        "minimum_eigenvalue": float(eig[0]),
        "maximum_eigenvalue": float(eig[-1]),
    }


def endpoint(module: Any, directory: Path) -> dict[str, Any]:
    _summary, case, mesh, controls, adam_seed, seed_path = extract(module, directory)
    cfg = strict_cfg(module)
    base, objective, gradient, residual = evaluate(
        module, mesh, controls, adam_seed, cfg
    )
    direction = -gradient / np.linalg.norm(gradient)
    random = np.random.default_rng(20260901).standard_normal(controls.size)
    random /= np.linalg.norm(random)
    rows = []
    for name, vector in (("minus_gradient", direction), ("fixed_random", random)):
        predicted = float(gradient @ vector)
        for epsilon in EPSILONS:
            fixed_minus = evaluate(
                module, mesh, controls - epsilon * vector, adam_seed, cfg
            )
            fixed_plus = evaluate(
                module, mesh, controls + epsilon * vector, adam_seed, cfg
            )
            local_minus = evaluate(
                module, mesh, controls - epsilon * vector, base.u, cfg
            )
            local_plus = evaluate(
                module, mesh, controls + epsilon * vector, base.u, cfg
            )
            fixed_fd = (fixed_plus[1] - fixed_minus[1]) / (2 * epsilon)
            local_fd = (local_plus[1] - local_minus[1]) / (2 * epsilon)
            rows.append(
                {
                    "direction": name,
                    "epsilon": epsilon,
                    "adjoint_directional_derivative": predicted,
                    "fixed_seed_central_difference": fixed_fd,
                    "local_continuation_central_difference": local_fd,
                    "fixed_seed_relative_error": abs(fixed_fd - predicted)
                    / max(abs(fixed_fd), abs(predicted), 1e-14),
                    "local_continuation_relative_error": abs(local_fd - predicted)
                    / max(abs(local_fd), abs(predicted), 1e-14),
                    "branch_objective_gap_minus": fixed_minus[1] - local_minus[1],
                    "branch_objective_gap_plus": fixed_plus[1] - local_plus[1],
                    "fixed_seed_minus": {
                        "objective": fixed_minus[1],
                        "residual_rms": fixed_minus[3],
                        "iterations": fixed_minus[0].iterations,
                    },
                    "fixed_seed_plus": {
                        "objective": fixed_plus[1],
                        "residual_rms": fixed_plus[3],
                        "iterations": fixed_plus[0].iterations,
                    },
                    "local_minus": {
                        "objective": local_minus[1],
                        "residual_rms": local_minus[3],
                        "iterations": local_minus[0].iterations,
                    },
                    "local_plus": {
                        "objective": local_plus[1],
                        "residual_rms": local_plus[3],
                        "iterations": local_plus[0].iterations,
                    },
                }
            )
    return {
        "case": case.name,
        "source": str(directory.resolve()),
        "control_dofs": int(controls.size),
        "adam_seed": str(seed_path),
        "base": {
            "objective": objective,
            "gradient_l2": float(np.linalg.norm(gradient)),
            "gradient_inf": float(np.linalg.norm(gradient, np.inf)),
            "residual_rms": residual,
            "iterations": base.iterations,
            "min_det_f": float(base.det_f.min()),
            "hessian_inertia": inertia(base.h),
        },
        "derivatives": rows,
    }


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    module = runner()
    payload = {
        "method": "current src60 mechanics; strict 1e-10 solves; fixed stored-Adam seed versus base-endpoint local continuation",
        "epsilons": list(EPSILONS),
        "directions": ["minus_gradient", "fixed_random"],
        "endpoints": [endpoint(module, cfg.input_root / name) for name in CASES],
    }
    text = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    (cfg.output_dir / "receipt.json").write_text(text, encoding="utf-8")
    cherries.log_metrics(
        {
            "branch_audit/endpoints": len(payload["endpoints"]),
            "branch_audit/evaluations": len(CASES) * 1
            + len(CASES) * 2 * len(EPSILONS) * 4,
        }
    )


if __name__ == "__main__":
    cherries.main(main)
