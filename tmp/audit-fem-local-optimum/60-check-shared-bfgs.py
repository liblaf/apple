"""Isolated unconstrained BFGS restart audit for four 2-D shared-control cases."""

from __future__ import annotations

# ruff: noqa: BLE001, EM101, EM102, FBT003, SLF001, TRY003
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
import scipy.optimize as so
import scipy.sparse.linalg as spla

ROOT = Path("/home/liblaf/Projects/liblaf/apple")
SOURCE = (
    ROOT / "exp/2026/08/31/unreachable-pork-factor-study/src/60-run-pork-folding-2d.py"
)
DATA = ROOT / "exp/2026/08/31/unreachable-pork-factor-study/data/60-pork-folding-2d"
OUT = ROOT / "tmp/audit-fem-local-optimum/results-bfgs-adam"
NAMES = (
    "l010-band-shared-nu49",
    "l010-full-shared-nu35",
    "l100-band-shared-nu35",
    "l100-full-shared-nu49",
)


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("folding2d", SOURCE)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def evaluate(
    module: Any, mesh: Any, a: np.ndarray, u0: np.ndarray, case: Any
) -> dict[str, Any]:
    cfg = module.Config(
        max_steps=0,
        validate_derivatives=False,
        forward_max_iterations=1000,
        forward_tolerance=1.0e-8,
    )
    state = module.core.solve(mesh, a, "stable", u0, cfg)
    if not state.converged:
        raise RuntimeError(
            f"forward failed: {state.failure!r}, iterations={state.iterations}"
        )
    objective, du, _ = module.loss(mesh, state.u, module.HEIGHT, "l2", case.length)
    _, _, _, B, *_ = module.assembly(mesh, state.u, a, "stable", False, True)
    adjoint = spla.spsolve(state.h, -du)
    gradient = np.asarray(B.T @ adjoint).ravel()
    if not (np.isfinite(objective) and np.isfinite(gradient).all()):
        raise FloatingPointError("non-finite objective or implicit gradient")
    metrics = module.metrics(mesh, state.u, a, module.HEIGHT, case.length)
    return {
        "objective": float(objective),
        "gradient": gradient,
        "gradient_norm": float(np.linalg.norm(gradient)),
        "gradient_rms": float(np.linalg.norm(gradient) / math.sqrt(len(gradient))),
        "forward_iterations": int(state.iterations),
        "min_det_f": float(state.det_f.min()),
        "min_det_g": float(state.det_g.min()),
        "min_det_ainv": float(state.det_a.min()),
        "target_rms": metrics["top_target_rms"],
        "target_mae": metrics["top_target_mae"],
        "target_max": metrics["top_target_max"],
        "activation_rms": metrics["activation_rms"],
        "activation_neighbor_jump_rms": metrics["activation_neighbor_jump_rms"],
        "state": state,
    }


def summary(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value for key, value in record.items() if key not in {"gradient", "state"}
    }


def adam_restart(
    module: Any, mesh: Any, a0: np.ndarray, initial: dict[str, Any], case: Any
):
    """Run a fixed non-vanishing Adam restart with no bounds or repairs."""
    a, u = a0.copy(), initial["state"].u.copy()
    first_moment, second_moment = np.zeros_like(a), np.zeros_like(a)
    best, best_controls = initial, a.copy()
    history = []
    for step in range(301):
        current = evaluate(module, mesh, a, u, case)
        u = current["state"].u
        history.append({"step": step, "controls": a.tolist(), **summary(current)})
        if current["objective"] < best["objective"]:
            best, best_controls = current, a.copy()
        if step == 300:
            break
        first_moment = 0.9 * first_moment + 0.1 * current["gradient"]
        second_moment = 0.999 * second_moment + 0.001 * current["gradient"] ** 2
        lr = 0.003 * 0.995**step
        a -= (
            lr
            * (first_moment / (1.0 - 0.9 ** (step + 1)))
            / (np.sqrt(second_moment / (1.0 - 0.999 ** (step + 1))) + 1.0e-8)
        )
    return {
        "method": "Adam restart, unconstrained 3-vector, lr=0.003*0.995^step",
        "updates": 300,
        "evaluations": len(history),
        "final_controls": a.tolist(),
        "best_controls": best_controls.tolist(),
        "final": summary(current),
        "best": summary(best),
        "objective_change": float(current["objective"] - initial["objective"]),
        "objective_relative_change": float(
            (current["objective"] - initial["objective"])
            / max(abs(initial["objective"]), 1.0e-30)
        ),
        "target_rms_change": float(current["target_rms"] - initial["target_rms"]),
        "history": history,
    }


def audit_case(module: Any, case: Any) -> dict[str, Any]:
    module._SHARED = True
    mesh = module.build_mesh(case)
    frame = pv.read(DATA / case.name / "final.vtu")
    for name in ("ActivationXX", "ActivationYY", "ActivationXY", "Displacement"):
        if name not in frame.cell_data and name != "Displacement":
            raise KeyError(f"{case.name}/final.vtu lacks {name}")
        if name == "Displacement" and name not in frame.point_data:
            raise KeyError(f"{case.name}/final.vtu lacks displacement")
    fields = np.column_stack(
        [
            np.asarray(frame.cell_data["ActivationXX"]),
            np.asarray(frame.cell_data["ActivationYY"]),
            np.asarray(frame.cell_data["ActivationXY"]),
        ]
    )
    active = fields[mesh.muscle]
    a0 = active[0].copy()
    if not np.allclose(active, a0, rtol=0.0, atol=1.0e-12):
        raise AssertionError(f"{case.name} final activation is not one shared 3-vector")
    full_u = np.asarray(frame.point_data["Displacement"], float)[:, :2].ravel()
    u0 = full_u[mesh.free]
    initial = evaluate(module, mesh, a0, u0, case)
    evaluations: list[dict[str, Any]] = []
    best, best_controls = initial, a0.copy()

    def fun(a: np.ndarray) -> tuple[float, np.ndarray]:
        nonlocal best, best_controls
        result = evaluate(module, mesh, np.asarray(a, float), u0, case)
        evaluations.append(
            {
                "evaluation": len(evaluations),
                "controls": np.asarray(a, float).tolist(),
                **summary(result),
            }
        )
        if result["objective"] < best["objective"]:
            best, best_controls = result, np.asarray(a, float).copy()
        return result["objective"], result["gradient"]

    result = so.minimize(
        fun,
        a0,
        jac=True,
        method="BFGS",
        options={"gtol": 1.0e-10, "maxiter": 200, "disp": False},
    )
    final = evaluate(module, mesh, np.asarray(result.x, float), u0, case)
    adam = adam_restart(module, mesh, a0, initial, case)
    return {
        "case": case.name,
        "method": "scipy BFGS, unconstrained 3-vector shared activation",
        "solver_failures": 0,
        "initial_controls": a0.tolist(),
        "final_controls": np.asarray(result.x, float).tolist(),
        "best_controls": best_controls.tolist(),
        "initial": summary(initial),
        "final": summary(final),
        "best": summary(best),
        "best_objective_seen": float(best["objective"]),
        "objective_change": float(final["objective"] - initial["objective"]),
        "objective_relative_change": float(
            (final["objective"] - initial["objective"])
            / max(abs(initial["objective"]), 1.0e-30)
        ),
        "target_rms_change": float(final["target_rms"] - initial["target_rms"]),
        "bfgs": {
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "njev": int(result.njev),
        },
        "evaluations": evaluations,
        "adam_restart": adam,
    }


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    module = load_module()
    selected = {case.name: case for case in module.cases()}
    reports, failures = [], []
    for name in NAMES:
        try:
            reports.append(audit_case(module, selected[name]))
        except Exception as error:
            failures.append({"case": name, "error": repr(error)})
    report = {"reports": reports, "failures": failures}
    (OUT / "summary.json").write_text(json.dumps(clean(report), indent=2) + "\n")
    with (OUT / "evaluations.json").open("w") as stream:
        json.dump(clean(report), stream, indent=2)
        stream.write("\n")
    if failures:
        raise RuntimeError(f"restart audit has solver/evaluation failures: {failures}")


if __name__ == "__main__":
    main()
