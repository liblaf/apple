"""High-target 2-D band-muscle study with a shared-to-independent release.

The model deliberately remains unconstrained: no skin, contact, activation
bounds/regularizer, determinant barrier, or inversion repair.  Release cases
copy both the strict shared endpoint's activation and displacement, then
restart the optimizer after expanding the controls to one 3-DoF vector per
muscle triangle.
"""

from __future__ import annotations

# ruff: noqa: BLE001, C901, EM101, EM102, FBT001, FBT003, PLR0912, PLR0915, PLW0603, RUF007, TRY003, TRY301
import csv
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pydantic_settings as ps
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from liblaf import cherries

_SOURCE = Path(__file__).with_name("10-run-pork-2d.py")
_SPEC = importlib.util.spec_from_file_location("pork_2d_core", _SOURCE)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"cannot load {_SOURCE}")
core = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = core
_SPEC.loader.exec_module(core)

THICKNESS, CELL_SIZE = 0.1, 0.01
E_FAT, E_MUSCLE = 0.003, 0.03
_ORIGINAL_ASSEMBLY, _ORIGINAL_METRICS, _ORIGINAL_GRID = (
    core.assembly,
    core.metrics,
    core.grid,
)
_SHARED = False
_GROUPS: np.ndarray | None = None


@dataclass(frozen=True)
class Case:
    name: str
    length: float
    muscle_layout: Literal["band", "full"]
    activation_mode: Literal["per_cell", "shared"]
    poisson: float
    height: float
    protocol: Literal[
        "direct", "shared", "shared_then_release", "shared_then_release_zero_u"
    ]

    @property
    def resolution(self) -> tuple[int, int]:
        return (round(self.length / CELL_SIZE), round(THICKNESS / CELL_SIZE))


def cases() -> tuple[Case, ...]:
    # h=.20 and .30 are respectively 2x and 3x the prior high target (.10).
    # The common long, band-muscle, nu=.49 geometry isolates initialization and
    # control dimensionality; full-muscle cases are deliberately out of scope.
    return tuple(
        Case(
            f"h{int(height * 100):03d}-{protocol.replace('_then_', '-')}",
            1.0,
            "band",
            mode,
            0.49,
            height,
            protocol,
        )
        for height in (0.20, 0.30)
        for protocol, mode in (
            ("direct", "per_cell"),
            ("shared", "shared"),
            ("shared_then_release", "per_cell"),
            ("shared_then_release_zero_u", "per_cell"),
        )
    )


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("10-pork-shared-release-high-targets", mkdir=True)
    cases: str = "all"
    # Canonical horizon from the validated OFAT runner.  Shorter runs are
    # explicitly pilots and must override this value on the command line.
    max_steps: int = 1200
    learning_rate: float = 0.03
    lr_decay: float = 0.99
    forward_tolerance: float = 1e-8
    forward_max_iterations: int = 3000
    refinement_max_iterations: int = 1000
    refinement_max_function_evaluations: int = 100000
    refinement_max_restarts: int = 100
    refinement_max_stalled_restarts: int = 5
    refinement_forward_tolerance: float = 1e-10
    refinement_forward_initialization: Literal[
        "fixed_adam", "accepted_continuation"
    ] = "fixed_adam"
    refinement_optimizer_gradient_inf_tolerance: float = 1e-12
    refinement_acceptance_gradient_inf_tolerance: float = 2e-8
    refinement_acceptance_gradient_rms_tolerance: float = 1e-8
    refinement_max_line_search_steps: int = 50
    tail_absolute_tolerance: float = 1e-10
    validate_derivatives: bool = True
    require_inverse_convergence: bool = True
    smoke: bool = False


def write_json(path: Path, data: Any) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [clean(v) for v in value]
        if isinstance(value, np.ndarray):
            return clean(value.tolist())
        if isinstance(value, np.generic):
            return clean(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.write_text(json.dumps(clean(data), indent=2, sort_keys=True) + "\n")


def file_digest(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def strict_observation(
    case: Case, controls: np.ndarray, initial_u: np.ndarray, cfg: Config
) -> tuple[Any, dict[str, Any]]:
    """Strict forward/adjoint receipt used to make release branch choice explicit."""
    global _GROUPS, _SHARED
    _SHARED = case.activation_mode == "shared"
    mesh = build_mesh(case)
    _GROUPS = group_map(mesh, _SHARED)
    strict_cfg = cfg.model_copy(update={"forward_tolerance": cfg.refinement_forward_tolerance})
    state = core.solve(mesh, controls, "stable", initial_u, strict_cfg)
    if not state.converged:
        raise RuntimeError(f"strict handoff forward failure: {state.failure!r}")
    value, du, _ = loss(mesh, state.u, case.height, "l2", case.length)
    _, _, _, mixed, *_ = assembly(mesh, state.u, controls, "stable", False, True)
    gradient = np.asarray(mixed.T @ spla.spsolve(state.h, -du)).ravel()
    if not (np.isfinite(value) and np.isfinite(gradient).all()):
        raise FloatingPointError("non-finite strict handoff inverse result")
    return state, {
        "objective": float(value),
        "gradient_inf": float(np.linalg.norm(gradient, np.inf)),
        "gradient_rms": float(np.linalg.norm(gradient) / math.sqrt(len(gradient))),
        "equilibrium_residual_rms": float(np.linalg.norm(state.r) / math.sqrt(mesh.nfree)),
        "forward_iterations": int(state.iterations),
        "min_det_f": float(state.det_f.min()),
        "min_det_ainv": float(state.det_a.min()),
        "min_det_g": float(state.det_g.min()),
        "displacement_sha256": hashlib.sha256(state.u.tobytes()).hexdigest(),
    }


def _edges(tri: np.ndarray, muscle: np.ndarray) -> tuple[tuple[int, int], ...]:
    owner: dict[tuple[int, int], int] = {}
    out = []
    for loc, e in enumerate(np.flatnonzero(muscle)):
        for a, b in (
            (tri[e][0], tri[e][1]),
            (tri[e][1], tri[e][2]),
            (tri[e][2], tri[e][0]),
        ):
            key = tuple(sorted((int(a), int(b))))
            if key in owner:
                out.append((owner.pop(key), loc))
            else:
                owner[key] = loc
    return tuple(out)


def build_mesh(case: Case):
    nx, ny = case.resolution
    if nx * CELL_SIZE != case.length or ny * CELL_SIZE != THICKNESS:
        raise ValueError("requested domain is not an exact CELL_SIZE grid")
    mesh = core.build_mesh(nx, ny)
    p = mesh.p.copy()
    p[:, 0] *= case.length
    grad, area = np.empty_like(mesh.grad), np.empty_like(mesh.area)
    for e, nodes in enumerate(mesh.tri):
        dm = np.c_[p[nodes[1]] - p[nodes[0]], p[nodes[2]] - p[nodes[0]]]
        area[e] = np.linalg.det(dm) / 2
        inv = np.linalg.inv(dm)
        grad[e, 1:], grad[e, 0] = inv, -inv.sum(0)
    muscle = (
        np.ones(len(mesh.tri), bool)
        if case.muscle_layout == "full"
        else ((p[mesh.tri].mean(1)[:, 1] >= 0.04) & (p[mesh.tri].mean(1)[:, 1] <= 0.06))
    )
    young = np.where(muscle, E_MUSCLE, E_FAT)
    mu = young / (2 * (1 + case.poisson))
    lam = young * case.poisson / ((1 + case.poisson) * (1 - 2 * case.poisson))
    local = np.full(len(mesh.tri), -1)
    local[np.flatnonzero(muscle)] = np.arange(muscle.sum())
    return replace(
        mesh,
        p=p,
        grad=grad,
        area=area,
        muscle=muscle,
        young=young,
        mu=mu,
        lam=lam,
        muscle_local=local,
        edges=_edges(mesh.tri, muscle),
    )


def field_activation(mesh: Any, controls: np.ndarray) -> np.ndarray:
    out = np.zeros((len(mesh.tri), 3))
    values = (
        np.broadcast_to(controls, (mesh.muscle.sum(), 3))
        if _SHARED
        else controls.reshape(-1, 3)
    )
    out[mesh.muscle] = values
    return out


def assembly(
    mesh: Any, u: np.ndarray, a: np.ndarray, kind: str, hessian: bool, mixed: bool
):
    result = _ORIGINAL_ASSEMBLY(mesh, u, a, kind, hessian, mixed)
    if not (_SHARED and mixed):
        return result
    B = result[3].tocoo()
    collapsed = sp.coo_matrix(
        (B.data, (B.row, B.col % 3)), shape=(mesh.nfree, 3)
    ).tocsc()
    return (*result[:3], collapsed, *result[4:])


def target_y(mesh: Any, height: float, length: float) -> np.ndarray:
    s = mesh.p[:, 0] / length
    return height * 4 * s * (1 - s)


def loss(mesh: Any, u: np.ndarray, height: float, _name: str, length: float):
    d = core.unpack(mesh, u)[mesh.top]
    target = np.c_[np.zeros(len(mesh.top)), target_y(mesh, height, length)[mesh.top]]
    err = d - target
    value = float(np.mean(np.sum(err * err, axis=1)))
    grad = np.zeros(mesh.nfree)
    for node, value_at_node in zip(mesh.top, 2 * err / len(err), strict=True):
        for q in range(2):
            k = mesh.lookup[2 * node + q]
            if k >= 0:
                grad[k] = value_at_node[q]
    return value, grad, target


def metrics(
    mesh: Any, u: np.ndarray, a: np.ndarray, height: float, length: float
) -> dict[str, float]:
    # Reuse every existing diagnostic after expanding a shared group to cells.
    result = _ORIGINAL_METRICS(
        mesh, u, field_activation(mesh, a)[mesh.muscle].ravel(), height
    )
    top = core.unpack(mesh, u)[mesh.top]
    target = np.c_[np.zeros(len(mesh.top)), target_y(mesh, height, length)[mesh.top]]
    vector_error = top - target
    error = vector_error[:, 1]
    result.update(
        {
            "top_target_mae": float(np.mean(np.linalg.norm(vector_error, axis=1))),
            "top_target_rms": float(
                np.sqrt(np.mean(np.sum(vector_error * vector_error, axis=1)))
            ),
            "top_target_max": float(np.max(np.linalg.norm(vector_error, axis=1))),
            "top_error_rms": float(np.sqrt(np.mean(error * error))),
        }
    )
    return result


def grid(mesh: Any, a: np.ndarray, state: Any, step: int, height: float, length: float):
    out = _ORIGINAL_GRID(mesh, a, state, step, height)
    target = np.zeros_like(out.point_data["TargetDisplacement"])
    target[mesh.top, 1] = target_y(mesh, height, length)[mesh.top]
    out.point_data["TargetDisplacement"] = target
    if _GROUPS is not None:
        out.cell_data["ActivationGroup"] = _GROUPS
    return out


# ``solve`` is intentionally reused, so install the parameterized activation
# and collapsed shared-control Jacobian in its source module too.
core.activation = field_activation
core.assembly = assembly


def group_map(mesh: Any, shared: bool) -> np.ndarray:
    out = np.full(len(mesh.tri), -1, dtype=np.int64)
    out[mesh.muscle] = 0 if shared else np.arange(mesh.muscle.sum())
    return out


def determinant_modes(mesh: Any, state: Any) -> dict[str, float]:
    """Report every observed orientation mode without preventing it."""
    detf = np.asarray(state.det_f)
    deta = np.asarray(state.det_a)
    detg = np.asarray(state.det_g)

    def fractions(mask: np.ndarray, name: str) -> dict[str, float]:
        return {
            f"{name}_cell_fraction": float(mask.mean()),
            f"{name}_rest_measure_fraction": float(mesh.area[mask].sum() / mesh.area.sum()),
        }

    return {
        **fractions(deta < 0, "ainv_negative"),
        **fractions(detg < 0, "g_negative"),
        # det(F)<0 and det(Ainv)<0 gives the documented double-inversion mode.
        **fractions((detf < 0) & (deta < 0), "double_inverted"),
    }


def run_case(
    case: Case,
    cfg: Config,
    *,
    initial_controls: np.ndarray | None = None,
    initial_u: np.ndarray | None = None,
    initialization: str = "zero activation and zero displacement",
) -> dict[str, Any]:
    global _GROUPS, _SHARED
    _SHARED = case.activation_mode == "shared"
    mesh = build_mesh(case)
    out = cfg.output_dir / case.name
    out.mkdir()
    (out / "frames").mkdir()
    groups = group_map(mesh, _SHARED)
    _GROUPS = groups
    group_count = int(groups.max() + 1)
    a = (
        np.zeros(3 * group_count)
        if initial_controls is None
        else np.asarray(initial_controls, dtype=float).copy()
    )
    if a.shape != (3 * group_count,):
        raise ValueError(
            f"initial controls shape {a.shape} does not match {(3 * group_count,)}"
        )
    moment = np.zeros_like(a)
    variance = np.zeros_like(a)
    u = np.zeros(mesh.nfree) if initial_u is None else np.asarray(initial_u, dtype=float).copy()
    if u.shape != (mesh.nfree,):
        raise ValueError(f"initial displacement shape {u.shape} does not match {(mesh.nfree,)}")
    target_state = core.State(
        np.zeros(mesh.nfree),
        0.0,
        np.zeros(mesh.nfree),
        sp.csc_matrix((mesh.nfree, mesh.nfree)),
        *(np.ones(len(mesh.tri)) for _ in range(4)),
        0,
        True,
        None,
        0,
    )
    grid(mesh, a, target_state, 0, case.height, case.length).save(out / "target.vtu")
    rows: list[dict[str, Any]] = []
    series = []
    best = (math.inf, 0, None, None)
    best_op = best
    previous_recorded_a: np.ndarray | None = None
    persisted_row_count = 0

    def save_trace():
        nonlocal persisted_row_count
        mode = "w" if persisted_row_count == 0 else "a"
        with (out / "trace.csv").open(mode, newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            if persisted_row_count == 0:
                w.writeheader()
            w.writerows(rows[persisted_row_count:])
        persisted_row_count = len(rows)

    def persist_partial(*, force_series: bool = False):
        save_trace()
        if force_series or len(rows) % 50 == 0:
            write_json(
                out / "history.vtu.series",
                {"file-series-version": "1.0", "files": series},
            )

    for step in range(cfg.max_steps + 1):
        state = None
        error = None
        value = math.inf
        grad = np.zeros_like(a)
        try:
            state = core.solve(mesh, a, "stable", u, cfg)
            value, du, _ = loss(mesh, state.u, case.height, "l2", case.length)
            if not state.converged:
                raise RuntimeError(f"forward failed: {state.failure}")
            _, _, _, B, *_ = assembly(mesh, state.u, a, "stable", False, True)
            grad = np.asarray(B.T @ spla.spsolve(state.h, -du)).ravel()
            if not (np.isfinite(value) and np.isfinite(grad).all()):
                raise FloatingPointError("non-finite inverse gradient")
        except Exception as exc:
            error = repr(exc)
        if state is None:
            raise RuntimeError(f"no forward state at {step}: {error}")
        u = state.u
        detf = state.det_f
        inverted = detf < 0
        row = {
            "step": step,
            "optimizer_phase": "adam",
            "optimizer_iteration": step,
            "optimizer_evaluation": step + 1,
            "learning_rate": cfg.learning_rate * cfg.lr_decay**step
            if step < cfg.max_steps
            else None,
            "evaluation_success": int(error is None),
            "objective": value,
            "gradient_rms": float(np.linalg.norm(grad) / math.sqrt(max(1, len(grad)))),
            "gradient_inf": float(np.linalg.norm(grad, ord=np.inf)),
            "activation_update_rms": 0.0
            if previous_recorded_a is None
            else float(
                np.linalg.norm(a - previous_recorded_a) / math.sqrt(max(1, len(a)))
            ),
            "forward_converged": int(state.converged),
            "forward_iterations": state.iterations,
            "forward_failure": error or state.failure or "",
            "line_search_failures": state.line_search_failures,
            "equilibrium_residual_rms": float(
                np.linalg.norm(state.r) / math.sqrt(max(1, mesh.nfree))
            ),
            "min_det_f": float(detf.min()),
            "min_det_g": float(state.det_g.min()),
            "min_det_ainv": float(state.det_a.min()),
            "inverted_cell_fraction": float(inverted.mean()),
            "inverted_rest_measure_fraction": float(
                mesh.area[inverted].sum() / mesh.area.sum()
            ),
            "negative_det_f_mean": float(
                np.sum(mesh.area * np.maximum(-detf, 0.0)) / mesh.area.sum()
            ),
            **determinant_modes(mesh, state),
            **metrics(mesh, u, a, case.height, case.length),
        }
        row.update(
            {
                "detF/min": row["min_det_f"],
                "detG/min": row["min_det_g"],
                "detAinv/min": row["min_det_ainv"],
                "target/rms": row["top_target_rms"],
                "target/mae": row["top_target_mae"],
                "target/max": row["top_target_max"],
                "activation/rms": row["activation_rms"],
                "activation/neighbor_jump_rms": row["activation_neighbor_jump_rms"],
            }
        )
        rows.append(row)
        previous_recorded_a = a.copy()
        frame = out / "frames" / f"step-{step:04d}.vtu"
        grid(mesh, a, state, step, case.height, case.length).save(frame)
        series.append({"name": str(frame.relative_to(out)), "time": float(step)})
        persist_partial()
        if value < best[0]:
            best = (value, step, a.copy(), state)
        if (
            state.converged
            and row["min_det_f"] > 0
            and row["min_det_g"] > 0
            and row["min_det_ainv"] > 0
            and value < best_op[0]
        ):
            best_op = (value, step, a.copy(), state)
        cherries.set_step(step)
        cherries.log_metrics({f"{case.name}/objective": value})
        if error is not None:
            persist_partial(force_series=True)
            write_json(out / "failure.json", row)
            raise RuntimeError(f"inverse evaluation failed at {step}")
        if step < cfg.max_steps:
            moment = 0.9 * moment + 0.1 * grad
            variance = 0.999 * variance + 0.001 * grad * grad
            lr = cfg.learning_rate * cfg.lr_decay**step
            a -= (
                lr
                * (moment / (1 - 0.9 ** (step + 1)))
                / (np.sqrt(variance / (1 - 0.999 ** (step + 1))) + 1e-8)
            )

    refinement_cfg = cfg.model_copy(
        update={"forward_tolerance": cfg.refinement_forward_tolerance}
    )
    refinement_seed_u = u.copy()
    accepted_u = u.copy()
    accepted_a = a.copy()
    accepted_state = state
    refinement_callback_count = 0
    refinement_function_evaluation_count = 0
    refinement_forward_evaluation_count = 0
    refinement_adjoint_evaluation_count = 0
    refinement_trial_forward_failures = 0
    refinement_trial_max_forward_iterations = 0
    refinement_trial_max_equilibrium_residual_rms = 0.0
    refinement_attempts: list[dict[str, Any]] = []
    stalled_refinement_restarts = 0
    last_refinement_evaluation: tuple[np.ndarray, Any, float, np.ndarray] | None = None

    def persist_refinement_failure(
        stage: str,
        error: Exception | str,
        controls: np.ndarray,
        state_at_controls: Any | None,
    ) -> None:
        persist_partial(force_series=True)
        payload = {
            "case": case.name,
            "optimizer_phase": "lbfgs_unbounded",
            "stage": stage,
            "error": repr(error) if isinstance(error, Exception) else error,
            "accepted_frame_count": len(rows),
            "refinement_function_evaluations": refinement_function_evaluation_count,
            "control_count": len(controls),
            "controls_sha256": hashlib.sha256(controls.tobytes()).hexdigest(),
            "controls_rms": float(np.sqrt(np.mean(controls * controls))),
            "partial_trace": "trace.csv",
            "partial_series": "history.vtu.series",
        }
        if state_at_controls is not None:
            payload.update(
                {
                    "forward_converged": bool(state_at_controls.converged),
                    "forward_iterations": state_at_controls.iterations,
                    "forward_failure": state_at_controls.failure,
                    "equilibrium_residual_rms": float(
                        np.linalg.norm(state_at_controls.r)
                        / math.sqrt(max(1, mesh.nfree))
                    ),
                }
            )
        write_json(out / "failure.json", payload)

    def evaluate_refinement(
        controls: np.ndarray, initial: np.ndarray
    ) -> tuple[Any, float, np.ndarray]:
        nonlocal refinement_trial_forward_failures
        nonlocal refinement_forward_evaluation_count, refinement_adjoint_evaluation_count
        refinement_forward_evaluation_count += 1
        try:
            state_at_controls = core.solve(
                mesh, controls, "stable", initial, refinement_cfg
            )
        except Exception as error:
            refinement_trial_forward_failures += 1
            persist_refinement_failure("forward_exception", error, controls, None)
            raise
        if not state_at_controls.converged:
            refinement_trial_forward_failures += 1
            error = RuntimeError(
                "refinement forward failed: "
                f"iterations={state_at_controls.iterations}, "
                f"failure={state_at_controls.failure!r}"
            )
            persist_refinement_failure(
                "forward_nonconvergence", error, controls, state_at_controls
            )
            raise error
        try:
            value_at_controls, du, _ = loss(
                mesh, state_at_controls.u, case.height, "l2", case.length
            )
            _, _, _, mixed, *_ = assembly(
                mesh, state_at_controls.u, controls, "stable", False, True
            )
            gradient_at_controls = np.asarray(
                mixed.T @ spla.spsolve(state_at_controls.h, -du)
            ).ravel()
            refinement_adjoint_evaluation_count += 1
        except Exception as error:
            persist_refinement_failure(
                "inverse_or_adjoint_exception", error, controls, state_at_controls
            )
            raise
        if not (
            np.isfinite(value_at_controls) and np.isfinite(gradient_at_controls).all()
        ):
            error = FloatingPointError("non-finite refinement objective or gradient")
            persist_refinement_failure(
                "inverse_or_adjoint_nonfinite", error, controls, state_at_controls
            )
            raise error
        return state_at_controls, value_at_controls, gradient_at_controls

    def record_refinement(
        controls: np.ndarray,
        state_at_controls: Any,
        value_at_controls: float,
        gradient_at_controls: np.ndarray,
        iteration: int,
        evaluation: int,
    ) -> None:
        nonlocal accepted_a, accepted_state, accepted_u
        nonlocal best, best_op, previous_recorded_a
        accepted_a = controls.copy()
        accepted_state = state_at_controls
        accepted_u = state_at_controls.u.copy()
        detf = state_at_controls.det_f
        inverted = detf < 0
        step_at_controls = len(rows)
        row = {
            "step": step_at_controls,
            "optimizer_phase": "lbfgs_unbounded",
            "optimizer_iteration": iteration,
            "optimizer_evaluation": evaluation,
            "learning_rate": None,
            "evaluation_success": 1,
            "objective": value_at_controls,
            "gradient_rms": float(
                np.linalg.norm(gradient_at_controls)
                / math.sqrt(max(1, len(gradient_at_controls)))
            ),
            "gradient_inf": float(np.linalg.norm(gradient_at_controls, ord=np.inf)),
            "activation_update_rms": float(
                np.linalg.norm(controls - previous_recorded_a)
                / math.sqrt(max(1, len(controls)))
            ),
            "forward_converged": 1,
            "forward_iterations": state_at_controls.iterations,
            "forward_failure": state_at_controls.failure or "",
            "line_search_failures": state_at_controls.line_search_failures,
            "equilibrium_residual_rms": float(
                np.linalg.norm(state_at_controls.r) / math.sqrt(max(1, mesh.nfree))
            ),
            "min_det_f": float(detf.min()),
            "min_det_g": float(state_at_controls.det_g.min()),
            "min_det_ainv": float(state_at_controls.det_a.min()),
            "inverted_cell_fraction": float(inverted.mean()),
            "inverted_rest_measure_fraction": float(
                mesh.area[inverted].sum() / mesh.area.sum()
            ),
            "negative_det_f_mean": float(
                np.sum(mesh.area * np.maximum(-detf, 0.0)) / mesh.area.sum()
            ),
            **determinant_modes(mesh, state_at_controls),
            **metrics(mesh, accepted_u, accepted_a, case.height, case.length),
        }
        row.update(
            {
                "detF/min": row["min_det_f"],
                "detG/min": row["min_det_g"],
                "detAinv/min": row["min_det_ainv"],
                "target/rms": row["top_target_rms"],
                "target/mae": row["top_target_mae"],
                "target/max": row["top_target_max"],
                "activation/rms": row["activation_rms"],
                "activation/neighbor_jump_rms": row["activation_neighbor_jump_rms"],
            }
        )
        if tuple(row) != tuple(rows[0]):
            raise RuntimeError("Adam and L-BFGS trace schemas differ")
        rows.append(row)
        previous_recorded_a = controls.copy()
        frame = out / "frames" / f"step-{step_at_controls:04d}.vtu"
        grid(
            mesh,
            controls,
            state_at_controls,
            step_at_controls,
            case.height,
            case.length,
        ).save(frame)
        series.append(
            {"name": str(frame.relative_to(out)), "time": float(step_at_controls)}
        )
        persist_partial()
        if value_at_controls < best[0]:
            best = (
                value_at_controls,
                step_at_controls,
                controls.copy(),
                state_at_controls,
            )
        if (
            row["min_det_f"] > 0
            and row["min_det_g"] > 0
            and row["min_det_ainv"] > 0
            and value_at_controls < best_op[0]
        ):
            best_op = (
                value_at_controls,
                step_at_controls,
                controls.copy(),
                state_at_controls,
            )
        cherries.set_step(step_at_controls)
        cherries.log_metrics({f"{case.name}/objective": value_at_controls})

    strict_state, strict_value, strict_gradient = evaluate_refinement(
        a, refinement_seed_u
    )
    refinement_start_step = len(rows)
    record_refinement(a, strict_state, strict_value, strict_gradient, 0, 0)
    refinement_start_objective = strict_value
    refinement_objective_scale = 1.0 / max(abs(strict_value), 1.0e-12)
    last_refinement_evaluation = (
        a.copy(),
        strict_state,
        strict_value,
        strict_gradient,
    )

    trial_fields = (
        "evaluation",
        "objective",
        "gradient_rms",
        "gradient_inf",
        "forward_iterations",
        "equilibrium_residual_rms",
        "min_det_f",
        "min_det_g",
        "min_det_ainv",
    )
    with (out / "refinement-evaluations.csv").open("w", newline="") as trial_file:
        trial_writer = csv.DictWriter(trial_file, fieldnames=trial_fields)
        trial_writer.writeheader()

        def refinement_objective(
            controls: np.ndarray,
        ) -> tuple[float, np.ndarray]:
            nonlocal last_refinement_evaluation
            nonlocal refinement_function_evaluation_count
            nonlocal refinement_trial_max_equilibrium_residual_rms
            nonlocal refinement_trial_max_forward_iterations
            if last_refinement_evaluation is not None and np.array_equal(
                controls, last_refinement_evaluation[0]
            ):
                state_at_controls = last_refinement_evaluation[1]
                value_at_controls = last_refinement_evaluation[2]
                gradient_at_controls = last_refinement_evaluation[3]
            else:
                evaluation_seed = (
                    refinement_seed_u
                    if cfg.refinement_forward_initialization == "fixed_adam"
                    else accepted_u
                )
                state_at_controls, value_at_controls, gradient_at_controls = (
                    evaluate_refinement(controls, evaluation_seed)
                )
                last_refinement_evaluation = (
                    controls.copy(),
                    state_at_controls,
                    value_at_controls,
                    gradient_at_controls,
                )
            refinement_function_evaluation_count += 1
            residual = float(
                np.linalg.norm(state_at_controls.r) / math.sqrt(max(1, mesh.nfree))
            )
            refinement_trial_max_forward_iterations = max(
                refinement_trial_max_forward_iterations,
                state_at_controls.iterations,
            )
            refinement_trial_max_equilibrium_residual_rms = max(
                refinement_trial_max_equilibrium_residual_rms, residual
            )
            trial_writer.writerow(
                {
                    "evaluation": refinement_function_evaluation_count,
                    "objective": value_at_controls,
                    "gradient_rms": float(
                        np.linalg.norm(gradient_at_controls)
                        / math.sqrt(max(1, len(gradient_at_controls)))
                    ),
                    "gradient_inf": float(
                        np.linalg.norm(gradient_at_controls, ord=np.inf)
                    ),
                    "forward_iterations": state_at_controls.iterations,
                    "equilibrium_residual_rms": residual,
                    "min_det_f": float(state_at_controls.det_f.min()),
                    "min_det_g": float(state_at_controls.det_g.min()),
                    "min_det_ainv": float(state_at_controls.det_a.min()),
                }
            )
            trial_file.flush()
            return (
                refinement_objective_scale * value_at_controls,
                refinement_objective_scale * gradient_at_controls,
            )

        def accept_refinement(controls: np.ndarray) -> None:
            nonlocal refinement_callback_count
            if last_refinement_evaluation is None or not np.array_equal(
                controls, last_refinement_evaluation[0]
            ):
                raise RuntimeError(
                    "L-BFGS callback does not match its last objective evaluation"
                )
            refinement_callback_count += 1
            record_refinement(
                controls,
                last_refinement_evaluation[1],
                last_refinement_evaluation[2],
                last_refinement_evaluation[3],
                refinement_callback_count,
                refinement_function_evaluation_count,
            )

        optimizer = None
        refinement_termination = "restart_limit"
        for attempt in range(1, cfg.refinement_max_restarts + 1):
            current = rows[-1]
            if (
                current["gradient_inf"]
                <= cfg.refinement_acceptance_gradient_inf_tolerance
                and current["gradient_rms"]
                <= cfg.refinement_acceptance_gradient_rms_tolerance
                and current["equilibrium_residual_rms"]
                <= cfg.refinement_forward_tolerance
            ):
                refinement_termination = "physical_stationarity"
                break
            remaining_iterations = (
                cfg.refinement_max_iterations - refinement_callback_count
            )
            remaining_evaluations = (
                cfg.refinement_max_function_evaluations
                - refinement_function_evaluation_count
            )
            if remaining_iterations <= 0:
                refinement_termination = "accepted_iteration_budget"
                break
            if remaining_evaluations <= 0:
                refinement_termination = "function_evaluation_budget"
                break
            callbacks_before = refinement_callback_count
            evaluations_before = refinement_function_evaluation_count
            attempt_start_objective = rows[-1]["objective"]
            optimizer = opt.minimize(
                refinement_objective,
                accepted_a,
                jac=True,
                method="L-BFGS-B",
                callback=accept_refinement,
                options={
                    "maxiter": remaining_iterations,
                    "maxfun": remaining_evaluations,
                    "ftol": 0.0,
                    "gtol": refinement_objective_scale
                    * cfg.refinement_optimizer_gradient_inf_tolerance,
                    "maxls": cfg.refinement_max_line_search_steps,
                    "maxcor": 20,
                },
            )
            accepted_this_attempt = refinement_callback_count - callbacks_before
            evaluations_this_attempt = (
                refinement_function_evaluation_count - evaluations_before
            )
            physical_improvement = attempt_start_objective - rows[-1]["objective"]
            meaningful_improvement = physical_improvement > max(
                1e-14, abs(attempt_start_objective) * 1e-10
            )
            if accepted_this_attempt != optimizer.nit:
                raise RuntimeError(
                    "L-BFGS callback count differs from accepted iterations"
                )
            if evaluations_this_attempt != optimizer.nfev:
                raise RuntimeError("L-BFGS evaluation count differs from scipy receipt")
            if not np.array_equal(np.asarray(optimizer.x), accepted_a):
                raise RuntimeError("L-BFGS result differs from last accepted iterate")
            refinement_attempts.append(
                {
                    "attempt": attempt,
                    "accepted_iterations": accepted_this_attempt,
                    "function_evaluations": evaluations_this_attempt,
                    "scipy_success": bool(optimizer.success),
                    "scipy_status": int(optimizer.status),
                    "scipy_message": str(optimizer.message),
                    "scaled_objective": float(optimizer.fun),
                    "physical_objective": float(
                        optimizer.fun / refinement_objective_scale
                    ),
                    "physical_gradient_inf": rows[-1]["gradient_inf"],
                    "physical_gradient_rms": rows[-1]["gradient_rms"],
                    "physical_objective_improvement": physical_improvement,
                    "meaningful_objective_improvement": meaningful_improvement,
                }
            )
            if accepted_this_attempt == 0 or not meaningful_improvement:
                stalled_refinement_restarts += 1
            else:
                stalled_refinement_restarts = 0
            if stalled_refinement_restarts >= cfg.refinement_max_stalled_restarts:
                refinement_termination = "stalled_restart_limit"
                break
        if (
            refinement_termination == "restart_limit"
            and refinement_callback_count >= cfg.refinement_max_iterations
        ):
            refinement_termination = "accepted_iteration_budget"
    a = accepted_a
    state = accepted_state
    u = accepted_u
    if best_op[2] is None:
        raise RuntimeError("no orientation-preserving evaluation")
    persist_partial(force_series=True)
    _, best_step, best_a, best_state = best
    _, best_op_step, best_op_a, best_op_state = best_op
    grid(mesh, best_a, best_state, best_step, case.height, case.length).save(
        out / "best.vtu"
    )
    grid(mesh, best_op_a, best_op_state, best_op_step, case.height, case.length).save(
        out / "best-orientation-preserving.vtu"
    )
    final_step = len(rows) - 1
    grid(mesh, a, state, final_step, case.height, case.length).save(out / "final.vtu")
    np.savez_compressed(
        out / "final-state.npz", controls=a, displacement_free=state.u, step=final_step
    )
    tail = rows[-50:]
    tail_absolute_range = max(r["objective"] for r in tail) - min(
        r["objective"] for r in tail
    )
    tail_range = tail_absolute_range / max(
        abs(min(r["objective"] for r in tail)), 1e-30
    )
    tail_stabilization_gate = bool(
        all(r["forward_converged"] and r["evaluation_success"] for r in tail)
        and (tail_range <= 0.01 or tail_absolute_range <= cfg.tail_absolute_tolerance)
    )
    refinement_rows = rows[refinement_start_step:]
    objective_increase_tolerance = max(1e-14, abs(refinement_start_objective) * 1e-10)
    accepted_objective_increase_count = sum(
        later["objective"] - earlier["objective"] > objective_increase_tolerance
        for earlier, later in zip(refinement_rows, refinement_rows[1:], strict=False)
    )
    practical_stationarity_gate = bool(
        rows[-1]["gradient_inf"] <= cfg.refinement_acceptance_gradient_inf_tolerance
        and rows[-1]["gradient_rms"] <= cfg.refinement_acceptance_gradient_rms_tolerance
        and rows[-1]["equilibrium_residual_rms"] <= cfg.refinement_forward_tolerance
        and rows[-1]["objective"]
        <= refinement_start_objective + objective_increase_tolerance
        and accepted_objective_increase_count == 0
        and refinement_trial_forward_failures == 0
        and all(r["forward_converged"] and r["evaluation_success"] for r in rows)
    )
    digest = hashlib.sha256(groups.tobytes()).hexdigest()
    sizes = np.bincount(groups[groups >= 0], minlength=group_count)
    report = {
        "case": {
            "name": case.name,
            "length": case.length,
            "domain_id": f"L{case.length:g}",
            "muscle_layout": case.muscle_layout,
            "activation_mode": case.activation_mode,
            "poisson": case.poisson,
            "height": case.height,
            "protocol": case.protocol,
        },
        "geometry": {
            "geometry_id": "short" if case.length == 0.1 else "long",
            "muscle_extent_id": case.muscle_layout,
            "domain": [case.length, THICKNESS],
            "resolution": list(case.resolution),
            "active_band_y": [0.04, 0.06]
            if case.muscle_layout == "band"
            else [0.0, THICKNESS],
            "fixed": "bottom and sides, xy",
            "top_and_interior": "free",
            "target": "uy=h*4*(x/L)*(1-x/L)",
        },
        "physics": {
            "elasticity": "stable_neo_hookean",
            "target_loss": "l2",
            "target_observation": "free top nodes, xy components",
        },
        "materials": {
            "fat": None
            if case.muscle_layout == "full"
            else {"E_MPa": E_FAT, "nu": case.poisson},
            "muscle": {"E_MPa": E_MUSCLE, "nu": case.poisson},
            "skin_energy": None,
        },
        "activation": {
            "sharing_id": case.activation_mode,
            "initialization": initialization,
            "group_count": group_count,
            "cells_per_group_min": int(sizes.min()),
            "cells_per_group_max": int(sizes.max()),
            "raw_dofs_per_group": 3,
            "raw_activation_dofs": len(a),
            "cell_to_group_sha256": digest,
            "bounds": None,
            "regularizer": None,
            "det_constraint": None,
        },
        "counts": {
            "triangles": len(mesh.tri),
            "muscle_triangles": int(mesh.muscle.sum()),
            "fat_triangles": int((~mesh.muscle).sum()),
            "free_dofs": mesh.nfree,
            "observed_top_nodes": len(mesh.top),
            "observed_top_components": int(2 * len(mesh.top)),
            "activation_dofs": len(a),
        },
        "inverse": {
            "schedule": "fixed Adam exploration then objective-normalized, automatically restarted unbounded L-BFGS refinement",
            "updates": cfg.max_steps + refinement_callback_count,
            "evaluations": len(rows),
            "cost": {
                "accepted_trace_states": len(rows),
                "adam_objective_forward_adjoint_evaluations": cfg.max_steps + 1,
                "strict_refinement_forward_adjoint_evaluations": 1,
                "lbfgs_function_calls": refinement_function_evaluation_count,
                "lbfgs_actual_forward_evaluations": refinement_forward_evaluation_count - 1,
                "lbfgs_actual_adjoint_evaluations": refinement_adjoint_evaluation_count - 1,
                "lbfgs_rejected_forward_adjoint_evaluations": max(
                    0, refinement_forward_evaluation_count - (1 + refinement_callback_count)
                ),
                "note": "Rejected L-BFGS trials are counted separately from exact accepted trace states.",
            },
            "failures": {
                "forward": sum(not r["forward_converged"] for r in rows),
                "inverse": sum(not r["evaluation_success"] for r in rows),
                "adjoint": 0,
                "nonfinite": sum(not r["evaluation_success"] for r in rows),
                "refinement_trial_forward": refinement_trial_forward_failures,
            },
            "best_step": best[1],
            "best_loss": best[0],
            "best_orientation_preserving_step": best_op[1],
            "best_orientation_preserving_loss": best_op[0],
            "adam": {
                "updates": cfg.max_steps,
                "evaluations": cfg.max_steps + 1,
                "learning_rate": cfg.learning_rate,
                "learning_rate_decay": cfg.lr_decay,
                "forward_tolerance": cfg.forward_tolerance,
            },
            "refinement": {
                "method": "automatically restarted L-BFGS-B used without bounds",
                "bounds": None,
                "forward_initialization": cfg.refinement_forward_initialization,
                "start_step": refinement_start_step,
                "start_objective": refinement_start_objective,
                "objective_scale": refinement_objective_scale,
                "accepted_iterations": refinement_callback_count,
                "function_evaluations": refinement_function_evaluation_count,
                "forward_evaluations": refinement_forward_evaluation_count,
                "adjoint_evaluations": refinement_adjoint_evaluation_count,
                "rejected_forward_adjoint_evaluations": max(
                    0,
                    refinement_forward_evaluation_count
                    - (1 + refinement_callback_count),
                ),
                "callback_matches_accepted_iterations": bool(
                    refinement_callback_count
                    == sum(item["accepted_iterations"] for item in refinement_attempts)
                ),
                "termination": refinement_termination,
                "attempt_count": len(refinement_attempts),
                "attempts": refinement_attempts,
                "stalled_restarts": stalled_refinement_restarts,
                "scipy_success": bool(optimizer.success)
                if optimizer is not None
                else True,
                "scipy_status": int(optimizer.status) if optimizer is not None else 0,
                "scipy_message": str(optimizer.message)
                if optimizer is not None
                else "not run: strict seed met physical stationarity",
                "scipy_reported_objective": float(
                    optimizer.fun / refinement_objective_scale
                )
                if optimizer is not None
                else refinement_start_objective,
                "max_iterations": cfg.refinement_max_iterations,
                "max_function_evaluations": cfg.refinement_max_function_evaluations,
                "max_restarts": cfg.refinement_max_restarts,
                "max_stalled_restarts": cfg.refinement_max_stalled_restarts,
                "optimizer_gradient_inf_tolerance": cfg.refinement_optimizer_gradient_inf_tolerance,
                "acceptance_gradient_inf_tolerance": cfg.refinement_acceptance_gradient_inf_tolerance,
                "acceptance_gradient_rms_tolerance": cfg.refinement_acceptance_gradient_rms_tolerance,
                "forward_tolerance": cfg.refinement_forward_tolerance,
                "max_line_search_steps": cfg.refinement_max_line_search_steps,
                "trial_forward_failures": refinement_trial_forward_failures,
                "trial_max_forward_iterations": refinement_trial_max_forward_iterations,
                "trial_max_equilibrium_residual_rms": refinement_trial_max_equilibrium_residual_rms,
                "accepted_objective_increase_count": accepted_objective_increase_count,
            },
            "convergence": {
                "criterion": "all accepted evaluations valid, no failed refinement trials, nonincreasing accepted refinement objective, strict forward residual, and final gradient infinity/RMS norms within tolerance",
                "final_gradient_inf": rows[-1]["gradient_inf"],
                "gradient_inf_tolerance": cfg.refinement_acceptance_gradient_inf_tolerance,
                "final_gradient_rms": rows[-1]["gradient_rms"],
                "gradient_rms_tolerance": cfg.refinement_acceptance_gradient_rms_tolerance,
                "final_equilibrium_residual_rms": rows[-1]["equilibrium_residual_rms"],
                "equilibrium_residual_tolerance": cfg.refinement_forward_tolerance,
                "practical_stationarity_gate": practical_stationarity_gate,
            },
            "tail": {
                "window": len(tail),
                "relative_range": tail_range,
                "absolute_range": tail_absolute_range,
                "absolute_tolerance": cfg.tail_absolute_tolerance,
                "criterion": "all valid and (relative range <= 1% or absolute L2 range <= tolerance)",
                "inverse_converged_1pct_tail_gate": tail_stabilization_gate,
            },
            "minimum_det_f": min(r["min_det_f"] for r in rows),
            "minimum_det_g": min(r["min_det_g"] for r in rows),
            "minimum_det_ainv": min(r["min_det_ainv"] for r in rows),
            "final_inverted_cell_fraction": rows[-1]["inverted_cell_fraction"],
            "final_inverted_rest_measure_fraction": rows[-1][
                "inverted_rest_measure_fraction"
            ],
            "final_negative_det_f_mean": rows[-1]["negative_det_f_mean"],
            "first_inversion_step": next(
                (r["step"] for r in rows if r["inverted_cell_fraction"] > 0), None
            ),
            "last_inversion_step": next(
                (r["step"] for r in reversed(rows) if r["inverted_cell_fraction"] > 0),
                None,
            ),
            "inverted_frame_count": sum(r["inverted_cell_fraction"] > 0 for r in rows),
            "inverted_frame_fraction": float(
                sum(r["inverted_cell_fraction"] > 0 for r in rows) / len(rows)
            ),
            "peak_inverted_cell_fraction": max(
                r["inverted_cell_fraction"] for r in rows
            ),
            "peak_inverted_rest_measure_fraction": max(
                r["inverted_rest_measure_fraction"] for r in rows
            ),
            "peak_negative_det_f_mean": max(r["negative_det_f_mean"] for r in rows),
            "orientation_modes": {
                mode: {
                    "first_step": next(
                        (r["step"] for r in rows if r[f"{mode}_cell_fraction"] > 0),
                        None,
                    ),
                    "last_step": next(
                        (r["step"] for r in reversed(rows) if r[f"{mode}_cell_fraction"] > 0),
                        None,
                    ),
                    "peak_cell_fraction": max(r[f"{mode}_cell_fraction"] for r in rows),
                    "peak_rest_measure_fraction": max(
                        r[f"{mode}_rest_measure_fraction"] for r in rows
                    ),
                    "final_cell_fraction": rows[-1][f"{mode}_cell_fraction"],
                    "final_rest_measure_fraction": rows[-1][f"{mode}_rest_measure_fraction"],
                }
                for mode in ("ainv_negative", "g_negative", "double_inverted")
            },
        },
        "metrics": {
            "best": rows[best[1]],
            "final": rows[-1],
            "best_orientation_preserving": rows[best_op[1]],
        },
        "paraview": {"fps": 30, "frames": len(rows), "series": "history.vtu.series"},
        "reproducibility": {
            "runner": file_digest(Path(__file__)),
            "equilibrium_core": file_digest(_SOURCE),
            "config": cfg.model_dump(mode="json"),
            "config_sha256": hashlib.sha256(
                json.dumps(cfg.model_dump(mode="json"), sort_keys=True, default=str).encode()
            ).hexdigest(),
        },
        "artifacts": {
            "series": "history.vtu.series",
            "target": "target.vtu",
            "best": "best.vtu",
            "best_orientation_preserving": "best-orientation-preserving.vtu",
            "final": "final.vtu",
            "final_state": "final-state.npz",
            "refinement_evaluations": "refinement-evaluations.csv",
        },
    }
    write_json(out / "summary.json", report)
    return report


def derivative_check(case: Case) -> dict[str, Any]:
    global _SHARED
    _SHARED = case.activation_mode == "shared"
    mesh = build_mesh(case)
    groups = 1 if _SHARED else int(mesh.muscle.sum())
    cfg = Config(
        max_steps=2,
        validate_derivatives=False,
        forward_max_iterations=120,
        forward_tolerance=1e-10,
    )
    a = 0.001 * np.sin(np.arange(3 * groups))
    state = core.solve(mesh, a, "stable", np.zeros(mesh.nfree), cfg)
    _value, du, _ = loss(mesh, state.u, case.height, "l2", case.length)
    _, _, _, B, *_ = assembly(mesh, state.u, a, "stable", False, True)
    g = np.asarray(B.T @ spla.spsolve(state.h, -du)).ravel()
    i = int(np.argmax(abs(g)))
    eps = 1e-5
    vals = []
    perturbed_converged = []
    for sign in (-1, 1):
        aa = a.copy()
        aa[i] += sign * eps
        ss = core.solve(mesh, aa, "stable", state.u, cfg)
        perturbed_converged.append(bool(ss.converged))
        vals.append(loss(mesh, ss.u, case.height, "l2", case.length)[0])
    fd = (vals[1] - vals[0]) / (2 * eps)
    err = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-14)
    return {
        "activation_mode": case.activation_mode,
        "poisson": case.poisson,
        "base_forward_converged": bool(state.converged),
        "perturbed_forward_converged": perturbed_converged,
        "relative_error": float(err),
        "passed": bool(state.converged and all(perturbed_converged) and err < 1e-4),
    }


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError("refusing nonempty output")
    available = {case.name: case for case in cases()}
    names = (
        tuple(available)
        if cfg.cases == "all"
        else tuple(part.strip() for part in cfg.cases.split(",") if part.strip())
    )
    unknown = [name for name in names if name not in available]
    if unknown:
        raise ValueError(f"unknown cases: {unknown}")
    if len(set(names)) != len(names):
        raise ValueError("case names must be unique")
    selected = tuple(available[name] for name in names)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    checks = (
        [
            derivative_check(Case("check", 1.0, "band", mode, 0.49, 0.20, "direct"))
            for mode in ("per_cell", "shared")
        ]
        if cfg.validate_derivatives
        else []
    )
    if checks and not all(x["passed"] for x in checks):
        raise RuntimeError(f"derivative failure: {checks}")
    if cfg.smoke:
        selected = selected[:2]
        cfg.max_steps = min(3, cfg.max_steps)
        cfg.refinement_max_iterations = min(3, cfg.refinement_max_iterations)
        cfg.refinement_max_function_evaluations = min(
            100, cfg.refinement_max_function_evaluations
        )
    reports: list[dict[str, Any]] = []
    completed: dict[float, dict[str, Any]] = {}
    for case in selected:
        if not case.protocol.startswith("shared_then_release"):
            report = run_case(case, cfg)
            reports.append(report)
            if case.protocol == "shared":
                completed[case.height] = report
            continue
        shared = completed.get(case.height)
        if shared is None:
            raise RuntimeError(
                f"{case.name} requires its shared phase earlier in --cases; "
                "use the default ordered matrix"
            )
        if not shared["inverse"]["convergence"]["practical_stationarity_gate"]:
            raise RuntimeError(
                f"refusing to release nonstationary shared seed {shared['case']['name']}"
            )
        source = cfg.output_dir / shared["case"]["name"] / "final-state.npz"
        if not source.is_file():
            raise FileNotFoundError(source)
        seed = np.load(source)
        shared_controls = np.asarray(seed["controls"], dtype=float)
        shared_u = np.asarray(seed["displacement_free"], dtype=float)
        shared_case = available[shared["case"]["name"]]
        shared_state, shared_strict = strict_observation(
            shared_case, shared_controls, shared_u, cfg
        )
        shared_final = shared["metrics"]["final"]
        shared_u_delta = float(np.linalg.norm(shared_state.u - shared_u, np.inf))
        shared_objective_delta = abs(shared_strict["objective"] - shared_final["objective"])
        if shared_u_delta > 1e-8 or shared_objective_delta > 1e-10:
            raise RuntimeError(
                "stored shared endpoint did not reproduce under strict handoff: "
                f"u_inf={shared_u_delta}, objective={shared_objective_delta}"
            )
        mesh = build_mesh(case)
        expanded = np.tile(shared_controls, int(mesh.muscle.sum()))
        shared_u_state, shared_u_branch = strict_observation(
            case, expanded, shared_state.u, cfg
        )
        branch_u_delta = float(np.linalg.norm(shared_u_state.u - shared_state.u, np.inf))
        branch_objective_delta = abs(
            shared_u_branch["objective"] - shared_strict["objective"]
        )
        branch_det_deltas = {
            name: abs(shared_u_branch[name] - shared_strict[name])
            for name in ("min_det_f", "min_det_ainv", "min_det_g")
        }
        if (
            branch_u_delta > 1e-8
            or branch_objective_delta > 1e-10
            or max(branch_det_deltas.values()) > 1e-8
        ):
            raise RuntimeError(
                "tiled per-cell controls failed to reproduce the strict shared branch: "
                f"u_inf={branch_u_delta}, objective={branch_objective_delta}, "
                f"dets={branch_det_deltas}"
            )
        zero_u_state, zero_u_branch = strict_observation(
            case, expanded, np.zeros(mesh.nfree), cfg
        )
        use_shared_displacement = case.protocol == "shared_then_release"
        report = run_case(
            case,
            cfg,
            initial_controls=expanded,
            initial_u=shared_u_state.u if use_shared_displacement else None,
            initialization=(
                f"expanded converged shared activation from {source.relative_to(cfg.output_dir)}; "
                + (
                    "stored strict shared forward displacement transferred"
                    if use_shared_displacement
                    else "released forward solve initialized at zero displacement"
                )
                + "; new independent optimizer state restarted after dimension change"
            ),
        )
        report["continuation"] = {
            "seed_case": shared["case"]["name"],
            "seed_final_state": str(source.relative_to(cfg.output_dir)),
            "seed_controls_sha256": hashlib.sha256(shared_controls.tobytes()).hexdigest(),
            "expanded_controls_sha256": hashlib.sha256(expanded.tobytes()).hexdigest(),
            "seed_displacement_sha256": hashlib.sha256(shared_u.tobytes()).hexdigest(),
            "forward_initialization": (
                "stored strict shared endpoint displacement"
                if use_shared_displacement
                else "zero displacement"
            ),
            "optimizer_restart": True,
            "dimension_change": [int(shared_controls.size), int(expanded.size)],
            "handoff": {
                "strict_tolerance": cfg.refinement_forward_tolerance,
                "shared_endpoint_reproduction": {
                    "strict": shared_strict,
                    "stored_u_inf_delta": shared_u_delta,
                    "stored_objective_delta": shared_objective_delta,
                    "asserted": True,
                },
                "tiled_shared_u_branch": {
                    "strict": shared_u_branch,
                    "u_inf_delta_from_shared": branch_u_delta,
                    "objective_delta_from_shared": branch_objective_delta,
                    "determinant_deltas_from_shared": branch_det_deltas,
                    "asserted": True,
                },
                "tiled_zero_u_branch": {
                    "strict": zero_u_branch,
                    "u_inf_delta_from_shared_u_branch": float(
                        np.linalg.norm(zero_u_state.u - shared_u_state.u, np.inf)
                    ),
                    "objective_gap_from_shared_u_branch": float(
                        zero_u_branch["objective"] - shared_u_branch["objective"]
                    ),
                    "determinant_deltas_from_shared_u_branch": {
                        name: float(zero_u_branch[name] - shared_u_branch[name])
                        for name in ("min_det_f", "min_det_ainv", "min_det_g")
                    },
                    "asserted": False,
                },
                "strict_forward_adjoint_evaluations": 3,
            },
            "cost": {
                "shared_evaluations": int(shared["inverse"]["evaluations"]),
                "release_evaluations": int(report["inverse"]["evaluations"]),
                "end_to_end_evaluations": int(
                    shared["inverse"]["evaluations"]
                    + report["inverse"]["evaluations"]
                ),
                "end_to_end_forward_adjoint_evaluations": int(
                    shared["inverse"]["cost"]["adam_objective_forward_adjoint_evaluations"]
                    + shared["inverse"]["cost"]["strict_refinement_forward_adjoint_evaluations"]
                    + shared["inverse"]["cost"]["lbfgs_actual_forward_evaluations"]
                    + report["inverse"]["cost"]["adam_objective_forward_adjoint_evaluations"]
                    + report["inverse"]["cost"]["strict_refinement_forward_adjoint_evaluations"]
                    + report["inverse"]["cost"]["lbfgs_actual_forward_evaluations"]
                    + 3
                ),
                "comparison_note": (
                    "direct is a conditional endpoint comparator; shared-plus-release "
                    "has this additional shared-stage cost"
                ),
            },
        }
        write_json(cfg.output_dir / case.name / "summary.json", report)
        reports.append(report)
    write_json(
        cfg.output_dir / "summary.json",
        {
            "design": "2d-band-muscle-shared-then-release-high-targets",
            "derivative_checks": checks,
            "cases": reports,
        },
    )
    failed = [
        report["case"]["name"]
        for report in reports
        if not report["inverse"]["convergence"]["practical_stationarity_gate"]
    ]
    if cfg.require_inverse_convergence and not cfg.smoke and failed:
        raise RuntimeError(
            f"practical-stationarity gate failed after recording full matrix: {failed}"
        )


if __name__ == "__main__":
    cherries.main(main)
