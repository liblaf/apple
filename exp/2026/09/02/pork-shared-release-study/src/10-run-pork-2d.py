"""Unconstrained 2-D pork inverse-physics factor study.

Each frame is an inverse evaluation, including failed nonlinear forward solves;
this is deliberate diagnostic data, not a filtered "physical" trajectory.
"""

from __future__ import annotations

# ruff: noqa: ANN001, ARG005, B007, BLE001, C901, E741, EM101, EM102, FBT001, FBT003, PLR0912, PLR0915, RUF007, RUF059, S112, TRY003, TRY301
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.ndimage as ndi
import scipy.optimize as opt
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from liblaf import cherries

E_FAT, E_MUSCLE, NU = 0.003, 0.03, 0.49
BASES = np.array(
    [[[1.0, 0.0], [0.0, 0.0]], [[0.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]]
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("10-pork-2d", mkdir=True)
    # Controlled OFAT: each entry differs from baseline in exactly one factor.
    cases: str = (
        "baseline:stable:l2:100x10:.05,"
        "height-low:stable:l2:100x10:.025,"
        "height-high:stable:l2:100x10:.1,"
        "loss-l1:stable:l1:100x10:.05,"
        "loss-linf:stable:linf:100x10:.05,"
        "mesh-medium:stable:l2:50x5:.05,"
        "mesh-dense:stable:l2:200x20:.05,"
        "energy-linear:linear:l2:100x10:.05"
    )
    max_steps: int = 1200
    learning_rate: float = 0.03
    lr_decay: float = 0.99
    forward_tolerance: float = 1e-8
    forward_max_iterations: int = 3000
    refinement_max_iterations: int = 500
    refinement_max_function_evaluations: int = 30000
    refinement_forward_tolerance: float = 1e-10
    refinement_optimizer_gradient_inf_tolerance: float = 1e-12
    refinement_acceptance_gradient_inf_tolerance: float = 2e-8
    refinement_acceptance_gradient_rms_tolerance: float = 1e-8
    refinement_max_line_search_steps: int = 50
    refinement_total_accepted_iterations: int = 1000
    refinement_stalled_restart_cap: int = 5
    refinement_max_stagnant_accepted_iterations: int = 5
    refinement_minimum_control_update_rms: float = 1e-12
    refinement_minimum_relative_control_update: float = 1e-10
    validate_derivatives: bool = True
    require_inverse_convergence: bool = False
    smoke: bool = False


@dataclass(frozen=True)
class Mesh:
    nx: int
    ny: int
    p: np.ndarray
    tri: np.ndarray
    grad: np.ndarray
    area: np.ndarray
    muscle: np.ndarray
    young: np.ndarray
    lam: np.ndarray
    mu: np.ndarray
    edof: np.ndarray
    free: np.ndarray
    lookup: np.ndarray
    top: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    ee: np.ndarray
    lr: np.ndarray
    lc: np.ndarray
    muscle_local: np.ndarray
    edges: tuple[tuple[int, int], ...]

    @property
    def nfree(self) -> int:
        return int(self.free.size)


@dataclass
class State:
    u: np.ndarray
    energy: float
    r: np.ndarray
    h: sp.csc_matrix
    det_f: np.ndarray
    det_g: np.ndarray
    det_a: np.ndarray
    min_sv_a: np.ndarray
    iterations: int
    converged: bool
    failure: str | None
    line_search_failures: int


def write_json(path: Path, data: Any) -> None:
    def sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
            return None
        if isinstance(value, np.integer):
            return int(value)
        return value

    path.write_text(
        json.dumps(sanitize(data), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )


def build_mesh(nx: int, ny: int) -> Mesh:
    x, y = np.linspace(0, 1, nx + 1), np.linspace(0, 0.1, ny + 1)
    xx, yy = np.meshgrid(x, y)
    p = np.c_[xx.ravel(), yy.ravel()]
    tri = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            tri.extend(((a, a + 1, a + nx + 2), (a, a + nx + 2, a + nx + 1)))
    tri = np.asarray(tri, dtype=np.int64)
    grad = np.empty((len(tri), 3, 2))
    area = np.empty(len(tri))
    for e, nodes in enumerate(tri):
        dm = np.c_[p[nodes[1]] - p[nodes[0]], p[nodes[2]] - p[nodes[0]]]
        if np.linalg.det(dm) <= 0:
            raise ValueError("nonpositive reference triangle")
        area[e] = np.linalg.det(dm) / 2
        inv = np.linalg.inv(dm)
        grad[e, 1:], grad[e, 0] = inv, -inv.sum(0)
    muscle = (p[tri].mean(1)[:, 1] >= 0.04) & (p[tri].mean(1)[:, 1] <= 0.06)
    young = np.where(muscle, E_MUSCLE, E_FAT)
    mu = young / (2 * (1 + NU))
    lam = young * NU / ((1 + NU) * (1 - 2 * NU))
    edof = np.empty((len(tri), 6), dtype=np.int64)
    edof[:, 0::2] = 2 * tri
    edof[:, 1::2] = 2 * tri + 1
    fixed_node = np.flatnonzero(
        np.isclose(p[:, 1], 0) | np.isclose(p[:, 0], 0) | np.isclose(p[:, 0], 1)
    )
    fixed = np.r_[2 * fixed_node, 2 * fixed_node + 1]
    all_dof = np.arange(2 * len(p))
    free = np.setdiff1d(all_dof, fixed)
    lookup = np.full(2 * len(p), -1)
    lookup[free] = np.arange(len(free))
    ef = lookup[edof]
    ee = []
    lr = []
    lc = []
    rows = []
    cols = []
    for e in range(len(tri)):
        for a in range(6):
            for b in range(6):
                if ef[e, a] >= 0 and ef[e, b] >= 0:
                    ee.append(e)
                    lr.append(a)
                    lc.append(b)
                    rows.append(ef[e, a])
                    cols.append(ef[e, b])
    me = np.flatnonzero(muscle)
    ml = np.full(len(tri), -1)
    ml[me] = np.arange(len(me))
    owners = {}
    edges = []
    for loc, e in enumerate(me):
        ns = tri[e]
        for a, b in ((ns[0], ns[1]), (ns[1], ns[2]), (ns[2], ns[0])):
            key = tuple(sorted((int(a), int(b))))
            if key in owners:
                edges.append((owners.pop(key), loc))
            else:
                owners[key] = loc
    top = np.flatnonzero(
        np.isclose(p[:, 1], 0.1) & ~np.isclose(p[:, 0], 0) & ~np.isclose(p[:, 0], 1)
    )
    return Mesh(
        nx,
        ny,
        p,
        tri,
        grad,
        area,
        muscle,
        young,
        lam,
        mu,
        edof,
        free,
        lookup,
        top,
        np.array(rows),
        np.array(cols),
        np.array(ee),
        np.array(lr),
        np.array(lc),
        ml,
        tuple(edges),
    )


def unpack(m: Mesh, uf: np.ndarray) -> np.ndarray:
    u = np.zeros(2 * len(m.p))
    u[m.free] = uf
    return u.reshape(-1, 2)


def activation(m: Mesh, a: np.ndarray) -> np.ndarray:
    out = np.zeros((len(m.tri), 3))
    out[m.muscle] = a.reshape(-1, 3)
    return out


def cof(a: np.ndarray) -> np.ndarray:
    out = np.empty_like(a)
    out[..., 0, 0] = a[..., 1, 1]
    out[..., 0, 1] = -a[..., 1, 0]
    out[..., 1, 0] = -a[..., 0, 1]
    out[..., 1, 1] = a[..., 0, 0]
    return out


def assembly(
    m: Mesh,
    uf: np.ndarray,
    controls: np.ndarray,
    energy_kind: str,
    hessian: bool,
    mixed: bool,
):
    u = unpack(m, uf)
    f = np.einsum("eia,eib->eab", (m.p + u)[m.tri], m.grad)
    av = activation(m, controls)
    activation_delta = np.einsum("ec,cij->eij", av, BASES)
    ainv = np.eye(2)[None] + activation_delta
    if energy_kind == "linear":
        # Small strain: eps = sym(F - I + A), with symmetric A=Ainv-I.
        g = f + activation_delta
        q = g - np.eye(2)[None]
        e = 0.5 * (q + np.swapaxes(q, 1, 2))
        tr = np.trace(e, axis1=1, axis2=2)
        s = 2 * m.mu[:, None, None] * e + m.lam[:, None, None] * tr[
            :, None, None
        ] * np.eye(2)
        density = m.mu * np.sum(e * e, (1, 2)) + 0.5 * m.lam * tr * tr
        detg = np.linalg.det(g)
        deta = np.linalg.det(ainv)
        pf = s  # d e / d F is identity in this linearized active model
    else:
        g = f @ ainv
        j = np.linalg.det(g)
        c = cof(g)
        k = -m.mu + m.lam * (j - 1)
        pg = m.mu[:, None, None] * g + k[:, None, None] * c
        pf = pg @ np.swapaxes(ainv, 1, 2)
        density = (
            0.5 * m.mu * (np.sum(g * g, (1, 2)) - 2)
            - m.mu * (j - 1)
            + 0.5 * m.lam * (j - 1) ** 2
        )
        detg = j
        deta = np.linalg.det(ainv)
    local = m.area[:, None, None] * np.einsum("eab,eib->eia", pf, m.grad)
    r = np.zeros(m.nfree)
    lf = m.lookup[m.edof]
    mask = lf >= 0
    np.add.at(r, lf[mask], local.reshape(-1, 6)[mask])
    H = sp.csc_matrix((m.nfree, m.nfree))
    if hessian:
        lh = np.empty((len(m.tri), 6, 6))
        for n in range(3):
            for q in range(2):
                df = np.zeros_like(f)
                df[:, q, :] = m.grad[:, n, :]
                if energy_kind == "linear":
                    deps = 0.5 * (df + np.swapaxes(df, 1, 2))
                    dp = 2 * m.mu[:, None, None] * deps + m.lam[
                        :, None, None
                    ] * np.trace(deps, axis1=1, axis2=2)[:, None, None] * np.eye(2)
                else:
                    dg = df @ ainv
                    dj = np.sum(cof(g) * dg, (1, 2))
                    dpg = (
                        m.mu[:, None, None] * dg
                        + m.lam[:, None, None] * dj[:, None, None] * cof(g)
                        + k[:, None, None] * cof(dg)
                    )
                    dp = dpg @ np.swapaxes(ainv, 1, 2)
                lh[:, :, 2 * n + q] = (
                    m.area[:, None, None] * np.einsum("eab,eib->eia", dp, m.grad)
                ).reshape(-1, 6)
        lh = 0.5 * (lh + lh.swapaxes(1, 2))
        vals = lh[m.ee, m.lr, m.lc]
        H = sp.coo_matrix((vals, (m.rows, m.cols)), shape=(m.nfree, m.nfree)).tocsc()
    B = sp.csc_matrix((m.nfree, 3 * m.muscle.sum()))
    if mixed:
        rr = []
        cc = []
        vv = []
        for loc, eidx in enumerate(np.flatnonzero(m.muscle)):
            for q, da in enumerate(BASES):
                if energy_kind == "linear":
                    dp = 2 * m.mu[eidx] * da + m.lam[eidx] * np.trace(da) * np.eye(2)
                else:
                    dg = f[eidx] @ da
                    c = cof(g[eidx])
                    dj = np.sum(c * dg)
                    dpg = m.mu[eidx] * dg + m.lam[eidx] * dj * c + k[eidx] * cof(dg)
                    dp = dpg @ ainv[eidx].T + pg[eidx] @ da.T
                dl = (m.area[eidx] * np.einsum("ab,ib->ia", dp, m.grad[eidx])).ravel()
                for ld, gd in enumerate(lf[eidx]):
                    if gd >= 0:
                        rr.append(gd)
                        cc.append(3 * loc + q)
                        vv.append(dl[ld])
        B = sp.coo_matrix((vv, (rr, cc)), shape=(m.nfree, 3 * m.muscle.sum())).tocsc()
    return (
        float(m.area @ density),
        r,
        H,
        B,
        np.linalg.det(f),
        detg,
        deta,
        np.linalg.svd(ainv, compute_uv=False)[:, -1],
    )


def solve(m: Mesh, a: np.ndarray, kind: str, initial: np.ndarray, cfg: Config) -> State:
    if kind == "linear":
        E, r, H, *tail = assembly(m, np.zeros(m.nfree), a, kind, True, False)
        try:
            u = np.asarray(spla.spsolve(H, -r))
            failure = None
        except Exception as exc:
            u = initial.copy()
            failure = repr(exc)
        E, r, H, _B, *tail = assembly(m, u, a, kind, True, False)
        return State(u, E, r, H, *tail, 1, failure is None, failure, 0)
    u = initial.copy()
    failure = None
    ls = 0
    converged = False
    it = 0
    for it in range(cfg.forward_max_iterations):
        E, r, H, *_ = assembly(m, u, a, kind, True, False)
        norm = np.linalg.norm(r) / math.sqrt(max(1, m.nfree))
        if not np.isfinite(E) or not np.all(np.isfinite(r)):
            failure = "non-finite energy/residual"
            break
        if norm <= cfg.forward_tolerance:
            converged = True
            break
        scale = max(np.max(np.abs(H.diagonal())), 1e-12)
        direction = None
        # Continue the same Levenberg damping sequence far enough to dominate
        # strongly indefinite artifact states; successful lower-damping solves
        # remain unchanged.
        for power in range(17):
            damp = 0 if power == 0 else scale * 10 ** (power - 12)
            try:
                d = np.asarray(
                    spla.spsolve(H + damp * sp.eye(m.nfree, format="csc"), -r)
                )
            except Exception:
                continue
            if np.all(np.isfinite(d)) and r @ d < 0:
                direction = d
                break
        if direction is None:
            failure = "no descent Newton direction"
            break
        slope = r @ direction
        accepted = False
        for _ in range(30):
            trial = u + (1 if _ == 0 else 0.5**_) * direction
            Et, *__ = assembly(m, trial, a, kind, False, False)
            if np.isfinite(Et) and Et <= E + 1e-4 * (1 if _ == 0 else 0.5**_) * slope:
                u = trial
                accepted = True
                break
        if not accepted:
            failure = "Armijo line search exhausted"
            ls += 1
            break
    E, r, H, _B, *tail = assembly(m, u, a, kind, True, False)
    return State(
        u,
        E,
        r,
        H,
        *tail,
        it + 1,
        converged
        or np.linalg.norm(r) / math.sqrt(max(1, m.nfree)) <= cfg.forward_tolerance,
        failure,
        ls,
    )


def loss(m: Mesh, u: np.ndarray, height: float, name: str):
    d = unpack(m, u)[m.top]
    target = np.c_[
        np.zeros(len(m.top)), height * 4 * m.p[m.top, 0] * (1 - m.p[m.top, 0])
    ]
    err = d - target
    norm = np.linalg.norm(err, axis=1)
    grad = np.zeros(m.nfree)
    n = len(norm)
    if name == "l2":
        value = np.mean(norm**2)
        de = 2 * err / n
    elif name == "l1":
        value = np.mean(norm)
        de = (
            np.divide(
                err, norm[:, None], out=np.zeros_like(err), where=norm[:, None] > 0
            )
            / n
        )
    elif name == "linf":
        index = int(np.argmax(norm))
        value = float(norm[index])
        de = np.zeros_like(err)
        if norm[index] > 0:
            de[index] = err[index] / norm[index]
    else:
        raise ValueError(name)
    for n, v in zip(m.top, de, strict=True):
        for q in range(2):
            k = m.lookup[2 * n + q]
            if k >= 0:
                grad[k] = v[q]
    return float(value), grad, target


def metrics(m: Mesh, u: np.ndarray, a: np.ndarray, h: float):
    p = m.p[m.top]
    top = unpack(m, u)[m.top]
    target = np.c_[np.zeros(len(p)), h * 4 * p[:, 0] * (1 - p[:, 0])]
    err = top - target
    error_norm = np.linalg.norm(err, axis=1)
    uy = top[:, 1]
    dx = np.mean(np.diff(p[:, 0]))
    smooth = ndi.gaussian_filter1d(uy, max(0.02 / dx, 0.5))
    slope = np.gradient(uy, p[:, 0])
    curv = np.gradient(slope, p[:, 0])
    aa = a.reshape(-1, 3)
    jumps = [aa[i] - aa[j] for i, j in m.edges]
    return {
        "top_target_mae": float(np.mean(error_norm)),
        "top_target_rms": float(np.sqrt(np.mean(error_norm**2))),
        "top_target_max": float(np.max(error_norm)),
        "top_error_rms": float(np.sqrt(np.mean((uy - target[:, 1]) ** 2))),
        "top_highpass_rms": float(np.sqrt(np.mean((uy - smooth) ** 2))),
        "top_slope_rms": float(np.sqrt(np.mean(slope * slope))),
        "top_curvature_rms": float(np.sqrt(np.mean(curv * curv))),
        "activation_neighbor_jump_rms": float(np.sqrt(np.mean(np.square(jumps))))
        if jumps
        else 0.0,
        "activation_rms": float(np.sqrt(np.mean(a * a))),
    }


def grid(m: Mesh, a: np.ndarray, s: State, step: int, height: float):
    cells = np.c_[np.full(len(m.tri), 3), m.tri].ravel()
    g = pv.UnstructuredGrid(
        cells,
        np.full(len(m.tri), pv.CellType.TRIANGLE, dtype=np.uint8),
        np.c_[m.p, np.zeros(len(m.p))],
    )
    u = unpack(m, s.u)
    av = activation(m, a)
    target = np.zeros((len(m.p), 3))
    top = np.isclose(m.p[:, 1], 0.1)
    target[top, 1] = height * 4 * m.p[top, 0] * (1 - m.p[top, 0])
    g.point_data["Displacement"] = np.c_[u, np.zeros(len(u))]
    g.point_data["TargetDisplacement"] = target
    for n, v in {
        "MuscleMask": m.muscle.astype(np.uint8),
        "YoungMPa": m.young,
        "DetF": s.det_f,
        "DetG": s.det_g,
        "DetAinv": s.det_a,
        "MinSingularAinv": s.min_sv_a,
        "ActivationXX": av[:, 0],
        "ActivationYY": av[:, 1],
        "ActivationXY": av[:, 2],
    }.items():
        g.cell_data[n] = v
    g.field_data["InverseStep"] = np.array([step])
    return g


def target_grid(m: Mesh, height: float):
    a = np.zeros(3 * m.muscle.sum())
    s = State(
        np.zeros(m.nfree),
        0.0,
        np.zeros(m.nfree),
        sp.csc_matrix((m.nfree, m.nfree)),
        np.ones(len(m.tri)),
        np.ones(len(m.tri)),
        np.ones(len(m.tri)),
        np.ones(len(m.tri)),
        0,
        True,
        None,
        0,
    )
    return grid(m, a, s, 0, height)


def run_case(
    name: str,
    kind: str,
    lname: str,
    resolution: str,
    height: float,
    out: Path,
    cfg: Config,
):
    nx, ny = map(int, resolution.split("x"))
    m = build_mesh(nx, ny)
    case = out / name
    case.mkdir()
    frames = case / "frames"
    frames.mkdir()
    target_grid(m, height).save(case / "target.vtu")
    a = np.zeros(3 * m.muscle.sum())
    moment = np.zeros_like(a)
    variance = np.zeros_like(a)
    u = np.zeros(m.nfree)
    previous_a = a.copy()
    rows = []
    series = []
    best = (math.inf, 0, None, None)
    best_converged = (math.inf, 0, None, None)
    best_orientation_preserving = (math.inf, 0, None, None)
    failure_count = 0
    persisted_row_count = 0

    def write_trace() -> None:
        nonlocal persisted_row_count
        if not rows:
            return
        mode = "w" if persisted_row_count == 0 else "a"
        with (case / "trace.csv").open(mode, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            if persisted_row_count == 0:
                writer.writeheader()
            writer.writerows(rows[persisted_row_count:])
        persisted_row_count = len(rows)

    def persist(*, force_series: bool = False) -> None:
        write_trace()
        if force_series or len(rows) % 50 == 0:
            write_json(
                case / "history.vtu.series",
                {"file-series-version": "1.0", "files": series},
            )

    # Step zero is the undeformed control. Thereafter every accepted Adam update
    # gets one forward/adjoint evaluation and one frame; no patience exit.
    for step in range(cfg.max_steps + 1):
        state = None
        val = math.inf
        grad = np.zeros_like(a)
        evaluation_error = None
        try:
            state = solve(m, a, kind, u, cfg)
            val, du, _ = loss(m, state.u, height, lname)
            if not state.converged:
                raise RuntimeError(
                    f"forward equilibrium did not converge at inverse step {step}: "
                    f"iterations={state.iterations}, failure={state.failure!r}"
                )
            _, _, _, B, *_ = assembly(m, state.u, a, kind, False, True)
            adj = spla.spsolve(state.h, -du)
            grad = np.asarray(B.T @ adj).ravel()
            if not (np.all(np.isfinite(grad)) and np.isfinite(val)):
                raise FloatingPointError("non-finite adjoint gradient")
        except Exception as exc:
            evaluation_error = repr(exc)
            if state is None:
                state = State(
                    u.copy(),
                    math.nan,
                    np.full(m.nfree, np.nan),
                    sp.csc_matrix((m.nfree, m.nfree)),
                    np.full(len(m.tri), np.nan),
                    np.full(len(m.tri), np.nan),
                    np.full(len(m.tri), np.nan),
                    np.full(len(m.tri), np.nan),
                    0,
                    False,
                    evaluation_error,
                    0,
                )
            failure_count += 1
        assert state is not None
        u = state.u
        res = (
            float(np.linalg.norm(state.r) / math.sqrt(max(1, m.nfree)))
            if np.all(np.isfinite(state.r))
            else math.inf
        )
        row = {
            "step": step,
            "optimizer_phase": "adam",
            "optimizer_iteration": step,
            "optimizer_evaluation": step + 1,
            "learning_rate": cfg.learning_rate * cfg.lr_decay**step
            if step < cfg.max_steps
            else None,
            "evaluation_success": int(evaluation_error is None),
            "objective": val,
            "gradient_rms": float(np.linalg.norm(grad) / math.sqrt(max(1, len(grad)))),
            "gradient_inf": float(np.linalg.norm(grad, ord=np.inf)),
            "activation_update_rms": 0.0
            if step == 0
            else float(np.linalg.norm(a - previous_a) / math.sqrt(max(1, len(a)))),
            "forward_converged": int(state.converged),
            "forward_iterations": state.iterations,
            "forward_failure": evaluation_error or state.failure or "",
            "line_search_failures": state.line_search_failures,
            "equilibrium_residual_rms": res,
            "min_det_f": float(np.nanmin(state.det_f)),
            "min_det_g": float(np.nanmin(state.det_g)),
            "min_det_ainv": float(np.nanmin(state.det_a)),
            "min_singular_ainv": float(np.nanmin(state.min_sv_a)),
            **metrics(m, u, a, height),
        }
        rows.append(row)
        f = frames / f"step-{step:04d}.vtu"
        grid(m, a, state, step, height).save(f)
        series.append({"name": str(f.relative_to(case)), "time": float(step)})
        persist()
        if val < best[0]:
            best = (val, step, a.copy(), state)
        if state.converged and val < best_converged[0]:
            best_converged = (val, step, a.copy(), state)
        orientation_preserving = (
            state.converged
            and row["min_det_f"] > 0
            and row["min_det_g"] > 0
            and row["min_det_ainv"] > 0
        )
        if orientation_preserving and val < best_orientation_preserving[0]:
            best_orientation_preserving = (val, step, a.copy(), state)
        cherries.set_step(step)
        cherries.log_metrics(
            {
                f"2d/{name}/objective": val,
                f"2d/{name}/highpass": row["top_highpass_rms"],
            }
        )
        if evaluation_error is not None:
            persist(force_series=True)
            write_json(case / "failure.json", row)
            raise RuntimeError(
                f"inverse evaluation {step} failed; partial trace retained at {case}"
            )
        if step < cfg.max_steps:
            previous_a = a.copy()
            moment = 0.9 * moment + 0.1 * grad
            variance = 0.999 * variance + 0.001 * grad * grad
            lr = cfg.learning_rate * cfg.lr_decay**step
            a = a - lr * (moment / (1 - 0.9 ** (step + 1))) / (
                np.sqrt(variance / (1 - 0.999 ** (step + 1))) + 1e-8
            )
    refinement_cfg = Config(
        forward_tolerance=cfg.refinement_forward_tolerance,
        forward_max_iterations=cfg.forward_max_iterations,
        validate_derivatives=False,
    )
    refinement_seed = u.copy()
    accepted_a, accepted_state = a.copy(), state
    refinement_failure: str | None = None
    refinement_trial_failures = 0
    refinement_evaluations = 0
    refinement_callbacks = 0
    refinement_attempts = []
    refinement_termination = "not_started"
    stalled = 0
    refinement_intra_attempt_stagnation_stops = 0
    refinement_subthreshold_control_progress_count = 0
    attempt_stagnant_accepted = 0
    stopped_for_intra_attempt_stagnation = False
    linf_certificate = {
        "near_active_count": None,
        "unique_max_margin": None,
        "minimum_norm_clarke_subgradient_norm": None,
    }
    cache = None
    refinement_start_objective = math.inf

    def strict_value_gradient(controls: np.ndarray):
        nonlocal linf_certificate
        state_at_controls = solve(m, controls, kind, refinement_seed, refinement_cfg)
        if not state_at_controls.converged:
            raise RuntimeError(
                f"strict refinement forward failed: {state_at_controls.failure}"
            )
        value_at_controls, du, _ = loss(m, state_at_controls.u, height, lname)
        _, _, _, mixed, *_ = assembly(
            m, state_at_controls.u, controls, kind, False, True
        )
        if lname == "linf":
            top = unpack(m, state_at_controls.u)[m.top]
            target = np.c_[
                np.zeros(len(m.top)), height * 4 * m.p[m.top, 0] * (1 - m.p[m.top, 0])
            ]
            error = top - target
            norms = np.linalg.norm(error, axis=1)
            maximum = float(norms.max())
            near = np.flatnonzero(maximum - norms <= 1e-10 * max(1.0, maximum))
            displacement_candidates = []
            for i in near:
                candidate = np.zeros(m.nfree)
                if norms[i] > 0:
                    for q in range(2):
                        dof = m.lookup[2 * m.top[i] + q]
                        if dof >= 0:
                            candidate[dof] = error[i, q] / norms[i]
                displacement_candidates.append(candidate)
            matrix = np.column_stack(
                [
                    np.asarray(
                        mixed.T @ spla.spsolve(state_at_controls.h, -candidate)
                    ).ravel()
                    for candidate in displacement_candidates
                ]
            )
            if len(near) == 1:
                gradient_at_controls = matrix[:, 0]
            else:
                weights = opt.minimize(
                    lambda w: 0.5 * float(np.sum((matrix @ w) ** 2)),
                    np.full(len(near), 1 / len(near)),
                    jac=lambda w: matrix.T @ (matrix @ w),
                    bounds=[(0.0, 1.0)] * len(near),
                    constraints={
                        "type": "eq",
                        "fun": lambda w: w.sum() - 1,
                        "jac": lambda w: np.ones(len(near)),
                    },
                    method="SLSQP",
                )
                if not weights.success:
                    raise RuntimeError(f"Clarke QP failed: {weights.message}")
                gradient_at_controls = matrix @ weights.x
            sorted_norms = np.sort(norms)
            linf_certificate = {
                "near_active_count": len(near),
                "unique_max_margin": float(maximum - sorted_norms[-2])
                if len(norms) > 1
                else math.inf,
                "minimum_norm_clarke_subgradient_norm": float(
                    np.linalg.norm(gradient_at_controls)
                ),
            }
        else:
            gradient_at_controls = np.asarray(
                mixed.T @ spla.spsolve(state_at_controls.h, -du)
            ).ravel()
        if not (
            np.isfinite(value_at_controls) and np.isfinite(gradient_at_controls).all()
        ):
            raise FloatingPointError(
                "non-finite strict refinement objective or gradient"
            )
        return state_at_controls, value_at_controls, gradient_at_controls

    def record_refinement(
        controls,
        state_at_controls,
        value_at_controls,
        gradient_at_controls,
        iteration,
        evaluation,
    ):
        nonlocal \
            accepted_a, \
            accepted_state, \
            best, \
            best_converged, \
            best_orientation_preserving
        prior = rows[-1]
        detf = state_at_controls.det_f
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
                np.linalg.norm(controls - accepted_a) / math.sqrt(max(1, len(controls)))
            )
            if iteration
            else 0.0,
            "forward_converged": 1,
            "forward_iterations": state_at_controls.iterations,
            "forward_failure": state_at_controls.failure or "",
            "line_search_failures": state_at_controls.line_search_failures,
            "equilibrium_residual_rms": float(
                np.linalg.norm(state_at_controls.r) / math.sqrt(max(1, m.nfree))
            ),
            "min_det_f": float(detf.min()),
            "min_det_g": float(state_at_controls.det_g.min()),
            "min_det_ainv": float(state_at_controls.det_a.min()),
            "min_singular_ainv": float(state_at_controls.min_sv_a.min()),
            **metrics(m, state_at_controls.u, controls, height),
        }
        if tuple(row) != tuple(prior):
            raise RuntimeError("Adam/refinement trace schemas differ")
        rows.append(row)
        accepted_a, accepted_state = controls.copy(), state_at_controls
        if lname == "linf":
            # Recompute after accepting so rejected trial diagnostics cannot leak.
            strict_value_gradient(accepted_a)
        f = frames / f"step-{step_at_controls:04d}.vtu"
        grid(m, controls, state_at_controls, step_at_controls, height).save(f)
        series.append(
            {"name": str(f.relative_to(case)), "time": float(step_at_controls)}
        )
        persist()
        if value_at_controls < best[0]:
            best = (
                value_at_controls,
                step_at_controls,
                controls.copy(),
                state_at_controls,
            )
        if value_at_controls < best_converged[0]:
            best_converged = (
                value_at_controls,
                step_at_controls,
                controls.copy(),
                state_at_controls,
            )
        if (
            row["min_det_f"] > 0
            and row["min_det_g"] > 0
            and row["min_det_ainv"] > 0
            and value_at_controls < best_orientation_preserving[0]
        ):
            best_orientation_preserving = (
                value_at_controls,
                step_at_controls,
                controls.copy(),
                state_at_controls,
            )

    try:
        strict_state, strict_value, strict_gradient = strict_value_gradient(a)
        refinement_start_objective = strict_value
        record_refinement(a, strict_state, strict_value, strict_gradient, 0, 0)
        cache = (a.copy(), strict_state, strict_value, strict_gradient)

        def objective(controls):
            nonlocal cache, refinement_evaluations, refinement_trial_failures
            try:
                if cache is None or not np.array_equal(controls, cache[0]):
                    cache = (controls.copy(), *strict_value_gradient(controls))
                refinement_evaluations += 1
                return scale * cache[2], scale * cache[3]
            except Exception:
                refinement_trial_failures += 1
                raise

        def callback(controls):
            nonlocal \
                attempt_stagnant_accepted, \
                refinement_callbacks, \
                refinement_intra_attempt_stagnation_stops, \
                refinement_subthreshold_control_progress_count, \
                stopped_for_intra_attempt_stagnation
            if cache is None or not np.array_equal(controls, cache[0]):
                raise RuntimeError("L-BFGS callback/cache mismatch")
            prior_objective = rows[-1]["objective"]
            refinement_callbacks += 1
            record_refinement(
                controls,
                cache[1],
                cache[2],
                cache[3],
                refinement_callbacks,
                refinement_evaluations,
            )
            physical_improvement = prior_objective - cache[2]
            objective_progress = physical_improvement > max(
                1e-14, abs(prior_objective) * 1e-10
            )
            control_update_threshold = max(
                cfg.refinement_minimum_control_update_rms,
                cfg.refinement_minimum_relative_control_update
                * float(np.linalg.norm(controls) / math.sqrt(max(1, len(controls)))),
            )
            control_progress = (
                rows[-1]["activation_update_rms"] > control_update_threshold
            )
            if objective_progress and not control_progress:
                refinement_subthreshold_control_progress_count += 1
            if objective_progress and control_progress:
                attempt_stagnant_accepted = 0
            else:
                attempt_stagnant_accepted += 1
            if (
                attempt_stagnant_accepted
                >= cfg.refinement_max_stagnant_accepted_iterations
            ):
                stopped_for_intra_attempt_stagnation = True
                refinement_intra_attempt_stagnation_stops += 1
                raise StopIteration

        scale = 1 / max(abs(strict_value), 1e-30)
        stalled = 0
        refinement_termination = "accepted_iteration_budget"
        while refinement_callbacks < cfg.refinement_total_accepted_iterations:
            before_callbacks, before_evaluations = (
                refinement_callbacks,
                refinement_evaluations,
            )
            attempt_stagnant_accepted = 0
            stopped_for_intra_attempt_stagnation = False
            attempt_start_objective = rows[-1]["objective"]
            optimizer = opt.minimize(
                objective,
                accepted_a,
                jac=True,
                method="L-BFGS-B",
                callback=callback,
                options={
                    "maxiter": min(
                        cfg.refinement_max_iterations,
                        cfg.refinement_total_accepted_iterations - refinement_callbacks,
                    ),
                    "maxfun": cfg.refinement_max_function_evaluations,
                    "ftol": 0.0,
                    "gtol": scale * cfg.refinement_optimizer_gradient_inf_tolerance,
                    "maxls": cfg.refinement_max_line_search_steps,
                },
            )
            refinement_attempts.append(
                {
                    "success": bool(optimizer.success),
                    "status": int(optimizer.status),
                    "message": str(optimizer.message),
                    "accepted_iterations": refinement_callbacks - before_callbacks,
                    "function_evaluations": refinement_evaluations - before_evaluations,
                    "objective_scale": scale,
                    "physical_objective": rows[-1]["objective"],
                    "physical_gradient_inf": rows[-1]["gradient_inf"],
                    "physical_gradient_rms": rows[-1]["gradient_rms"],
                    "terminated_by_intra_attempt_stagnation": (
                        stopped_for_intra_attempt_stagnation
                    ),
                }
            )
            accepted_this_attempt = refinement_callbacks - before_callbacks
            evaluations_this_attempt = refinement_evaluations - before_evaluations
            physical_improvement = attempt_start_objective - rows[-1]["objective"]
            meaningful_improvement = physical_improvement > max(
                1e-14, abs(attempt_start_objective) * 1e-10
            )
            refinement_attempts[-1].update(
                {
                    "physical_objective_improvement": physical_improvement,
                    "meaningful_objective_improvement": meaningful_improvement,
                }
            )
            if optimizer.nit != accepted_this_attempt:
                raise RuntimeError("L-BFGS receipt mismatch")
            if optimizer.nfev != evaluations_this_attempt:
                raise RuntimeError("L-BFGS evaluation receipt mismatch")
            if not np.array_equal(np.asarray(optimizer.x), accepted_a):
                raise RuntimeError("L-BFGS result differs from last accepted iterate")
            final_row = rows[-1]
            physical = (
                final_row["equilibrium_residual_rms"]
                <= cfg.refinement_forward_tolerance
                and final_row["gradient_inf"]
                <= cfg.refinement_acceptance_gradient_inf_tolerance
                and final_row["gradient_rms"]
                <= cfg.refinement_acceptance_gradient_rms_tolerance
            )
            if physical:
                refinement_termination = "physical_stationarity"
                break
            if (
                accepted_this_attempt == 0
                or not meaningful_improvement
                or stopped_for_intra_attempt_stagnation
            ):
                stalled += 1
                if stalled >= cfg.refinement_stalled_restart_cap:
                    refinement_termination = "stalled_restart_limit"
                    break
            else:
                stalled = 0
            cache = None
    except Exception as exc:
        refinement_failure = repr(exc)
        refinement_termination = "exception"
        persist(force_series=True)
        write_json(
            case / "refinement-failure.json",
            {
                "failure": refinement_failure,
                "accepted_iterations": refinement_callbacks,
                "trial_failures": refinement_trial_failures,
            },
        )
    if lname == "linf" and refinement_failure is None:
        certificate_state, certificate_value, certificate_gradient = (
            strict_value_gradient(accepted_a)
        )
        if not (
            math.isclose(
                certificate_value,
                rows[-1]["objective"],
                rel_tol=1e-8,
                abs_tol=1e-12,
            )
            and math.isclose(
                float(np.linalg.norm(certificate_gradient, ord=np.inf)),
                rows[-1]["gradient_inf"],
                rel_tol=1e-6,
                abs_tol=1e-10,
            )
            and math.isclose(
                float(
                    np.linalg.norm(certificate_gradient)
                    / math.sqrt(max(1, len(certificate_gradient)))
                ),
                rows[-1]["gradient_rms"],
                rel_tol=1e-6,
                abs_tol=1e-10,
            )
            and certificate_state.converged
        ):
            raise RuntimeError(
                "final L-infinity certificate differs from accepted state"
            )
    a, state, u = accepted_a, accepted_state, accepted_state.u
    _, bstep, ba, bs = best
    _, bcstep, bca, bcs = best_converged
    _, bopstep, bopa, bops = best_orientation_preserving
    if ba is None or bs is None:
        raise RuntimeError(f"{name} produced no finite inverse evaluation")
    if bca is None or bcs is None:
        raise RuntimeError(f"{name} produced no converged forward evaluation")
    if bopa is None or bops is None:
        raise RuntimeError(
            f"{name} produced no orientation-preserving converged evaluation"
        )
    grid(m, ba, bs, bstep, height).save(case / "best.vtu")
    grid(m, bca, bcs, bcstep, height).save(case / "best-converged.vtu")
    grid(m, bopa, bops, bopstep, height).save(case / "best-orientation-preserving.vtu")
    final_step = len(rows) - 1
    grid(m, a, state, final_step, height).save(case / "final.vtu")
    np.savez_compressed(
        case / "best-state.npz", controls=ba, displacement_free=bs.u, step=bstep
    )
    np.savez_compressed(
        case / "best-converged-state.npz",
        controls=bca,
        displacement_free=bcs.u,
        step=bcstep,
    )
    np.savez_compressed(
        case / "best-orientation-preserving-state.npz",
        controls=bopa,
        displacement_free=bops.u,
        step=bopstep,
    )
    np.savez_compressed(
        case / "final-state.npz",
        controls=a,
        displacement_free=state.u,
        step=final_step,
    )
    persist(force_series=True)
    tail = rows[-min(50, len(rows)) :]
    finite_tail = [r["objective"] for r in tail if math.isfinite(r["objective"])]
    tail_fraction = float(np.mean([r["forward_converged"] for r in tail]))
    tail_range = (
        ((max(finite_tail) - min(finite_tail)) / max(abs(min(finite_tail)), 1e-30))
        if finite_tail
        else math.inf
    )
    first_inversion = next(
        (
            r["step"]
            for r in rows
            if r["min_det_f"] <= 0 or r["min_det_g"] <= 0 or r["min_det_ainv"] <= 0
        ),
        None,
    )
    tail_gate = bool(tail_fraction == 1 and tail_range <= 0.01)
    accepted_refinement = rows[cfg.max_steps + 1 :]
    objective_increases = sum(
        later["objective"]
        > earlier["objective"] + max(1e-14, abs(refinement_start_objective) * 1e-10)
        for earlier, later in zip(
            accepted_refinement, accepted_refinement[1:], strict=False
        )
    )
    l1_differentiable = lname != "l1" or bool(
        np.all(
            np.linalg.norm(
                unpack(m, state.u)[m.top]
                - np.c_[
                    np.zeros(len(m.top)),
                    height * 4 * m.p[m.top, 0] * (1 - m.p[m.top, 0]),
                ],
                axis=1,
            )
            > 1e-12
        )
    )
    stationarity_gate = bool(
        l1_differentiable
        and refinement_failure is None
        and refinement_trial_failures == 0
        and objective_increases == 0
        and rows[-1]["equilibrium_residual_rms"] <= cfg.refinement_forward_tolerance
        and rows[-1]["gradient_inf"] <= cfg.refinement_acceptance_gradient_inf_tolerance
        and rows[-1]["gradient_rms"] <= cfg.refinement_acceptance_gradient_rms_tolerance
    )
    summary = {
        "name": name,
        "energy": kind,
        "loss": lname,
        "resolution": [nx, ny],
        "height": height,
        "n_triangles": len(m.tri),
        "n_muscle_triangles": int(m.muscle.sum()),
        "activation_dofs": len(a),
        "evaluations": len(rows),
        "best_step": bstep,
        "best": rows[bstep],
        "best_converged_step": bcstep,
        "best_converged": rows[bcstep],
        "best_orientation_preserving_step": bopstep,
        "best_orientation_preserving": rows[bopstep],
        "final": rows[-1],
        "forward_failure_count": sum(not r["forward_converged"] for r in rows),
        "inverse_evaluation_failure_count": failure_count,
        "all_forwards_converged": all(r["forward_converged"] for r in rows),
        "first_inversion_step": first_inversion,
        "minimum_det_f": min(r["min_det_f"] for r in rows),
        "minimum_det_g": min(r["min_det_g"] for r in rows),
        "minimum_det_ainv": min(r["min_det_ainv"] for r in rows),
        "tail_convergence": {
            "window_steps": len(tail),
            "forward_converged_fraction": tail_fraction,
            "objective_first": tail[0]["objective"],
            "objective_last": tail[-1]["objective"],
            "objective_change": tail[-1]["objective"] - tail[0]["objective"],
            "objective_relative_range": tail_range,
            "gradient_rms_last": tail[-1]["gradient_rms"],
            "equilibrium_residual_rms_max": max(
                r["equilibrium_residual_rms"] for r in tail
            ),
            "inverse_converged_1pct_tail_gate": tail_gate,
        },
        "refinement": {
            "method": "objective-normalized, automatically restarted L-BFGS-B used without bounds",
            "fixed_forward_seed": True,
            "forward_tolerance": cfg.refinement_forward_tolerance,
            "objective_scale": 1 / max(abs(refinement_start_objective), 1e-30),
            "termination": refinement_termination,
            "accepted_iterations": refinement_callbacks,
            "total_accepted_iteration_budget": cfg.refinement_total_accepted_iterations,
            "function_evaluations": refinement_evaluations,
            "stalled_restarts": stalled,
            "stalled_restart_cap": cfg.refinement_stalled_restart_cap,
            "max_stagnant_accepted_iterations": (
                cfg.refinement_max_stagnant_accepted_iterations
            ),
            "minimum_control_update_rms": (cfg.refinement_minimum_control_update_rms),
            "minimum_relative_control_update": (
                cfg.refinement_minimum_relative_control_update
            ),
            "intra_attempt_stagnation_stops": (
                refinement_intra_attempt_stagnation_stops
            ),
            "accepted_objective_progress_with_subthreshold_control_update": (
                refinement_subthreshold_control_progress_count
            ),
            "trial_forward_failures": refinement_trial_failures,
            "failure": refinement_failure,
            "attempts": refinement_attempts,
            "accepted_objective_increase_count": objective_increases,
            "l1_differentiable_at_final": l1_differentiable,
            "linf_clarke": linf_certificate if lname == "linf" else None,
        },
        "stationarity": {
            "gradient_inf_tolerance": cfg.refinement_acceptance_gradient_inf_tolerance,
            "gradient_rms_tolerance": cfg.refinement_acceptance_gradient_rms_tolerance,
            "passed": stationarity_gate,
        },
        "artifacts": {
            "series": "history.vtu.series",
            "target": "target.vtu",
            "best": "best.vtu",
            "best_converged": "best-converged.vtu",
            "best_orientation_preserving": "best-orientation-preserving.vtu",
            "final": "final.vtu",
        },
    }
    write_json(case / "summary.json", summary)
    if cfg.require_inverse_convergence and not cfg.smoke and not stationarity_gate:
        raise RuntimeError(
            f"{name} completed refinement but failed the physical stationarity gate"
        )
    return summary


def parse_cases(text: str):
    out = []
    for part in text.split(","):
        n, k, l, r, h = part.split(":")
        if k not in {"linear", "stable"} or l not in {"l1", "l2", "linf"}:
            raise ValueError(f"invalid case {part}")
        nx, ny = map(int, r.split("x"))
        if nx < 50 or ny < 5:
            raise ValueError("resolutions must be meaningful (at least 50x5)")
        out.append((n, k, l, r, float(h)))
    if len({x[0] for x in out}) != len(out):
        raise ValueError("case names must be unique")
    return out


def derivative_check(energy_kind: str) -> dict[str, Any]:
    # A small assembled implicit-adjoint check against central differences.
    m = build_mesh(50, 5)
    cfg = Config(
        max_steps=2,
        validate_derivatives=False,
        forward_tolerance=1e-10,
        forward_max_iterations=120,
    )
    a = 0.001 * np.sin(np.arange(3 * m.muscle.sum()))
    s = solve(m, a, energy_kind, np.zeros(m.nfree), cfg)
    v, du, _ = loss(m, s.u, 0.05, "l2")
    _, _, _, B, *_ = assembly(m, s.u, a, energy_kind, False, True)
    g = np.asarray(B.T @ spla.spsolve(s.h, -du)).ravel()
    i = int(np.argmax(abs(g)))
    eps = 1e-5
    vals = []
    for sign in (-1, 1):
        aa = a.copy()
        aa[i] += sign * eps
        ss = solve(m, aa, energy_kind, s.u, cfg)
        vals.append(loss(m, ss.u, 0.05, "l2")[0])
    fd = (vals[1] - vals[0]) / (2 * eps)
    err = abs(fd - g[i]) / max(abs(fd), abs(g[i]), 1e-14)
    return {
        "energy": energy_kind,
        "relative_error": float(err),
        "forward_converged": bool(s.converged),
        "passed": bool(s.converged and err < 1e-4),
    }


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(f"refusing nonempty {cfg.output_dir}")
    cases = parse_cases(cfg.cases)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    check = (
        {kind: derivative_check(kind) for kind in ("linear", "stable")}
        if cfg.validate_derivatives
        else {"skipped": True, "passed": True}
    )
    if cfg.validate_derivatives and not all(item["passed"] for item in check.values()):
        raise RuntimeError(f"derivative check failed: {check}")
    if cfg.smoke:
        cases = cases[:1]
        cfg.max_steps = min(cfg.max_steps, 3)
        cfg.refinement_max_iterations = min(cfg.refinement_max_iterations, 3)
        cfg.refinement_total_accepted_iterations = min(
            cfg.refinement_total_accepted_iterations, 3
        )
    results = [run_case(*c, cfg.output_dir, cfg) for c in cases]
    write_json(
        cfg.output_dir / "summary.json",
        {
            "design": "2d-unreachable-pork-controlled-OFAT",
            "geometry": {
                "domain": [1, 0.1],
                "muscle_band_centroid_y": [0.04, 0.06],
                "fixed": "bottom, left, right; both components",
                "top_and_inner": "free",
                "target": "ux=0; uy=h*4*x*(1-x)",
            },
            "materials": {
                "fat_E_MPa": E_FAT,
                "muscle_E_MPa": E_MUSCLE,
                "poisson": NU,
                "skin_energy": None,
            },
            "activation": "exactly 3 raw unbounded symmetric Ainv-I DoFs per muscle triangle; no regularization or inversion constraint",
            "inverse": {
                "adam_updates_per_case": cfg.max_steps,
                "adam_evaluations_per_case": cfg.max_steps + 1,
                "early_stopping": "physical stationarity only; otherwise accepted-step budget or explicit stalled-restart/failure receipt",
                "optimizer": "fixed Adam exploration followed by objective-normalized, automatically restarted unbounded L-BFGS-B",
                "forward_linear": "exact sparse solve",
                "forward_stable": "warm-start damped Newton with Armijo; any nonconverged equilibrium hard-fails the case",
                "completion_gate": "strict forward residual, nonincreasing accepted refinement objective, no failed trial, and physical gradient infinity/RMS thresholds",
                "stationarity_pass_cases": [
                    result["name"]
                    for result in results
                    if result["stationarity"]["passed"]
                ],
                "stationarity_fail_cases": [
                    result["name"]
                    for result in results
                    if not result["stationarity"]["passed"]
                ],
                "checkpoint_selection_is_posthoc_only": True,
            },
            "derivative_check": check,
            "cases": results,
            "elapsed_seconds": time.perf_counter() - started,
            "paraview": "open each case/history.vtu.series; source time is the integer optimization step; encode at 30 fps",
        },
    )


if __name__ == "__main__":
    cherries.main(main)
