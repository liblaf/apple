"""Matched forward replays after conservative within-muscle field diffusion."""

# ruff: noqa: EM101, PLR0915, PT018, TRY003

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import scipy.sparse as sp
from experiment_profile import ProfileCometNoCommit
from historical_adam_physics import FacePhysics, configure
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
LOG = logging.getLogger(__name__)
SPEC = importlib.util.spec_from_file_location(
    "historical_base_for_diffusion", HERE / "src/30-run-historical-adam.py"
)
assert SPEC is not None and SPEC.loader is not None
BASE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = BASE
SPEC.loader.exec_module(BASE)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = HERE / "data/12-historical-fixture"
    source_checkpoint: Path = (
        HERE / "data/38-historical-adam-raw6-continuation/final.npz"
    )
    output_dir: Path = cherries.output("80-forward-field-diffusion", mkdir=True)
    smooth_length: float = 0.005


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": BASE.sha256(path),
    }


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes()).hexdigest()


class Diffusion:
    """Solve (M + tau L) q = M q0 for all six packed tensor components."""

    def __init__(self, graph: dict, q0: np.ndarray, length: float) -> None:
        self.graph, self.q0, self.length = graph, q0, length
        i, j, w, mass = (graph[k] for k in ("i", "j", "weight", "volume"))
        assert np.all(mass > 0) and np.all(w > 0)
        self.mass = mass
        self.adjacency = sp.coo_matrix(
            (np.r_[w, w], (np.r_[i, j], np.r_[j, i])), shape=(len(q0), len(q0))
        ).tocsr()
        self.degree = np.asarray(self.adjacency.sum(axis=1)).ravel()
        self.laplacian = sp.diags(self.degree) - self.adjacency
        self.n_components, self.components = connected_components(self.adjacency)
        self.component_mass = np.bincount(self.components, weights=mass)
        self.mean0 = self.means(q0)
        self.r0 = BASE.smoothness_numpy(q0, graph, length)
        self.trials = []

    def means(self, q: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                np.bincount(self.components, weights=self.mass * q[:, k])
                / self.component_mass
                for k in range(6)
            ],
            axis=1,
        )

    def solve(self, tau: float) -> tuple[np.ndarray, dict]:
        matrix = sp.diags(self.mass) + tau * self.laplacian
        preconditioner = sp.diags(1.0 / (self.mass + tau * self.degree))
        q = np.empty_like(self.q0)
        residuals = []
        for k in range(6):
            rhs = self.mass * self.q0[:, k]
            q[:, k], info = cg(
                matrix,
                rhs,
                x0=self.q0[:, k],
                M=preconditioner,
                rtol=1e-11,
                atol=0.0,
                maxiter=3000,
            )
            assert info == 0, (tau, k, info)
            residuals.append(
                float(np.linalg.norm(matrix @ q[:, k] - rhs) / np.linalg.norm(rhs))
            )
        mean_error = float(np.max(np.abs(self.means(q) - self.mean0)))
        # This only removes iterative-solve roundoff in the conserved nullspace.
        q += (self.mean0 - self.means(q))[self.components]
        final_mean_error = float(np.max(np.abs(self.means(q) - self.mean0)))
        assert mean_error < 1e-8 and final_mean_error < 1e-11
        ratio = BASE.smoothness_numpy(q, self.graph, self.length) / self.r0
        receipt = {
            "tau_m2": tau,
            "roughness_ratio": ratio,
            "cg_relative_residuals": residuals,
            "component_mean_error_before_roundoff_correction": mean_error,
            "component_mean_error_after_roundoff_correction": final_mean_error,
        }
        self.trials.append(receipt)
        LOG.info("Field diffusion tau=%.6g, R/R0=%.6f", tau, ratio)
        return q, receipt

    def target(self, ratio: float) -> tuple[np.ndarray, dict]:
        lower, upper = 0.0, 1e-9
        q, receipt = self.solve(upper)
        while receipt["roughness_ratio"] > ratio:
            lower, upper = upper, upper * 4.0
            q, receipt = self.solve(upper)
        for _ in range(35):
            if abs(receipt["roughness_ratio"] - ratio) < 1e-4:
                return q, {**receipt, "requested_roughness_ratio": ratio}
            if receipt["roughness_ratio"] > ratio:
                lower = receipt["tau_m2"]
            else:
                upper = receipt["tau_m2"]
            q, receipt = self.solve((lower + upper) / 2.0)
        raise RuntimeError("field roughness target was not reached")


def cofactor(f: np.ndarray) -> np.ndarray:
    return np.stack(
        (
            np.cross(f[:, :, 1], f[:, :, 2]),
            np.cross(f[:, :, 2], f[:, :, 0]),
            np.cross(f[:, :, 0], f[:, :, 1]),
        ),
        axis=2,
    )


def stress_metrics(p: FacePhysics, q: np.ndarray, u: np.ndarray) -> dict:
    x = p.points + u
    tets = p.tets[p.ids]
    ds = np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1))
    f = ds @ p.dm_inv[p.ids]
    a = BASE.activation_matrix(q)
    mu = p.material_spec["muscle_mu_code_MPa"]
    lam = p.material_spec["muscle_lambda_code_MPa"]
    g = f @ a

    def piola(k: np.ndarray) -> np.ndarray:
        return mu * k + (-mu + lam * (np.linalg.det(k) - 1))[:, None, None] * cofactor(
            k
        )

    active = piola(g) @ a.transpose(0, 2, 1) - piola(f)
    weight = p.volumes / p.volumes.sum()
    return {
        "definition": "P(F,Ainv)-P(F,I), pure-muscle constituent weighted by rest volume times MuscleFraction",
        "active_piola_rms_MPa": float(
            np.sqrt(np.sum(weight[:, None, None] * active**2))
        ),
        "mean_active_piola_MPa": np.einsum("i,ijk->jk", weight, active).tolist(),
        "field_offset_frobenius_rms": float(
            np.sqrt(np.sum(weight[:, None, None] * (a - np.eye(3)) ** 2))
        ),
    }


def surface_metrics(p: FacePhysics, u: np.ndarray) -> dict:
    pred, target, w = u[p.top], p.target[p.top], p.weights
    return {
        "area_fit_rms_mm": float(
            1000 * np.sqrt(np.sum(w[:, None] * (pred - target) ** 2))
        ),
        "area_motion_rms_mm": float(1000 * np.sqrt(np.sum(w[:, None] * pred**2))),
        "area_target_projection": float(
            np.sum(w[:, None] * pred * target) / np.sum(w[:, None] * target**2)
        ),
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Choose an empty output directory"
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    sources = out / "sources"
    sources.mkdir()
    paths = [
        Path(__file__),
        HERE / "src/30-run-historical-adam.py",
        HERE / "src/historical_adam_physics.py",
        HERE / "src/experiment_profile.py",
    ]
    paths += sorted((BASE.REPO / "src/liblaf/apple").rglob("*.py"))
    for path in paths:
        relative = path.relative_to(BASE.REPO)
        copied = sources / relative
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, copied)
    inputs = [
        cfg.source_checkpoint,
        cfg.source_checkpoint.parent / "summary.json",
        cfg.source_checkpoint.parent / "provenance.json",
        *(cfg.fixture / name for name in ("volume.vtu", "skin.vtp", "summary.json")),
    ]
    provenance = {
        "inputs": [record(path) for path in inputs],
        "sources": [record(path) for path in paths],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "command": sys.argv,
    }
    write_json(out / "provenance.json", provenance)
    source_provenance = json.loads(
        (cfg.source_checkpoint.parent / "provenance.json").read_text()
    )
    for key, name in (
        ("fixture_volume", "volume.vtu"),
        ("fixture_skin", "skin.vtp"),
        ("fixture_summary", "summary.json"),
    ):
        assert (
            BASE.sha256(cfg.fixture / name)
            == source_provenance["frozen_inputs"][key]["sha256"]
        )
    with np.load(cfg.source_checkpoint) as saved:
        q0, seed = saved["q"].copy(), saved["u"].copy()
        assert bool(saved["solver_valid"])
        assert np.array_equal(saved["Ainv"], BASE.activation_matrix(q0))
    assert q0.shape == (288235, 6) and seed.shape == (228660, 3)
    mesh = pv.read(cfg.fixture / "volume.vtu")
    graph = BASE.graph_arrays(mesh)
    diffuser = Diffusion(graph, q0, cfg.smooth_length)
    fields = [("baseline", q0, {"tau_m2": 0.0, "roughness_ratio": 1.0})]
    for name, ratio in (("mild", 0.9), ("strong", 0.5)):
        q, receipt = diffuser.target(ratio)
        fields.append((name, q, receipt))
    preflight = {
        "status": "field_preparation_complete",
        "active_tetrahedra": len(q0),
        "scalar_controls": q0.size,
        "same_muscle_edges": len(graph["i"]),
        "connected_components": int(diffuser.n_components),
        "source_roughness": diffuser.r0,
        "trials": diffuser.trials,
        "seed_sha256": array_hash(seed),
        "selected": {name: rec for name, _, rec in fields},
    }
    write_json(out / "field-preparation.json", preflight)
    configure()
    import torch

    p = FacePhysics(cfg.fixture)
    assert np.array_equal(p.ids, graph["active"])
    assert np.array_equal(p.graph[0], graph["i"]) and np.array_equal(
        p.graph[1], graph["j"]
    )
    assert np.allclose(p.volumes, graph["volume"], rtol=1e-12, atol=0.0)
    cases = []
    for step, (name, q, field_receipt) in enumerate(fields):
        cherries.set_step(step)
        case_dir = out / name
        case_dir.mkdir()
        start = time.perf_counter()
        seed_stress = stress_metrics(p, q, seed)
        u = p.solve(torch.as_tensor(q), seed.copy()).detach().cpu().numpy().copy()
        forward = p.last_forward
        ainv = BASE.activation_matrix(q)
        np.savez_compressed(
            case_dir / "state.npz",
            q=q,
            u=u,
            Ainv=ainv,
            forward_success=np.asarray(forward["success"]),
            solver_valid=np.asarray(forward["success"]),
        )
        p.save_mesh(case_dir / "final.vtu", u, ainv)
        det = p.detf(u)
        eig = np.linalg.eigvalsh(ainv)
        metrics = surface_metrics(p, u)
        row = {
            "id": name,
            "status": "equilibrium_valid"
            if forward["success"]
            else "forward_not_converged",
            "forward": forward,
            "field": field_receipt,
            "roughness": BASE.smoothness_numpy(q, graph, cfg.smooth_length),
            **metrics,
            "stress_at_common_seed": seed_stress,
            "stress_at_equilibrium": stress_metrics(p, q, u),
            "physical_detF_min": float(det.min()),
            "inverted_tetrahedra": int((det <= 0).sum()),
            "non_spd_activation_tetrahedra": int((eig[:, 0] <= 0).sum()),
            "seed_displacement_delta_area_rms_mm": float(
                1000
                * np.sqrt(np.sum(p.weights[:, None] * (u[p.top] - seed[p.top]) ** 2))
            ),
            "q_sha256": array_hash(q),
            "seed_sha256": array_hash(seed),
            "wall_s": time.perf_counter() - start,
            "state": record(case_dir / "state.npz"),
            "mesh": record(case_dir / "final.vtu"),
        }
        write_json(case_dir / "diagnostics.json", row)
        cases.append(row)
        cherries.log_metrics({name: metrics})
        LOG.info(
            "%s: valid=%s, fit %.6f mm, motion %.6f mm, R %.6f",
            name,
            forward["success"],
            metrics["area_fit_rms_mm"],
            metrics["area_motion_rms_mm"],
            row["roughness"],
        )
    summary = {
        "schema_version": 1,
        "status": "completed"
        if all(c["forward"]["success"] for c in cases)
        else "completed_with_invalid_forward",
        "scope": "forward-only conservative activation diffusion; no output geometry smoothing",
        "geometry_rejection_enabled": False,
        "all_cases_same_seed": True,
        "materials": p.material_spec,
        "field_preparation": preflight,
        "cases": cases,
        "provenance": provenance,
    }
    write_json(out / "summary.json", summary)
    for path in (
        out / "summary.json",
        out / "field-preparation.json",
        out / "provenance.json",
    ):
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
