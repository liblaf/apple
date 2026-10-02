# ruff: noqa: PLR0915
"""Bounded material sensitivities for the no-skin c50 smile prescription."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import ModuleType

import activation_models as am
import numpy as np
import pydantic_settings as ps
import torch
from experiment_profile import ProfileCometNoCommit
from face_physics import ROOT, FacePhysics, ForwardConvergenceError, configure

from liblaf import cherries

LOG = logging.getLogger(__name__)
VARIANTS = {
    "selected-muscle-lame-x10": ("muscle", 10.0, "selected"),
    "fat-lame-x0.1": ("fat", 0.1, "all"),
    "aponeurosis-lame-x0.1": ("aponeurosis", 0.1, "all"),
}
SMILE_IDS = (57, 58, 63, 64, 142, 143, 218, 219, 283, 284)


def load_manual_module() -> ModuleType:
    path = Path(__file__).with_name("10-manual-activation.py")
    spec = importlib.util.spec_from_file_location("manual_activation", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MANUAL = load_manual_module()


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = (
        Path(__file__).resolve().parents[2]
        / "face-activation-materials/data/10-fixture"
    )
    baseline: Path = (
        Path(__file__).resolve().parents[1]
        / "data/10-manual-activation/skin-0.00/smile-elevators/c50/state.npz"
    )
    baseline_diagnostics: Path = (
        Path(__file__).resolve().parents[1]
        / "data/10-manual-activation/skin-0.00/smile-elevators/c50/diagnostics.json"
    )
    output_dir: Path = cherries.output("11-manual-sensitivities", mkdir=True)
    fat_factor: float = 1.0
    muscle_factor: float = 0.8
    soft_nu: float = 0.46
    skin_nu: float = 0.46
    fat_model: str = "stable"
    fat_nu: float = 0.49
    forward_rtol: float = 1e-5
    forward_atol: float = 1e-12
    target_name: str = "Smile"


def array_receipt(value: torch.Tensor) -> dict[str, object]:
    array = value.detach().cpu().numpy()
    assert np.isfinite(array).all()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        "min": float(array.min()),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def material_receipt(p: FacePhysics) -> dict[str, object]:
    return {
        name: {field: array_receipt(materials[field]) for field in ("lmbda", "mu")}
        for name, materials in p.materials.items()
    }


def apply_variant(
    p: FacePhysics,
    variant: str,
    selected_cells: np.ndarray,
) -> dict[str, object]:
    potential, factor, scope = VARIANTS[variant]
    before = material_receipt(p)
    if scope == "selected":
        indices = torch.as_tensor(selected_cells)
    else:
        indices = torch.arange(p.mesh.n_cells)
    changed = {}
    with torch.no_grad():
        for field in ("lmbda", "mu"):
            old = p.materials[potential][field]
            new = old.clone()
            new[indices] *= factor
            p.materials[potential][field] = new
            changed[field] = {
                "before_selected": array_receipt(old[indices]),
                "after_selected": array_receipt(new[indices]),
            }
    after = material_receipt(p)
    for field in ("lmbda", "mu"):
        assert before[potential][field]["sha256"] != after[potential][field]["sha256"]
    return {
        "variant": variant,
        "potential": potential,
        "factor": factor,
        "scope": scope,
        "changed_cell_count": len(indices),
        "interpretation": (
            "Scaling selected muscle lambda and mu changes both selected passive stiffness "
            "and induced eigenstrain actuation; this is not a pure active-stress scaling."
            if potential == "muscle"
            else "Scaling both lambda and mu preserves Poisson ratio while scaling Young modulus."
        ),
        "before": before,
        "after": after,
        "changed_arrays": changed,
    }


def write_json(path: Path, data: object) -> None:
    path.write_text(
        json.dumps(
            data,
            indent=2,
            allow_nan=False,
            default=lambda value: (
                value.item() if isinstance(value, np.generic) else str(value)
            ),
        )
        + "\n"
    )


def static_preflight(cfg: Config) -> dict[str, object]:
    base = np.load(cfg.baseline)
    q = np.asarray(base["q"])
    u = np.asarray(base["u"])
    ainv = np.asarray(base["Ainv"])
    diagnostics = json.loads(cfg.baseline_diagnostics.read_text())
    fixture = MANUAL.static_preflight(cfg.fixture)
    expected_a = -math.log(0.5)
    nonzero = np.flatnonzero(q)
    assert q.shape == (fixture["active_tets"],)
    assert u.shape == (fixture["n_vertices"], 3)
    assert ainv.shape == (fixture["active_tets"], 3, 3)
    assert len(nonzero) == fixture["patterns"]["smile-elevators"]["active_cell_count"]
    assert np.all(q[nonzero] == expected_a)
    assert np.all(q[np.setdiff1d(np.arange(len(q)), nonzero)] == 0.0)
    q_hash = hashlib.sha256(q.tobytes()).hexdigest()
    assert q_hash == diagnostics["activation_inverse"]["control_sha256"]
    assert diagnostics["skin_factor"] == 0.0
    assert diagnostics["contraction"] == 0.5
    return {
        "fixture": fixture,
        "baseline_control_sha256": q_hash,
        "baseline_state_sha256": hashlib.sha256(cfg.baseline.read_bytes()).hexdigest(),
        "baseline_diagnostics_sha256": hashlib.sha256(
            cfg.baseline_diagnostics.read_bytes()
        ).hexdigest(),
        "nonzero_control_count": len(nonzero),
        "fiber_amplitude_a": expected_a,
        "prescribed_natural_contraction": 0.5,
        "baseline_skin_factor": 0.0,
        "variant_definitions": VARIANTS,
    }


def main(cfg: Config) -> None:
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Choose an empty output directory"
    assert cfg.forward_rtol == 1e-5
    assert cfg.forward_atol == 1e-12
    assert cfg.fat_nu == 0.49
    preflight = static_preflight(cfg)
    write_json(out / "preflight.json", preflight)
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    sources_dir = out / "sources"
    sources_dir.mkdir()
    source_hashes = {}
    for path in sorted(Path(__file__).parent.glob("*.py")):
        shutil.copy2(path, sources_dir / path.name)
        source_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        out / "provenance.json",
        {
            "sources": source_hashes,
            "inputs": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (
                    cfg.baseline,
                    cfg.baseline_diagnostics,
                    *(
                        cfg.fixture / name
                        for name in ("volume.vtu", "skin.vtp", "summary.json")
                    ),
                )
            },
            "git_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
        },
    )

    configure()
    baseline = np.load(cfg.baseline)
    q = np.asarray(baseline["q"])
    baseline_u = np.asarray(baseline["u"])
    q_hash = hashlib.sha256(q.tobytes()).hexdigest()
    start = time.perf_counter()
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for variant in VARIANTS:
        p = FacePhysics(
            cfg.fixture,
            skin_factor=0.0,
            fat_factor=cfg.fat_factor,
            muscle_factor=cfg.muscle_factor,
            rtol=cfg.forward_rtol,
            atol=cfg.forward_atol,
            soft_nu=cfg.soft_nu,
            skin_nu=cfg.skin_nu,
            target_name=cfg.target_name,
            fat_model=cfg.fat_model,
            fat_nu=cfg.fat_nu,
        )
        active_muscle_id = np.asarray(p.mesh.cell_data["MuscleId"], dtype=int)[p.ids]
        selected_local = np.flatnonzero(np.isin(active_muscle_id, SMILE_IDS))
        selected_cells = p.ids[selected_local]
        assert np.array_equal(selected_local, np.flatnonzero(q))
        materials = apply_variant(p, variant, selected_cells)
        q_t = torch.as_tensor(q[:, None])
        ainv_t, _ = am.matrices(q_t, "F", p.fibers)
        ainv = ainv_t.detach().cpu().numpy()
        assert q_hash == preflight["baseline_control_sha256"]
        assert np.array_equal(ainv, baseline["Ainv"])
        case_path = out / variant
        LOG.info(
            "Solving %s from the exact baseline geometry and c50 controls", variant
        )
        try:
            u = p.solve(am.packed(ainv_t), baseline_u).detach().cpu().numpy()
        except ForwardConvergenceError as error:
            failure = {
                "variant": variant,
                "error": str(error),
                "forward": p.last_forward,
                "materials": materials,
                "control_sha256": q_hash,
            }
            case_path.mkdir(parents=True, exist_ok=False)
            write_json(case_path / "failure.json", failure)
            failures.append(failure)
            rows.append({"variant": variant, "status": "failed"})
            write_json(out / "failures.json", failures)
            LOG.exception("Variant %s failed; continuing independent variants", variant)
            continue
        diagnostics, arrays = MANUAL.case_diagnostics(p, u, ainv, selected_local)
        diagnostics.update(
            {
                "variant": variant,
                "status": "success",
                "skin_factor": 0.0,
                "pattern": "smile-elevators",
                "contraction": 0.5,
                "fiber_amplitude_a": -math.log(0.5),
                "control_sha256": q_hash,
                "baseline_seed_sha256": hashlib.sha256(
                    baseline_u.tobytes()
                ).hexdigest(),
                "material_override": materials,
                "forward": p.last_forward,
                "interpretation": "A one-at-a-time diagnostic material sensitivity, not a calibrated recommendation.",
            }
        )
        MANUAL.save_case(
            p,
            case_path,
            u,
            q,
            ainv,
            selected_local,
            diagnostics,
            arrays,
        )
        physical = diagnostics["physical_deformation"]
        surface = diagnostics["surface_motion"]
        muscle = diagnostics["selected_muscle_motion"]
        row = {
            "variant": variant,
            "status": "success",
            "control_sha256": q_hash,
            "forward_steps": p.last_forward["steps"],
            "forward_grad_norm": p.last_forward["grad_norm"],
            "detF_min_all": physical["detF_all"]["min"],
            "inverted_tets_all": physical["inverted_tets_all"],
            "fiber_stretch_F_weighted_mean": physical[
                "fiber_stretch_F_selected_fraction_volume_weighted_mean"
            ],
            "muscle_centroid_rms_mm": muscle["centroid_displacement_rms_mm"],
            "surface_rms_mm": surface["weighted_rms_mm"],
            "smile_projection": surface["smile_target_projection_amplitude"],
            "wall_s": time.perf_counter() - start,
        }
        rows.append(row)
        with (out / "trace.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=sorted({key for item in rows for key in item}),
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        cherries.log_metrics(
            {
                "sensitivity/surface_rms_mm": row["surface_rms_mm"],
                "sensitivity/smile_projection": row["smile_projection"],
                "sensitivity/detF_min": row["detF_min_all"],
                "sensitivity/fiber_stretch_weighted_mean": row[
                    "fiber_stretch_F_weighted_mean"
                ],
            },
            step=len(rows),
        )
        LOG.info("Completed %s: %s", variant, row)
        del p
        torch.cuda.empty_cache()

    summary = {
        "status": "completed" if not failures else "completed_with_failures",
        "control_sha256": q_hash,
        "exact_baseline_control_replay": True,
        "baseline": {
            "path": str(cfg.baseline),
            "seed": "saved no-skin c50 smile equilibrium",
            "comparison_metrics": json.loads(cfg.baseline_diagnostics.read_text()),
        },
        "solver": {
            "max_steps": 10000,
            "rtol": cfg.forward_rtol,
            "atol": cfg.forward_atol,
            "geometry_rejection": False,
        },
        "rows": rows,
        "failure_count": len(failures),
        "wall_s": time.perf_counter() - start,
    }
    write_json(out / "summary.json", summary)
    LOG.info("Completed material sensitivity sweep: %s", summary["status"])


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
