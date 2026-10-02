"""Forward-only skin-transmission replay for saved baseline and strong fields."""

# ruff: noqa: PLR0915

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
UPSTREAM_PHYSICS = HERE / "src/historical_adam_physics.py"
LOCAL_PHYSICS = HERE / "src/85-historical-adam-skin-physics.py"
FIELD_RUN = HERE / "data/80-forward-field-diffusion"
SOURCE_CHECKPOINT = HERE / "data/38-historical-adam-raw6-continuation/final.npz"
SKIN_FACTOR = 0.12


def load_module(name: str, path: Path):
    """Load a numbered experiment-local source file without changing imports."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SKIN = load_module("historical_adam_skin_transmission", LOCAL_PHYSICS)
FIELD = load_module(
    "field_diffusion_helpers", HERE / "src/80-forward-field-diffusion.py"
)


class Config(cherries.BaseConfig):
    """Fixed input contract for the approved skin-transmission branch."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    fixture: Path = HERE / "data/12-historical-fixture"
    source_checkpoint: Path = SOURCE_CHECKPOINT
    field_run: Path = FIELD_RUN
    output_dir: Path = cherries.output("85-forward-field-skin-transmission", mkdir=True)


def sha256(path: Path) -> str:
    """Hash an exact file used by the forward replay."""
    return FIELD.BASE.sha256(path)


def record(path: Path) -> dict[str, Any]:
    """Capture an immutable input or output reference."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: object) -> None:
    """Write finite JSON evidence."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_npz(path: Path) -> dict[str, np.ndarray]:
    """Copy checkpoint fields outside the NPZ context manager."""
    with np.load(path) as archive:
        return {key: archive[key].copy() for key in archive.files}


def assert_one_guard_deletion() -> dict[str, str]:
    """Prove the local physics copy differs only by its nonzero-skin rejection."""
    deleted = (
        "        if skin_factor != 0.0:\n"
        '            raise ValueError("matched historical comparison requires zero skin energy")\n'
    )
    upstream, local = UPSTREAM_PHYSICS.read_text(), LOCAL_PHYSICS.read_text()
    assert upstream.count(deleted) == 1
    assert local == upstream.replace(deleted, "", 1)
    return {
        "upstream_sha256": sha256(UPSTREAM_PHYSICS),
        "local_sha256": sha256(LOCAL_PHYSICS),
        "only_source_delta": "deleted nonzero skin_factor rejection guard",
    }


def case_reference(
    field_run: Path, name: str
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a valid saved no-skin field and its diagnostic receipt."""
    path = field_run / name
    state, diagnostics = (
        load_npz(path / "state.npz"),
        json.loads((path / "diagnostics.json").read_text()),
    )
    assert bool(state["solver_valid"])
    assert bool(diagnostics["forward"]["success"])
    assert diagnostics["status"] == "equilibrium_valid"
    assert np.array_equal(state["Ainv"], FIELD.BASE.activation_matrix(state["q"]))
    return state, {
        "state": record(path / "state.npz"),
        "mesh": record(path / "final.vtu"),
        "diagnostics": record(path / "diagnostics.json"),
        "field": diagnostics["field"],
        "forward_tolerance": diagnostics["forward"]["tolerance"],
    }


def assert_field_run_inputs(field_run: Path, paths: list[Path]) -> None:
    """Require inputs still match the immutable data80 provenance receipt."""
    records = json.loads((field_run / "provenance.json").read_text())["inputs"]
    for path in paths:
        matching = [
            row for row in records if Path(row["path"]).resolve() == path.resolve()
        ]
        assert len(matching) == 1, f"data80 lacks a unique input record for {path}"
        assert sha256(path) == matching[0]["sha256"], (
            f"data80 input hash changed: {path}"
        )


def main(cfg: Config) -> None:
    """Run only the two approved skin-on equilibria from one shared seed."""
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), "Choose an empty output directory"
    assert cfg.fixture == HERE / "data/12-historical-fixture"
    assert cfg.source_checkpoint == SOURCE_CHECKPOINT
    assert cfg.field_run == FIELD_RUN
    assert_one_guard_deletion()
    summary = json.loads((cfg.field_run / "summary.json").read_text())
    assert summary["status"] == "completed"
    source = load_npz(cfg.source_checkpoint)
    q0, seed = source["q"], source["u"]
    assert bool(source["solver_valid"])
    assert np.array_equal(source["Ainv"], FIELD.BASE.activation_matrix(q0))
    baseline, baseline_reference = case_reference(cfg.field_run, "baseline")
    strong, strong_reference = case_reference(cfg.field_run, "strong")
    assert np.array_equal(baseline["q"], q0)
    assert baseline_reference["field"]["roughness_ratio"] == 1.0
    assert abs(strong_reference["field"]["roughness_ratio"] - 0.5) < 1e-4
    assert baseline["u"].shape == strong["u"].shape == seed.shape
    assert baseline["q"].shape == strong["q"].shape == q0.shape
    assert_field_run_inputs(
        cfg.field_run,
        [
            cfg.source_checkpoint,
            cfg.source_checkpoint.parent / "summary.json",
            cfg.source_checkpoint.parent / "provenance.json",
            *(cfg.fixture / x for x in ("volume.vtu", "skin.vtp", "summary.json")),
        ],
    )
    inputs = [
        cfg.source_checkpoint,
        cfg.source_checkpoint.parent / "summary.json",
        cfg.source_checkpoint.parent / "provenance.json",
        *(cfg.fixture / x for x in ("volume.vtu", "skin.vtp", "summary.json")),
        cfg.field_run / "summary.json",
        cfg.field_run / "field-preparation.json",
    ]
    provenance = {
        "inputs": [record(path) for path in inputs],
        "source_delta": assert_one_guard_deletion(),
        "sources": [
            record(path)
            for path in (
                Path(__file__),
                LOCAL_PHYSICS,
                UPSTREAM_PHYSICS,
                HERE / "src/80-forward-field-diffusion.py",
            )
        ],
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "command": sys.argv,
    }
    write_json(out / "config.json", cfg.model_dump(mode="json"))
    write_json(out / "provenance.json", provenance)
    sources = out / "sources"
    sources.mkdir()
    for path in (
        Path(__file__),
        LOCAL_PHYSICS,
        UPSTREAM_PHYSICS,
        HERE / "src/80-forward-field-diffusion.py",
    ):
        destination = sources / path.name
        shutil.copy2(path, destination)

    SKIN.configure()
    import torch

    physics = SKIN.FacePhysics(cfg.fixture, skin_factor=SKIN_FACTOR)
    expected_materials = {
        "skin_E_MPa": 0.024,
        "skin_nu": 0.46,
        "skin_thickness_m": 0.001,
        "skin_prestrain": 0.0,
        "contact_enabled": False,
        "fat_E_MPa": 0.003,
        "muscle_E_MPa": 0.03,
        "fat_nu": 0.49,
        "muscle_nu": 0.49,
    }
    assert all(
        physics.material_spec[key] == value for key, value in expected_materials.items()
    )
    assert physics.forward_tolerance == baseline_reference["forward_tolerance"]
    assert physics.forward_tolerance == {
        "max_steps": 5000,
        "rtol": 5e-4,
        "atol": 1e-10,
        "line_search_max_steps": 10,
    }
    cases = []
    no_skin_cells = {
        "baseline_skin_0": baseline_reference,
        "strong_skin_0": strong_reference,
    }
    for step, (name, state, reference) in enumerate(
        (
            ("baseline_skin_0.12", baseline, baseline_reference),
            ("strong_skin_0.12", strong, strong_reference),
        )
    ):
        cherries.set_step(step)
        directory = out / name
        directory.mkdir()
        start = time.perf_counter()
        q = state["q"]
        seed_stress = FIELD.stress_metrics(physics, q, seed)
        try:
            u = (
                physics.solve(torch.as_tensor(q), seed.copy())
                .detach()
                .cpu()
                .numpy()
                .copy()
            )
        except SKIN.ForwardConvergenceError as error:
            with (out / "solver-stdout.log").open("a") as file:
                file.write(f"\n===== {name}: forward failure =====\n")
                file.write(str((error.receipt or {}).get("stdout", "")))
            raise
        forward = physics.last_forward
        with (out / "solver-stdout.log").open("a") as file:
            file.write(f"\n===== {name}: completed forward =====\n")
            file.write(forward["stdout"])
        ainv = FIELD.BASE.activation_matrix(q)
        np.savez_compressed(
            directory / "state.npz",
            q=q,
            u=u,
            Ainv=ainv,
            solver_valid=np.asarray(forward["success"]),
            forward_success=np.asarray(forward["success"]),
        )
        physics.save_mesh(directory / "final.vtu", u, ainv)
        det, eig = physics.detf(u), np.linalg.eigvalsh(ainv)
        diagnostics = {
            "id": name,
            "status": "equilibrium_valid"
            if forward["success"]
            else "forward_not_converged",
            "forward": forward,
            "field_reference": reference,
            "skin_factor": SKIN_FACTOR,
            "surface": FIELD.surface_metrics(physics, u),
            "stress_at_common_seed": seed_stress,
            "stress_at_equilibrium": FIELD.stress_metrics(physics, q, u),
            "physical_detF_min": float(det.min()),
            "inverted_tetrahedra": int((det <= 0).sum()),
            "non_spd_activation_tetrahedra": int((eig[:, 0] <= 0).sum()),
            "seed_sha256": FIELD.array_hash(seed),
            "q_sha256": FIELD.array_hash(q),
            "state": record(directory / "state.npz"),
            "mesh": record(directory / "final.vtu"),
            "wall_s": time.perf_counter() - start,
        }
        write_json(directory / "diagnostics.json", diagnostics)
        cases.append(diagnostics)
        cherries.log_metrics({name: diagnostics["surface"]})
    summary = {
        "schema_version": 1,
        "status": "completed"
        if all(c["forward"]["success"] for c in cases)
        else "completed_with_invalid_forward",
        "scope": "forward-only saved-field skin transmission; no inverse update, output smoothing, or geometry rejection",
        "geometry_rejection_enabled": False,
        "skin_factor": SKIN_FACTOR,
        "materials": physics.material_spec,
        "solver": physics.forward_tolerance,
        "all_four_cells_same_seed": True,
        "seed_sha256": FIELD.array_hash(seed),
        "existing_no_skin_cells": no_skin_cells,
        "new_skin_cells": cases,
        "provenance": provenance,
    }
    write_json(out / "summary.json", summary)
    for path in (
        out / "summary.json",
        out / "provenance.json",
        out / "solver-stdout.log",
    ):
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
