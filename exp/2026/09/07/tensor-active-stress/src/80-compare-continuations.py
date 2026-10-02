# ruff: noqa: EM101, EM102, TRY003
"""Validate and compare actual endpoints from variable-length continuations."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
import math
import re
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit
from tensor_controls import matrices

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
DEFAULT_MANIFEST = EXPERIMENT / "docs/80-comparison-manifest.json"
DEFAULT_OUTPUT = EXPERIMENT / "data/80-continuation-comparison"
ROLES = {
    "baseline16probe",
    "calibrated16probe",
    "selectedcontinuationblock",
    "laterregularization",
}
TRACE_COLUMNS = (
    "step",
    "local_step",
    "data_objective_mm2",
    "area_fit_rms_mm",
    "area_motion_rms_mm",
    "detF_min",
    "inverted_tetrahedra",
    "solver_valid",
    "projected_gradient_mapping_rms",
    "physical_update_rms_mpa",
    "cumulative_physical_update_rms_mpa",
)
PHYSICS_SOURCE_KEYS = (
    "experiment/20-face-inverse.py",
    "experiment/face_physics.py",
    "experiment/tensor_active.py",
    "experiment/tensor_controls.py",
)
INTERMEDIATE_CHECKPOINT = re.compile(r"optimizer-step-(\d{4,})\.pt")
COMPLETED = False


class Config(cherries.BaseConfig):
    """Completed-only comparison paths."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    manifest: Path = DEFAULT_MANIFEST
    output_dir: Path | None = None


def sha256(path: Path) -> str:
    """Return the streaming SHA-256 of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash exact contiguous typed-array bytes."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe an immutable required file."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def read_json(path: Path) -> dict[str, Any]:
    """Read an object-only JSON receipt."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    """Write finite JSON atomically."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def resolve(manifest_path: Path, value: str) -> Path:
    """Resolve one manifest-relative path."""
    if not isinstance(value, str) or not value:
        raise TypeError("path must be a nonempty string")
    return (manifest_path.parent / value).resolve()


def require_equal(actual: Any, expected: Any, label: str) -> None:
    """Fail visibly when a receipt differs from its declared contract."""
    if actual != expected:
        raise ValueError(f"{label} differs")


def finite(value: Any, label: str) -> float:
    """Parse one finite scalar."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must be numeric") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{label} must be finite")
    return parsed


def as_bool(value: Any, label: str) -> bool:
    """Parse explicit CSV boolean forms only."""
    if value in (True, "True", "true", "1", 1):
        return True
    if value in (False, "False", "false", "0", 0):
        return False
    raise TypeError(f"{label} must be boolean")


def load_script(name: str, path: Path) -> ModuleType:
    """Load the frozen surface helper quietly."""
    if not path.is_file():
        raise FileNotFoundError(path)
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with io.StringIO() as quiet:
        stdout, stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = sys.stderr = quiet
            spec.loader.exec_module(module)
        finally:
            sys.stdout, sys.stderr = stdout, stderr
    return module


def checkpoint_state(path: Path) -> dict[str, Any]:
    """Read checkpoint fields needed for recursive continuation identity."""
    state = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(state, dict) or not {
        "step",
        "q",
        "u",
        "optimizer",
        "config",
    } <= set(state):
        raise ValueError(f"invalid optimizer checkpoint: {path}")
    q, u = state["q"], np.asarray(state["u"])
    if (
        not isinstance(q, torch.Tensor)
        or q.shape != (288235, 6)
        or u.shape != (228660, 3)
    ):
        raise ValueError(f"checkpoint tensor shapes differ: {path}")
    if not torch.isfinite(q).all() or not np.isfinite(u).all():
        raise FloatingPointError(f"nonfinite checkpoint state: {path}")
    return {
        "step": int(state["step"]),
        "q_sha256": array_sha256(q.numpy()),
        "u_sha256": array_sha256(u),
        "state": state,
    }


def npz_state(path: Path) -> dict[str, Any]:
    """Read final/history NPZ state hashes without inventing receipt aliases."""
    with np.load(path, allow_pickle=False) as saved:
        required = {"q", "u", "Q", "active_ids", "step", "solver_valid"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{path} lacks {sorted(missing)}")
        if not bool(saved["solver_valid"]):
            raise ValueError(f"endpoint is not solver-valid: {path}")
        return {
            "step": int(saved["step"]),
            "q_sha256": array_sha256(saved["q"]),
            "u_sha256": array_sha256(saved["u"]),
            "Q_sha256": array_sha256(saved["Q"]),
            "active_ids_sha256": array_sha256(saved["active_ids"]),
        }


def validate_endpoint_binding(  # noqa: C901, PLR0912, PLR0915
    npz_path: Path,
    vtu_path: Path,
    reference: pv.UnstructuredGrid,
    stress_reference_mpa: float,
    label: str,
) -> dict[str, Any]:
    """Bind one saved tensor state to its physical VTU arrays and topology."""
    with np.load(npz_path, allow_pickle=False) as saved:
        required = {"q", "u", "Q", "active_ids", "step", "solver_valid"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{label} NPZ lacks {sorted(missing)}")
        q = np.asarray(saved["q"])
        u = np.asarray(saved["u"])
        stress = np.asarray(saved["Q"])
        active_ids = np.asarray(saved["active_ids"])
        step = np.asarray(saved["step"])
        solver_valid = np.asarray(saved["solver_valid"])
    if (
        q.dtype != np.float64
        or u.dtype != np.float64
        or stress.dtype != np.float64
        or active_ids.dtype != np.int64
        or step.shape != ()
        or solver_valid.shape != ()
        or not np.issubdtype(step.dtype, np.integer)
        or solver_valid.dtype != np.bool_
    ):
        raise TypeError(f"{label} NPZ typed-array contract differs")
    if (
        q.ndim != 2
        or q.shape[1] != 6
        or u.shape != (reference.n_points, 3)
        or stress.shape != (len(q), 3, 3)
        or active_ids.shape != (len(q),)
    ):
        raise ValueError(f"{label} NPZ array shape differs")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(u).all()
        or not np.isfinite(stress).all()
    ):
        raise FloatingPointError(f"{label} NPZ state is nonfinite")
    if (
        np.any(active_ids < 0)
        or np.any(active_ids >= reference.n_cells)
        or not np.all(active_ids[1:] > active_ids[:-1])
    ):
        raise ValueError(f"{label} active IDs are not sorted unique cell IDs")
    physical = stress_reference_mpa * matrices(torch.from_numpy(q)).numpy()
    tolerance = 8.0 * np.finfo(np.float64).eps * stress_reference_mpa
    if not np.allclose(
        stress, physical, rtol=8.0 * np.finfo(np.float64).eps, atol=tolerance
    ):
        raise ValueError(f"{label} saved Q differs from q in physical MPa")

    grid = pv.read(vtu_path)
    if not isinstance(grid, pv.UnstructuredGrid):
        raise TypeError(f"{label} VTU is not an UnstructuredGrid")
    if (
        grid.n_points != reference.n_points
        or grid.n_cells != reference.n_cells
        or not np.array_equal(grid.celltypes, reference.celltypes)
        or not np.array_equal(grid.cells, reference.cells)
    ):
        raise ValueError(f"{label} VTU topology differs from the frozen fixture")
    for key in ("RestPosition", "Displacement"):
        if key not in grid.point_data:
            raise ValueError(f"{label} VTU lacks {key}")
    for key in ("ActivationMask", "ActiveStressMatrixMPa"):
        if key not in grid.cell_data:
            raise ValueError(f"{label} VTU lacks {key}")
    rest = np.asarray(grid.point_data["RestPosition"])
    displacement = np.asarray(grid.point_data["Displacement"])
    points = np.asarray(grid.points)
    active = np.asarray(grid.cell_data["ActivationMask"])
    vtk_stress = np.asarray(grid.cell_data["ActiveStressMatrixMPa"])
    if (
        rest.dtype != np.float64
        or displacement.dtype != np.float64
        or points.dtype != np.float64
        or active.dtype != np.bool_
        or vtk_stress.dtype != np.float64
    ):
        raise TypeError(f"{label} VTU typed-array contract differs")
    if not np.array_equal(rest, np.asarray(reference.points, dtype=np.float64)):
        raise ValueError(f"{label} VTU RestPosition differs from the frozen fixture")
    if not np.array_equal(displacement, u):
        raise ValueError(f"{label} VTU Displacement differs from NPZ u")
    if not np.array_equal(points, rest + u):
        raise ValueError(f"{label} VTU points differ from RestPosition + NPZ u")
    if active.shape != (reference.n_cells,) or not np.array_equal(
        np.flatnonzero(active), active_ids
    ):
        raise ValueError(f"{label} VTU active IDs differ from the NPZ")
    if vtk_stress.shape != (reference.n_cells, 9):
        raise ValueError(f"{label} VTU stress array shape differs")
    vtk_stress = vtk_stress.reshape(-1, 3, 3)
    if not np.array_equal(vtk_stress[active], stress):
        raise ValueError(f"{label} VTU active stress differs from NPZ Q")
    if np.count_nonzero(vtk_stress[~active]):
        raise ValueError(f"{label} VTU inactive stress is nonzero")
    return {
        "step": int(step),
        "stress_reference_mpa": stress_reference_mpa,
        "checks": [
            "typed NPZ arrays",
            "q to physical Q",
            "frozen fixture topology and RestPosition",
            "Displacement and state points",
            "active IDs and VTK stress",
        ],
    }


def validate_parent_geometry(
    checkpoint: dict[str, Any], npz_path: Path, vtu_path: Path, label: str
) -> None:
    """Bind an explicit parent checkpoint to its saved NPZ and VTU state."""
    npz = npz_state(npz_path)
    require_equal(checkpoint["step"], npz["step"], f"{label} NPZ step")
    require_equal(checkpoint["q_sha256"], npz["q_sha256"], f"{label} NPZ q")
    require_equal(checkpoint["u_sha256"], npz["u_sha256"], f"{label} NPZ u")
    grid = pv.read(vtu_path)
    if not isinstance(grid, pv.UnstructuredGrid):
        raise TypeError(f"{label} VTU is not an UnstructuredGrid")
    for key in ("RestPosition", "Displacement"):
        if key not in grid.point_data:
            raise ValueError(f"{label} VTU lacks {key}")
    with np.load(npz_path, allow_pickle=False) as saved:
        u = np.asarray(saved["u"], dtype=np.float64)
    rest = np.asarray(grid.point_data["RestPosition"], dtype=np.float64)
    displacement = np.asarray(grid.point_data["Displacement"], dtype=np.float64)
    points = np.asarray(grid.points, dtype=np.float64)
    if rest.shape != u.shape or not np.allclose(points - rest, u, rtol=0.0, atol=3e-15):
        raise ValueError(f"{label} VTU points do not bind NPZ u")
    if not np.allclose(displacement, u, rtol=0.0, atol=3e-15):
        raise ValueError(f"{label} VTU displacement does not bind NPZ u")


def parent_owner(
    manifest_path: Path, runs: list[dict[str, Any]], parent: dict[str, Any]
) -> dict[str, Any] | None:
    """Return a listed source run for a non-root parent checkpoint."""
    parent_path = resolve(manifest_path, parent["path"])
    matches = [
        run for run in runs if resolve(manifest_path, run["path"]) == parent_path.parent
    ]
    if len(matches) != 1:
        return None
    return matches[0]


def validate_explicit_parent(  # noqa: C901
    manifest_path: Path, runs: list[dict[str, Any]], parent: dict[str, Any]
) -> dict[str, Any]:
    """Validate root, ordinary-final, or listed regularization intermediate parent."""
    path = resolve(manifest_path, parent["path"])
    if sha256(path) != parent["sha256"]:
        raise ValueError("parent checkpoint hash differs")
    checkpoint = checkpoint_state(path)
    require_equal(checkpoint["step"], parent["global_step"], "parent checkpoint step")
    common = manifest_path  # Resolve the one permitted external chain root exactly.
    root = read_json(manifest_path).get("common_root_checkpoint")
    del common
    if not isinstance(root, dict):
        raise TypeError("common root is required")
    root_path = resolve(manifest_path, root["path"])
    if path == root_path:
        if (
            parent["sha256"] != root["sha256"]
            or parent["global_step"] != root["global_step"]
        ):
            raise ValueError("common root parent receipt differs")
        return {"kind": "common_root", "checkpoint": checkpoint}
    owner = parent_owner(manifest_path, runs, parent)
    if owner is None:
        raise ValueError("parent checkpoint is not owned by a listed run")
    if path.name == "optimizer-latest.pt":
        if parent["global_step"] != owner["expected"]["final_global_step"]:
            raise ValueError("ordinary optimizer-latest parent is not its listed final")
        directory = resolve(manifest_path, owner["path"])
        validate_parent_geometry(
            checkpoint,
            directory / "final.npz",
            directory / "final.vtu",
            "listed final parent",
        )
        return {"kind": "listed_final", "owner": owner, "checkpoint": checkpoint}
    matched = INTERMEDIATE_CHECKPOINT.fullmatch(path.name)
    if matched is None or int(matched.group(1)) != parent["global_step"]:
        raise ValueError(
            "parent must be root, listed optimizer-latest final, or optimizer-step-NNNN"
        )
    directory = resolve(manifest_path, owner["path"])
    summary = read_json(directory / "summary.json")
    config = read_json(directory / "config.json")
    if (
        summary.get("status") != "completed_fixed_budget_continuation"
        or config.get("phase") != "regularization"
    ):
        raise ValueError(
            "intermediate parent must be a listed completed regularization run"
        )
    trace = read_trace(directory / "trace.csv", owner["id"])
    matches = [row for row in trace if int(row["step"]) == parent["global_step"]]
    if len(matches) != 1:
        raise ValueError("intermediate parent lacks one solver-valid trace row")
    stem = f"step-{parent['global_step']:04d}"
    npz, vtu = directory / f"{stem}.npz", directory / f"{stem}.vtu"
    if not npz.is_file() or not vtu.is_file():
        raise FileNotFoundError("intermediate parent lacks saved NPZ/VTU pair")
    validate_parent_geometry(checkpoint, npz, vtu, "intermediate parent")
    return {
        "kind": "listed_regularization_intermediate",
        "owner": owner,
        "checkpoint": checkpoint,
        "trace_row": matches[0],
        "npz": record(npz),
        "vtu": record(vtu),
    }


def source_mapping(provenance: dict[str, Any]) -> dict[str, Any]:
    """Use standard base20 source/input mappings, nested or direct."""
    value = provenance.get("base20", provenance)
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("sources"), dict)
        or not isinstance(value.get("inputs"), dict)
    ):
        raise TypeError("provenance needs base20 sources and inputs mappings")
    return value


def validate_manifest(  # noqa: C901, PLR0912
    manifest: dict[str, Any], path: Path
) -> tuple[list[dict[str, Any]], Path]:
    """Validate only the concise recursive-chain manifest schema."""
    require_equal(manifest.get("schema_version"), 1, "schema_version")
    if not isinstance(manifest.get("title"), str) or not manifest["title"]:
        raise ValueError("manifest title is required")
    root = manifest.get("common_root_checkpoint")
    if not isinstance(root, dict):
        raise TypeError("common_root_checkpoint is required")
    for key in ("path", "sha256"):
        if not isinstance(root.get(key), str) or not root[key]:
            raise ValueError(f"common_root_checkpoint.{key} is required")
    if not isinstance(root.get("global_step"), int) or root["global_step"] < 0:
        raise ValueError("common root global_step must be nonnegative int")
    surface = manifest.get("surface_protocol")
    if not isinstance(surface, dict) or tuple(surface.get("scales_mm", ())) != (
        2,
        5,
        10,
    ):
        raise ValueError("surface protocol must retain scales [2, 5, 10]")
    if surface.get("primary_scale_mm") != 5 or surface.get("mouth_radius_mm") != 10:
        raise ValueError("surface protocol must retain primary5/mouth10")
    if not all(
        isinstance(surface.get(key), str) for key in ("fixture_vtu", "skin_vtp")
    ):
        raise TypeError("surface protocol fixture and skin paths are required")
    runs = manifest.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("manifest must list runs")
    ids: set[str] = set()
    for run in runs:
        if not isinstance(run, dict):
            raise TypeError("run must be an object")
        for key in ("id", "label", "role", "path"):
            if not isinstance(run.get(key), str) or not run[key]:
                raise ValueError(f"run.{key} is required")
        if run["id"] in ids or run["role"] not in ROLES:
            raise ValueError(f"invalid duplicate ID or role for {run['id']}")
        ids.add(run["id"])
        parent, expected = run.get("parent_checkpoint"), run.get("expected")
        if not isinstance(parent, dict) or not isinstance(expected, dict):
            raise TypeError(f"run {run['id']} needs parent_checkpoint and expected")
        for key in ("path", "sha256"):
            if not isinstance(parent.get(key), str) or not parent[key]:
                raise ValueError(f"run {run['id']} parent {key} is required")
        for key in ("global_step",):
            if not isinstance(parent.get(key), int) or parent[key] < 0:
                raise ValueError(
                    f"run {run['id']} parent {key} must be nonnegative int"
                )
        for key in ("final_global_step", "final_local_step"):
            if not isinstance(expected.get(key), int) or expected[key] < 0:
                raise ValueError(
                    f"run {run['id']} expected {key} must be nonnegative int"
                )
    return runs, resolve(path, manifest.get("output_dir", str(DEFAULT_OUTPUT)))


def read_trace(path: Path, run_id: str) -> list[dict[str, Any]]:
    """Require native source74 trace columns and valid increasing steps."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or set(TRACE_COLUMNS) - set(reader.fieldnames):
            raise ValueError(f"trace {run_id} lacks required native columns")
        rows = list(reader)
    if not rows:
        raise ValueError(f"trace {run_id} is empty")
    previous = (-1, -1)
    for index, row in enumerate(rows):
        for name in TRACE_COLUMNS:
            if name == "solver_valid":
                if not as_bool(row[name], f"{run_id}:{index}:{name}"):
                    raise ValueError(f"{run_id} has an invalid solver row")
            elif name == "inverted_tetrahedra":
                row[name] = int(row[name])
                if row[name] < 0:
                    raise ValueError(f"{run_id} has negative inversion count")
            else:
                row[name] = finite(row[name], f"{run_id}:{index}:{name}")
        key = (int(row["step"]), int(row["local_step"]))
        if key[0] <= previous[0] or key[1] <= previous[1]:
            raise ValueError(f"{run_id} trace steps must strictly increase")
        previous = key
    return rows


def assert_parent_physics(
    parent: dict[str, Any], current: dict[str, Any], label: str
) -> None:
    """Require source74's preserved base20 physical code and inputs."""
    old, new = source_mapping(parent), source_mapping(current)
    for key in PHYSICS_SOURCE_KEYS:
        require_equal(
            new["sources"].get(key), old["sources"].get(key), f"{label} source {key}"
        )
    for key, digest in old["sources"].items():
        if key.startswith("liblaf/apple/"):
            require_equal(
                new["sources"].get(key), digest, f"{label} library source {key}"
            )
    require_equal(new["inputs"], old["inputs"], f"{label} inputs")


def surface_metrics(
    manifest_path: Path, manifest: dict[str, Any], endpoint: Path
) -> dict[str, Any]:
    """Apply source40's finite frozen-skin high-pass measure to an actual final."""
    source40 = load_script(
        "continuation_surface40", EXPERIMENT / "src/40-measure-surface.py"
    )
    helper = source40.load_script(
        "continuation_surface41", source40.FROZEN_ROUGHNESS_HELPER
    )
    protocol = manifest["surface_protocol"]
    source_fixture, skin = (
        pv.read(resolve(manifest_path, protocol["fixture_vtu"])),
        pv.read(resolve(manifest_path, protocol["skin_vtp"])),
    )
    if not isinstance(source_fixture, pv.UnstructuredGrid) or not isinstance(
        skin, pv.PolyData
    ):
        raise TypeError("surface fixture type changed")
    fixture, target_contract = source40.sanitize_smile_target(source_fixture, skin)
    final = pv.read(endpoint)
    if not isinstance(final, pv.UnstructuredGrid):
        raise TypeError("final endpoint must be an unstructured grid")
    helper.assert_rest_geometry(fixture, skin, final)
    audit = helper.load_sibling_audit()
    values, _ = audit.surface_diagnostics(
        fixture, final, skin, audit.cotangent_operators(skin), (0.002, 0.005, 0.010)
    )
    helper.assert_nonnegative_highpass(values)
    values = source40.add_normalized_ratios(values)
    if not helper.finite_numbers(values):
        raise FloatingPointError("surface measurement is nonfinite")
    return {"target_contract": target_contract, "surface": values}


def endpoint_row(endpoint: dict[str, Any]) -> dict[str, Any]:
    """Flatten joint fit/motion/stress/normalized-HP values for CSV."""
    row = {
        key: endpoint[key]
        for key in ("id", "label", "role", "matched_group", "run_dir")
    }
    row.update(endpoint["primary_endpoint"])
    primary = endpoint["surface_measurement"]["surface"]["primary_5mm"]
    for field in ("normal_displacement", "normal_residual"):
        for roi in ("full_face", "mouth_10mm"):
            value = primary[field][roi]
            prefix = f"{field}_{roi}_5mm"
            row[f"{prefix}_highpass_rms_mm"] = value["highpass_rms_mm"]
            row[f"{prefix}_highpass_over_total_normal_rms"] = value[
                "highpass_over_total_normal_rms"
            ]["ratio"]
    return row


def validate_run(  # noqa: PLR0915
    manifest_path: Path,
    manifest: dict[str, Any],
    runs: list[dict[str, Any]],
    run: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate actual source74 receipt fields and final endpoint bindings."""
    root = resolve(manifest_path, run["path"])
    names = (
        "config.json",
        "resume.json",
        "provenance.json",
        "summary.json",
        "trace.csv",
        "latest.npz",
        "best.npz",
        "final.npz",
        "best.vtu",
        "final.vtu",
        "optimizer-latest.pt",
    )
    files = {name: root / name for name in names}
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config, resume, provenance, summary = (
        read_json(files[name])
        for name in ("config.json", "resume.json", "provenance.json", "summary.json")
    )
    require_equal(config.get("model"), "tensor", f"{run['id']} model")
    stress_reference_mpa = finite(
        config.get("stress_reference_mpa"), f"{run['id']} stress reference"
    )
    if stress_reference_mpa <= 0:
        raise ValueError(f"{run['id']} stress reference must be positive")
    require_equal(
        summary.get("config", {}).get("stress_reference_mpa"),
        stress_reference_mpa,
        f"{run['id']} summary stress reference",
    )
    parent = run["parent_checkpoint"]
    declared = resume.get("parent_checkpoint")
    if not isinstance(declared, dict):
        raise TypeError(f"{run['id']} resume lacks parent checkpoint")
    require_equal(declared.get("sha256"), parent["sha256"], f"{run['id']} parent hash")
    require_equal(
        resolve(manifest_path, parent["path"]),
        Path(declared.get("path", "")).resolve(),
        f"{run['id']} parent path",
    )
    require_equal(
        resume.get("parent_global_step"),
        parent["global_step"],
        f"{run['id']} parent step",
    )
    parent_path = resolve(manifest_path, parent["path"])
    parent_validation = validate_explicit_parent(manifest_path, runs, parent)
    parent_state = parent_validation["checkpoint"]
    require_equal(
        resume.get("controls_sha256"),
        parent_state["q_sha256"],
        f"{run['id']} resumed q",
    )
    require_equal(
        resume.get("seed_displacement_sha256"),
        parent_state["u_sha256"],
        f"{run['id']} resumed seed",
    )
    require_equal(
        summary.get("status"),
        "completed_fixed_budget_continuation",
        f"{run['id']} status",
    )
    if summary.get("inverse_convergence_claimed") is not False:
        raise ValueError(f"{run['id']} claims inverse convergence")
    trace = read_trace(files["trace.csv"], run["id"])
    first, last, expected = trace[0], trace[-1], run["expected"]
    require_equal(
        (int(first["step"]), int(first["local_step"])),
        (parent["global_step"], 0),
        f"{run['id']} trace start",
    )
    require_equal(
        (int(last["step"]), int(last["local_step"])),
        (expected["final_global_step"], expected["final_local_step"]),
        f"{run['id']} trace final",
    )
    require_equal(
        resume.get("requested_additional_updates"),
        expected["final_local_step"],
        f"{run['id']} requested updates",
    )
    require_equal(
        resume.get("requested_final_global_step"),
        expected["final_global_step"],
        f"{run['id']} requested final step",
    )
    primary, initial = summary.get("primary_endpoint"), summary.get("initial_endpoint")
    if not isinstance(primary, dict) or not isinstance(initial, dict):
        raise TypeError(f"{run['id']} summary lacks primary/initial endpoint")
    for key in (
        "step",
        "local_step",
        "data_objective_mm2",
        "area_fit_rms_mm",
        "area_motion_rms_mm",
        "detF_min",
        "inverted_tetrahedra",
    ):
        require_equal(primary.get(key), last.get(key), f"{run['id']} primary {key}")
        require_equal(initial.get(key), first.get(key), f"{run['id']} initial {key}")
    final_state, latest_state = (
        npz_state(files["final.npz"]),
        npz_state(files["latest.npz"]),
    )
    require_equal(final_state, latest_state, f"{run['id']} final/latest state")
    require_equal(
        final_state["step"],
        expected["final_global_step"],
        f"{run['id']} final NPZ step",
    )
    history = root / f"step-{expected['final_global_step']:04d}.npz"
    history_vtu = history.with_suffix(".vtu")
    if not history.is_file() or not history_vtu.is_file():
        raise FileNotFoundError(f"{run['id']} lacks final history pair")
    require_equal(npz_state(history), final_state, f"{run['id']} history/final state")
    reference = pv.read(
        resolve(manifest_path, manifest["surface_protocol"]["fixture_vtu"])
    )
    if not isinstance(reference, pv.UnstructuredGrid):
        raise TypeError("surface fixture must be an UnstructuredGrid")
    final_binding = validate_endpoint_binding(
        files["final.npz"],
        files["final.vtu"],
        reference,
        stress_reference_mpa,
        f"{run['id']} final",
    )
    history_binding = validate_endpoint_binding(
        history,
        history_vtu,
        reference,
        stress_reference_mpa,
        f"{run['id']} final history",
    )
    require_equal(final_binding, history_binding, f"{run['id']} VTU bindings")
    latest_checkpoint = checkpoint_state(files["optimizer-latest.pt"])
    require_equal(
        latest_checkpoint["step"],
        final_state["step"],
        f"{run['id']} optimizer final step",
    )
    require_equal(
        latest_checkpoint["q_sha256"],
        final_state["q_sha256"],
        f"{run['id']} optimizer/final q",
    )
    require_equal(
        latest_checkpoint["u_sha256"],
        final_state["u_sha256"],
        f"{run['id']} optimizer/final u",
    )
    parent_provenance = read_json(parent_path.parent / "provenance.json")
    assert_parent_physics(parent_provenance, provenance, run["id"])
    measured = surface_metrics(manifest_path, manifest, files["final.vtu"])
    endpoint = {
        "id": run["id"],
        "label": run["label"],
        "role": run["role"],
        "matched_group": run.get("matched_group"),
        "run_dir": str(root),
        "parent_checkpoint": parent,
        "parent_validation": {
            **parent_validation,
            "checkpoint": {
                key: value
                for key, value in parent_validation["checkpoint"].items()
                if key != "state"
            },
        },
        "primary_endpoint": primary,
        "initial_endpoint": initial,
        "final_state": final_state,
        "endpoint_binding": final_binding,
        "surface_measurement": measured,
        "files": {name: record(path) for name, path in files.items()},
        "final_history": {"npz": record(history), "vtu": record(history_vtu)},
    }
    return endpoint, trace


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a rectangular evidence table."""
    if not rows:
        raise ValueError("cannot write empty CSV")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot(path: Path, title: str, traces: list[list[dict[str, Any]]]) -> None:
    """Plot actual fixed-budget traces against global step."""
    figure, axes = plt.subplots(
        2, 2, figsize=(12, 8), layout="constrained", sharex=True
    )
    panels = (
        ("area_fit_rms_mm", "Area fit RMS (mm)"),
        ("area_motion_rms_mm", "Area motion RMS (mm)"),
        ("Q_rms_mpa", "Q RMS (MPa)"),
        (
            "cumulative_physical_update_rms_mpa",
            "Within-block cumulative ΔQ RMS (MPa)",
        ),
    )
    for axis, (key, ylabel) in zip(axes.flat, panels, strict=True):
        found = False
        for rows in traces:
            if key in rows[0]:
                axis.plot(
                    [int(row["step"]) for row in rows],
                    [finite(row[key], key) for row in rows],
                    marker="o",
                    markersize=2,
                    label=rows[0]["label"],
                )
                found = True
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        if not found:
            axis.text(
                0.5,
                0.5,
                "not recorded",
                transform=axis.transAxes,
                ha="center",
                va="center",
            )
    for axis in axes[1]:
        axis.set_xlabel("Global optimizer step")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        title
        + "\nSaved actual endpoints; variable blocks, no inverse-convergence claim"
    )
    figure.savefig(path.with_suffix(".png"), dpi=180)
    figure.savefig(path.with_suffix(".pdf"))
    plt.close(figure)


def validate_chain(
    manifest_path: Path, manifest: dict[str, Any], runs: list[dict[str, Any]]
) -> None:
    """Require every declared parent to be root or a listed exact saved checkpoint."""
    root = manifest["common_root_checkpoint"]
    root_path = resolve(manifest_path, root["path"])
    require_equal(sha256(root_path), root["sha256"], "common root checkpoint bytes")
    require_equal(
        checkpoint_state(root_path)["step"],
        root["global_step"],
        "common root checkpoint step",
    )
    by_id = {run["id"]: run for run in runs}
    for run in runs:
        seen: set[str] = {run["id"]}
        parent = run["parent_checkpoint"]
        while True:
            path = resolve(manifest_path, parent["path"])
            key = (str(path), parent["sha256"])
            if key == (str(root_path), root["sha256"]):
                break
            owner = parent_owner(manifest_path, runs, parent)
            if owner is None or owner["id"] in seen:
                raise ValueError(f"{run['id']} parent is unlisted or creates a cycle")
            validate_explicit_parent(manifest_path, runs, parent)
            seen.add(owner["id"])
            parent = by_id[owner["id"]]["parent_checkpoint"]


def main(cfg: Config) -> None:
    """Create receipt tables and plot only after all manifests/runs exist."""
    global COMPLETED  # noqa: PLW0603
    manifest_path, manifest = cfg.manifest.resolve(), read_json(cfg.manifest.resolve())
    runs, declared_output = validate_manifest(manifest, manifest_path)
    validate_chain(manifest_path, manifest, runs)
    output = (cfg.output_dir or declared_output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cherries.log_input(manifest_path)
    endpoints: list[dict[str, Any]] = []
    plotted: list[list[dict[str, Any]]] = []
    trace_rows: list[dict[str, Any]] = []
    for run in runs:
        endpoint, trace = validate_run(manifest_path, manifest, runs, run)
        endpoints.append(endpoint)
        labeled = [
            {"id": run["id"], "label": run["label"], "role": run["role"], **row}
            for row in trace
        ]
        plotted.append(labeled)
        trace_rows.extend(labeled)
    write_csv(
        output / "continuation-endpoints.csv",
        [endpoint_row(item) for item in endpoints],
    )
    write_csv(output / "continuation-traces.csv", trace_rows)
    plot(output / "continuation-optimization", manifest["title"], plotted)
    summary = {
        "schema_version": 1,
        "status": "completed_postprocessing",
        "scope": "actual saved finals only; no solve, control mutation, mesh mutation, or best substitution",
        "interpretation_limits": [
            "numerical validation does not imply inverse convergence or physical acceptability",
            "fixed-budget fit/motion/stress values do not establish capacity",
            "regularization requires an explicit matched_group and retains fit/motion/stress/normalized high-pass together",
            "normal high-pass content is descriptive and target detail can be real",
        ],
        "manifest": record(manifest_path),
        "common_root_checkpoint": manifest["common_root_checkpoint"],
        "surface_protocol": manifest["surface_protocol"],
        "endpoint_policy": "summary.primary_endpoint plus final NPZ/VTU; best is provenance only",
        "endpoints": endpoints,
        "artifacts": {
            name: record(output / name)
            for name in (
                "continuation-endpoints.csv",
                "continuation-traces.csv",
                "continuation-optimization.png",
                "continuation-optimization.pdf",
            )
        },
    }
    write_json(output / "summary.json", summary)
    for name in (
        "summary.json",
        "continuation-endpoints.csv",
        "continuation-traces.csv",
        "continuation-optimization.png",
        "continuation-optimization.pdf",
    ):
        cherries.log_output(output / name)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
