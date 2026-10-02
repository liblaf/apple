# Copyright (c) 2026 liblaf
# ruff: noqa: C901, EM101, EM102, TRY003
"""Match completed regularization arms and make the conditional rank decision."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "docs/82-regularization-match-manifest.json"
WEIGHTS = {"control": 0.0, "weak": 1.4762928671047126, "full": 5.9051714684188505}
SOURCE74_TRACE_COLUMNS = {
    "step",
    "local_step",
    "data_objective_mm2",
    "smoothness",
    "magnitude",
    "rank_penalty",
    "objective",
    "fit_gradient_rms",
    "gradient_rms",
    "regularizer_gradient_rms",
    "fit_rms_mm",
    "area_fit_rms_mm",
    "area_motion_rms_mm",
    "target_projection",
    "detF_min",
    "detF_max",
    "inverted_tetrahedra",
    "Q_eigen_min_mpa",
    "Q_eigen_max_mpa",
    "Q_trace_mean_mpa",
    "Q_rms_mpa",
    "upper_cap_cell_fraction",
    "principal_tension_fraction",
    "rank_mixing_fraction",
    "zero_stress_cell_fraction",
    "projection_rms",
    "projected_negative_eigenvalue_fraction",
    "projected_upper_eigenvalue_fraction",
    "unprojected_update_rms",
    "actual_update_rms",
    "physical_update_rms_mpa",
    "physical_update_frobenius_max_mpa",
    "projected_gradient_mapping_rms",
    "projected_gradient_mapping_max_abs",
    "projected_gradient_eta",
    "projected_to_raw_gradient_rms_ratio",
    "adam_step",
    "adam_learning_rate",
    "adam_eps",
    "adam_sqrt_vhat_median",
    "adam_sqrt_vhat_p90",
    "adam_sqrt_vhat_p99",
    "adam_sqrt_vhat_max",
    "adam_eps_dominant_coordinate_fraction",
    "cumulative_physical_update_rms_mpa",
    "net_physical_update_rms_mpa",
    "projected_gradient_relative_to_start",
    "forward_steps",
    "forward_grad_norm",
    "solver_valid",
    "elapsed_s",
}
INTEGER_TRACE_FIELDS = {
    "step",
    "local_step",
    "inverted_tetrahedra",
    "adam_step",
    "forward_steps",
}
COMPLETED = False


class Config(cherries.BaseConfig):
    """Immutable manifest and empty output for CPU-only post-processing."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    manifest: Path = DEFAULT_MANIFEST
    output_dir: Path | None = None


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    """Hash exact contiguous typed-array bytes."""
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe an existing immutable input or output."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    temporary.replace(path)


def load_source80() -> ModuleType:
    """Load the already-frozen surface calculation without starting a solver."""
    path = ROOT / "src/80-compare-continuations.py"
    spec = importlib.util.spec_from_file_location("regularization_source80", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve(manifest: Path, raw: object, label: str) -> Path:
    """Resolve a manifest-relative path within this experiment."""
    if not isinstance(raw, str) or not raw:
        raise TypeError(f"{label} must be a nonempty string")
    path = (manifest.parent / raw).resolve()
    try:
        path.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"{label} escapes experiment root: {path}") from error
    return path


def finite(raw: object, label: str) -> float:
    """Parse one finite trace scalar."""
    value = float(raw)
    if not np.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def load_trace(  # noqa: PLR0912
    path: Path, identity: str
) -> list[dict[str, Any]]:
    """Read and type every field in source74's real solver-valid trace."""
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if (
            reader.fieldnames is None
            or set(reader.fieldnames) != SOURCE74_TRACE_COLUMNS
        ):
            raise ValueError(f"source74 trace schema differs: {identity}")
        raw_rows = list(reader)
    if len(raw_rows) != 17:
        raise ValueError(f"source74 trace must have local steps 0..16: {identity}")
    rows: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        row: dict[str, Any] = {}
        for name, value in raw.items():
            if name in INTEGER_TRACE_FIELDS:
                if value is None or str(int(value)) != value:
                    raise ValueError(
                        f"source74 integer field differs: {identity}:{index}:{name}"
                    )
                row[name] = int(value)
            elif name == "solver_valid":
                if value not in {"True", "true", "1"}:
                    raise ValueError(
                        f"source74 trace is solver-invalid: {identity}:{index}"
                    )
                row[name] = True
            elif value == "":
                row[name] = None
            else:
                row[name] = finite(value, f"{identity}:{index}:{name}")
        if row["local_step"] != index or row["step"] < 0:
            raise ValueError(f"source74 trace sequence differs: {identity}")
        if index and row["step"] != rows[-1]["step"] + 1:
            raise ValueError(f"source74 global steps are not contiguous: {identity}")
        if row["smoothness"] < 0 or not 0 <= row["rank_mixing_fraction"] <= 1:
            raise ValueError(f"source74 metric range differs: {identity}:{index}")
        rows.append(row)
    return rows


def state_files(
    directory: Path,
    row: dict[str, Any],
    identity: str,
    config: dict[str, Any],
) -> dict[str, Path]:
    """Require source74's exact saved solver-valid state pair at a trace row."""
    stem = f"step-{row['step']:04d}"
    paths = {
        "npz": directory / f"{stem}.npz",
        "vtu": directory / f"{stem}.vtu",
        "optimizer": directory / f"optimizer-{stem}.pt",
    }
    for suffix, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{identity} lacks saved {suffix}: {path}")
    with np.load(paths["npz"], allow_pickle=False) as saved:
        if int(saved["step"]) != row["step"] or not bool(saved["solver_valid"]):
            raise ValueError(f"{identity} state binding differs at {row['step']}")
        q, u = saved["q"].copy(), saved["u"].copy()
    checkpoint = torch.load(paths["optimizer"], map_location="cpu", weights_only=False)
    if (
        not isinstance(checkpoint, dict)
        or int(checkpoint.get("step", -1)) != row["step"]
        or checkpoint.get("config") != config
    ):
        raise ValueError(
            f"{identity} optimizer checkpoint step differs at {row['step']}"
        )
    if not np.array_equal(checkpoint["q"].numpy(), q) or not np.array_equal(
        np.asarray(checkpoint["u"]), u
    ):
        raise ValueError(
            f"{identity} optimizer checkpoint state differs at {row['step']}"
        )
    groups = checkpoint.get("optimizer", {}).get("param_groups", [])
    if len(groups) != 1 or len(groups[0].get("params", [])) != 1:
        raise ValueError(
            f"{identity} optimizer checkpoint shape differs at {row['step']}"
        )
    state = checkpoint["optimizer"].get("state", {}).get(groups[0]["params"][0])
    if not isinstance(state, dict) or int(state.get("step", -1)) != row["adam_step"]:
        raise ValueError(f"{identity} Adam counter differs at {row['step']}")
    if row["adam_step"] != row["step"]:
        raise ValueError(f"{identity} trace Adam counter differs at {row['step']}")
    if (
        groups[0].get("lr") != config["learning_rate"]
        or groups[0].get("eps") != config["adam_eps"]
        or row["adam_learning_rate"] != config["learning_rate"]
        or row["adam_eps"] != config["adam_eps"]
        or row["projected_gradient_eta"] != config["projected_gradient_eta"]
    ):
        raise ValueError(f"{identity} optimizer settings differ at {row['step']}")
    return paths


def load_arm(  # noqa: PLR0912
    manifest: Path, role: str, raw: object
) -> dict[str, Any]:
    """Validate one completed source74 regularization branch."""
    if not isinstance(raw, dict):
        raise TypeError(f"{role} arm must be an object")
    identity, label = raw.get("id"), raw.get("label")
    if (
        not isinstance(identity, str)
        or not identity
        or not isinstance(label, str)
        or not label
    ):
        raise ValueError(f"{role} arm needs id and label")
    directory = resolve(manifest, raw.get("path"), f"{role} path")
    files = {
        name: directory / name
        for name in (
            "config.json",
            "provenance.json",
            "resume.json",
            "summary.json",
            "trace.csv",
            "optimizer-latest.pt",
        )
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    config, provenance, resume, summary = (
        json.loads(files[name].read_text())
        for name in ("config.json", "provenance.json", "resume.json", "summary.json")
    )
    if (
        summary.get("status") != "completed_fixed_budget_continuation"
        or summary.get("inverse_convergence_claimed") is not False
    ):
        raise ValueError(
            f"{identity} is not a completed non-converged source74 continuation"
        )
    if config.get("phase") != "regularization" or summary.get("config") != config:
        raise ValueError(f"{identity} is not a source74 regularization arm")
    if summary.get("provenance") != provenance:
        raise ValueError(f"{identity} provenance file differs from summary")
    if (
        config.get("smoothness_weight") != WEIGHTS[role]
        or config.get("rank_weight") != 0
        or config.get("magnitude_weight") != 0
    ):
        raise ValueError(f"{identity} has wrong regularization weights")
    if config.get("steps") != 16 or config.get("checkpoint_interval") != 1:
        raise ValueError(f"{identity} does not retain all 16 saved states")
    if config.get("projected_gradient_eta") != 1.0:
        raise ValueError(f"{identity} projected-gradient eta differs")
    parent_receipt = resume.get("parent_checkpoint", {})
    if (
        Path(config.get("resume", "")).resolve()
        != Path(parent_receipt.get("path", "")).resolve()
    ):
        raise ValueError(f"{identity} config resume differs from parent receipt")
    trace = load_trace(files["trace.csv"], identity)
    if (
        summary.get("initial_endpoint") != trace[0]
        or summary.get("primary_endpoint") != trace[-1]
    ):
        raise ValueError(f"{identity} summary endpoint differs from trace")
    if (
        resume.get("requested_additional_updates") != 16
        or resume.get("requested_final_global_step") != trace[-1]["step"]
    ):
        raise ValueError(f"{identity} resume length differs")
    if resume.get("parent_global_step") != trace[0]["step"]:
        raise ValueError(f"{identity} resume start differs")
    snapshots = [state_files(directory, row, identity, config) for row in trace]
    endpoint = state_files(directory, trace[-1], identity, config)
    if sha256(directory / "final.npz") != sha256(endpoint["npz"]) or sha256(
        directory / "final.vtu"
    ) != sha256(endpoint["vtu"]):
        raise ValueError(f"{identity} final state differs from final trace state")
    return {
        "role": role,
        "id": identity,
        "label": label,
        "directory": directory,
        "files": files,
        "config": config,
        "resume": resume,
        "summary": summary,
        "provenance": provenance,
        "trace": trace,
        "snapshots": snapshots,
    }


def same_state(left: Any, right: Any) -> bool:
    """Compare nested optimizer state, including tensors, exactly."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            same_state(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            same_state(a, b) for a, b in zip(left, right, strict=True)
        )
    return bool(left == right)


def common_fork(  # noqa: PLR0912
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Require exact common source state, optimizer settings, and physics inputs."""
    control = arms["control"]
    keys = (
        "parent_checkpoint",
        "parent_global_step",
        "controls_sha256",
        "seed_displacement_sha256",
        "old_learning_rate",
        "old_adam_eps",
        "new_learning_rate",
        "new_adam_eps",
    )
    for role, arm in arms.items():
        for key in keys:
            if arm["resume"].get(key) != control["resume"].get(key):
                raise ValueError(f"{role} does not share control fork {key}")
        for key in (
            "fixture",
            "model",
            "stress_reference_mpa",
            "stress_cap_mpa",
            "smooth_length_m",
        ):
            if arm["config"].get(key) != control["config"].get(key):
                raise ValueError(f"{role} changes physical configuration {key}")
        for key in ("sources", "inputs", "git_sha", "python", "torch", "cuda"):
            if arm["summary"].get("provenance", {}).get(key) != control["summary"].get(
                "provenance", {}
            ).get(key):
                raise ValueError(f"{role} provenance differs from common fork {key}")
        comparable = dict(arm["config"])
        control_comparable = dict(control["config"])
        for value in (comparable, control_comparable):
            del value["output_dir"]
            del value["smoothness_weight"]
        if comparable != control_comparable:
            raise ValueError(f"{role} changes non-smoothness configuration")
        if arm["config"]["learning_rate"] != arm["resume"].get(
            "new_learning_rate"
        ) or arm["config"]["adam_eps"] != arm["resume"].get("new_adam_eps"):
            raise ValueError(f"{role} config differs from resumed Adam settings")

    receipt = control["resume"]["parent_checkpoint"]
    if not isinstance(receipt, dict):
        raise TypeError("common parent checkpoint receipt must be an object")
    parent = Path(receipt.get("path", "")).resolve()
    try:
        parent.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"common parent escapes experiment root: {parent}") from error
    if not parent.is_file() or sha256(parent) != receipt.get("sha256"):
        raise ValueError("common parent checkpoint bytes differ")
    parent_state = torch.load(parent, map_location="cpu", weights_only=False)
    if (
        not isinstance(parent_state, dict)
        or int(parent_state.get("step", -1)) != control["resume"]["parent_global_step"]
    ):
        raise ValueError("common parent checkpoint step differs")
    parent_config = parent_state.get("config", {})
    if tuple(
        parent_config.get(key)
        for key in ("smoothness_weight", "rank_weight", "magnitude_weight")
    ) != (0, 0, 0):
        raise ValueError("common parent is not fit-only")
    for key in (
        "fixture",
        "model",
        "learning_rate",
        "adam_eps",
        "stress_reference_mpa",
        "stress_cap_mpa",
        "smooth_length_m",
        "projected_gradient_eta",
    ):
        if parent_config.get(key) != control["config"].get(key):
            raise ValueError(f"common parent configuration differs: {key}")
    parent_q = parent_state.get("q")
    parent_u = np.asarray(parent_state.get("u"))
    if (
        not isinstance(parent_q, torch.Tensor)
        or array_sha256(parent_q.numpy()) != control["resume"]["controls_sha256"]
        or array_sha256(parent_u) != control["resume"]["seed_displacement_sha256"]
    ):
        raise ValueError("common parent q/u differs from resume receipt")
    for role, arm in arms.items():
        initial = torch.load(
            arm["snapshots"][0]["optimizer"], map_location="cpu", weights_only=False
        )
        if not torch.equal(initial["q"], parent_q) or not same_state(
            initial["optimizer"], parent_state["optimizer"]
        ):
            raise ValueError(f"{role} did not restore common parent Adam state")
    return {
        **{key: control["resume"][key] for key in keys},
        "verified_parent": record(parent),
    }


def surface_sources(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Bind live source80/source40 and frozen helpers to recorded source bytes."""
    live = {
        "experiment/80-compare-continuations.py": ROOT
        / "src/80-compare-continuations.py",
        "experiment/40-measure-surface.py": ROOT / "src/40-measure-surface.py",
    }
    expected = arms["control"]["provenance"]["sources"]
    for role, arm in arms.items():
        for key, path in live.items():
            archived = arm["directory"] / "sources" / key
            digest = expected.get(key)
            if digest is None or sha256(path) != digest or sha256(archived) != digest:
                raise ValueError(f"{role} archived/live surface source differs: {key}")
    helper_receipt = ROOT / "data/06-helper-sources/summary.json"
    helper_summary = json.loads(helper_receipt.read_text())
    helpers = {}
    for key in (
        "face-actuation-diagnosis/src/41-surface-roughness.py",
        "face-activation-materials/src/40-audit-face-results.py",
    ):
        source = ROOT.parent / key
        expected_record = helper_summary["helpers"][key]["source"]
        if sha256(source) != expected_record["sha256"]:
            raise ValueError(f"frozen surface helper differs: {key}")
        helpers[key] = record(source)
    return {
        "source80": record(live["experiment/80-compare-continuations.py"]),
        "source40": record(live["experiment/40-measure-surface.py"]),
        "helper_receipt": record(helper_receipt),
        "helpers": helpers,
    }


def tolerances(control: dict[str, Any]) -> tuple[float, float]:
    """Return preregistered fit/motion tolerances for one control row."""
    return max(0.02, 0.005 * control["area_fit_rms_mm"]), max(
        0.02, 0.01 * control["area_motion_rms_mm"]
    )


def pairs(control: dict[str, Any], smooth: dict[str, Any]) -> list[dict[str, Any]]:
    """Inventory all positive-local-step control/smooth comparisons."""
    rows: list[dict[str, Any]] = []
    for left in control["trace"][1:]:
        for right in smooth["trace"][1:]:
            fit_tol, motion_tol = tolerances(left)
            fit_difference = abs(right["area_fit_rms_mm"] - left["area_fit_rms_mm"])
            motion_difference = abs(
                right["area_motion_rms_mm"] - left["area_motion_rms_mm"]
            )
            distance = (fit_difference / fit_tol) ** 2 + (
                motion_difference / motion_tol
            ) ** 2
            rows.append(
                {
                    "control_local_step": left["local_step"],
                    "control_global_step": left["step"],
                    "smooth_local_step": right["local_step"],
                    "smooth_global_step": right["step"],
                    "equal_step": left["local_step"] == right["local_step"],
                    "control_fit_rms_mm": left["area_fit_rms_mm"],
                    "smooth_fit_rms_mm": right["area_fit_rms_mm"],
                    "fit_difference_mm": fit_difference,
                    "fit_tolerance_mm": fit_tol,
                    "control_motion_rms_mm": left["area_motion_rms_mm"],
                    "smooth_motion_rms_mm": right["area_motion_rms_mm"],
                    "motion_difference_mm": motion_difference,
                    "motion_tolerance_mm": motion_tol,
                    "scaled_squared_distance": distance,
                    "admissible": fit_difference <= fit_tol
                    and motion_difference <= motion_tol,
                }
            )
    return rows


def selected_pair(inventory: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Apply the frozen distance and deterministic tie-breaking rule."""
    admitted = [row for row in inventory if row["admissible"]]
    return (
        min(
            admitted,
            key=lambda row: (
                row["scaled_squared_distance"],
                row["smooth_local_step"],
                row["control_local_step"],
            ),
        )
        if admitted
        else None
    )


def surface_ratio(
    source80: ModuleType, manifest: Path, document: dict[str, Any], vtu: Path
) -> dict[str, Any]:
    """Measure exact source40 normalized 5-mm full-face normal displacement HP."""
    measured = source80.surface_metrics(manifest, document, vtu)
    ratio = measured["surface"]["normalized_highpass"]["scales"]["5mm"][
        "normal_displacement"
    ]["full_face"]
    if not ratio["defined"]:
        raise ValueError("frozen 5-mm full-face high-pass ratio is undefined")
    return {"measurement": measured, "ratio": float(ratio["ratio"])}


def evaluate(
    source80: ModuleType,
    manifest: Path,
    document: dict[str, Any],
    reference: pv.UnstructuredGrid,
    control: dict[str, Any],
    smooth: dict[str, Any],
    pair: dict[str, Any] | None,
) -> dict[str, Any]:
    """Compute decision evidence for one selected admissible smoothness pair."""
    result: dict[str, Any] = {"pair": pair, "qualifies_for_rank": False, "reasons": []}
    if pair is None:
        result["reasons"].append("no admissible positive-local-step fit/motion pair")
        return result
    control_row, smooth_row = (
        control["trace"][pair["control_local_step"]],
        smooth["trace"][pair["smooth_local_step"]],
    )
    control_state = state_files(
        control["directory"], control_row, control["id"], control["config"]
    )
    smooth_state = state_files(
        smooth["directory"], smooth_row, smooth["id"], smooth["config"]
    )
    control_binding = source80.validate_endpoint_binding(
        control_state["npz"],
        control_state["vtu"],
        reference,
        control["config"]["stress_reference_mpa"],
        f"{control['id']} selected",
    )
    smooth_binding = source80.validate_endpoint_binding(
        smooth_state["npz"],
        smooth_state["vtu"],
        reference,
        smooth["config"]["stress_reference_mpa"],
        f"{smooth['id']} selected",
    )
    if (
        control_binding["step"] != control_row["step"]
        or smooth_binding["step"] != smooth_row["step"]
    ):
        raise ValueError("selected surface binding step differs")
    control_surface = surface_ratio(source80, manifest, document, control_state["vtu"])
    smooth_surface = surface_ratio(source80, manifest, document, smooth_state["vtu"])
    variation_ratio = (
        smooth_row["smoothness"] / control_row["smoothness"]
        if control_row["smoothness"] > 0
        else None
    )
    highpass_ratio = (
        smooth_surface["ratio"] / control_surface["ratio"]
        if control_surface["ratio"] > 0
        else None
    )
    checks = {
        "tensor_variation_reduced_at_least_10_percent": variation_ratio is not None
        and variation_ratio <= 0.9,
        "full_face_normal_displacement_highpass_reduced_at_least_10_percent": highpass_ratio
        is not None
        and highpass_ratio <= 0.9,
        "rank_mixing_fraction_above_0_05": smooth_row["rank_mixing_fraction"] > 0.05,
    }
    for name, passed in checks.items():
        if not passed:
            result["reasons"].append(name)
    result.update(
        {
            "control_checkpoint": {
                "npz": record(control_state["npz"]),
                "vtu": record(control_state["vtu"]),
                "optimizer": record(control_state["optimizer"]),
                "trace": control_row,
                "binding": control_binding,
            },
            "smooth_checkpoint": {
                "npz": record(smooth_state["npz"]),
                "vtu": record(smooth_state["vtu"]),
                "optimizer": record(smooth_state["optimizer"]),
                "trace": smooth_row,
                "binding": smooth_binding,
            },
            "tensor_variation": {
                "definition": "source74 normalized-coordinate geometric smoothness before applying its weight",
                "control": control_row["smoothness"],
                "smooth": smooth_row["smoothness"],
                "smooth_over_control": variation_ratio,
            },
            "full_face_normal_displacement_highpass": {
                "definition": "source40/source80 frozen 5-mm normalized normal-displacement high-pass ratio",
                "control": control_surface,
                "smooth": smooth_surface,
                "smooth_over_control": highpass_ratio,
            },
            "rank_mixing_fraction": smooth_row["rank_mixing_fraction"],
            "checks": checks,
            "qualifies_for_rank": all(checks.values()),
        }
    )
    return result


def main(cfg: Config) -> None:
    """Validate all completed arms, select any rank-eligible smooth checkpoint."""
    global COMPLETED  # noqa: PLW0603
    manifest_path = cfg.manifest.resolve()
    document = json.loads(manifest_path.read_text())
    if document.get("schema_version") != 1 or not isinstance(
        document.get("title"), str
    ):
        raise ValueError("regularization manifest schema/title differs")
    if (
        document.get("surface_protocol", {}).get("scales_mm") != [2, 5, 10]
        or document["surface_protocol"].get("primary_scale_mm") != 5
        or document["surface_protocol"].get("mouth_radius_mm") != 10
    ):
        raise ValueError(
            "surface protocol must retain frozen [2,5,10], primary5, mouth10"
        )
    output = (
        cfg.output_dir
        or resolve(manifest_path, document.get("output_dir"), "output_dir")
    ).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    cherries.log_input(manifest_path)
    raw_arms = document.get("arms")
    if not isinstance(raw_arms, dict) or set(raw_arms) != set(WEIGHTS):
        raise ValueError("manifest must declare control, weak, and full arms")
    arms = {role: load_arm(manifest_path, role, raw_arms[role]) for role in WEIGHTS}
    if len({arm["id"] for arm in arms.values()}) != 3:
        raise ValueError("arm IDs must be unique")
    fork = common_fork(arms)
    loaded_surface_sources = surface_sources(arms)
    source80 = load_source80()
    reference = pv.read(
        source80.resolve(manifest_path, document["surface_protocol"]["fixture_vtu"])
    )
    if not isinstance(reference, pv.UnstructuredGrid):
        raise TypeError("surface fixture must be an UnstructuredGrid")
    decisions: dict[str, Any] = {}
    inventories: dict[str, list[dict[str, Any]]] = {}
    for role in ("weak", "full"):
        inventory = pairs(arms["control"], arms[role])
        inventories[role] = inventory
        decisions[role] = evaluate(
            source80,
            manifest_path,
            document,
            reference,
            arms["control"],
            arms[role],
            selected_pair(inventory),
        )
        with (output / f"{role}-pair-inventory.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(inventory[0]))
            writer.writeheader()
            writer.writerows(inventory)
    selected_role = (
        "weak"
        if decisions["weak"]["qualifies_for_rank"]
        else "full"
        if decisions["full"]["qualifies_for_rank"]
        else None
    )
    decision = {
        "rank_branch": "eligible" if selected_role else "skipped",
        "selected_smoothness_role": selected_role,
        "selected": decisions[selected_role] if selected_role else None,
        "skip_reasons": []
        if selected_role
        else {role: decisions[role]["reasons"] for role in ("weak", "full")},
    }
    summary = {
        "schema_version": 1,
        "status": "completed_regularization_match",
        "scope": "completed source74 states only; no solve, checkpoint mutation, or rank run",
        "manifest": record(manifest_path),
        "common_fork": fork,
        "surface_sources": loaded_surface_sources,
        "arms": {
            role: {
                "id": arm["id"],
                "label": arm["label"],
                "summary": record(arm["files"]["summary.json"]),
                "trace": record(arm["files"]["trace.csv"]),
                "config": record(arm["files"]["config.json"]),
                "resume": record(arm["files"]["resume.json"]),
            }
            for role, arm in arms.items()
        },
        "matching_rule": "positive local steps; fit <= max(0.02 mm, 0.5% control), motion <= max(0.02 mm, 1% control); min scaled squared distance, then earlier smooth/control",
        "pair_inventory": {
            role: record(output / f"{role}-pair-inventory.csv") for role in inventories
        },
        "decisions": decisions,
        "rank_decision": decision,
    }
    write_json(output / "summary.json", summary)
    for path in [
        output / "summary.json",
        *(output / f"{role}-pair-inventory.csv" for role in inventories),
    ]:
        cherries.log_output(path)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
