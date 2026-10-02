# ruff: noqa: C901, EM101, EM102, PLR0912, TRY003
"""Collect inverse endpoints, diagnostic exports, and CPU-audit receipts."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
REPO_ROOT = HERE.parents[5]
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    """Explicit result/audit inputs and one comparison-table output."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)

    result_dirs: str
    audit_dirs: str = ""
    output_dir: Path = cherries.output("45-comparisons", mkdir=True)
    partial: bool = False
    partial_reason: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def parse_paths(value: str, *, label: str, required: bool) -> list[Path]:
    paths = [Path(item.strip()).resolve() for item in value.split(",") if item.strip()]
    if required and not paths:
        raise ValueError(f"{label} must name at least one directory")
    if len(paths) != len(set(paths)):
        raise ValueError(f"{label} must not contain duplicate directories")
    for path in paths:
        if not path.is_dir():
            raise NotADirectoryError(path)
    return paths


def require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a JSON object")
    return value


def require_finite_number(value: Any, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def select_present(source: Mapping[str, Any], keys: Sequence[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def file_receipt(path: Path, *, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "path": str(path),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def stored_path(value: Any) -> Path:
    """Resolve paths written by experiment stages from their experiment root."""
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (EXPERIMENT / path).resolve()


ENDPOINT_KEYS = (
    "step",
    "objective",
    "magnitude",
    "smoothness",
    "fit_rms_over_D",
    "fit_rms_mm",
    "motion_rms_mm",
    "target_projection_amplitude",
    "target_projection_residual_over_D",
    "detF_min",
    "detF_max",
    "inverted_tets",
    "detG_min",
    "activation_eigen_min",
    "activation_eigen_max",
    "activation_det_min",
    "activation_det_max",
    "forward_steps",
    "forward_grad_norm",
)


def validate_trace_endpoint(path: Path, endpoint: Mapping[str, Any]) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"trace has no endpoint row: {path}")
    final_row = rows[-1]
    for key in ENDPOINT_KEYS:
        if key not in endpoint:
            continue
        if key not in final_row:
            raise ValueError(f"trace endpoint lacks summary field {key!r}: {path}")
        expected = require_finite_number(endpoint[key], label=f"summary.final.{key}")
        observed = float(final_row[key])
        if isinstance(expected, int) and not isinstance(expected, bool):
            if observed != expected:
                raise ValueError(
                    f"trace/summary mismatch for {key}: {observed} != {expected}"
                )
        elif not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-15):
            raise ValueError(
                f"trace/summary mismatch for {key}: {observed} != {expected}"
            )


def fd_receipt(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"finite-difference receipt must be a nonempty list: {path}")
    for index, item in enumerate(payload):
        row = require_mapping(item, label=f"gradient-audit[{index}]")
        for key in ("epsilon", "analytic", "finite_difference", "relative_error"):
            require_finite_number(row[key], label=f"gradient-audit[{index}].{key}")
    return {
        "input": file_receipt(path, role="finite_difference_receipt"),
        "records": payload,
    }


def q_receipt(path: Path) -> tuple[int, list[int]]:
    with np.load(path, allow_pickle=False) as arrays:
        if "q" not in arrays:
            raise ValueError(f"final.npz lacks q: {path}")
        shape = list(arrays["q"].shape)
    return int(np.prod(shape)), shape


def parameterization_receipt(
    summary: Mapping[str, Any], parameter_count: int, q_shape: Sequence[int]
) -> dict[str, Any]:
    declared_count = summary.get("controls")
    if declared_count is not None and parameter_count != declared_count:
        raise ValueError(
            f"summary/final.npz parameter count mismatch: {declared_count} != {parameter_count}"
        )
    explicit = summary.get("parameterization")
    if explicit is None:
        stored_parameters = parameter_count
        free_parameters = parameter_count
        source = "final.npz/q shape; all stored coordinates are free"
        details: dict[str, Any] = {}
    else:
        details = dict(require_mapping(explicit, label="summary.parameterization"))
        stored_parameters = details.get("stored_parameters")
        free_parameters = details.get("free_parameters")
        if not isinstance(stored_parameters, int) or isinstance(
            stored_parameters, bool
        ):
            raise TypeError(
                "summary.parameterization.stored_parameters must be an integer"
            )
        if not isinstance(free_parameters, int) or isinstance(free_parameters, bool):
            raise TypeError(
                "summary.parameterization.free_parameters must be an integer"
            )
        if stored_parameters != parameter_count:
            raise ValueError(
                "summary.parameterization.stored_parameters differs from final q"
            )
        if not 0 < free_parameters <= stored_parameters:
            raise ValueError("free_parameters must lie in (0, stored_parameters]")
        declared_shape = details.get("stored_shape")
        if declared_shape is not None and list(declared_shape) != list(q_shape):
            raise ValueError(
                "summary.parameterization.stored_shape differs from final q"
            )
        source = "summary.parameterization, verified against final.npz/q"
    return {
        **select_present(summary, ("controls", "regions", "n_active", "n_tets")),
        "q_shape": list(q_shape),
        "stored_parameters": stored_parameters,
        "free_parameters": free_parameters,
        "parameter_count_verified": parameter_count,
        "count_source": source,
        **({"details": details} if details else {}),
    }


def execution_receipt(
    config: Mapping[str, Any], status: str, endpoint: Mapping[str, Any]
) -> dict[str, Any]:
    requested_steps = config.get("steps")
    endpoint_step = endpoint.get("step")
    if requested_steps == 0:
        return {
            "classification": "fixed_control_replay",
            "optimization_status": "not_performed",
            "requested_optimizer_steps": 0,
            "endpoint_step": endpoint_step,
            "source_stop_status": status,
            "interpretation": "fixed controls were re-equilibrated; no optimizer update was requested",
        }
    return {
        "classification": "optimization_run",
        "optimization_status": status,
        "requested_optimizer_steps": requested_steps,
        "endpoint_step": endpoint_step,
    }


def collect_result(path: Path, manifest: list[dict[str, Any]]) -> dict[str, Any]:
    summary_path = path / "summary.json"
    summary = require_mapping(load_json(summary_path), label=f"{summary_path}")
    if summary.get("status") == "diagnostic_stop_not_converged":
        return collect_diagnostic_result(path, summary, manifest)
    config = require_mapping(summary.get("config"), label=f"{summary_path}: config")
    materials = require_mapping(
        summary.get("materials"), label=f"{summary_path}: materials"
    )
    endpoint = require_mapping(summary.get("final"), label=f"{summary_path}: final")
    status = summary.get("status")
    if not isinstance(status, str) or not status:
        raise ValueError(f"{summary_path}: status must be a nonempty string")
    for key in ENDPOINT_KEYS:
        if key in endpoint:
            require_finite_number(endpoint[key], label=f"{summary_path}: final.{key}")

    trace_path = path / "trace.csv"
    validate_trace_endpoint(trace_path, endpoint)
    final_npz = path / "final.npz"
    final_vtu = path / "final.vtu"
    if not final_npz.is_file() or not final_vtu.is_file():
        raise FileNotFoundError(f"completed endpoint files are missing in {path}")
    parameter_count, q_shape = q_receipt(final_npz)

    inputs = [
        file_receipt(summary_path, role="inverse_summary"),
        file_receipt(trace_path, role="inverse_trace"),
        file_receipt(final_npz, role="inverse_endpoint_arrays"),
        file_receipt(final_vtu, role="inverse_endpoint_mesh"),
    ]
    for name, role in (
        ("config.json", "inverse_config"),
        ("provenance.json", "inverse_provenance"),
        ("reset.npz", "inverse_branch_reset"),
    ):
        candidate = path / name
        if candidate.is_file():
            inputs.append(file_receipt(candidate, role=role))
    gradient_path = path / "gradient-audit.json"
    gradient = fd_receipt(gradient_path) if gradient_path.is_file() else None
    if gradient is not None:
        inputs.append(gradient["input"])
    manifest.extend(inputs)

    record: dict[str, Any] = {
        "case_id": path.name,
        "result_dir": str(path),
        "stop_status": status,
        "parameterization": {
            **select_present(config, ("method",)),
            **parameterization_receipt(summary, parameter_count, q_shape),
        },
        "execution": execution_receipt(config, status, endpoint),
        "config": dict(config),
        "materials": dict(materials),
        "endpoint": select_present(endpoint, ENDPOINT_KEYS),
        "inputs": inputs,
    }
    if "kkt" in endpoint:
        projected_gradient = {
            "value": require_finite_number(
                endpoint["kkt"], label=f"{summary_path}: final.kkt"
            ),
            "source_field": "summary.final.kkt",
            "scope": "activation parameter bounds only; excludes detF admissibility",
        }
        if "kkt_tol" in config:
            projected_gradient["tolerance"] = require_finite_number(
                config["kkt_tol"], label=f"{summary_path}: config.kkt_tol"
            )
        record["activation_only_projected_gradient"] = projected_gradient
    branch = select_present(
        summary,
        (
            "accepted_trajectory_min_detF",
            "rest_reset_difference_over_D",
            "reset_forward",
        ),
    )
    if branch:
        record["branch_reset"] = {"status": "performed", **branch}
    else:
        record["branch_reset"] = {
            "status": "missing",
            "rest_reset_difference_over_D": "missing",
        }
    if gradient is not None:
        record["finite_difference_receipt"] = gradient
    else:
        record["finite_difference_receipt"] = {"status": "missing"}
    return record


def endpoint_from_diagnostic_trace(source: Mapping[str, Any]) -> dict[str, Any]:
    integer_keys = {"step", "inverted_tets", "forward_steps"}
    endpoint: dict[str, Any] = {}
    for key in ENDPOINT_KEYS:
        if key not in source:
            continue
        raw = source[key]
        try:
            value = int(raw) if key in integer_keys else float(raw)
        except (TypeError, ValueError) as error:
            raise TypeError(f"diagnostic trace field {key!r} is not numeric") from error
        endpoint[key] = require_finite_number(
            value, label=f"diagnostic trace endpoint.{key}"
        )
    return endpoint


def collect_diagnostic_result(
    path: Path,
    summary: Mapping[str, Any],
    manifest: list[dict[str, Any]],
) -> dict[str, Any]:
    """Collect a preserved accepted checkpoint without a convergence claim."""
    summary_path = path / "summary.json"
    if summary.get("optimization_complete") is not False:
        raise ValueError("diagnostic export must state optimization_complete=false")
    if summary.get("physics_resolve_performed") is not False:
        raise ValueError("diagnostic export must state physics_resolve_performed=false")
    if summary.get("fresh_rest_branch_check_performed") is not False:
        raise ValueError(
            "diagnostic export must state fresh_rest_branch_check_performed=false"
        )
    if summary.get("accepted_equilibrium") is not True:
        raise ValueError("diagnostic export must identify an accepted equilibrium")

    config = require_mapping(
        summary.get("source_config"), label=f"{summary_path}: source_config"
    )
    materials = require_mapping(
        summary.get("materials"), label=f"{summary_path}: materials"
    )
    evidence = require_mapping(
        summary.get("evidence"), label=f"{summary_path}: evidence"
    )
    trace_evidence = require_mapping(
        evidence.get("trace"), label=f"{summary_path}: evidence.trace"
    )
    trace_row = require_mapping(
        trace_evidence.get("row"), label=f"{summary_path}: evidence.trace.row"
    )
    endpoint = endpoint_from_diagnostic_trace(trace_row)
    checkpoint = require_mapping(
        summary.get("checkpoint"), label=f"{summary_path}: checkpoint"
    )
    if endpoint.get("step") != checkpoint.get("step"):
        raise ValueError("diagnostic trace and checkpoint steps differ")

    trace_path = path / "source-run/trace.csv"
    validate_trace_endpoint(trace_path, endpoint)
    final_npz = path / "final.npz"
    final_vtu = path / "final.vtu"
    if not final_npz.is_file() or not final_vtu.is_file():
        raise FileNotFoundError(f"diagnostic endpoint files are missing in {path}")
    parameter_count, q_shape = q_receipt(final_npz)
    final_npz_hash = sha256(final_npz)
    if checkpoint.get("final_npz_sha256") != final_npz_hash:
        raise ValueError("diagnostic checkpoint hash differs from final.npz")

    inputs = [
        file_receipt(summary_path, role="diagnostic_summary"),
        file_receipt(trace_path, role="diagnostic_source_trace"),
        file_receipt(final_npz, role="inverse_endpoint_arrays"),
        file_receipt(final_vtu, role="inverse_endpoint_mesh"),
    ]
    for name, role in (
        ("artifact-manifest.json", "diagnostic_artifact_manifest"),
        ("source-run/config.json", "diagnostic_source_config"),
        ("source-run/provenance.json", "diagnostic_source_provenance"),
        ("source-run/diagnostic-stop.json", "diagnostic_stop_receipt"),
    ):
        candidate = path / name
        if candidate.is_file():
            inputs.append(file_receipt(candidate, role=role))
    manifest.extend(inputs)

    status = str(summary["status"])
    stop = require_mapping(summary.get("stop"), label=f"{summary_path}: stop")
    if stop.get("kkt_criterion_met") is not False:
        raise ValueError("diagnostic stop must preserve unmet KKT criterion")
    return {
        "case_id": path.name,
        "result_dir": str(path),
        "stop_status": status,
        "parameterization": {
            **select_present(config, ("method",)),
            **parameterization_receipt(summary, parameter_count, q_shape),
        },
        "execution": {
            "classification": "diagnostic_stop_export",
            "optimization_status": status,
            "optimization_complete": False,
            "accepted_equilibrium": True,
            "physics_resolve_status": "not_performed",
            "requested_optimizer_steps": config.get("steps"),
            "endpoint_step": endpoint.get("step"),
            "interpretation": summary.get("description"),
        },
        "config": dict(config),
        "materials": dict(materials),
        "endpoint": endpoint,
        "activation_only_projected_gradient": {
            "value": require_finite_number(stop["kkt"], label="stop.kkt"),
            "source_field": "summary.stop.kkt",
            "scope": "activation parameter bounds only; excludes detF admissibility",
            "tolerance": require_finite_number(
                stop["kkt_tolerance"], label="stop.kkt_tolerance"
            ),
            "criterion_met": False,
        },
        "branch_reset": {
            "status": "not_performed",
            "fresh_rest_branch_check_performed": False,
            "rest_reset_difference_over_D": "not_performed",
        },
        "finite_difference_receipt": {"status": "missing"},
        "diagnostic_stop": {
            "stop": dict(stop),
            "checkpoint": dict(checkpoint),
            "independent_recomputation": summary.get("independent_recomputation"),
        },
        "inputs": inputs,
    }


INTERSECTION_DOMAIN_KEYS = (
    "vertices",
    "edges",
    "triangles",
    "broad_phase_candidates",
    "rest_edge_face_hits",
    "deformed_edge_face_hits",
    "new_edge_face_hits",
    "resolved_edge_face_hits",
    "rest_has_intersections",
    "deformed_has_intersections",
)


def extract_intersections(source: Mapping[str, Any]) -> dict[str, Any]:
    output = select_present(
        source, ("method", "temporal_scope", "adjacency_policy", "ipctk_version")
    )
    domains = source.get("domains")
    if isinstance(domains, Mapping):
        output["domains"] = {
            str(name): select_present(
                require_mapping(value, label=f"intersection domain {name}"),
                INTERSECTION_DOMAIN_KEYS,
            )
            for name, value in domains.items()
        }
    return output


def extract_high_pass(surface: Mapping[str, Any]) -> dict[str, Any]:
    output = select_present(surface, ("interpretation_limit", "operator"))
    scales = surface.get("scales")
    if not isinstance(scales, Mapping):
        return output
    selected_scales: dict[str, Any] = {}
    for scale, value in scales.items():
        raw = require_mapping(value, label=f"surface.scales.{scale}")
        selected = select_present(
            raw,
            (
                "heat_time_m2",
                "normal_displacement_highpass",
                "normal_residual_highpass",
            ),
        )
        if selected:
            selected_scales[str(scale)] = selected
    if selected_scales:
        output["scales"] = selected_scales
    return output


def collect_audits(
    paths: Sequence[Path], manifest: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    by_result: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        summary_path = path / "summary.json"
        summary = require_mapping(load_json(summary_path), label=f"{summary_path}")
        rows = summary.get("results")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"audit summary has no result records: {summary_path}")
        summary_receipt = file_receipt(summary_path, role="audit_summary")
        manifest.append(summary_receipt)
        for index, item in enumerate(rows):
            raw = require_mapping(item, label=f"{summary_path}: results[{index}]")
            source_input = require_mapping(
                raw.get("input"), label=f"{summary_path}: results[{index}].input"
            )
            result_dir = stored_path(source_input["result_dir"])
            endpoint = stored_path(source_input["endpoint"])
            endpoint_receipt = file_receipt(endpoint, role="audited_endpoint_mesh")
            declared_hash = source_input.get("endpoint_sha256")
            if declared_hash != endpoint_receipt["sha256"]:
                raise ValueError(f"audit endpoint hash mismatch: {endpoint}")
            manifest.append(endpoint_receipt)
            metrics_path = raw.get("metrics_path")
            metric_receipt = None
            if metrics_path is not None:
                metric_receipt = file_receipt(
                    stored_path(metrics_path), role="audit_case_metrics"
                )
                manifest.append(metric_receipt)
            audit: dict[str, Any] = {
                "audit_dir": str(path),
                "audit_summary": summary_receipt,
                "endpoint": endpoint_receipt,
                "endpoint_name": endpoint.name,
                "saved_inverse_summary_sha256": source_input.get(
                    "saved_summary_sha256"
                ),
            }
            intersections = raw.get("intersections")
            if isinstance(intersections, Mapping):
                audit["intersections"] = extract_intersections(intersections)
            surface = raw.get("surface")
            if isinstance(surface, Mapping):
                high_pass = extract_high_pass(surface)
                if high_pass:
                    audit["high_pass"] = high_pass
            if metric_receipt is not None:
                audit["metrics_input"] = metric_receipt
            by_result.setdefault(str(result_dir), []).append(audit)
    return by_result


def flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, sort_keys=True, separators=(",", ":"))
    elif value is not None:
        output[prefix] = value


def csv_rows(
    records: Sequence[Mapping[str, Any]], coverage: Mapping[str, Any]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        base_source = select_present(
            record,
            (
                "case_id",
                "result_dir",
                "stop_status",
                "parameterization",
                "execution",
                "config",
                "materials",
                "endpoint",
                "activation_only_projected_gradient",
                "branch_reset",
                "audit_status",
            ),
        )
        base_source["coverage"] = coverage
        fd = record.get("finite_difference_receipt")
        if isinstance(fd, Mapping):
            base_source["finite_difference_status"] = fd.get("status", "performed")
            fd_records = fd.get("records")
            if isinstance(fd_records, list):
                errors = [
                    require_finite_number(row["relative_error"], label="FD error")
                    for row in fd_records
                ]
                base_source["finite_difference"] = {
                    "receipt_sha256": require_mapping(fd["input"], label="FD input")[
                        "sha256"
                    ],
                    "records": len(fd_records),
                    "best_relative_error": min(errors),
                    "worst_relative_error": max(errors),
                }
        audits = record.get("audits")
        audit_rows = audits if isinstance(audits, list) and audits else [None]
        for audit in audit_rows:
            source = dict(base_source)
            source["audit_present"] = audit is not None
            if audit is not None:
                source["audit"] = audit
            row: dict[str, Any] = {}
            flatten("", source, row)
            rows.append(row)
    return rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(cfg: Config) -> None:
    if cfg.partial and not cfg.partial_reason.strip():
        raise ValueError("partial_reason is required when partial is true")
    if not cfg.partial and cfg.partial_reason.strip():
        raise ValueError("partial_reason requires partial=true")
    result_dirs = parse_paths(cfg.result_dirs, label="result_dirs", required=True)
    audit_dirs = parse_paths(cfg.audit_dirs, label="audit_dirs", required=False)
    output_dir = cfg.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output_dir}")

    source_dir = output_dir / "sources"
    source_dir.mkdir()
    source_paths = (Path(__file__), HERE / "experiment_profile.py")
    for source in source_paths:
        shutil.copy2(source, source_dir / source.name)
    manifest = [
        file_receipt(source, role="collector_source") for source in source_paths
    ]

    records = [collect_result(path, manifest) for path in result_dirs]
    audits = collect_audits(audit_dirs, manifest)
    known_results = {record["result_dir"] for record in records}
    unknown_audits = sorted(set(audits) - known_results)
    if unknown_audits:
        raise ValueError(
            f"audits reference result directories not requested: {unknown_audits}"
        )
    for record in records:
        attached = audits.get(record["result_dir"], [])
        endpoint_hash = next(
            item["sha256"]
            for item in record["inputs"]
            if item["role"] == "inverse_endpoint_mesh"
        )
        summary_hash = next(
            item["sha256"]
            for item in record["inputs"]
            if item["role"] in {"inverse_summary", "diagnostic_summary"}
        )
        for audit in attached:
            audit["matches_inverse_endpoint"] = (
                audit["endpoint"]["sha256"] == endpoint_hash
            )
            saved_hash = audit.get("saved_inverse_summary_sha256")
            if saved_hash is not None:
                audit["saved_inverse_summary_matches_current"] = (
                    saved_hash == summary_hash
                )
        if attached:
            record["audits"] = attached
            record["audit_status"] = "performed"
        else:
            record["audit_status"] = "missing"

    classifications: dict[str, int] = {}
    for record in records:
        execution = require_mapping(record["execution"], label="record.execution")
        classification = str(execution["classification"])
        classifications[classification] = classifications.get(classification, 0) + 1
    coverage = {
        "partial": cfg.partial,
        **({"reason": cfg.partial_reason.strip()} if cfg.partial else {}),
        "inverse_results": len(records),
        "audit_records": sum(len(value) for value in audits.values()),
        "result_classifications": classifications,
        "policy": "explicit inputs only; no method ranking",
    }
    comparison = {
        "schema_version": 2,
        "coverage": coverage,
        "git_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "records": records,
        "sha256_manifest": manifest,
    }
    json_path = output_dir / "comparisons.json"
    csv_path = output_dir / "comparisons.csv"
    write_json(json_path, comparison)
    rows = csv_rows(records, coverage)
    write_csv(csv_path, rows)
    LOG.info(
        "Wrote %d explicit inverse results and %d audit records to %s",
        len(records),
        sum(len(value) for value in audits.values()),
        output_dir,
    )
    cherries.log_metrics(
        {
            "comparisons/inverse_results": len(records),
            "comparisons/audit_records": sum(len(value) for value in audits.values()),
            "comparisons/partial": int(cfg.partial),
        }
    )


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
