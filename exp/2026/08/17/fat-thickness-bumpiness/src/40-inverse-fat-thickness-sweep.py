from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import pydantic_settings as ps
import pyvista as pv
from _common import case_lookup, resolve_recorded_path, slugify, toy

from liblaf import cherries

logger = logging.getLogger(__name__)

ZERO_ACTIVATION_INV = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
EXPECTED_LABELS = ("thin", "current", "thick")
TRACE_FINITE_KEYS = (
    "loss/total",
    "loss/data",
    "target/error_rms",
    "target/error_max",
    "activation_inv/rms",
    "activation_inv/max_abs",
    "grad/norm",
    "forward/grad_norm",
    "forward/relative_grad_norm",
    "adjoint/absolute_residual",
    "adjoint/relative_residual",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_manifest: Path = cherries.input("10-prepare-manifest.json")
    output_manifest: Path = cherries.output("40-inverse-manifest.json", mkdir=True)
    output_table: Path = cherries.output("40-inverse-table.md", mkdir=True)

    labels: tuple[str, ...] = EXPECTED_LABELS
    inverse_lr: float = 0.03
    inverse_max_steps: int = 200
    inverse_loss_min_delta: float = 1.0e-8
    require_convergence: bool = True


def validate_config(cfg: Config) -> tuple[str, ...]:
    labels = tuple(slugify(label) for label in cfg.labels)
    if not labels:
        msg = "at least one inverse case is required"
        raise ValueError(msg)
    if len(labels) != len(set(labels)):
        msg = f"inverse case labels are not unique: {labels}"
        raise ValueError(msg)
    unknown = sorted(set(labels) - set(EXPECTED_LABELS))
    if unknown:
        msg = f"unknown inverse case labels {unknown}; expected {EXPECTED_LABELS}"
        raise ValueError(msg)
    if cfg.inverse_lr <= 0.0 or not math.isfinite(cfg.inverse_lr):
        msg = f"inverse_lr must be finite and positive, got {cfg.inverse_lr}"
        raise ValueError(msg)
    if cfg.inverse_max_steps < 0:
        msg = f"inverse_max_steps must be non-negative, got {cfg.inverse_max_steps}"
        raise ValueError(msg)
    if cfg.inverse_loss_min_delta < 0.0 or not math.isfinite(
        cfg.inverse_loss_min_delta
    ):
        msg = (
            "inverse_loss_min_delta must be finite and non-negative, got "
            f"{cfg.inverse_loss_min_delta}"
        )
        raise ValueError(msg)
    return labels


def make_case(
    label: str, mesh: pv.UnstructuredGrid
) -> tuple[toy.ToyCase, toy.ResolutionSpec]:
    mesh_resolution = toy.mesh_resolution(mesh)
    resolution = toy.ResolutionSpec(
        name=f"{label}-{mesh_resolution.name}",
        lr=mesh_resolution.lr,
    )
    variant = toy.LossVariant(
        name="l2",
        skin_energy=False,
        skin_prestrain=False,
        activation_mode="per-tet",
    )
    case = toy.ToyCase(
        resolution=resolution,
        mode="squash",
        variant=variant,
        target_y=-toy.SQUASH_TARGET_MAGNITUDE,
    )
    return case, mesh_resolution


def case_config(
    *,
    cfg: Config,
    mesh_path: Path,
    case: toy.ToyCase,
    case_dir: Path,
) -> toy.InverseConfig:
    stem = case.stem
    return toy.InverseConfig(
        _cli_parse_args=False,
        input_mesh=mesh_path,
        output_summary=case_dir / f"{stem}-summary.json",
        output_table=cfg.output_table,
        mode="squash",
        loss_variant="l2",
        skin_energy_enabled=False,
        skin_prestrain_enabled=False,
        activation_mode="per-tet",
        inverse_lr=cfg.inverse_lr,
        inverse_max_steps=cfg.inverse_max_steps,
        inverse_loss_min_delta=cfg.inverse_loss_min_delta,
        initial_activation_inv=ZERO_ACTIVATION_INV,
        require_convergence=False,
    )


def artifact_paths(case: toy.ToyCase, case_dir: Path) -> dict[str, Path]:
    stem = case.stem
    return {
        "target/path": case_dir / f"{stem}-target.vtu",
        "result/path": case_dir / f"{stem}.vtu",
        "history/path": case_dir / f"{stem}-steps.vtkhdf",
        "summary/path": case_dir / f"{stem}-summary.json",
    }


def finite(value: Any) -> bool:
    return isinstance(value, int | float) and math.isfinite(float(value))


def validate_activation(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if row.get("activation/mode") != "per-tet" or bool(
        row.get("activation/shared", True)
    ):
        errors.append("activation is not independent per-tet")

    n_active = row.get("n_active_tets")
    n_dofs = row.get("n_activation_parameter_dofs")
    if not isinstance(n_active, int) or not isinstance(n_dofs, int):
        errors.append("activation counts are missing or non-integral")
    elif n_dofs != 6 * n_active:
        errors.append(f"activation DOFs are {n_dofs}, expected {6 * n_active}")

    for name, expected in zip(
        ("x", "y", "z", "xy", "yz", "xz"),
        ZERO_ACTIVATION_INV,
        strict=True,
    ):
        value = row.get(f"inverse/initial_activation_inv/{name}")
        if value != expected:
            errors.append(f"initial activation {name} is {value!r}, expected zero")
    return errors


def validate_trace(row: dict[str, Any]) -> list[str]:
    trace = row.get("trace")
    if not isinstance(trace, list) or not trace:
        return ["inverse trace is missing or empty"]
    errors: list[str] = []
    bad_forward_steps = [
        int(step.get("step", index))
        for index, step in enumerate(trace)
        if step.get("forward/success") is not True
    ]
    if bad_forward_steps:
        errors.append(f"forward solve failed at steps {bad_forward_steps}")
    bad_adjoint_steps = [
        int(step.get("step", index))
        for index, step in enumerate(trace)
        if step.get("adjoint/success") is not True
    ]
    if bad_adjoint_steps:
        errors.append(f"adjoint failed at steps {bad_adjoint_steps}")
    nonfinite_steps = [
        (int(step.get("step", index)), key)
        for index, step in enumerate(trace)
        for key in TRACE_FINITE_KEYS
        if not finite(step.get(key))
    ]
    if nonfinite_steps:
        preview = ", ".join(f"step {step} {key}" for step, key in nonfinite_steps[:8])
        suffix = " ..." if len(nonfinite_steps) > 8 else ""
        errors.append(f"non-finite trace metrics: {preview}{suffix}")
    return errors


def add_solver_status(row: dict[str, Any]) -> None:
    trace = row.get("trace")
    if not isinstance(trace, list) or not trace:
        row["forward/success"] = False
        row["adjoint/success"] = False
        return
    row["forward/success"] = all(step.get("forward/success") is True for step in trace)
    row["adjoint/success"] = all(step.get("adjoint/success") is True for step in trace)


def validate_artifacts(
    row: dict[str, Any], case: toy.ToyCase, case_dir: Path
) -> list[str]:
    errors: list[str] = []
    for key, expected_path in artifact_paths(case, case_dir).items():
        recorded = Path(str(row.get(key, "")))
        if recorded.resolve() != expected_path.resolve():
            errors.append(f"{key} records {recorded}, expected {expected_path}")
        elif not recorded.is_file():
            errors.append(f"{key} artifact does not exist: {recorded}")
    return errors


def validate_result(
    row: dict[str, Any],
    *,
    label: str,
    case: toy.ToyCase,
    case_dir: Path,
) -> list[str]:
    errors: list[str] = []
    if row.get("case") != case.stem:
        errors.append(f"case stem is {row.get('case')!r}, expected {case.stem!r}")
    errors.extend(validate_activation(row))
    errors.extend(validate_trace(row))
    errors.extend(
        f"summary metric {key} is non-finite"
        for key in (
            "best/loss",
            "best/error_rms",
            "best/error_rms_fraction_of_target",
            "activation_inv/rms",
            "activation_inv/max_abs",
        )
        if not finite(row.get(key))
    )
    errors.extend(validate_artifacts(row, case, case_dir))

    if errors:
        logger.error("Validation failed for %s: %s", label, "; ".join(errors))
    return errors


def table_value(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else str(value)
    return str(value)


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| label | fat min | fat center | active tets | 6-DoF params | evaluations | stop | converged | best loss | error RMS | error/target | status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        values = (
            row.get("label", ""),
            row.get("fat_thickness/min", ""),
            row.get("fat_thickness/center", ""),
            row.get("n_active_tets", ""),
            row.get("n_activation_parameter_dofs", ""),
            row.get("inverse/evaluations", ""),
            row.get("inverse/stop_reason", row.get("error/type", "")),
            row.get("inverse/converged", False),
            row.get("best/loss", ""),
            row.get("best/error_rms", ""),
            row.get("best/error_rms_fraction_of_target", ""),
            row.get("status", ""),
        )
        escaped = [table_value(value).replace("|", "\\|") for value in values]
        lines.append("| " + " | ".join(escaped) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(
    cfg: Config,
    *,
    labels: tuple[str, ...],
    rows: list[dict[str, Any]],
    hard_failures: list[str],
    convergence_failures: list[str],
) -> None:
    manifest = {
        "schema_version": 1,
        "kind": "fat-thickness-independent-inverse-results",
        "source_manifest": str(cfg.input_manifest),
        "setup": "no-skin-l2-per-tet-6dof",
        "labels": list(labels),
        "initial_activation_inv": list(ZERO_ACTIVATION_INV),
        "fresh_optimizer_per_case": True,
        "inverse/lr": cfg.inverse_lr,
        "inverse/max_steps": cfg.inverse_max_steps,
        "inverse/loss_min_delta": cfg.inverse_loss_min_delta,
        "inverse/patience": toy.INVERSE_PATIENCE,
        "require_convergence": cfg.require_convergence,
        "complete": not hard_failures
        and (not convergence_failures or not cfg.require_convergence),
        "hard_failures": hard_failures,
        "convergence_failures": convergence_failures,
        "cases": rows,
    }
    cfg.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_manifest)
    logger.info("Wrote %s", cfg.output_table)


def main(cfg: Config) -> None:
    labels = validate_config(cfg)
    manifest = json.loads(cfg.input_manifest.read_text(encoding="utf-8"))
    source_cases = case_lookup(manifest)
    missing_labels = [label for label in labels if label not in source_cases]
    if missing_labels:
        msg = f"prepare manifest is missing requested cases: {missing_labels}"
        raise ValueError(msg)

    cfg.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    cfg.output_table.parent.mkdir(parents=True, exist_ok=True)
    output_root = cfg.output_manifest.parent / "40-inverse" / "no-skin"
    rows: list[dict[str, Any]] = []
    hard_failures: list[str] = []
    convergence_failures: list[str] = []

    toy.configure_runtime()
    for label in labels:
        source_case = source_cases[label]
        mesh_path = resolve_recorded_path(
            cfg.input_manifest, str(source_case.get("mesh_path", ""))
        )
        base_row: dict[str, Any] = {
            "label": label,
            "source_mesh_path": str(mesh_path),
            "fat_thickness/min": source_case.get("fat_thickness/min"),
            "fat_thickness/center": source_case.get("fat_thickness/center"),
        }
        if not mesh_path.is_file():
            message = f"{label}: input mesh does not exist: {mesh_path}"
            hard_failures.append(message)
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "error/type": "FileNotFoundError",
                    "error/message": message,
                }
            )
            continue

        try:
            mesh = pv.read(mesh_path)
            if not isinstance(mesh, pv.UnstructuredGrid):
                mesh = mesh.cast_to_unstructured_grid()
            case, source_resolution = make_case(label, mesh)
            case_dir = output_root / label
            case_dir.mkdir(parents=True, exist_ok=True)
            inverse_cfg = case_config(
                cfg=cfg,
                mesh_path=mesh_path,
                case=case,
                case_dir=case_dir,
            )
            row = {
                **base_row,
                "source_resolution": source_resolution.name,
                **toy.solve_case(case, mesh, inverse_cfg),
            }
            add_solver_status(row)
            validation_errors = validate_result(
                row,
                label=label,
                case=case,
                case_dir=case_dir,
            )
            row["validation/errors"] = validation_errors
            row["status"] = "ok" if not validation_errors else "invalid"
            rows.append(row)
            hard_failures.extend(f"{label}: {error}" for error in validation_errors)
            if not bool(row.get("inverse/converged", False)):
                convergence_failures.append(
                    f"{label}: {row.get('inverse/stop_reason', 'unknown stop')}"
                )
        except Exception as error:
            logger.exception("Inverse case %s failed", label)
            message = f"{label}: {type(error).__name__}: {error}"
            hard_failures.append(message)
            rows.append(
                {
                    **base_row,
                    "status": "error",
                    "error/type": type(error).__name__,
                    "error/message": str(error),
                }
            )

    write_outputs(
        cfg,
        labels=labels,
        rows=rows,
        hard_failures=hard_failures,
        convergence_failures=convergence_failures,
    )
    if hard_failures:
        raise RuntimeError(
            "inverse sweep validation failed: " + "; ".join(hard_failures)
        )
    if convergence_failures and cfg.require_convergence:
        raise RuntimeError(
            "inverse cases did not hit the 20-step loss plateau: "
            + "; ".join(convergence_failures)
        )


if __name__ == "__main__":
    cherries.main(main)
