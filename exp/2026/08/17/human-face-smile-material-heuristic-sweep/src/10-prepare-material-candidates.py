from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from _human_face_config import SKIN_NU
from _material_heuristics import (
    SCHEMA_VERSION,
    MaterialCandidate,
    candidate_field_metrics,
    candidate_fields,
    default_candidates,
    file_sha256,
    make_candidate_skin,
    make_signed_heat_field,
    prepare_surface_geometry,
    skin_material_content_hash,
    skin_solver_content_hash,
    skin_topology_content_hash,
)
from _reference import PREPARED_MESH

from liblaf import cherries, melon

logger = logging.getLogger(__name__)

EXPECTED_YOUNG_SCALES = (1.0, 0.25)
EXPECTED_PRESTRAIN_GAINS = (0.0, 0.5, 1.0)
EXPECTED_LABELS = tuple(candidate.label for candidate in default_candidates())
EXPECTED_HEURISTIC = {
    "area_deadband": 0.01,
    "cap_quantile": 0.99,
    "diffusion_sigma_m": 0.005,
}
EXPECTED_MATERIAL_GATES = {
    "max_normalized_interior_jump_q99": 0.08,
    "max_normalized_interior_jump": 0.20,
    "max_normalized_boundary_jump_q99": 0.08,
    "max_normalized_boundary_jump": 0.20,
    "max_e_edge_jump_mpa": 0.06,
    "max_activation_edge_jump": 0.08,
    "max_e_edge_rms_mpa": 0.012,
    "max_activation_edge_rms": 0.02,
    "max_singleton_components": 20,
    "max_small_component_area_fraction": 0.005,
}
LAME_CONVERSION = (
    "existing 3D isotropic convention: "
    "lambda = E * nu / ((1 + nu) * (1 - 2 * nu)); "
    "mu = E / (2 * (1 + nu))"
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True)

    input_mesh: Path = cherries.input(PREPARED_MESH)
    output_manifest: Path = cherries.output(
        "10-material-candidates-manifest.json", mkdir=True
    )
    output_table: Path = cherries.output("10-material-candidates-table.md", mkdir=True)
    output_dir_name: str = "10-material-candidates"

    young_min_scales: str = "1.0,0.25"
    prestrain_gains: str = "0.0,0.5,1.0"
    area_deadband: float = 0.01
    cap_quantile: float = 0.99
    diffusion_sigma_m: float = 0.005

    max_normalized_interior_jump_q99: float = 0.08
    max_normalized_interior_jump: float = 0.20
    max_normalized_boundary_jump_q99: float = 0.08
    max_normalized_boundary_jump: float = 0.20
    max_e_edge_jump_mpa: float = 0.06
    max_activation_edge_jump: float = 0.08
    max_e_edge_rms_mpa: float = 0.012
    max_activation_edge_rms: float = 0.02
    max_singleton_components: int = 20
    max_small_component_area_fraction: float = 0.005

    sensitivity_sigma_mm: str = "2.5,5,10"
    sensitivity_deadbands: str = "0.005,0.01,0.02"
    sensitivity_cap_quantiles: str = "0.975,0.99,0.995"


def parse_float_list(value: str, *, name: str) -> list[float]:
    try:
        result = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        msg = f"{name} must be a comma-separated float list, got {value!r}"
        raise ValueError(msg) from error
    if not result:
        msg = f"{name} must select at least one value"
        raise ValueError(msg)
    if not all(math.isfinite(item) for item in result):
        msg = f"{name} must contain only finite values: {result}"
        raise ValueError(msg)
    if len(set(result)) != len(result):
        msg = f"{name} contains duplicate values: {result}"
        raise ValueError(msg)
    return result


def validate_fixed_design(cfg: Config) -> None:
    heuristic = {
        "area_deadband": cfg.area_deadband,
        "cap_quantile": cfg.cap_quantile,
        "diffusion_sigma_m": cfg.diffusion_sigma_m,
    }
    gates = {
        "max_normalized_interior_jump_q99": cfg.max_normalized_interior_jump_q99,
        "max_normalized_interior_jump": cfg.max_normalized_interior_jump,
        "max_normalized_boundary_jump_q99": cfg.max_normalized_boundary_jump_q99,
        "max_normalized_boundary_jump": cfg.max_normalized_boundary_jump,
        "max_e_edge_jump_mpa": cfg.max_e_edge_jump_mpa,
        "max_activation_edge_jump": cfg.max_activation_edge_jump,
        "max_e_edge_rms_mpa": cfg.max_e_edge_rms_mpa,
        "max_activation_edge_rms": cfg.max_activation_edge_rms,
        "max_singleton_components": cfg.max_singleton_components,
        "max_small_component_area_fraction": cfg.max_small_component_area_fraction,
    }
    numeric_values = (*heuristic.values(), *gates.values())
    if not all(math.isfinite(float(value)) for value in numeric_values):
        msg = "formal material design parameters and gates must all be finite"
        raise ValueError(msg)
    if heuristic != EXPECTED_HEURISTIC:
        msg = (
            "formal material preparation requires the fixed 1%/99%/5mm "
            f"heuristic {EXPECTED_HEURISTIC}, got {heuristic}"
        )
        raise ValueError(msg)
    if gates != EXPECTED_MATERIAL_GATES:
        msg = (
            "formal material preparation requires the calibrated fixed gates "
            f"{EXPECTED_MATERIAL_GATES}, got {gates}"
        )
        raise ValueError(msg)


def selected_candidates(cfg: Config) -> list[MaterialCandidate]:
    young_scales = parse_float_list(cfg.young_min_scales, name="young_min_scales")
    gains = parse_float_list(cfg.prestrain_gains, name="prestrain_gains")
    candidates = [
        MaterialCandidate(young_min_scale=young_scale, prestrain_gain=gain)
        for young_scale in young_scales
        for gain in gains
    ]
    labels = [candidate.label for candidate in candidates]
    if len(set(labels)) != len(labels):
        msg = f"candidate labels collide after percentage formatting: {labels}"
        raise ValueError(msg)
    expected_grid = (list(EXPECTED_YOUNG_SCALES), list(EXPECTED_PRESTRAIN_GAINS))
    if (young_scales, gains) != expected_grid:
        msg = (
            "formal material preparation requires the fixed interpretable 2x3 "
            f"grid {expected_grid}, got {(young_scales, gains)}"
        )
        raise ValueError(msg)
    if tuple(labels) != EXPECTED_LABELS:
        msg = f"candidate labels/order differ from fixed grid: {labels}"
        raise ValueError(msg)
    return candidates


def readback_metrics(  # noqa: C901, PLR0912, PLR0915
    path: Path, expected: dict[str, Any]
) -> dict[str, Any]:
    from liblaf.apple.common import (
        ACTIVATION_INV,
        FRACTION,
        GLOBAL_POINT_ID,
        LAMBDA,
        MU,
    )

    skin = pv.read(path)
    if not isinstance(skin, pv.PolyData):
        msg = f"{path} read back as {type(skin).__name__}, expected PolyData"
        raise TypeError(msg)
    required_cell = {
        LAMBDA.vtk,
        MU.vtk,
        FRACTION.vtk,
        ACTIVATION_INV.vtk,
        "SkinYoungModulusMPa",
        "SkinActivationInvDiag",
        "StressFreeAreaRatio",
        "LogAreaRaw",
        "LogAreaDeadbanded",
        "LogAreaCapped",
        "LogAreaDiffused",
        "ExpansionMaterialMask",
        "ContractionPrestrainMask",
    }
    missing_cell = sorted(required_cell - set(skin.cell_data))
    missing_point = sorted({GLOBAL_POINT_ID.vtk} - set(skin.point_data))
    errors: list[str] = []
    if missing_cell:
        errors.append(f"missing required cell arrays: {missing_cell}")
    if missing_point:
        errors.append(f"missing required point arrays: {missing_point}")
    if errors:
        return {
            "readback/ok": False,
            "readback/errors": errors,
            "readback/n_points": int(skin.n_points),
            "readback/n_triangles": int(skin.n_cells),
            "readback/file_identity": {
                "size_bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            },
        }

    lambda_ = np.asarray(skin.cell_data[LAMBDA.vtk], dtype=np.float64)
    mu = np.asarray(skin.cell_data[MU.vtk], dtype=np.float64)
    fraction = np.asarray(skin.cell_data[FRACTION.vtk], dtype=np.float64)
    activation = np.asarray(skin.cell_data[ACTIVATION_INV.vtk], dtype=np.float64)
    young = np.asarray(skin.cell_data["SkinYoungModulusMPa"], dtype=np.float64)
    activation_diag = np.asarray(
        skin.cell_data["SkinActivationInvDiag"], dtype=np.float64
    )
    stress_free = np.asarray(skin.cell_data["StressFreeAreaRatio"], dtype=np.float64)
    expected_lambda = young * SKIN_NU / ((1.0 + SKIN_NU) * (1.0 - 2.0 * SKIN_NU))
    expected_mu = young / (2.0 * (1.0 + SKIN_NU))
    expected_stress_free = np.reciprocal(np.square(1.0 + activation_diag))

    if activation.shape != (skin.n_cells, 3):
        errors.append(f"ActivationInv shape {activation.shape} != {(skin.n_cells, 3)}")
    if skin.n_points != int(expected["content/n_points"]):
        errors.append("point count changed during VTP readback")
    if skin.n_cells != int(expected["content/n_triangles"]):
        errors.append("triangle count changed during VTP readback")
    finite_arrays = {
        "points": np.asarray(skin.points),
        "Lambda": lambda_,
        "Mu": mu,
        "Fraction": fraction,
        "ActivationInv": activation,
        "SkinYoungModulusMPa": young,
        "SkinActivationInvDiag": activation_diag,
        "StressFreeAreaRatio": stress_free,
    }
    finite_arrays.update(
        {
            name: np.asarray(skin.cell_data[name], dtype=np.float64)
            for name in (
                "LogAreaRaw",
                "LogAreaDeadbanded",
                "LogAreaCapped",
                "LogAreaDiffused",
            )
        }
    )
    nonfinite = sorted(
        name for name, values in finite_arrays.items() if not np.isfinite(values).all()
    )
    if nonfinite:
        errors.append(f"non-finite readback arrays: {nonfinite}")
    if lambda_.shape != (skin.n_cells,) or not np.allclose(
        lambda_, expected_lambda, rtol=1.0e-13, atol=1.0e-14
    ):
        errors.append("Lambda readback is inconsistent with E and nu")
    if mu.shape != (skin.n_cells,) or not np.allclose(
        mu, expected_mu, rtol=1.0e-13, atol=1.0e-14
    ):
        errors.append("Mu readback is inconsistent with E and nu")
    if fraction.shape != (skin.n_cells,) or not np.allclose(
        fraction, 1.0, rtol=0.0, atol=0.0
    ):
        errors.append("Fraction readback must be exactly one on every skin triangle")
    if activation.shape == (skin.n_cells, 3):
        if not np.allclose(
            activation[:, 0], activation_diag, rtol=0.0, atol=0.0
        ) or not np.allclose(activation[:, 1], activation_diag, rtol=0.0, atol=0.0):
            errors.append("ActivationInv in-plane entries differ from helper diag")
        if not np.allclose(activation[:, 2], 0.0, rtol=0.0, atol=0.0):
            errors.append("ActivationInv out-of-plane readback entry is nonzero")
    if stress_free.shape != (skin.n_cells,) or not np.allclose(
        stress_free, expected_stress_free, rtol=1.0e-13, atol=1.0e-14
    ):
        errors.append("stress-free area ratio is inconsistent with helper diag")

    topology_sha256 = skin_topology_content_hash(skin)
    material_sha256 = skin_material_content_hash(skin)
    solver_sha256 = skin_solver_content_hash(skin)
    for name, actual in (
        ("topology", topology_sha256),
        ("material", material_sha256),
        ("solver", solver_sha256),
    ):
        if actual != str(expected[f"content/{name}_sha256"]):
            errors.append(f"{name} content hash changed during VTP readback")
    return {
        "readback/ok": not errors,
        "readback/errors": errors,
        "readback/n_points": int(skin.n_points),
        "readback/n_triangles": int(skin.n_cells),
        "readback/content/topology_sha256": topology_sha256,
        "readback/content/material_sha256": material_sha256,
        "readback/content/solver_sha256": solver_sha256,
        "readback/file_identity": {
            "size_bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        },
        "readback/formula_consistent": not any(
            "inconsistent" in error or "helper" in error for error in errors
        ),
    }


def sensitivity_summary(
    geometry: Any,
    cfg: Config,
) -> list[dict[str, Any]]:
    sigma_mm = parse_float_list(cfg.sensitivity_sigma_mm, name="sensitivity_sigma_mm")
    deadbands = parse_float_list(
        cfg.sensitivity_deadbands, name="sensitivity_deadbands"
    )
    cap_quantiles = parse_float_list(
        cfg.sensitivity_cap_quantiles,
        name="sensitivity_cap_quantiles",
    )
    if (sigma_mm, deadbands, cap_quantiles) != (
        [2.5, 5.0, 10.0],
        [0.005, 0.01, 0.02],
        [0.975, 0.99, 0.995],
    ):
        msg = "field sensitivity requires the fixed 3x3x3 calibrated grid"
        raise ValueError(msg)

    combined = MaterialCandidate(young_min_scale=0.25, prestrain_gain=1.0)
    rows: list[dict[str, Any]] = []
    for sigma in sigma_mm:
        for deadband in deadbands:
            for cap_quantile in cap_quantiles:
                signed_field = make_signed_heat_field(
                    geometry,
                    area_deadband=deadband,
                    cap_quantile=cap_quantile,
                    diffusion_sigma=1.0e-3 * sigma,
                    max_normalized_interior_jump_q99=(
                        cfg.max_normalized_interior_jump_q99
                    ),
                    max_normalized_interior_jump=(cfg.max_normalized_interior_jump),
                    max_normalized_boundary_jump_q99=(
                        cfg.max_normalized_boundary_jump_q99
                    ),
                    max_normalized_boundary_jump=(cfg.max_normalized_boundary_jump),
                )
                fields = candidate_fields(geometry, signed_field, combined)
                metrics = candidate_field_metrics(geometry, combined, fields)
                heat = signed_field.metrics
                numeric_values = (
                    heat["expansion_cap"],
                    heat["contraction_cap"],
                    heat["area_weighted_rms_attenuation"],
                    heat["interior_normalized_jump_q99"],
                    heat["interior_normalized_jump_max"],
                    heat["boundary_normalized_jump_q99"],
                    heat["boundary_normalized_jump_max"],
                    metrics["skin/E_MPa_min"],
                    metrics["skin/E_MPa_area_weighted_mean"],
                    metrics["skin/activation_inv_diag_max"],
                    metrics["skin/stress_free_area_ratio_min"],
                )
                finite = bool(np.isfinite(numeric_values).all())
                rows.append(
                    {
                        "sigma_mm": sigma,
                        "area_deadband": deadband,
                        "cap_quantile": cap_quantile,
                        "heat/expansion_cap": heat["expansion_cap"],
                        "heat/contraction_cap": heat["contraction_cap"],
                        "heat/rms_attenuation": heat["area_weighted_rms_attenuation"],
                        "heat/interior_jump_q99": heat["interior_normalized_jump_q99"],
                        "heat/interior_jump_max": heat["interior_normalized_jump_max"],
                        "heat/boundary_jump_q99": heat["boundary_normalized_jump_q99"],
                        "heat/boundary_jump_max": heat["boundary_normalized_jump_max"],
                        "combined/E_MPa_min": metrics["skin/E_MPa_min"],
                        "combined/E_MPa_area_weighted_mean": metrics[
                            "skin/E_MPa_area_weighted_mean"
                        ],
                        "combined/activation_inv_diag_max": metrics[
                            "skin/activation_inv_diag_max"
                        ],
                        "combined/stress_free_area_ratio_min": metrics[
                            "skin/stress_free_area_ratio_min"
                        ],
                        "evaluation/finite": finite,
                        "primary_jump_gates/ok": not signed_field.validation_errors,
                        "primary_jump_gates/errors": signed_field.validation_errors,
                    }
                )
    return rows


def validate_candidate_rows(  # noqa: C901
    candidates: list[MaterialCandidate], rows: list[dict[str, Any]]
) -> list[str]:
    errors: list[str] = []
    labels = [str(row["label"]) for row in rows]
    if tuple(labels) != EXPECTED_LABELS:
        errors.append(
            f"manifest labels/order {labels} != fixed grid {list(EXPECTED_LABELS)}"
        )
    if len(rows) != 6:
        errors.append(f"manifest requires six candidate rows, got {len(rows)}")
    if len({str(row["skin/path"]) for row in rows}) != len(rows):
        errors.append("candidate skin paths are not unique")
    if len({str(row["content/solver_sha256"]) for row in rows}) != len(rows):
        errors.append("candidate solver-content hashes are not unique")
    if len({str(row["content/topology_sha256"]) for row in rows}) != 1:
        errors.append("candidate topology-content hashes differ")
    for candidate, row in zip(candidates, rows, strict=True):
        if str(row["label"]) != candidate.label:
            errors.append(f"row label does not match candidate {candidate.label}")
        if not math.isclose(
            float(row["young_min_scale"]),
            candidate.young_min_scale,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            errors.append(f"{candidate.label}: Young scale changed in manifest")
        if not math.isclose(
            float(row["prestrain_gain"]),
            candidate.prestrain_gain,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            errors.append(f"{candidate.label}: prestrain gain changed in manifest")
        if not bool(row["readback/ok"]):
            errors.append(f"{candidate.label}: VTP readback validation failed")
        if not bool(row["validation/ok"]):
            errors.append(f"{candidate.label}: material validation failed")
    return errors


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "| candidate | E scale | gain | expansion area | contraction area | E min MPa | area-mean E MPa | Ainv max | E jump q99/max MPa | Ainv jump q99/max | validation | skin |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        e_jump = row["field/E_edge_jump_MPa"]
        activation_jump = row["field/activation_edge_jump"]
        lines.append(
            f"| {row['label']} | {row['young_min_scale']:.3g} | "
            f"{row['prestrain_gain']:.3g} | "
            f"{row['skin/expansion_rest_area_fraction']:.2%} | "
            f"{row['skin/contraction_rest_area_fraction']:.2%} | "
            f"{row['skin/E_MPa_min']:.6g} | "
            f"{row['skin/E_MPa_area_weighted_mean']:.6g} | "
            f"{row['skin/activation_inv_diag_max']:.6g} | "
            f"{e_jump['edge_length_weighted_q99']:.4g}/"
            f"{e_jump['max']:.4g} | "
            f"{activation_jump['edge_length_weighted_q99']:.4g}/"
            f"{activation_jump['max']:.4g} | "
            f"{'ok' if row['validation/ok'] else 'failed'} | "
            f"`{row['skin/path']}` |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(cfg: Config) -> None:
    validate_fixed_design(cfg)
    if cfg.input_mesh.resolve() != Path(PREPARED_MESH).resolve():
        msg = (
            "this fixed Smile experiment only accepts the prepared reference mesh "
            f"{Path(PREPARED_MESH).resolve()}, got {cfg.input_mesh.resolve()}"
        )
        raise ValueError(msg)
    input_identity_before = {
        "size_bytes": cfg.input_mesh.stat().st_size,
        "sha256": file_sha256(cfg.input_mesh),
    }
    mesh = pv.read(cfg.input_mesh)
    if not isinstance(mesh, pv.UnstructuredGrid):
        mesh = mesh.cast_to_unstructured_grid()
    candidates = selected_candidates(cfg)
    geometry = prepare_surface_geometry(mesh)
    signed_field = make_signed_heat_field(
        geometry,
        area_deadband=cfg.area_deadband,
        cap_quantile=cfg.cap_quantile,
        diffusion_sigma=cfg.diffusion_sigma_m,
        max_normalized_interior_jump_q99=(cfg.max_normalized_interior_jump_q99),
        max_normalized_interior_jump=cfg.max_normalized_interior_jump,
        max_normalized_boundary_jump_q99=(cfg.max_normalized_boundary_jump_q99),
        max_normalized_boundary_jump=cfg.max_normalized_boundary_jump,
    )
    output_dir = cfg.output_manifest.parent / cfg.output_dir_name
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        skin, metrics = make_candidate_skin(
            geometry,
            signed_field,
            candidate,
            max_e_edge_jump=cfg.max_e_edge_jump_mpa,
            max_activation_edge_jump=cfg.max_activation_edge_jump,
            max_e_edge_rms=cfg.max_e_edge_rms_mpa,
            max_activation_edge_rms=cfg.max_activation_edge_rms,
            max_singleton_components=cfg.max_singleton_components,
            max_small_component_area_fraction=(cfg.max_small_component_area_fraction),
        )
        path = output_dir / f"skin-{candidate.label}.vtp"
        melon.save(skin, path)
        readback = readback_metrics(path, metrics)
        errors = [
            *metrics["validation/errors"],
            *(f"readback: {error}" for error in readback["readback/errors"]),
        ]
        row = {
            **metrics,
            "skin/path": str(path.relative_to(cfg.output_manifest.parent)),
            "skin/file_identity": readback["readback/file_identity"],
            **readback,
            "validation/errors": errors,
            "validation/ok": not errors,
        }
        rows.append(row)
        cherries.log_output(path)
        logger.info(
            "%s E=[%.6g, %.6g] MPa Ainv_max=%.6g validation=%s",
            candidate.label,
            row["skin/E_MPa_min"],
            row["skin/E_MPa_max"],
            row["skin/activation_inv_diag_max"],
            row["validation/ok"],
        )

    sensitivity = sensitivity_summary(geometry, cfg)
    input_identity_after = {
        "size_bytes": cfg.input_mesh.stat().st_size,
        "sha256": file_sha256(cfg.input_mesh),
    }
    manifest_errors = list(signed_field.validation_errors)
    if input_identity_after != input_identity_before:
        manifest_errors.append(
            "prepared input mesh changed during candidate preparation"
        )
    if len(sensitivity) != 27:
        manifest_errors.append(
            f"field sensitivity requires 27 rows, got {len(sensitivity)}"
        )
    if not all(bool(row["evaluation/finite"]) for row in sensitivity):
        manifest_errors.append("field sensitivity contains non-finite summary values")
    manifest_errors.extend(validate_candidate_rows(candidates, rows))
    candidate_errors = {
        str(row["label"]): row["validation/errors"]
        for row in rows
        if row["validation/errors"]
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "complete": not manifest_errors and not candidate_errors,
        "input_mesh": str(cfg.input_mesh),
        "input_mesh_identity": input_identity_before,
        "input_mesh_identity_verified_stable": (
            input_identity_before == input_identity_after
        ),
        "target": "Smile",
        "selection": "all surface-triangle vertices are finite IsFace points",
        "grid": {
            "young_min_scales": parse_float_list(
                cfg.young_min_scales, name="young_min_scales"
            ),
            "prestrain_gains": parse_float_list(
                cfg.prestrain_gains, name="prestrain_gains"
            ),
        },
        "heuristic": {
            "area_deadband": cfg.area_deadband,
            "cap_quantile": cfg.cap_quantile,
            "diffusion_sigma_m": cfg.diffusion_sigma_m,
            "sequence": [
                "signed log(target/rest triangle area) on every eligible triangle",
                "symmetric soft log-domain deadband",
                "separate rest-area-weighted positive and negative caps",
                "single signed finite-volume implicit heat diffusion",
                "decode positive softening and negative isotropic prestrain",
            ],
            "young_rule": (
                "E0 * exp(log(EminScale) * "
                "positive(diffused_signed_log_area) / positive_cap)"
            ),
            "prestrain_rule": (
                "Ainv_diag = exp(0.5 * gain * positive(-diffused_signed_log_area)) - 1"
            ),
            "diffusion_domain": (
                "full eligible finite IsFace component; zero Dirichlet boundary"
            ),
            "lame_conversion": LAME_CONVERSION,
            "lame_interpretation": (
                "plane-strain-like coefficient convention retained from the existing "
                "Smile baseline for a controlled relative material-map comparison; "
                "not a thin-shell plane-stress claim"
            ),
        },
        "material_gates": {
            "max_normalized_interior_jump_q99": (cfg.max_normalized_interior_jump_q99),
            "max_normalized_interior_jump": cfg.max_normalized_interior_jump,
            "max_normalized_boundary_jump_q99": (cfg.max_normalized_boundary_jump_q99),
            "max_normalized_boundary_jump": cfg.max_normalized_boundary_jump,
            "max_e_edge_jump_mpa": cfg.max_e_edge_jump_mpa,
            "max_activation_edge_jump": cfg.max_activation_edge_jump,
            "max_e_edge_rms_mpa": cfg.max_e_edge_rms_mpa,
            "max_activation_edge_rms": cfg.max_activation_edge_rms,
            "max_singleton_components": cfg.max_singleton_components,
            "max_small_component_area_fraction": (
                cfg.max_small_component_area_fraction
            ),
        },
        "surface_geometry": geometry.geometry_metrics,
        "primary_signed_heat_field": {
            "metrics": signed_field.metrics,
            "validation/ok": not signed_field.validation_errors,
            "validation/errors": signed_field.validation_errors,
        },
        "field_sensitivity": {
            "candidate": "e025-p100",
            "grid": {
                "sigma_mm": [2.5, 5.0, 10.0],
                "area_deadband": [0.005, 0.01, 0.02],
                "cap_quantile": [0.975, 0.99, 0.995],
            },
            "complete": len(sensitivity) == 27
            and all(bool(row["evaluation/finite"]) for row in sensitivity),
            "rows": sensitivity,
        },
        "n_candidates": len(rows),
        "validation_errors": manifest_errors,
        "candidate_validation_errors": candidate_errors,
        "candidates": rows,
    }
    cfg.output_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    write_table(cfg.output_table, rows)
    logger.info("Wrote %s", cfg.output_manifest)
    logger.info("Wrote %s", cfg.output_table)
    if manifest_errors or candidate_errors:
        msg = (
            "material candidate validation failed: "
            f"manifest={manifest_errors}, candidates={candidate_errors}"
        )
        raise RuntimeError(msg)


if __name__ == "__main__":
    cherries.main(run)
