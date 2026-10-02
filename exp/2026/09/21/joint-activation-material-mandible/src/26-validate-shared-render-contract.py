"""CPU validation of Spatial80 snapshot metadata and baseline VTK exports."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_fields import BULK_TISSUES, symmetric_matrices
from joint_spatial_fields import (
    EXPECTED_MESH_CELL_COUNT,
    SpatialSharedFieldParameters,
)

from liblaf import cherries


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    basis_path: Path = (
        GROUP / "data/spatial-baseline-audit-005/basis-and-normal-equations.npz"
    )
    audit_summary_path: Path = GROUP / "data/spatial-baseline-audit-005/summary.json"
    spatial_fd_receipt: Path = (
        GROUP / "data/spatial-face-gradient-validation-002/summary.json"
    )
    output_dir: Path


def _load_renderer() -> ModuleType:
    path = Path(__file__).with_name("31-render-final.py")
    spec = importlib.util.spec_from_file_location("joint_render_final", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_scalar(value: object) -> np.ndarray:
    return np.asarray(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )


def _shared_field(receipt: dict) -> dict:
    anchor_counts = receipt["anchor_counts"]
    return {
        "schema": "joint-additive-spatial-stress-fields-v1",
        "basis": "spatial80",
        "coefficient_count": 80,
        "bulk_coordinate_slices": {
            "fat": [0, 24],
            "aponeurosis": [24, 48],
            "muscle": [48, 78],
        },
        "bulk_anchor_counts": anchor_counts,
        "skin_baseline_index": 78,
        "skin_log_multiplier_index": 79,
        "fixed_coordinate_ids": [78],
        "free_coordinate_ids": [*range(78), 79],
        "spatial_smoothness": {
            "weight": 100.0,
            "factor": 0.5,
            "regularizer": "bulk_spatial_roughness",
            "objective_term": "0.5 * beta * mean_t tr(C_t^T G_t C_t)",
        },
    }


def main(cfg: Config) -> None:  # noqa: PLR0915
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    renderer = _load_renderer()
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    fields = SpatialSharedFieldParameters(
        cfg.basis_path,
        cfg.audit_summary_path,
        EXPECTED_MESH_CELL_COUNT,
        device="cpu",
        dtype=torch.float64,
    )
    directional = json.loads(cfg.spatial_fd_receipt.read_text())
    assert directional["success"] is True
    coefficients = torch.as_tensor(
        directional["base_coefficients"], dtype=torch.float64
    )
    assert coefficients.shape == (80,)
    with torch.no_grad():
        fields.coefficients.copy_(coefficients)
    basis = fields.basis_receipt()
    shared_field = _shared_field(basis)
    materials = fields.config
    protocol = {
        "stage": "render_contract_fixture",
        "materials": materials,
        "shared_basis": "spatial80",
        "shared_field": shared_field,
        "spatial_basis": basis,
    }
    snapshot = cfg.output_dir / "visualization-fixture.npz"
    np.savez_compressed(
        snapshot,
        schema=np.asarray("joint-optimization-visualization-snapshot-v2"),
        stage=np.asarray(protocol["stage"]),
        checkpoint_label=np.asarray("fixture"),
        status=np.asarray("render_contract_fixture_not_scientific_result"),
        shared=coefficients.numpy(),
        shared_basis=np.asarray("spatial80"),
        shared_coefficient_count=np.asarray(80, dtype=np.int64),
        shared_field_json=_json_scalar(shared_field),
        material_config_json=_json_scalar(materials),
        spatial_basis_json=_json_scalar(basis),
        spatial_smoothness_weight=np.asarray(100.0, dtype=np.float64),
        spatial_smoothness_factor=np.asarray(0.5, dtype=np.float64),
    )
    with np.load(snapshot) as data:
        assets, export = renderer._render_shared_baseline(  # noqa: SLF001
            snapshot, data, prepared, protocol, cfg.output_dir
        )

    volume = pv.read(cfg.output_dir / "shared-baseline-fixture.vtu")
    skin = pv.read(cfg.output_dir / "shared-skin-fixture.vtp")
    assert assets == ["shared-baseline-fixture.vtu", "shared-skin-fixture.vtp"]
    assert volume.n_cells == EXPECTED_MESH_CELL_COUNT
    assert np.array_equal(
        volume.cell_data["GlobalCellId"], np.arange(volume.n_cells, dtype=np.int64)
    )
    minimum = np.asarray(volume.cell_data["baseline_min_principal_mpa"])
    maximum = np.asarray(volume.cell_data["baseline_max_principal_mpa"])
    assert np.isfinite(minimum).all()
    assert np.isfinite(maximum).all()
    assert np.all(minimum <= maximum)
    reference_volume = pv.read(prepared.volume_path)
    sample_ids = np.unique(
        np.asarray(
            [
                0,
                volume.n_cells - 1,
                *(int(fields.cell_ids(name)[0]) for name in BULK_TISSUES),
                *(int(fields.cell_ids(name)[-1]) for name in BULK_TISSUES),
            ],
            dtype=np.int64,
        )
    )
    expected = np.zeros((len(sample_ids), 3, 3), dtype=np.float64)
    sampled_tissue_error = 0.0
    tissue_off_support_error = 0.0
    for tissue_index, tissue in enumerate(BULK_TISSUES):
        ids = fields.cell_ids(tissue).numpy()
        positions = np.searchsorted(ids, sample_ids)
        inside = positions < len(ids)
        inside[inside] &= ids[positions[inside]] == sample_ids[inside]
        if not inside.any():
            continue
        phi = fields.basis_weights(tissue)[positions[inside]]
        coordinates = phi @ fields.anchor_coordinates(tissue)
        stress = fields.bulk_mu_mpa[tissue_index] * symmetric_matrices(coordinates)
        fraction = np.asarray(
            reference_volume.cell_data[f"{tissue.title()}Fraction"], dtype=np.float64
        )[sample_ids[inside]]
        expected[inside] += fraction[:, None, None] * stress.detach().numpy()
        expected_tissue = np.zeros(len(sample_ids), dtype=np.float64)
        expected_tissue[inside] = torch.linalg.eigvalsh(stress)[:, -1].detach().numpy()
        actual_tissue = np.asarray(
            volume.cell_data[f"{tissue}_baseline_max_principal_mpa"]
        )
        sampled_tissue_error = max(
            sampled_tissue_error,
            float(np.max(np.abs(actual_tissue[sample_ids] - expected_tissue))),
        )
        full_fraction = np.asarray(
            reference_volume.cell_data[f"{tissue.title()}Fraction"], dtype=np.float64
        )
        tissue_off_support_error = max(
            tissue_off_support_error,
            float(np.max(np.abs(actual_tissue[full_fraction <= 0.0]))),
        )
    expected_eigenvalues = np.linalg.eigvalsh(expected)
    sampled_error = max(
        float(np.max(np.abs(minimum[sample_ids] - expected_eigenvalues[:, 0]))),
        float(np.max(np.abs(maximum[sample_ids] - expected_eigenvalues[:, -1]))),
    )
    assert sampled_error <= 5.0e-15
    assert sampled_tissue_error <= 5.0e-15
    assert tissue_off_support_error == 0.0
    assert np.array_equal(
        skin.point_data["GlobalPointId"],
        pv.read(prepared.skin_path).point_data["GlobalPointId"],
    )
    effective_young = np.asarray(skin.point_data["effective_young_mpa"])
    assert np.isfinite(effective_young).all()
    assert np.max(np.abs(effective_young - export["skin_effective_young_mpa"])) == 0

    legacy = cfg.output_dir / "invalid-spatial-v1.npz"
    np.savez_compressed(
        legacy,
        schema=np.asarray("joint-optimization-visualization-snapshot-v1"),
        shared=coefficients.numpy(),
    )
    try:
        with np.load(legacy) as data:
            renderer._snapshot_shared_contract(data, protocol)  # noqa: SLF001
    except AssertionError:
        pass
    else:
        msg = "spatial snapshot v1 was accepted without required metadata"
        raise AssertionError(msg)
    legacy.unlink()

    receipt = {
        "schema": "joint-shared-render-contract-validation-v1",
        "success": True,
        "status": "passed_cpu_spatial80_export_validation",
        "test_fixture_only": True,
        "scientific_result": False,
        "source_spatial_fd_receipt": str(cfg.spatial_fd_receipt),
        "source_spatial_fd_receipt_sha256": sha256(cfg.spatial_fd_receipt),
        "snapshot_sha256": sha256(snapshot),
        "basis": basis,
        "shared_field": shared_field,
        "field_export": export,
        "checks": {
            "sampled_principal_stress_maximum_absolute_error_mpa": sampled_error,
            "sampled_principal_stress_tolerance_mpa": 5.0e-15,
            "sampled_tissue_principal_stress_maximum_absolute_error_mpa": (
                sampled_tissue_error
            ),
            "tissue_off_support_maximum_absolute_mpa": tissue_off_support_error,
            "spatial_v1_rejected": True,
            "global_cell_ids_exact": True,
            "global_skin_point_ids_exact": True,
            "effective_young_finite_uniform": True,
        },
        "sources": {
            str(Path(__file__)): sha256(Path(__file__)),
            str(Path(__file__).with_name("31-render-final.py")): sha256(
                Path(__file__).with_name("31-render-final.py")
            ),
            str(Path(__file__).with_name("joint_spatial_fields.py")): sha256(
                Path(__file__).with_name("joint_spatial_fields.py")
            ),
        },
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_metrics({"shared_render/sampled_stress_error_mpa": sampled_error})
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
