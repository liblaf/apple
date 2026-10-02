"""Render joint-optimization trends and VTK fields without another FEM solve."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_data import PreparedInputs
from joint_fields import BULK_TISSUES, SharedFieldParameters, symmetric_matrices
from joint_spatial_fields import SpatialSharedFieldParameters

from liblaf import cherries

mpl.use("Agg")


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    run_dir: Path
    prepared_dir: Path = GROUP / "data/prepared"
    output_dir: Path = cherries.output("joint-final-visuals", mkdir=True)


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=220)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def _series(trace: list[dict], path: tuple[str, ...]) -> list[float]:
    values = []
    for row in trace:
        value = row
        for key in path:
            value = value[key]
        values.append(float(value))
    return values


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _snapshot_shared_contract(
    data: np.lib.npyio.NpzFile,
    protocol: dict,
) -> tuple[str, dict, dict, dict | None]:
    """Validate one snapshot's tagged shared-field metadata."""
    schema = str(data["schema"].item())
    shared_basis = protocol.get("shared_basis", "constant20")
    assert shared_basis in {"constant20", "spatial80"}
    shared_field = protocol.get("shared_field")
    if shared_field is None:
        assert shared_basis == "constant20"
        shared_field = {
            "schema": "joint-additive-stress-fields-v1",
            "basis": "constant20",
            "coefficient_count": 20,
            "skin_baseline_index": 18,
            "skin_log_multiplier_index": 19,
        }
    materials = protocol["materials"]
    spatial_basis = protocol.get("spatial_basis")
    expected_count = int(shared_field["coefficient_count"])
    assert shared_field["basis"] == shared_basis
    assert data["shared"].shape == (expected_count,)
    if schema == "joint-optimization-visualization-snapshot-v1":
        assert shared_basis == "constant20", (
            "Spatial80 snapshots require the v2 metadata contract"
        )
        return shared_basis, shared_field, materials, None
    assert schema == "joint-optimization-visualization-snapshot-v2"
    assert str(data["shared_basis"].item()) == shared_basis
    assert int(data["shared_coefficient_count"].item()) == expected_count
    assert json.loads(str(data["shared_field_json"].item())) == shared_field
    assert json.loads(str(data["material_config_json"].item())) == materials
    assert json.loads(str(data["spatial_basis_json"].item())) == spatial_basis
    expected_weight = 100.0 if shared_basis == "spatial80" else 0.0
    expected_factor = 0.5 if shared_basis == "spatial80" else 0.0
    assert float(data["spatial_smoothness_weight"].item()) == expected_weight
    assert float(data["spatial_smoothness_factor"].item()) == expected_factor
    return shared_basis, shared_field, materials, spatial_basis


def _shared_from_snapshot(
    data: np.lib.npyio.NpzFile,
    prepared: PreparedInputs,
    protocol: dict,
) -> tuple[SharedFieldParameters | SpatialSharedFieldParameters, dict[str, str]]:
    shared_basis, shared_field, materials, spatial_basis = _snapshot_shared_contract(
        data, protocol
    )
    if shared_basis == "spatial80":
        assert spatial_basis is not None
        shared: SharedFieldParameters | SpatialSharedFieldParameters = (
            SpatialSharedFieldParameters(
                Path(spatial_basis["basis_path"]),
                Path(spatial_basis["audit_summary_path"]),
                int(spatial_basis["mesh_cell_count"]),
                device="cpu",
                dtype=torch.float64,
            )
        )
        assert shared.basis_receipt() == spatial_basis
        assert shared.config == materials
        assert int(spatial_basis["mesh_cell_count"]) == int(
            prepared.manifest["fixture"]["tetrahedra"]
        )
    else:
        assert spatial_basis is None
        shared = SharedFieldParameters(materials, device="cpu", dtype=torch.float64)
    with torch.no_grad():
        shared.coefficients.copy_(torch.as_tensor(data["shared"], dtype=torch.float64))
    metadata_hashes = {
        "shared_field_sha256": _canonical_sha256(shared_field),
        "material_config_sha256": _canonical_sha256(materials),
        "spatial_basis_sha256": _canonical_sha256(spatial_basis),
    }
    return shared, metadata_hashes


@torch.no_grad()
def _render_shared_baseline(
    snapshot: Path,
    data: np.lib.npyio.NpzFile,
    prepared: PreparedInputs,
    protocol: dict,
    output: Path,
) -> tuple[list[str], dict]:
    """Export actual fraction-weighted shared stress and skin stiffness."""
    shared, metadata_hashes = _shared_from_snapshot(data, prepared, protocol)
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path)
    baseline = np.zeros((volume.n_cells, 3, 3), dtype=np.float64)
    tissue_maximum_principal: dict[str, np.ndarray] = {}
    for tissue in BULK_TISSUES:
        fraction = np.asarray(
            volume.cell_data[f"{tissue.title()}Fraction"], dtype=np.float64
        )
        if isinstance(shared, SpatialSharedFieldParameters):
            stress = shared.bulk_stress_mpa(tissue).detach().cpu().numpy()
        else:
            tissue_index = BULK_TISSUES.index(tissue)
            stress = shared.bulk_stresses_mpa()[tissue_index].detach().cpu().numpy()
        if stress.shape == (3, 3):
            tissue_maximum = np.full(
                volume.n_cells,
                np.linalg.eigvalsh(stress)[-1],
                dtype=np.float64,
            )
        else:
            assert stress.shape == (volume.n_cells, 3, 3)
            tissue_maximum = np.linalg.eigvalsh(stress)[:, -1]
        tissue_maximum[fraction <= 0.0] = 0.0
        tissue_maximum_principal[tissue] = tissue_maximum
        baseline += fraction[:, None, None] * stress
        del stress
    eigenvalues = np.linalg.eigvalsh(baseline)
    assert eigenvalues.shape == (volume.n_cells, 3)
    assert np.isfinite(eigenvalues).all()
    label = snapshot.stem.removeprefix("visualization-")
    rendered = volume.copy(deep=True)
    rendered.cell_data["GlobalCellId"] = np.arange(volume.n_cells, dtype=np.int64)
    rendered.cell_data["baseline_min_principal_mpa"] = eigenvalues[:, 0]
    rendered.cell_data["baseline_max_principal_mpa"] = eigenvalues[:, -1]
    for tissue, values in tissue_maximum_principal.items():
        rendered.cell_data[f"{tissue}_baseline_max_principal_mpa"] = values
    volume_name = f"shared-baseline-{label}.vtu"
    rendered.save(output / volume_name)

    effective_young_mpa = float(
        shared.config["materials"]["skin"]["reference_map"]["young_mpa"]
    ) * float(shared.skin_stiffness_multiplier().detach())
    rendered_skin = skin.copy(deep=True)
    assert np.asarray(rendered_skin.point_data["GlobalPointId"]).shape == (
        skin.n_points,
    )
    rendered_skin.point_data["effective_young_mpa"] = np.full(
        skin.n_points, effective_young_mpa, dtype=np.float64
    )
    skin_name = f"shared-skin-{label}.vtp"
    rendered_skin.save(output / skin_name)
    export = {
        "snapshot": str(snapshot),
        "snapshot_sha256": sha256(snapshot),
        "checkpoint_label": label,
        "shared_basis": protocol.get("shared_basis", "constant20"),
        **metadata_hashes,
        "volume_asset": volume_name,
        "skin_asset": skin_name,
        "volume_cells": volume.n_cells,
        "skin_points": skin.n_points,
        "stress_definition": (
            "sum_t tissue_fraction_t * reconstructed_shared_baseline_stress_t"
        ),
        "tissue_stress_definition": (
            "raw reconstructed tissue Q before fraction weighting; zero outside "
            "the tissue's positive-fraction support"
        ),
        "skin_effective_young_mpa": effective_young_mpa,
    }
    return [volume_name, skin_name], export


def render_trends(  # noqa: C901, PLR0912, PLR0915 - one report composes all families.
    trace: list[dict], output: Path
) -> list[str]:
    updates = [row["update"] for row in trace]
    assets = []

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    for name, path in {
        "total": ("objective",),
        "data": ("data_loss",),
        "neutral": ("neutral_loss",),
    }.items():
        axes[0, 0].plot(updates, _series(trace, path), label=name)
    axes[0, 0].set_title("Objective families")
    axes[0, 0].legend()
    for name in (
        "weighted_smoothness",
        "weighted_magnitude",
        "weighted_shared_prior",
        "weighted_shared_spatial_roughness",
        "weighted_jaw_prior",
    ):
        if not all(name in row for row in trace):
            continue
        axes[0, 1].plot(
            updates, _series(trace, (name,)), label=name.removeprefix("weighted_")
        )
    axes[0, 1].set_title("Weighted regularizers")
    axes[0, 1].legend(fontsize=8)
    expression_count = len(trace[0]["expressions"])
    for index in range(expression_count):
        axes[1, 0].plot(
            updates,
            [row["expressions"][index]["data"] for row in trace],
            label=f"expression {index}",
        )
        axes[1, 1].plot(
            updates,
            [row["expressions"][index]["metrics"]["area_fit_rms_mm"] for row in trace],
            label=f"expression {index}",
        )
    axes[1, 0].set_title("Data loss by expression")
    axes[1, 1].set_title("Area-weighted fit RMS (mm)")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("accepted update")
        axis.grid(alpha=0.25)
    _save(fig, output, "objective-and-fit-trends")
    assets.append("objective-and-fit-trends")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    activation_relative = np.asarray(
        [
            [
                np.nan if value is None else value
                for value in row["convergence"]["activation_relative_l2"]
            ]
            for row in trace
        ]
    )
    jaw_relative = np.asarray(
        [
            [
                np.nan if value is None else value
                for value in row["convergence"]["jaw_relative_l2"]
            ]
            for row in trace
        ]
    )
    for index in range(expression_count):
        axes[0, 0].semilogy(
            updates, activation_relative[:, index], label=f"expr {index}"
        )
        axes[0, 1].semilogy(updates, jaw_relative[:, index], label=f"expr {index}")
    axes[0, 0].axhline(0.01, color="black", linestyle="--")
    axes[0, 1].axhline(0.01, color="black", linestyle="--")
    axes[0, 0].set_title("Activation projected-gradient ratio")
    axes[0, 1].set_title("Jaw projected-gradient ratio")
    axes[1, 0].semilogy(
        updates,
        [
            (
                np.nan
                if row["convergence"]["shared_relative_l2"] is None
                else row["convergence"]["shared_relative_l2"]
            )
            for row in trace
        ],
        label="shared",
    )
    axes[1, 0].axhline(0.01, color="black", linestyle="--")
    axes[1, 0].set_title("Shared projected-gradient ratio")
    axes[1, 1].semilogy(
        updates,
        [
            (
                np.nan
                if row["convergence"]["objective_relative_span"] is None
                else row["convergence"]["objective_relative_span"]
            )
            for row in trace
        ],
        label="objective span",
    )
    axes[1, 1].axhline(1.0e-4, color="black", linestyle="--")
    axes[1, 1].set_title("Five-evaluation objective span")
    for axis in axes.flat:
        axis.set_xlabel("accepted update")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    _save(fig, output, "stationarity-trends")
    assets.append("stationarity-trends")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    axes[0, 0].plot(
        updates,
        _series(trace, ("comparison_metrics", "detF_min_all_states")),
        label="minimum",
    )
    axes[0, 0].plot(
        updates,
        _series(trace, ("comparison_metrics", "detF_max_all_states")),
        label="maximum",
    )
    axes[0, 0].set_title("Deformation-gradient determinant")
    axes[0, 0].legend()
    axes[0, 1].plot(
        updates,
        _series(trace, ("comparison_metrics", "neutral_surface_motion_rms_mm")),
        label="surface",
    )
    axes[0, 1].plot(
        updates,
        _series(
            trace,
            ("comparison_metrics", "neutral_muscle_centroid_motion_rms_mm"),
        ),
        label="muscle centroid",
    )
    axes[0, 1].set_title("Neutral motion RMS (mm)")
    axes[0, 1].legend()
    for index in range(expression_count):
        spectra = [row["activation_spectrum"]["expressions"][index] for row in trace]
        axes[1, 0].plot(
            updates,
            [item["principal_stress_quantiles_mpa"]["p99"] for item in spectra],
            label=f"expr {index}",
        )
        axes[1, 1].plot(
            updates,
            [item["upper_cap_effective_volume_fraction"] for item in spectra],
            label=f"expr {index}",
        )
    axes[1, 0].set_title("Activation p99 principal stress (MPa)")
    axes[1, 1].set_title("Effective-volume cap occupancy")
    axes[1, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel("accepted update")
        axis.grid(alpha=0.25)
    _save(fig, output, "deformation-and-activation-trends")
    assets.append("deformation-and-activation-trends")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    pose = np.asarray([row["jaw_pose_rad_m"] for row in trace])
    for expression in range(expression_count):
        for component in range(3):
            axes[0, 0].plot(
                updates,
                np.rad2deg(pose[:, expression, component]),
                label=f"e{expression} r{component}",
            )
            axes[0, 1].plot(
                updates,
                1000.0 * pose[:, expression, component + 3],
                label=f"e{expression} t{component}",
            )
    axes[0, 0].set_title("Jaw rotation-vector components (degrees)")
    axes[0, 1].set_title("Jaw translation components (mm)")
    tissues = ("fat", "aponeurosis", "muscle")
    for tissue in tissues:
        axes[1, 0].plot(
            updates,
            [
                row["shared_prior_diagnostics"]["bulk"][tissue]["normalized_frobenius"]
                for row in trace
            ],
            label=tissue,
        )
    axes[1, 0].set_title("Shared bulk normalized Frobenius")
    axes[1, 0].legend()
    axes[1, 1].plot(
        updates,
        [row["shared_prior_diagnostics"]["skin"]["resultant_n_per_m"] for row in trace],
        label="resultant N/m",
    )
    axes[1, 1].plot(
        updates,
        [
            row["shared_prior_diagnostics"]["skin"]["stiffness_multiplier"]
            for row in trace
        ],
        label="stiffness multiplier",
    )
    axes[1, 1].set_title("Shared skin parameters")
    axes[1, 1].legend()
    for axis in axes.flat:
        axis.set_xlabel("accepted update")
        axis.grid(alpha=0.25)
    _save(fig, output, "jaw-and-shared-trends")
    assets.append("jaw-and-shared-trends")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
    all_contacts = [
        [row["neutral"]["contact"], *(item["contact"] for item in row["expressions"])]
        for row in trace
    ]
    axes[0, 0].plot(
        updates,
        [sum(item["active_contact_count"] for item in group) for group in all_contacts],
    )
    axes[0, 0].set_title("Active IPC pairs across states")
    axes[0, 1].plot(
        updates,
        [sum(item["barrier_energy"] for item in group) for group in all_contacts],
    )
    axes[0, 1].set_title("Summed barrier energy")
    axes[1, 0].plot(updates, _series(trace, ("forward_seconds",)), label="forward")
    axes[1, 0].plot(updates, _series(trace, ("adjoint_seconds",)), label="adjoint")
    axes[1, 0].set_title("Solve time (s)")
    axes[1, 0].legend()
    axes[1, 1].plot(
        updates,
        np.asarray(_series(trace, ("peak_cuda_memory_bytes",))) / 2**30,
    )
    axes[1, 1].set_title("Peak CUDA allocation (GiB)")
    for axis in axes.flat:
        axis.set_xlabel("accepted update")
        axis.grid(alpha=0.25)
    _save(fig, output, "contact-and-performance-trends")
    assets.append("contact-and-performance-trends")
    return assets


def render_spatial_fields(  # noqa: PLR0915
    snapshot: Path,
    prepared: PreparedInputs,
    protocol: dict,
    reference_mpa: float,
    cap_mpa: float,
    output: Path,
) -> tuple[list[str], dict]:
    data = np.load(snapshot)
    _snapshot_shared_contract(data, protocol)
    assert str(data["stage"].item()) == protocol["stage"]
    assert str(data["checkpoint_label"].item()) == snapshot.stem.removeprefix(
        "visualization-"
    )
    assert str(data["status"].item()) == "accepted_numerically_valid_state"
    assert math.isclose(
        float(data["activation_reference_mpa"]), reference_mpa, rel_tol=0, abs_tol=0
    )
    assert math.isclose(
        float(data["activation_cap_mpa"]), cap_mpa, rel_tol=0, abs_tol=0
    )
    assert data["symmetric_coordinate_order"].tolist() == [
        "xx",
        "yy",
        "zz",
        "sqrt2_xy",
        "sqrt2_yz",
        "sqrt2_xz",
    ]
    snapshot_label = snapshot.stem.removeprefix("visualization-")
    activation = torch.as_tensor(data["activation"])
    volume = pv.read(prepared.volume_path)
    skin = pv.read(prepared.skin_path)
    active_ids = data["active_cell_ids"]
    global_skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    observation_ids = data["observation_node_ids"]
    full_displacement = data["full_displacement_m"]
    assert full_displacement.shape == (
        activation.shape[0],
        volume.n_points,
        3,
    )
    assert full_displacement.dtype == np.float64
    assert np.isfinite(full_displacement).all()
    assert int(observation_ids.min()) >= 0
    assert int(observation_ids.max()) < volume.n_points
    assert np.array_equal(
        data["predicted_observation_displacement_m"],
        full_displacement[:, observation_ids],
    )
    observation_lookup = {
        int(value): index for index, value in enumerate(observation_ids)
    }
    assets = []
    for expression in range(activation.shape[0]):
        eigenvalues_tensor, eigenvectors_tensor = torch.linalg.eigh(
            symmetric_matrices(activation[expression])
        )
        eigenvalues = (reference_mpa * eigenvalues_tensor).numpy()
        principal_direction = eigenvectors_tensor[..., :, -1]
        largest_component = principal_direction.abs().argmax(dim=-1, keepdim=True)
        largest_sign = torch.gather(principal_direction, -1, largest_component).sign()
        largest_sign = torch.where(
            largest_sign == 0, torch.ones_like(largest_sign), largest_sign
        )
        principal_direction = (principal_direction * largest_sign).numpy()
        full_shape = volume.n_cells
        fields = {
            "activation_max_principal_mpa": eigenvalues[:, -1],
            "activation_trace_mpa": eigenvalues.sum(axis=1),
            "activation_anisotropy_mpa": eigenvalues[:, -1] - eigenvalues[:, 0],
            "activation_cap_occupancy": eigenvalues[:, -1] >= cap_mpa * (1 - 1.0e-6),
            "activation_max_principal_direction_xyz": principal_direction,
        }
        rendered = volume.copy(deep=True)
        for name, values in fields.items():
            fill = np.full((full_shape, *values.shape[1:]), np.nan)
            fill[active_ids] = values
            rendered.cell_data[name] = fill
        global_cell_ids = np.asarray(
            rendered.cell_data.get("GlobalCellId", np.arange(full_shape)),
            dtype=np.int64,
        )
        assert global_cell_ids.shape == (full_shape,)
        rendered.cell_data["GlobalCellId"] = global_cell_ids
        active_cell_index = np.full(full_shape, -1, dtype=np.int64)
        active_cell_index[active_ids] = active_ids
        rendered.cell_data["active_cell_index"] = active_cell_index
        volume_name = f"activation-{snapshot_label}-expression-{expression}.vtu"
        rendered.save(output / volume_name)
        assets.append(volume_name)

        residual = np.full(skin.n_points, np.nan)
        predicted = data["predicted_observation_displacement_m"][expression]
        target = data["target_displacement_m"][expression]
        residual_observation = 1000.0 * np.linalg.norm(predicted - target, axis=1)
        for skin_index, global_id in enumerate(global_skin_ids):
            observation_index = observation_lookup.get(int(global_id))
            if observation_index is not None:
                residual[skin_index] = residual_observation[observation_index]
        rendered_skin = skin.copy(deep=True)
        rendered_skin.point_data["surface_residual_mm"] = residual
        skin_name = f"surface-residual-{snapshot_label}-expression-{expression}.vtp"
        rendered_skin.save(output / skin_name)
        assets.append(skin_name)
    baseline_assets, export = _render_shared_baseline(
        snapshot,
        data,
        prepared,
        protocol,
        output,
    )
    assets.extend(baseline_assets)
    data.close()
    return assets, export


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = cfg.run_dir / "trace.json"
    protocol_path = cfg.run_dir / "protocol.json"
    trace = json.loads(trace_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz",
        cfg.prepared_dir / "manifest.json",
    )
    assert protocol["input_arrays_sha256"] == sha256(cfg.prepared_dir / "inputs.npz")
    assert protocol["input_manifest_sha256"] == sha256(
        cfg.prepared_dir / "manifest.json"
    )
    assets = render_trends(trace, cfg.output_dir)
    constraints = protocol["materials"]["constraints"]
    snapshots = sorted(cfg.run_dir.glob("visualization-*.npz"))
    assert snapshots
    field_exports = []
    for snapshot in snapshots:
        snapshot_assets, export = render_spatial_fields(
            snapshot,
            prepared,
            protocol,
            float(constraints["activation_reference_mpa"]),
            float(constraints["activation_cap_mpa"]),
            cfg.output_dir,
        )
        assets.extend(snapshot_assets)
        field_exports.append(export)
    receipt = {
        "success": True,
        "schema": "joint-final-visualization-summary-v1",
        "run_dir": str(cfg.run_dir),
        "trace_rows": len(trace),
        "snapshots": len(snapshots),
        "assets": assets,
        "shared_basis": protocol.get("shared_basis", "constant20"),
        "shared_field": protocol.get("shared_field"),
        "spatial_basis": protocol.get("spatial_basis"),
        "field_exports": field_exports,
        "trace_sha256": sha256(trace_path),
        "protocol_sha256": sha256(protocol_path),
        "anatomical_validation": False,
        "promotion_ready": False,
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
