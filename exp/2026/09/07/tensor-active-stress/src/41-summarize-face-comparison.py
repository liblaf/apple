"""Verify completed face evidence and plot the fixed-budget comparison."""

# ruff: noqa: PLR0915

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
ARMS = (
    ("raw6", "Raw6 reference", "20-raw6-reference", "#737b83"),
    ("psd", "PSD", "21-psd", "#0072b2"),
    ("psd-smooth", "PSD + smoothness", "22-psd-smooth", "#009e73"),
    ("psd-smooth-rank", "PSD + smoothness + rank", "23-psd-smooth-rank", "#c47a05"),
)
CORE_SOURCES = (
    "20-face-inverse.py",
    "tensor_active.py",
    "tensor_controls.py",
    "face_physics.py",
    "experiment_profile.py",
)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    measurements: Path = HERE / "data/40-surface-measurements/summary.json"
    settings: Path = HERE / "data/14-frozen-face-settings/summary.json"
    output_dir: Path = cherries.output("41-face-comparison", mkdir=True)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def finish_plot(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path.with_suffix(".png"), dpi=200, facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)


def clean_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(alpha=0.18, axis="y")


def unpack_tensor_coordinates(q: np.ndarray) -> np.ndarray:
    """Map orthonormal tensor coordinates to symmetric matrices."""
    assert q.ndim == 2
    assert q.shape[1] == 6
    result = np.empty((len(q), 3, 3), dtype=q.dtype)
    result[:, 0, 0] = q[:, 0]
    result[:, 1, 1] = q[:, 1]
    result[:, 2, 2] = q[:, 2]
    result[:, 0, 1] = result[:, 1, 0] = q[:, 3] / np.sqrt(2.0)
    result[:, 1, 2] = result[:, 2, 1] = q[:, 4] / np.sqrt(2.0)
    result[:, 0, 2] = result[:, 2, 0] = q[:, 5] / np.sqrt(2.0)
    return result


def raw6_matrices(q: np.ndarray) -> np.ndarray:
    """Reconstruct the historical direct symmetric active-strain offset."""
    scaled = q.copy()
    scaled[:, 3:] *= np.sqrt(2.0)
    return np.eye(3)[None] + unpack_tensor_coordinates(scaled)


def normalize_csv_bool(value: Any) -> bool:
    """Normalize the boolean representations accepted by a CSV round trip."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    assert isinstance(value, str)
    normalized = value.lower()
    assert normalized in {"true", "false"}
    return normalized == "true"


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a trace without adding a dataframe dependency to post-processing."""
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def numeric_column(records: list[dict[str, str]], field: str) -> np.ndarray:
    """Read one numeric CSV column, preserving blank values as NaN."""
    return np.asarray(
        [float(row[field]) if row[field] else np.nan for row in records],
        dtype=np.float64,
    )


def assert_csv_record(json_record: dict[str, Any], csv_record: dict[str, str]) -> None:
    """Require a JSON record to equal its CSV round trip."""
    assert set(json_record) == set(csv_record)
    for key, expected in json_record.items():
        actual = csv_record[key]
        if expected is None:
            assert not actual or actual.lower() == "nan", (key, actual, expected)
        elif isinstance(expected, bool):
            assert normalize_csv_bool(actual) is expected, (key, actual, expected)
        elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
            assert np.isclose(float(actual), float(expected), rtol=1e-12, atol=1e-12), (
                key,
                actual,
                expected,
            )
        else:
            assert actual == str(expected), (key, actual, expected)


def assert_close(actual: float | None, expected: float | None) -> None:
    """Compare an optional scalar using the frozen receipt tolerance."""
    if expected is None:
        assert actual is None
    else:
        assert actual is not None
        assert np.isclose(actual, expected, rtol=1e-12, atol=1e-12), (
            actual,
            expected,
        )


def assert_endpoint_artifacts(
    root: Path,
    identity: str,
    summary: dict[str, Any],
) -> str:
    """Bind controls, NPZ state, and VTK fields to one saved endpoint."""
    expected_model = "raw6" if identity == "raw6" else "tensor"
    assert summary["config"]["model"] == expected_model
    with np.load(root / "final.npz") as endpoint:
        assert int(endpoint["step"]) == 64
        assert normalize_csv_bool(endpoint["solver_valid"].item())
        q = np.asarray(endpoint["q"])
        u = np.asarray(endpoint["u"])
        active_ids = np.asarray(endpoint["active_ids"], dtype=np.int64)
        assert q.shape == (288235, 6)
        assert u.ndim == 2
        assert u.shape[1] == 3
        assert active_ids.shape == (288235,)
        assert np.isfinite(q).all()
        assert np.isfinite(u).all()
        if expected_model == "tensor":
            assert "Q" in endpoint.files
            assert "Ainv" not in endpoint.files
            active = np.asarray(endpoint["Q"])
            expected = float(summary["config"]["stress_reference_mpa"]) * (
                unpack_tensor_coordinates(q)
            )
            assert active.shape == (288235, 3, 3)
            assert np.allclose(active, expected, rtol=1e-12, atol=1e-12)
            assert np.max(np.abs(active - active.transpose(0, 2, 1))) < 1e-12
            eig = np.linalg.eigvalsh(active)
            assert eig.min() >= -1e-10
            assert eig.max() <= summary["config"]["stress_cap_mpa"] + 1e-10
        else:
            assert "Ainv" in endpoint.files
            assert "Q" not in endpoint.files
            active = np.asarray(endpoint["Ainv"])
            expected = raw6_matrices(q)
            assert active.shape == (288235, 3, 3)
            assert np.allclose(active, expected, rtol=1e-12, atol=1e-12)

    mesh = pv.read(root / "final.vtu")
    assert isinstance(mesh, pv.UnstructuredGrid)
    rest = np.asarray(mesh.point_data["RestPosition"])
    displacement = np.asarray(mesh.point_data["Displacement"])
    assert mesh.n_points == len(u)
    assert rest.shape == displacement.shape
    assert displacement.shape == u.shape
    assert np.array_equal(displacement, u)
    assert np.array_equal(np.asarray(mesh.points), rest + u)
    activation_mask = np.asarray(mesh.cell_data["ActivationMask"], dtype=bool)
    muscle_fraction = np.asarray(mesh.cell_data["MuscleFraction"])
    assert np.array_equal(activation_mask, muscle_fraction > 0)
    expected_ids = np.flatnonzero(activation_mask)
    assert np.array_equal(active_ids, expected_ids)
    assert np.all(mesh.celltypes == pv.CellType.TETRA)
    tets = np.asarray(mesh.cells).reshape(-1, 5)[:, 1:][active_ids]
    dm = np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1))
    volumes = np.linalg.det(dm) / 6.0
    assert np.all(volumes > 0)
    volumes *= muscle_fraction[active_ids]
    mass = volumes / volumes.sum()
    primary = summary["primary_endpoint"]
    coordinate_weights = (
        np.ones(6) if expected_model == "tensor" else np.asarray((1, 1, 1, 2, 2, 2))
    )
    q_norm2 = np.sum(coordinate_weights * q**2, axis=1)
    assert_close(primary["magnitude"], float(np.sum(mass * q_norm2)))
    assert_close(
        primary["rank_penalty"],
        float(np.sum(mass * (q[:, :3].sum(axis=1) ** 2 - q_norm2))),
    )
    if expected_model == "tensor":
        assert "ActivationInverseMatrix" not in mesh.cell_data
        field = np.asarray(mesh.cell_data["ActiveStressMatrixMPa"]).reshape(-1, 3, 3)
        assert np.array_equal(field[active_ids], active)
        assert np.count_nonzero(field[~activation_mask]) == 0
        assert np.array_equal(
            np.asarray(mesh.cell_data["ActiveStressTraceMPa"]),
            np.trace(field, axis1=1, axis2=2),
        )
        eig = np.linalg.eigvalsh(active)
        trace = eig.sum(axis=1)
        trace_squared_mean = float(np.sum(mass * trace**2))
        trace_mean = float(np.sum(mass * trace))
        cap = float(summary["config"]["stress_cap_mpa"])
        qref = float(summary["config"]["stress_reference_mpa"])
        assert_close(primary["Q_eigen_min_mpa"], float(eig.min()))
        assert_close(primary["Q_eigen_max_mpa"], float(eig.max()))
        assert_close(primary["Q_trace_mean_mpa"], trace_mean)
        assert_close(
            primary["Q_rms_mpa"],
            float(np.sqrt(np.sum(mass * np.sum(eig**2, axis=1)))),
        )
        assert_close(
            primary["upper_cap_cell_fraction"],
            float(np.mean(eig[:, -1] >= cap * (1 - 1e-7))),
        )
        assert_close(
            primary["principal_tension_fraction"],
            float(np.sum(mass * eig[:, -1]) / trace_mean) if trace_mean > 0 else None,
        )
        assert_close(
            primary["rank_mixing_fraction"],
            float(
                np.sum(mass * (trace**2 - np.sum(eig**2, axis=1))) / trace_squared_mean
            )
            if trace_squared_mean > 0
            else None,
        )
        assert_close(
            primary["zero_stress_cell_fraction"],
            float(np.mean(eig[:, -1] <= qref * 1e-8)),
        )
    else:
        assert "ActiveStressMatrixMPa" not in mesh.cell_data
        field = np.asarray(mesh.cell_data["ActivationInverseMatrix"]).reshape(-1, 3, 3)
        assert np.array_equal(field[active_ids], active)
        assert np.array_equal(
            field[~activation_mask],
            np.broadcast_to(np.eye(3), field[~activation_mask].shape),
        )
        assert np.allclose(
            np.asarray(mesh.cell_data["DetAinv"]),
            np.linalg.det(field),
            rtol=1e-12,
            atol=1e-12,
        )
        eig = np.linalg.eigvalsh(active)
        assert_close(primary["Ainv_eigen_min"], float(eig.min()))
        assert_close(primary["Ainv_eigen_max"], float(eig.max()))
        assert primary["non_spd_activation_tets"] == int(
            np.count_nonzero(eig[:, 0] <= 0)
        )
    detf = np.asarray(mesh.cell_data["DetF"])
    assert np.isclose(detf.min(), primary["detF_min"], rtol=1e-12, atol=1e-12)
    assert np.isclose(detf.max(), primary["detF_max"], rtol=1e-12, atol=1e-12)
    assert np.count_nonzero(detf <= 0) == primary["inverted_tetrahedra"]
    return hashlib.sha256(active_ids.tobytes()).hexdigest()


def plot_traces(traces: dict[str, list[dict[str, str]]], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), layout="constrained")
    fields = (
        ("area_fit_rms_mm", "Area-weighted fitting error", "RMS (mm)"),
        ("area_motion_rms_mm", "Realized motion", "RMS (mm)"),
        (
            "target_projection",
            "Motion projected onto the target",
            "projection coefficient",
        ),
        (
            "relative_smoothness",
            "Tensor variation / tensor magnitude",
            "S / M (dimensionless)",
        ),
        ("Q_rms_mpa", "Active-stress magnitude", "volume RMS (MPa)"),
        (
            "rank_mixing_fraction",
            "Mixing of principal tensions",
            "B / volume mean(trace Z squared)",
        ),
    )
    for identity, label, _, color in ARMS:
        trace = traces[identity]
        magnitude = numeric_column(trace, "magnitude")
        relative_smoothness = np.divide(
            numeric_column(trace, "smoothness"),
            magnitude,
            out=np.full(len(trace), np.nan),
            where=magnitude > 0,
        )
        step = numeric_column(trace, "step")
        for index, (field, title, ylabel) in enumerate(fields):
            axis = axes.flat[index]
            if index < 3 or identity != "raw6":
                values = (
                    relative_smoothness
                    if field == "relative_smoothness"
                    else numeric_column(trace, field)
                )
                axis.plot(step, values, color=color, label=label, lw=1.8)
            axis.set(title=title, xlabel="Adam updates", ylabel=ylabel, xlim=(0, 64))
            clean_axes(axis)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle(
        "64 updates from rest · all active-tetrahedron controls · no skin energy",
        fontsize=15,
    )
    finish_plot(fig, out / "optimization-traces")


def surface_row(case: dict[str, Any]) -> dict[str, Any]:
    surface = case["surface"]
    result = {"id": case["id"], "label": case["label"]}
    for field in ("normal_displacement", "normal_residual"):
        for roi in ("full_face", "mouth_10mm"):
            value = surface["primary_5mm"][field][roi]
            result[f"{field}_{roi}_5mm_hp_rms_mm"] = value["highpass_rms_mm"]
            result[f"{field}_{roi}_5mm_hp_ratio"] = value[
                "highpass_over_total_normal_rms"
            ]["ratio"]
    return result


def plot_surface(rows: list[dict[str, Any]], out: Path) -> None:
    order = [identity for identity, *_ in ARMS] + ["target"]
    by_id = {row["id"]: row for row in rows}
    labels = ["Raw6", "PSD", "PSD + smooth", "PSD + smooth\n+ rank", "Target"]
    colors = [color for *_, color in ARMS] + ["#aa4499"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.8), layout="constrained")
    for row_index, roi in enumerate(("full_face", "mouth_10mm")):
        for col_index, field in enumerate(("normal_displacement", "normal_residual")):
            axis = axes[row_index, col_index]
            values = [by_id[key][f"{field}_{roi}_5mm_hp_rms_mm"] for key in order]
            bars = axis.bar(labels, values, color=colors, width=0.65)
            axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
            axis.set(
                title=("Full face" if row_index == 0 else "Mouth, intrinsic 10 mm")
                + (" · displacement" if col_index == 0 else " · target residual"),
                ylabel="5 mm normal high-pass RMS (mm)",
            )
            axis.tick_params(axis="x", labelsize=8)
            axis.margins(y=0.2)
            clean_axes(axis)
    fig.suptitle(
        "Saved step-64 surfaces and the target · one frozen rest-surface operator",
        fontsize=14,
    )
    finish_plot(fig, out / "surface-highpass")


def main(cfg: Config) -> None:
    global COMPLETED  # noqa: PLW0603
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir())
    settings = json.loads(cfg.settings.read_text())
    measurements = json.loads(cfg.measurements.read_text())
    assert measurements["status"] == "completed"
    assert measurements["protocol"]["primary_scale_mm"] == 5.0
    assert measurements["protocol"]["rois"] == ["full_face", "mouth_10mm"]
    assert measurements["protocol"]["fields"] == [
        "normal_displacement",
        "normal_target_residual",
    ]
    assert len(measurements["cases"]) == 5
    measured = {row["id"]: row for row in measurements["cases"]}
    assert set(measured) == {"target", *(identity for identity, *_ in ARMS)}
    traces = {}
    comparisons = []
    initialization_hashes = set()
    active_id_hashes = set()
    fixture_hashes = set()
    core_sources: dict[str, set[str]] = {name: set() for name in CORE_SOURCES}
    optimizer_settings = set()
    passive_materials = set()
    summary_records = {}
    for identity, label, directory, _ in ARMS:
        root = HERE / "data" / directory
        summary = json.loads((root / "summary.json").read_text())
        trace = read_csv(root / "trace.csv")
        assert summary["status"] == "completed_fixed_budget"
        assert not summary["inverse_convergence_claimed"]
        assert summary["geometry_rejection_enabled"] is False
        assert summary["fiber_directions_used"] is False
        assert np.array_equal(numeric_column(trace, "step"), np.arange(65))
        assert all(normalize_csv_bool(row["solver_valid"]) for row in trace)
        assert int(trace[-1]["step"]) == summary["primary_endpoint"]["step"] == 64
        assert_csv_record(summary["primary_endpoint"], trace[-1])
        receipts = [
            json.loads(line)
            for line in (root / "solver-receipts.jsonl").read_text().splitlines()
        ]
        assert len(receipts) == 65
        assert [item["step"] for item in receipts] == list(range(65))
        assert all(
            item["forward"]["success"] is True and item["adjoint"]["success"] is True
            for item in receipts
        )
        active_id_hashes.add(assert_endpoint_artifacts(root, identity, summary))
        initialization_hashes.add(summary["initialization"]["control_sha256"])
        selection = settings["selection"]
        optimizer_settings.add(
            (
                summary["config"]["steps"],
                summary["config"]["learning_rate"],
                summary["config"]["adam_eps"],
            )
        )
        assert summary["config"]["steps"] == selection["final_steps"] == 64
        assert summary["config"]["learning_rate"] == selection["learning_rate"]
        assert summary["config"]["adam_eps"] == selection["adam_eps"]
        assert summary["materials"]["skin_E_MPa"] == 0.0
        assert summary["materials"]["contact_enabled"] is False
        passive_materials.add(
            json.dumps(
                {
                    key: value
                    for key, value in summary["materials"].items()
                    if key != "muscle_model"
                },
                sort_keys=True,
            )
        )
        fixture_hashes.add(
            tuple(
                summary["provenance"]["inputs"][name]["sha256"]
                for name in ("volume.vtu", "skin.vtp", "summary.json")
            )
        )
        assert measured[identity]["endpoint"]["sha256"] == sha256(root / "final.vtu")
        if identity != "raw6":
            assert summary["materials"]["muscle_model"] == "stable-tensor-active"
            assert (
                summary["config"]["stress_reference_mpa"]
                == selection["stress_reference_mpa"]
            )
            assert summary["config"]["stress_cap_mpa"] == selection["stress_cap_mpa"]
            assert summary["config"]["smooth_length_m"] == selection["smooth_length_m"]
            for key in ("smoothness_weight", "rank_weight", "magnitude_weight"):
                assert summary["config"][key] == settings["arms"][identity][key]
        else:
            assert summary["materials"]["muscle_model"] == "stable-active-strain"
            assert all(
                summary["config"][key] == 0.0
                for key in ("smoothness_weight", "rank_weight", "magnitude_weight")
            )
        for name in CORE_SOURCES:
            core_sources[name].add(
                summary["provenance"]["sources"][f"experiment/{name}"]
            )
        primary = summary["primary_endpoint"]
        row = {
            "id": identity,
            "label": label,
            "updates": 64,
            "area_fit_rms_mm": primary["area_fit_rms_mm"],
            "uniform_fit_rms_mm": primary["fit_rms_mm"],
            "area_motion_rms_mm": primary["area_motion_rms_mm"],
            "target_projection": primary["target_projection"],
            "detF_min": primary["detF_min"],
            "detF_max": primary["detF_max"],
            "inverted_tetrahedra": primary["inverted_tetrahedra"],
            "forward_and_adjoint_valid_states": len(receipts),
            "graph_variation": primary["smoothness"],
            "control_magnitude": primary["magnitude"],
            "relative_graph_variation": primary["smoothness"] / primary["magnitude"],
            "Q_rms_mpa": primary.get("Q_rms_mpa"),
            "Q_trace_mean_mpa": primary.get("Q_trace_mean_mpa"),
            "Q_eigen_max_mpa": primary.get("Q_eigen_max_mpa"),
            "upper_cap_cell_fraction": primary.get("upper_cap_cell_fraction"),
            "rank_mixing_fraction": primary.get("rank_mixing_fraction"),
            "principal_tension_fraction": primary.get("principal_tension_fraction"),
            "wall_s": summary["wall_s"],
            **{
                key: value
                for key, value in surface_row(measured[identity]).items()
                if key not in {"id", "label"}
            },
        }
        comparisons.append(row)
        traces[identity] = trace
        summary_records[identity] = {
            "directory": str(root),
            "summary_sha256": sha256(root / "summary.json"),
            "trace_sha256": sha256(root / "trace.csv"),
            "endpoint_npz_sha256": sha256(root / "final.npz"),
            "endpoint_vtu_sha256": sha256(root / "final.vtu"),
            "solver_receipts_sha256": sha256(root / "solver-receipts.jsonl"),
            "materials": summary["materials"],
        }
    assert (
        len(initialization_hashes)
        == len(active_id_hashes)
        == len(fixture_hashes)
        == len(optimizer_settings)
        == len(passive_materials)
        == 1
    )
    assert all(len(values) == 1 for values in core_sources.values())
    rows = [surface_row(measured[identity]) for identity, *_ in ARMS] + [
        surface_row(measured["target"])
    ]
    for name, records in (
        ("endpoint-comparison.csv", comparisons),
        ("surface-comparison.csv", rows),
    ):
        with (out / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42}
    )
    plot_traces(traces, out)
    plot_surface(rows, out)
    summary = {
        "status": "verified_completed_fixed_budget_comparison",
        "scope": "three internally matched tensor arms and one historical-model Raw6 reference, all from rest at actual update64",
        "checks": {
            "all_states_forward_and_adjoint_valid": True,
            "all_arms_have_65_states_including_rest": True,
            "common_fixture_active_ids_zero_control_initialization_and_optimizer": True,
            "active_ids_are_exactly_positive_muscle_fraction": True,
            "common_passive_materials_and_zero_skin_energy": True,
            "core_source_hashes_identical_across_all_runs": {
                key: next(iter(values)) for key, values in core_sources.items()
            },
            "all_tensor_endpoints_symmetric_and_in_spectral_box": True,
            "controls_npz_state_and_vtk_fields_are_bound": True,
            "surface_endpoint_hashes_match_final_meshes": True,
        },
        "settings": settings["selection"],
        "comparison_limit": settings["comparison_limit"],
        "cases": comparisons,
        "target_surface": surface_row(measured["target"]),
        "provenance": {
            "runs": summary_records,
            "measurement_summary_sha256": sha256(cfg.measurements),
            "settings_summary_sha256": sha256(cfg.settings),
            "script_sha256": sha256(Path(__file__)),
        },
    }
    write_json(out / "summary.json", summary)
    shutil.copy2(Path(__file__), out / Path(__file__).name)
    for path in sorted(out.iterdir()):
        if path.is_file():
            cherries.log_output(path)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
