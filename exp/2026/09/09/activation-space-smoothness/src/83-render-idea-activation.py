"""Render a common activation-field comparison for the selected model states."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, RUF001, TRY003

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from muscle_glyph_context import (
    MuscleRegionContext,
    RegionVisibility,
    build_muscle_region_context,
    save_muscle_region_context,
    save_region_visibility,
    set_parallel_camera,
    visible_region_mask,
)

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[6]
GROUP = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
SELECTION = GROUP / "data/80-idea-result-selection/selection.json"
AUDIT = GROUP / "data/82-idea-activation-audit-v2/summary.json"
OUTPUT = GROUP / "data/83-idea-activation"
VIEWS = ("side-context", "region1-mouth-corner")
WINDOW_SIZE = (1800, 1800)
CONTEXT_OPACITY = 0.06
BACKGROUND = "#f4f2ed"
DISPLAY_MAX_LENGTH_M = 0.0045
BACKWARD_ERROR_FACTOR = 64.0
MU_MPA = 0.03 / (2.0 * (1.0 + 0.49))
QREF_MPA = 3.0 * MU_MPA
LABELS = {
    "raw6-refit-off-200": "Corrected Raw6 refit · smoothness off",
    "raw6-refit-on-200": "Corrected Raw6 refit · smoothness on",
    "axis-on-128": "Learned axis · smoothness on",
    "axis-off-best-15": "Learned axis · smoothness off",
    "raw6-corrected-rest-start-200": "Corrected Raw6 · corrected-rest start",
    "psd-off-1024": "PSD active stress · smoothness off",
}


class Config(cherries.BaseConfig):
    """Output location and an optional one-state preview."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = OUTPUT
    state_id: str | None = None


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "bytes": resolved.stat().st_size,
        "sha256": _digest(resolved),
    }


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _snapshot(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    destination.chmod(0o444)
    if _digest(source) != _digest(destination):
        raise AssertionError("source snapshot differs from executed source")
    return {"live_at_generation": _record(source), "snapshot": _record(destination)}


def _raw6_b(q: np.ndarray) -> np.ndarray:
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError("Raw6 q must have shape (n, 6)")
    b = np.zeros((len(q), 3, 3), dtype=np.float64)
    b[:, 0, 0], b[:, 1, 1], b[:, 2, 2] = q[:, 0], q[:, 1], q[:, 2]
    b[:, 0, 1] = b[:, 1, 0] = q[:, 3]
    b[:, 1, 2] = b[:, 2, 1] = q[:, 4]
    b[:, 0, 2] = b[:, 2, 0] = q[:, 5]
    b += np.eye(3)
    return b


def _z_for(identifier: str, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    if identifier.startswith("raw6-"):
        b = _raw6_b(arrays["q"])
        z = b @ np.swapaxes(b, 1, 2) - np.eye(3)
        if "Ainv" in arrays and not np.allclose(b, arrays["Ainv"], atol=1e-13):
            raise ValueError(f"Raw6 B reconstruction disagrees with Ainv: {identifier}")
        if "Z" in arrays and not np.allclose(z, arrays["Z"], atol=1e-12):
            raise ValueError(
                f"Raw6 Z reconstruction disagrees with stored Z: {identifier}"
            )
        return z, "Z=B B^T-I from q"
    if identifier.startswith("axis-"):
        v = arrays["q"]
        z = (2.0 + np.sum(v * v, axis=1)[:, None, None]) * np.einsum("ni,nj->nij", v, v)
        if not np.allclose(z, arrays["Z"], atol=1e-12):
            raise ValueError(
                f"axis Z reconstruction disagrees with stored Z: {identifier}"
            )
        return z, "Z=(2+||v||^2)vv^T from saved v"
    if identifier == "psd-off-1024":
        q = arrays["q"]
        q_matrix = np.empty((len(q), 3, 3), dtype=np.float64)
        root2 = np.sqrt(2.0)
        q_matrix[:, 0, 0], q_matrix[:, 1, 1], q_matrix[:, 2, 2] = q.T[:3]
        q_matrix[:, 0, 1] = q_matrix[:, 1, 0] = q[:, 3] / root2
        q_matrix[:, 1, 2] = q_matrix[:, 2, 1] = q[:, 4] / root2
        q_matrix[:, 0, 2] = q_matrix[:, 2, 0] = q[:, 5] / root2
        if not np.allclose(arrays["Q"], QREF_MPA * q_matrix, atol=1e-13):
            raise ValueError("PSD Q disagrees with QREF times orthonormal q")
        return arrays["Q"] / MU_MPA, "Z=Q/mu from saved Q"
    raise ValueError(f"unknown state: {identifier}")


def _line_mesh(
    centroids: np.ndarray,
    axes: np.ndarray,
    amplitude: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> pv.PolyData:
    half = (0.5 * DISPLAY_MAX_LENGTH_M * amplitude)[:, None] * axes
    points = np.empty((2 * len(centroids), 3), dtype=np.float64)
    points[0::2] = centroids - half
    points[1::2] = centroids + half
    lines = np.column_stack(
        (
            np.full(len(centroids), 2, dtype=np.int64),
            2 * np.arange(len(centroids)),
            2 * np.arange(len(centroids)) + 1,
        )
    ).ravel()
    mesh = pv.PolyData(points, lines=lines)
    for name, value in arrays.items():
        mesh.cell_data[name] = value
    mesh.field_data["CoordinateFrame"] = np.asarray(["saved deformed coordinates"])
    mesh.field_data["LineSemantics"] = np.asarray(
        ["centered unoriented dominant positive eigenmode of Z, transported by F"]
    )
    mesh.field_data["LengthSemantics"] = np.asarray(
        ["0.0045 m * (1 - 1/sqrt(1 + max(lambda_max(Z), 0))); no floor"]
    )
    mesh.field_data["OmittedSquaredNormSemantics"] = np.asarray(
        ["1 - max(lambda_max(Z),0)^2 / ||Z||_F^2; zero only for exactly zero Z"]
    )
    return mesh


def _annotation(plotter: pv.Plotter, text: str) -> None:
    actor = plotter.add_text(text, position="upper_left", color="black", font_size=12)
    prop = actor.GetTextProperty()
    prop.SetBackgroundColor(244 / 255, 242 / 255, 237 / 255)
    prop.SetBackgroundOpacity(0.90)


def _render_geometry(
    skin: pv.PolyData,
    glyphs: pv.PolyData,
    visibility: RegionVisibility,
    camera: dict[str, Any],
    label: str,
    step: int,
    path: Path,
) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=WINDOW_SIZE, lighting="three lights"
    )
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(
        skin, color="#8d969b", opacity=CONTEXT_OPACITY, smooth_shading=False
    )
    shown = glyphs.extract_cells(visibility.mask)
    if shown.n_cells != visibility.retained_count:
        raise AssertionError("visible glyph extraction changed the retained count")
    plotter.add_mesh(
        shown,
        scalars="DominantModeAmplitudePercent",
        cmap="viridis",
        clim=(0.0, 100.0),
        lighting=False,
        line_width=1.0,
        render_lines_as_tubes=False,
        scalar_bar_args={
            "title": "Mode amplitude (%)",
            "color": "black",
            "title_font_size": 17,
            "label_font_size": 14,
            "n_labels": 5,
            "vertical": True,
            "background_color": BACKGROUND,
            "fill": True,
            "position_x": 0.80,
            "position_y": 0.10,
            "width": 0.08,
            "height": 0.45,
        },
    )
    _annotation(
        plotter,
        f"{label} · update {step}\n"
        "Dominant positive tensor mode on saved deformed shape\n"
        "One centered line per visible active tetrahedron · 100% = 4.5 mm",
    )
    set_parallel_camera(plotter, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _companion_surface(
    muscles: MuscleRegionContext,
    full_ids: np.ndarray,
    omitted_percent: np.ndarray,
    cell_count: int,
) -> pv.PolyData:
    lookup = np.full(cell_count, np.nan, dtype=np.float64)
    lookup[full_ids] = omitted_percent
    surface = muscles.combined.copy(deep=True)
    source_ids = np.asarray(
        surface.cell_data["SourceTetraGlobalCellId"], dtype=np.int64
    )
    values = lookup[source_ids]
    if not np.all(np.isfinite(values)):
        raise ValueError(
            "muscle surface contains a tetrahedron outside the active field"
        )
    surface.cell_data["NonDominantSquaredNormPercent"] = values
    return surface


def _render_companion(
    skin: pv.PolyData,
    surface: pv.PolyData,
    camera: dict[str, Any],
    label: str,
    step: int,
    path: Path,
) -> None:
    plotter = pv.Plotter(
        off_screen=True, window_size=WINDOW_SIZE, lighting="three lights"
    )
    plotter.set_background(BACKGROUND)
    plotter.add_mesh(
        skin, color="#8d969b", opacity=CONTEXT_OPACITY, smooth_shading=False
    )
    plotter.add_mesh(
        surface,
        scalars="NonDominantSquaredNormPercent",
        preference="cell",
        cmap="magma",
        clim=(0.0, 100.0),
        opacity=1.0,
        lighting=False,
        show_edges=False,
        smooth_shading=False,
        scalar_bar_args={
            "title": "Omitted norm² (%)",
            "color": "black",
            "title_font_size": 17,
            "label_font_size": 14,
            "n_labels": 5,
            "vertical": True,
            "background_color": BACKGROUND,
            "fill": True,
            "position_x": 0.80,
            "position_y": 0.10,
            "width": 0.08,
            "height": 0.45,
        },
    )
    _annotation(
        plotter,
        f"{label} · update {step}\n"
        "Squared Frobenius magnitude outside displayed line\n"
        "Opaque front muscle surfaces · fixed 0–100% scale",
    )
    set_parallel_camera(plotter, camera)
    path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(path)
    plotter.close()


def _readme(state_ids: list[str]) -> str:
    rows = "\n".join(
        f"- `{identifier}`: {LABELS[identifier]}" for identifier in state_ids
    )
    return f"""# Common activation fields on saved deformed shapes

Each complete VTP contains one centered line for every one of the 288,235 active
tetrahedra, without spatial sampling. All models are converted to the same
symmetric effective tensor Z. Let lambda_plus=max(lambda_max(Z),0) and let n be
the corresponding rest-coordinate eigenvector. The displayed amplitude is
a=1-1/sqrt(1+lambda_plus). Each line is centered at the deformed tetrahedron
centroid, points along normalize(F@n), and has length 0.0045 m*a. Its sign is
arbitrary. Color is a on one fixed linear 0–100% range for every state.

The companion panels show
1-lambda_plus^2/||Z||_F^2 on opaque deformed muscle-region surfaces, with a fixed
0–100% range. The value is defined as zero only for exactly zero Z. This quantity
is the squared tensor magnitude omitted by the single line. It is not mechanical
strain energy. Raw6 negative modes remain part of Z and therefore count toward
the omitted magnitude.

The deformed centers and directions use the saved displacement without
amplification. F maps reference tetrahedron edge columns to their saved deformed
edge columns. The faint skin has opacity 0.06. Primary views use a state-specific
opaque region-ID raster: a tetrahedron line is retained exactly when the front
region at its projected centroid equals its ActivationControlId. Same-region
interior tetrahedra remain visible; lines hidden by another region are removed.
No muscle surfaces or outlines appear behind the primary lines. Companion views
use opaque muscle surfaces without edges or silhouettes.

States:
{rows}

`glyphs/<state>.vtp` stores Z, all eigenvalues and eigenvectors, deformation data,
line amplitude, omitted fraction, and source IDs. `visibility/*.npz` retains the
complete raster and projection evidence. `summary.json` records checkpoint hashes,
audit statistics, output hashes, and visibility counts. `png-and-methods.zip`
contains the 24 presentation PNGs, methods text and JSON, and frozen source files;
the large VTP and visibility data stay local.
"""


def _main(cfg: Config) -> dict[str, Any]:
    output = cfg.output_dir
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    helper_source = Path(__file__).with_name("muscle_glyph_context.py")
    profile_source = Path(__file__).with_name("experiment_profile.py")
    required = [
        FIXTURE / "volume.vtu",
        FIXTURE / "skin.vtp",
        CAMERAS,
        SELECTION,
        AUDIT,
        helper_source,
        profile_source,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    source_receipts = {
        "renderer": _snapshot(
            Path(__file__), output / "sources/83-render-idea-activation.py"
        ),
        "muscle_context": _snapshot(
            helper_source, output / "sources/muscle_glyph_context.py"
        ),
        "profile": _snapshot(profile_source, output / "sources/experiment_profile.py"),
        "selection": _snapshot(SELECTION, output / "sources/selection.json"),
        "audit_v2": _snapshot(AUDIT, output / "sources/audit-v2-summary.json"),
    }
    selection = json.loads(SELECTION.read_text())
    audit = json.loads(AUDIT.read_text())
    audit_by_id = {state["id"]: state for state in audit["states"]}
    states = [state for state in selection["states"] if state["kind"] == "full"]
    if cfg.state_id is not None:
        states = [state for state in states if state["id"] == cfg.state_id]
        if not states:
            raise ValueError(f"state not found among full selections: {cfg.state_id}")
    state_ids = [state["id"] for state in states]
    if set(state_ids) - set(LABELS) or set(state_ids) - set(audit_by_id):
        raise ValueError("selection, labels, and audit state IDs disagree")

    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    rest_points = np.asarray(volume.points, dtype=np.float64)
    cells = np.asarray(volume.cells, dtype=np.int64).reshape(-1, 5)
    if not np.all(cells[:, 0] == 4):
        raise ValueError("fixture contains a non-tetrahedral cell")
    tets = cells[:, 1:]
    full_ids = np.flatnonzero(
        np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    )
    muscle_ids = np.flatnonzero(
        np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64) > 0.0
    )
    if len(full_ids) != 288235 or not np.array_equal(full_ids, muscle_ids):
        raise ValueError("fixture active-cell contract changed")
    control_ids = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        full_ids
    ]
    active_muscle_ids = np.asarray(volume.cell_data["MuscleId"], dtype=np.int32)[
        full_ids
    ]
    muscle_fraction = np.asarray(volume.cell_data["MuscleFraction"], dtype=np.float64)[
        full_ids
    ]
    names = np.asarray(volume.field_data["MuscleName"]).astype(str)
    muscle_labels = names[active_muscle_ids]
    rest_tets = rest_points[tets[full_ids]]
    rest_centroids = rest_tets.mean(axis=1)
    rest_edges = np.swapaxes(rest_tets[:, 1:] - rest_tets[:, :1], 1, 2)
    inverse_rest_edges = np.linalg.inv(rest_edges)
    volumes = np.linalg.det(rest_edges) / 6.0 * muscle_fraction
    if not np.all(volumes > 0.0):
        raise ValueError("active muscle integration weights must be positive")
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    np.testing.assert_array_equal(skin.points, rest_points[skin_ids])
    camera_doc = json.loads(CAMERAS.read_text())
    cameras = {view["id"]: view["camera"] for view in camera_doc["views"]}
    if set(VIEWS) - set(cameras):
        raise ValueError("frozen camera receipt lacks a requested view")

    generated: list[Path] = []
    visibility_summary: dict[str, Any] = {}
    state_records: list[dict[str, Any]] = []
    for state in states:
        identifier = state["id"]
        logging.getLogger(__name__).info("Rendering %s", identifier)
        checkpoint = Path(state["path"])
        if _digest(checkpoint) != state["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {checkpoint}")
        with np.load(checkpoint, allow_pickle=False) as saved:
            arrays = {key: np.asarray(saved[key]) for key in saved.files}
        if int(arrays["step"]) != state["step"] or not bool(arrays["solver_valid"]):
            raise ValueError(
                f"selected checkpoint is not the valid named step: {identifier}"
            )
        active_ids = np.asarray(arrays["active_ids"], dtype=np.int64)
        if not np.array_equal(active_ids, full_ids):
            raise ValueError(f"active IDs differ from fixture: {identifier}")
        if "rest_points" in arrays and not np.array_equal(
            arrays["rest_points"], rest_points
        ):
            raise ValueError(
                f"checkpoint rest geometry differs from fixture: {identifier}"
            )
        displacement = np.asarray(arrays["u"], dtype=np.float64)
        if displacement.shape != rest_points.shape or not np.all(
            np.isfinite(displacement)
        ):
            raise ValueError(f"invalid checkpoint displacement: {identifier}")

        z, construction = _z_for(identifier, arrays)
        z = 0.5 * (z + np.swapaxes(z, 1, 2))
        eigenvalues, eigenvectors = np.linalg.eigh(z)
        lambda_plus = np.maximum(eigenvalues[:, 2], 0.0)
        amplitude = 1.0 - 1.0 / np.sqrt(1.0 + lambda_plus)
        total_squared = np.sum(z * z, axis=(1, 2))
        spectral_norm = np.max(np.abs(eigenvalues), axis=1)
        zero_tolerance = (
            BACKWARD_ERROR_FACTOR
            * np.finfo(np.float64).eps
            * np.maximum(spectral_norm, 1.0)
        )
        exact_zero = total_squared == 0.0
        omitted = np.divide(
            total_squared - lambda_plus * lambda_plus,
            total_squared,
            out=np.zeros_like(total_squared),
            where=~exact_zero,
        )
        if np.min(omitted) < -1e-12 or np.max(omitted) > 1.0 + 1e-12:
            raise ValueError(
                f"omitted squared-norm fraction outside [0,1]: {identifier}"
            )
        omitted = np.clip(omitted, 0.0, 1.0)
        negative_modes = eigenvalues < -zero_tolerance[:, None]

        deformed_points = rest_points + displacement
        deformed_tets = deformed_points[tets[full_ids]]
        centroids = deformed_tets.mean(axis=1)
        deformed_edges = np.swapaxes(deformed_tets[:, 1:] - deformed_tets[:, :1], 1, 2)
        deformation_gradient = deformed_edges @ inverse_rest_edges
        rest_axes = eigenvectors[:, :, 2]
        transported = np.einsum("nij,nj->ni", deformation_gradient, rest_axes)
        transport_norm = np.linalg.norm(transported, axis=1)
        if not np.all(np.isfinite(transport_norm)) or np.any(transport_norm <= 0.0):
            raise ValueError(f"dominant direction transport failed: {identifier}")
        spatial_axes = transported / transport_norm[:, None]

        deformed_skin = skin.copy(deep=True)
        deformed_skin.points = deformed_points[skin_ids]
        deformed_skin.point_data["RestPosition"] = rest_points[skin_ids]
        deformed_skin.point_data["Displacement"] = displacement[skin_ids]
        deformed_skin.field_data["CoordinateFrame"] = np.asarray(
            ["saved deformed coordinates"]
        )
        skin_path = output / "context" / f"{identifier}-skin.vtp"
        skin_path.parent.mkdir(parents=True, exist_ok=True)
        deformed_skin.save(skin_path, binary=True)
        deformed_volume = volume.copy(deep=True)
        deformed_volume.points = deformed_points
        muscles = build_muscle_region_context(deformed_volume)
        muscle_path = save_muscle_region_context(
            muscles, output / "context" / f"{identifier}-muscle-regions.vtp"
        )
        generated.extend((skin_path, muscle_path))

        line_arrays = {
            "GlobalCellId": full_ids,
            "MuscleId": active_muscle_ids,
            "MuscleLabel": muscle_labels,
            "ActivationControlId": control_ids,
            "MuscleFraction": muscle_fraction,
            "EffectiveTensorZ": z.reshape(-1, 9),
            "ZEigenvaluesAscending": eigenvalues,
            "ZEigenvectorsColumns": eigenvectors.reshape(-1, 9),
            "DominantEigenvectorRest": rest_axes,
            "DominantEigenvectorSpatial": spatial_axes,
            "DominantEigenvectorDyadRest": np.einsum(
                "ni,nj->nij", rest_axes, rest_axes
            ).reshape(-1, 9),
            "DominantEigenvectorDyadSpatial": np.einsum(
                "ni,nj->nij", spatial_axes, spatial_axes
            ).reshape(-1, 9),
            "DominantPositiveEigenvalue": lambda_plus,
            "DominantModeAmplitude": amplitude,
            "DominantModeAmplitudePercent": 100.0 * amplitude,
            "NonDominantSquaredNormFraction": omitted,
            "NonDominantSquaredNormPercent": 100.0 * omitted,
            "ExactZeroTensor": exact_zero.astype(np.uint8),
            "MaterialNegativeModeCount": np.sum(negative_modes, axis=1).astype(
                np.uint8
            ),
            "NegativeModeBackwardErrorTolerance": zero_tolerance,
            "DeformationGradient": deformation_gradient.reshape(-1, 9),
            "DetF": np.linalg.det(deformation_gradient),
            "DirectionTransportStretch": transport_norm,
            "RestCentroid": rest_centroids,
            "DisplayLineLengthM": DISPLAY_MAX_LENGTH_M * amplitude,
        }
        glyphs = _line_mesh(centroids, spatial_axes, amplitude, line_arrays)
        glyph_path = output / "glyphs" / f"{identifier}.vtp"
        glyph_path.parent.mkdir(parents=True, exist_ok=True)
        glyphs.save(glyph_path, binary=True)
        generated.append(glyph_path)
        companion = _companion_surface(
            muscles, full_ids, 100.0 * omitted, volume.n_cells
        )

        visibility_summary[identifier] = {}
        view_records: dict[str, Any] = {}
        for view_id in VIEWS:
            visibility = visible_region_mask(
                muscles,
                centroids,
                full_ids,
                control_ids,
                cameras[view_id],
                window_size=WINDOW_SIZE,
            )
            visibility_path = save_region_visibility(
                visibility, output / "visibility" / f"{identifier}--{view_id}.npz"
            )
            geometry_path = output / "geometry" / identifier / f"{view_id}.png"
            companion_path = output / "companions" / identifier / f"{view_id}.png"
            _render_geometry(
                deformed_skin,
                glyphs,
                visibility,
                cameras[view_id],
                LABELS[identifier],
                state["step"],
                geometry_path,
            )
            _render_companion(
                deformed_skin,
                companion,
                cameras[view_id],
                LABELS[identifier],
                state["step"],
                companion_path,
            )
            generated.extend((visibility_path, geometry_path, companion_path))
            record = {
                "retained_count": visibility.retained_count,
                "retained_interior_count": visibility.retained_interior_count,
                "retained_boundary_source_count": visibility.retained_boundary_source_count,
                "projected_inside_count": visibility.projected_inside_count,
                "projected_background_count": visibility.projected_background_count,
                "occluded_by_other_region_count": visibility.occluded_by_other_region_count,
                "front_surface_control_ids": visibility.front_surface_control_ids,
                "retained_control_ids": visibility.retained_control_ids,
                "visibility_file": _record(visibility_path),
            }
            visibility_summary[identifier][view_id] = record
            view_records[view_id] = {
                "geometry": _record(geometry_path),
                "companion": _record(companion_path),
                "visible_line_count": visibility.retained_count,
            }

        weighted_omitted = float(np.sum(volumes * omitted) / np.sum(volumes))
        audit_row = audit_by_id[identifier]
        if not np.isclose(
            weighted_omitted,
            audit_row["volume_weighted_non_dominant_energy_fraction"],
            rtol=0.0,
            atol=5e-15,
        ):
            raise ValueError(
                f"renderer omitted fraction disagrees with audit: {identifier}"
            )
        if not np.isclose(
            float(np.mean(negative_modes)),
            audit_row["negative_eigenvalue_fraction"],
            rtol=0.0,
            atol=5e-15,
        ):
            raise ValueError(
                f"renderer negative-mode classification disagrees with audit: {identifier}"
            )
        state_records.append(
            {
                "id": identifier,
                "label": LABELS[identifier],
                "step": state["step"],
                "checkpoint": _record(checkpoint),
                "z_construction": construction,
                "glyphs": _record(glyph_path),
                "audit_statistics": audit_row,
                "renderer_statistics": {
                    "lambda_plus_percentiles": {
                        str(p): float(np.percentile(lambda_plus, p))
                        for p in (0, 1, 50, 90, 99, 100)
                    },
                    "amplitude_percentiles": {
                        str(p): float(np.percentile(amplitude, p))
                        for p in (0, 1, 50, 90, 99, 100)
                    },
                    "omitted_squared_norm_fraction_percentiles": {
                        str(p): float(np.percentile(omitted, p))
                        for p in (0, 1, 50, 90, 99, 100)
                    },
                    "volume_weighted_mean_omitted_squared_norm_fraction": weighted_omitted,
                    "negative_eigenvalue_fraction": float(np.mean(negative_modes)),
                    "cells_with_negative_mode_fraction": float(
                        np.mean(np.any(negative_modes, axis=1))
                    ),
                    "exact_zero_tensor_count": int(np.count_nonzero(exact_zero)),
                    "max_point_displacement_mm": float(
                        1000.0 * np.linalg.norm(displacement, axis=1).max()
                    ),
                    "active_inverted_count": int(
                        np.count_nonzero(np.linalg.det(deformation_gradient) < 0.0)
                    ),
                },
                "deformed_skin": _record(skin_path),
                "deformed_muscle_regions": _record(muscle_path),
                "views": view_records,
            }
        )

    readme = output / "README.md"
    readme.write_text(_readme(state_ids))
    generated.append(readme)
    methods = output / "methods.json"
    methods_doc = {
        "state_ids": state_ids,
        "field": "common symmetric effective tensor Z",
        "dominant_line": "lambda_plus=max(lambda_max(Z),0); amplitude=1-1/sqrt(1+lambda_plus); direction=normalize(F@n_rest)",
        "omitted_squared_norm_fraction": "1-lambda_plus^2/||Z||_F^2; 0 only for exactly zero Z",
        "negative_mode_tolerance": "64*eps*max(||Z||_2,1) per tensor; used only to classify negative modes",
        "volume_weighted_summary": "mean of per-cell fractions weighted by rest tetrahedron volume times MuscleFraction",
        "visibility": "front opaque ActivationControlId at projected deformed cell centroid must equal cell ActivationControlId",
        "line_coverage": "one line for every active tetrahedron in the full VTP; no spatial sampling",
        "render_size_pixels": list(WINDOW_SIZE),
        "fixed_ranges_percent": {
            "dominant_amplitude": [0.0, 100.0],
            "omitted_squared_norm": [0.0, 100.0],
        },
        "state_records": state_records,
        "source_receipts": source_receipts,
    }
    _write_json(methods, methods_doc)
    generated.append(methods)
    archive = output / "png-and-methods.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
    ) as bundle:
        bundle.write(readme, "README.md")
        bundle.write(methods, "methods.json")
        for folder in ("geometry", "companions", "sources"):
            for path in sorted((output / folder).rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output))
    generated.append(archive)
    summary = {
        "status": "completed_activation_comparison_preview"
        if cfg.state_id
        else "completed_activation_comparison",
        "state_ids": state_ids,
        "semantics": methods_doc,
        "coverage": {
            "active_cell_count": len(full_ids),
            "line_count_per_state": len(full_ids),
            "every_active_tetrahedron_exported": True,
            "spatial_sampling": None,
            "activation_region_count": len(np.unique(control_ids)),
            "global_cell_ids_sha256": hashlib.sha256(full_ids.tobytes()).hexdigest(),
        },
        "render": {
            "views": list(VIEWS),
            "window_size": list(WINDOW_SIZE),
            "skin_opacity": CONTEXT_OPACITY,
            "maximum_line_length_m": DISPLAY_MAX_LENGTH_M,
            "primary_muscle_surface_or_outline": None,
            "companion_surface": "opaque, cell-scaled, no edges or silhouettes",
            "visibility": visibility_summary,
        },
        "states": state_records,
        "inputs": {str(path): _record(path) for path in required},
        "outputs": {str(path.relative_to(output)): _record(path) for path in generated},
        "archive": _record(archive),
        "output_root": str(output.resolve()),
    }
    _write_json(output / "summary.json", summary)
    return summary


def main(cfg: Config) -> None:
    summary = _main(cfg)
    cherries.log_metric("render/states", len(summary["states"]))
    cherries.log_metric(
        "render/lines_per_state", summary["coverage"]["line_count_per_state"]
    )
    cherries.log_output(cfg.output_dir)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.getenv("DEBUG") == "1" else ProfileCometNoCommit
    )
