"""Render and independently audit the saved Newton neutral endpoint on the CPU.

This is deliberately a receipt consumer.  It never imports a forward problem or
an optimizer: its only state input is ``terminal.npz`` written by 10-forward.
"""

# ruff: noqa: EM102, PT018, RUF005, TRY003

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.ticker import MaxNLocator
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT_SRC = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path.insert(0, str(JOINT_SRC))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_data import _collision_geometry  # noqa: E402

OLD_ENDPOINT = (
    ROOT
    / "exp/2026/09/21/joint-activation-material-mandible/data/eye-neutral-forward-002/"
    "checkpoint-terminal-step-02980.npz"
)
WINDOW = (1800, 750)


class Config(cherries.BaseConfig):
    """All paths are inputs, except ``output_dir`` below the new forward receipt."""

    run_dir: Path = GROUP / "data/forward-001"
    old_endpoint: Path = OLD_ENDPOINT


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _input(record: dict[str, Any], label: str) -> Path:
    path = Path(record["path"])
    assert path.is_file(), f"{label} does not exist: {path}"
    assert sha256(path) == record["sha256"], f"{label} SHA-256 mismatch: {path}"
    return path


def _triangles(surface: pv.PolyData) -> np.ndarray:
    faces = np.asarray(surface.faces).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3), "audit requires triangular faces"
    return faces[:, 1:]


def _surface_volume(volume: pv.UnstructuredGrid, points: np.ndarray) -> pv.PolyData:
    surface = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    surface.points = points[original]
    surface.point_data["GlobalPointId"] = original
    return surface


def _pure_soft(volume: pv.UnstructuredGrid, points: np.ndarray) -> pv.PolyData:
    boundary = _surface_volume(volume, points)
    ids = np.asarray(boundary.point_data["GlobalPointId"], dtype=np.int64)
    names = [
        str(name) for name in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    assert "Cranium" in names and "Mandible" in names
    groups = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[ids]
    faces = _triangles(boundary)
    bone = (groups[faces] == names.index("Cranium")) | (
        groups[faces] == names.index("Mandible")
    )
    return (
        boundary.extract_cells(np.flatnonzero(~np.any(bone, axis=1)))
        .extract_surface(algorithm=None)
        .triangulate()
    )


def _signed_clearance(points: np.ndarray, closed: pv.PolyData) -> np.ndarray:
    """Positive exterior distance; negative values identify eye containment."""
    assert closed.is_manifold and closed.n_open_edges == 0
    _, nearest = closed.find_closest_cell(points, return_closest_point=True)
    distance = np.linalg.norm(points - np.asarray(nearest), axis=1)
    enclosed = vtkSelectEnclosedPoints()
    enclosed.SetInputData(pv.PolyData(points))
    enclosed.SetSurfaceData(closed)
    enclosed.SetTolerance(0.0)
    enclosed.CheckSurfaceOn()
    enclosed.Update()
    inside = np.asarray(
        pv.wrap(enclosed.GetOutput()).point_data["SelectedPoints"], dtype=bool
    )
    return np.where(inside, -distance, distance)


def _eye_components(eyes_dir: Path) -> list[pv.PolyData]:
    """Weld exact duplicate vertices only, to obtain watertight containment meshes."""
    with np.load(eyes_dir / "eyes.npz", allow_pickle=False) as archive:
        points = np.asarray(archive["points_m"], dtype=np.float64)
        faces = np.asarray(archive["triangles"], dtype=np.int64)
        components = np.asarray(archive["vertex_component_ids"], dtype=np.int64)
    proxy_points, first, inverse = np.unique(
        points, axis=0, return_index=True, return_inverse=True
    )
    proxy_faces = inverse[faces]
    assert not np.any(
        (proxy_faces[:, 0] == proxy_faces[:, 1])
        | (proxy_faces[:, 1] == proxy_faces[:, 2])
        | (proxy_faces[:, 2] == proxy_faces[:, 0])
    )
    proxy_components = components[first]
    assert np.array_equal(proxy_components[inverse], components)
    proxy = pv.PolyData(
        proxy_points, np.column_stack((np.full(len(proxy_faces), 3), proxy_faces))
    )
    result: list[pv.PolyData] = []
    for component in np.unique(proxy_components):
        part = (
            proxy.extract_points(proxy_components == component, adjacent_cells=True)
            .extract_surface(algorithm=None)
            .triangulate()
        )
        assert part.is_manifold and part.n_open_edges == 0
        result.append(part)
    return result


def _camera(meshes: list[pv.DataSet], view: str) -> dict[str, Any]:
    bounds = np.asarray([mesh.bounds for mesh in meshes])
    low = np.minimum.reduce(bounds[:, ::2], axis=0)
    high = np.maximum.reduce(bounds[:, 1::2], axis=0)
    center = (low + high) / 2
    span = float(max(high - low))
    if view == "front":
        eye = center + (0.0, 0.0, 2.7 * span)
        horizontal = high[0] - low[0]
    elif view == "side":
        eye = center + (2.7 * span, 0.0, 0.0)
        horizontal = high[2] - low[2]
    else:
        raise ValueError(view)
    aspect = WINDOW[0] / WINDOW[1]
    return {
        "position": [eye.tolist(), center.tolist(), [0.0, 1.0, 0.0]],
        "scale": 1.08 * max((high[1] - low[1]) / 2, horizontal / (2 * aspect)),
    }


def _read_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            raise ValueError(f"invalid JSONL trace at line {number}") from None
        assert isinstance(value, dict)
        rows.append(value)
    return rows


def _geometry(path: Path) -> tuple[pv.PolyData, pv.PolyData]:
    with np.load(path, allow_pickle=False) as archive:
        cranium = pv.PolyData(
            archive["cranium_points_m"],
            np.column_stack(
                (np.full(len(archive["cranium_faces"]), 3), archive["cranium_faces"])
            ),
        )
        mandible = pv.PolyData(
            archive["mandible_points_m"],
            np.column_stack(
                (np.full(len(archive["mandible_faces"]), 3), archive["mandible_faces"])
            ),
        )
    return cranium, mandible


def _plot_comparison(
    output: Path, reference: pv.PolyData, old: pv.PolyData, latest: pv.PolyData
) -> list[str]:
    files = []
    for view in ("front", "side"):
        camera = _camera([reference, old, latest], view)
        plot = pv.Plotter(off_screen=True, shape=(1, 3), window_size=WINDOW)
        plot.set_background("#f7f7f5")
        for column, (title, mesh, color) in enumerate(
            (
                ("Constitutive reference", reference, "#878787"),
                ("Previous eye-neutral endpoint", old, "#4c8ca2"),
                ("Newton endpoint", latest, "#d6755c"),
            )
        ):
            plot.subplot(0, column)
            plot.add_text(
                f"{title}\n{view} · true scale",
                position="upper_left",
                font_size=14,
                color="#202124",
            )
            plot.add_mesh(mesh, color=color, smooth_shading=True)
            plot.camera_position = camera["position"]
            plot.camera.parallel_projection = True
            plot.camera.parallel_scale = camera["scale"]
            plot.reset_camera_clipping_range()
        name = f"neutral-comparison-{view}.png"
        plot.show(screenshot=output / name, auto_close=True)
        files.append(name)
    return files


def _plot_convergence(
    output: Path, trace: list[dict[str, Any]], summary: dict[str, Any]
) -> str:
    accepted = [row for row in trace if row.get("accepted") is True]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    if accepted:
        x = [row.get("iteration", index + 1) for index, row in enumerate(accepted)]
        axes[0].semilogy(x, [row["grad_norm"] for row in accepted], color="#087d81")
        axes[1].plot(x, [row["energy"] for row in accepted], color="#087d81")
    threshold = float(summary["effective_grad_threshold"])
    axes[0].axhline(threshold, color="#ae614a", linestyle="--", label="threshold")
    axes[0].scatter(
        [summary["accepted_steps"]],
        [summary["final_grad_norm"]],
        color="#171717",
        zorder=3,
    )
    axes[0].legend(fontsize=8)
    axes[0].set(
        xlabel="Accepted iteration",
        ylabel="Gradient norm",
        title="Accepted Newton convergence",
    )
    axes[1].set(
        xlabel="Accepted iteration", ylabel="Energy", title="Accepted-state energy"
    )
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(6))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    name = "convergence.png"
    fig.savefig(output / name, dpi=180)
    plt.close(fig)
    return name


def main(cfg: Config) -> None:  # noqa: PLR0915
    run_dir = cfg.run_dir.resolve()
    output = run_dir / "review"
    assert not output.exists(), f"refusing to overwrite review directory: {output}"
    summary_path = run_dir / "summary.json"
    protocol_path = run_dir / "protocol.json"
    terminal_path = run_dir / "terminal.npz"
    trace_path = run_dir / "trace.jsonl"
    assert all(
        path.is_file()
        for path in (summary_path, protocol_path, terminal_path, trace_path)
    )
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    inputs = protocol["inputs"]
    volume_path = _input(inputs["prepared_volume"], "prepared_volume")
    skin_path = _input(inputs["prepared_skin"], "prepared_skin")
    geometry_path = _input(inputs["geometry"], "geometry")
    eyes_dir = Path(inputs["eyes_dir"])
    assert (eyes_dir / "eyes.npz").is_file() and (eyes_dir / "eyes.vtp").is_file()
    with np.load(terminal_path, allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    assert displacement.shape == (228660, 3) and np.isfinite(displacement).all()
    with np.load(cfg.old_endpoint, allow_pickle=False) as archive:
        old_displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    assert (
        old_displacement.shape == displacement.shape
        and np.isfinite(old_displacement).all()
    )
    volume = pv.read(volume_path)
    skin = pv.read(skin_path).triangulate()
    assert volume.n_points == len(displacement)
    reference = np.asarray(volume.points).copy()
    skin_ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    reference_skin = skin.copy(deep=True)
    old_skin = skin.copy(deep=True)
    old_skin.points = reference[skin_ids] + old_displacement[skin_ids]
    latest_skin = skin.copy(deep=True)
    latest_skin.points = reference[skin_ids] + displacement[skin_ids]
    latest_skin.point_data["DisplacementFromConstitutiveReference_m"] = displacement[
        skin_ids
    ]
    latest_skin.point_data["DisplacementMagnitude_mm"] = 1000 * np.linalg.norm(
        displacement[skin_ids], axis=1
    )
    cells = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    dm = reference[cells[:, 1:]] - reference[cells[:, :1]]
    ds = (reference + displacement)[cells[:, 1:]] - (reference + displacement)[
        cells[:, :1]
    ]
    detf = np.linalg.det(ds) / np.linalg.det(dm)
    assert np.isfinite(detf).all()
    latest_volume = volume.copy(deep=True)
    latest_volume.points = reference + displacement
    latest_volume.point_data["DisplacementFromConstitutiveReference_m"] = displacement
    latest_volume.cell_data["PhysicalDetF"] = detf
    cranium, mandible = _geometry(geometry_path)
    eyes = pv.read(eyes_dir / "eyes.vtp").triangulate()
    soft = _pure_soft(volume, reference + displacement)
    intersections = {}
    for name, rigid in {"cranium": cranium, "mandible": mandible, "eyes": eyes}.items():
        pairs, _, _, lengths = _collision_geometry(soft, rigid)
        intersections[name] = {
            "pairs": len(pairs),
            "soft_triangles": len(np.unique(pairs[:, 0])) if len(pairs) else 0,
            "rigid_triangles": len(np.unique(pairs[:, 1])) if len(pairs) else 0,
            "segment_length_sum_m": float(lengths.sum()),
        }
    groups = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    names = [
        str(name) for name in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    free_ids = np.flatnonzero(
        (groups != names.index("Cranium")) & (groups != names.index("Mandible"))
    )
    containment = {}
    for component, eye in enumerate(_eye_components(eyes_dir)):
        boundary = _signed_clearance(np.asarray(soft.points), eye)
        nodes = _signed_clearance(reference[free_ids] + displacement[free_ids], eye)
        containment[str(component)] = {
            "minimum_boundary_m": float(boundary.min()),
            "q01_boundary_m": float(np.quantile(boundary, 0.01)),
            "inside_soft_boundary_vertices": int(np.count_nonzero(boundary < 0)),
            "minimum_nonrigid_fem_node_m": float(nodes.min()),
            "inside_nonrigid_fem_nodes": int(np.count_nonzero(nodes < 0)),
        }
    output.mkdir(parents=True)
    latest_volume.save(output / "neutral-volume.vtu")
    latest_skin.save(output / "neutral-skin.vtp")
    comparison_files = _plot_comparison(output, reference_skin, old_skin, latest_skin)
    trace = _read_trace(trace_path)
    convergence_file = _plot_convergence(output, trace, summary)
    passed = bool(summary["success"])
    receipt = {
        "schema": "neutral-newton-cpu-review-v1",
        "success": passed,
        "state_label": "converged endpoint"
        if passed
        else "FAILED forward endpoint; geometry is diagnostic only",
        "run": {
            "summary": _record(summary_path),
            "protocol": _record(protocol_path),
            "terminal": _record(terminal_path),
            "trace": _record(trace_path),
        },
        "inputs": {
            "prepared_volume": _record(volume_path),
            "prepared_skin": _record(skin_path),
            "geometry": _record(geometry_path),
            "eyes_dir": str(eyes_dir.resolve()),
            "old_comparison_endpoint": _record(cfg.old_endpoint),
        },
        "solver": {
            key: summary[key]
            for key in (
                "success",
                "status",
                "initial_grad_norm",
                "final_grad_norm",
                "effective_grad_threshold",
                "accepted_steps",
                "forward_seconds",
            )
        },
        "geometry": {
            "detF_min": float(detf.min()),
            "detF_max": float(detf.max()),
            "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
            "skin_rms_mm": float(
                1000 * np.sqrt(np.mean(np.sum(displacement[skin_ids] ** 2, axis=1)))
            ),
        },
        "independent_triangle_intersections": intersections,
        "eye_containment": containment,
        "nonrigid_fem_nodes_classified": len(free_ids),
        "assets": comparison_files
        + [convergence_file, "neutral-volume.vtu", "neutral-skin.vtp"],
        "scope": "CPU/offscreen saved-state review. VTK triangle-pair tests cover the complete cranium, mandible, and source-eye surfaces. Eye containment uses exact-coordinate-welded watertight copies only. A failed forward is labelled failed and its exported geometry is diagnostic, not an accepted equilibrium.",
    }
    write_json(output / "summary.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "review/detF_min": receipt["geometry"]["detF_min"],
            "review/inverted_tetrahedra": receipt["geometry"]["inverted_tetrahedra"],
            "review/triangle_pairs": sum(
                value["pairs"] for value in intersections.values()
            ),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
