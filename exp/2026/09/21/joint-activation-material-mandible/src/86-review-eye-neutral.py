"""Independently audit and visualize the fixed-eye neutral forward endpoint."""

from __future__ import annotations

import json
from pathlib import Path

import ipctk
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs, _collision_geometry
from joint_frozen_neutral import FrozenNeutral, load_script
from joint_rigid_eye_contact import build_eye_collision_physics
from matplotlib.ticker import MaxNLocator
from vtkmodules.vtkFiltersModeling import vtkSelectEnclosedPoints

from liblaf import cherries


class Config(cherries.BaseConfig):
    """Locations deliberately bind the review to the fixed-eye forward run."""

    run_dir: Path = GROUP / "data/eye-neutral-forward-002"
    output_dir: Path = cherries.output("eye-neutral-forward-review-006", mkdir=True)
    eyes_dir: Path = GROUP / "data/rigid-eyes-001"


def _triangles(poly: pv.PolyData) -> np.ndarray:
    faces = np.asarray(poly.faces).reshape(-1, 4)
    assert np.all(faces[:, 0] == 3)
    return faces[:, 1:]


def _trace(path: Path) -> list[dict]:
    rows: list[dict] = []
    lines = path.read_text().splitlines()
    for number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if number == len(lines):
                break
            raise
        assert isinstance(row, dict)
        rows.append(row)
    assert rows
    return rows


def _camera(
    mesh: pv.DataSet, direction: tuple[float, float, float], zoom: float = 1.0
) -> dict:
    center = np.asarray(mesh.center)
    span = max(mesh.length, 1e-3)
    eye = center + span * np.asarray(direction)
    return {
        "position": [eye.tolist(), center.tolist(), [0.0, 1.0, 0.0]],
        "scale": span / (1.9 * zoom),
    }


def _save(plot: pv.Plotter, path: Path, camera: dict) -> None:
    plot.camera_position = camera["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera["scale"]
    plot.reset_camera_clipping_range()
    plot.show(screenshot=path, auto_close=True)


def _plotter(title: str) -> pv.Plotter:
    plot = pv.Plotter(off_screen=True, window_size=(1600, 1200))
    plot.set_background("#f7f7f5")
    plot.enable_anti_aliasing("ssaa")
    plot.add_text(title, position="upper_left", font_size=15, color="#202124")
    return plot


def _surface_volume(volume: pv.UnstructuredGrid, points: np.ndarray) -> pv.PolyData:
    surface = volume.extract_surface(algorithm=None).triangulate()
    original = np.asarray(surface.point_data["vtkOriginalPointIds"], dtype=np.int64)
    surface.points = points[original]
    surface.point_data["GlobalPointId"] = original
    return surface


def _pure_soft(volume: pv.UnstructuredGrid, points: np.ndarray) -> pv.PolyData:
    boundary = _surface_volume(volume, points)
    ids = np.asarray(boundary.point_data["GlobalPointId"], dtype=np.int64)
    faces = _triangles(boundary)
    names = [
        str(name) for name in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    groups = np.asarray(volume.point_data["GroupId"], dtype=np.int32)[ids]
    bone = (groups[faces] == names.index("Cranium")) | (
        groups[faces] == names.index("Mandible")
    )
    return (
        boundary.extract_cells(np.flatnonzero(~np.any(bone, axis=1)))
        .extract_surface(algorithm=None)
        .triangulate()
    )


def _signed_clearance(points: np.ndarray, closed: pv.PolyData) -> np.ndarray:
    """Positive exterior clearance; negative means a point lies inside an eye."""
    assert closed.is_manifold
    assert closed.n_open_edges == 0
    _, closest = closed.find_closest_cell(points, return_closest_point=True)
    distance = np.linalg.norm(points - np.asarray(closest), axis=1)
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


def _record(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _containment_proxy(
    eyes_dir: Path, output_dir: Path
) -> tuple[list[pv.PolyData], dict]:
    """Exact-coordinate weld solely for containment; collision keeps raw triangles."""
    eye_npz = eyes_dir / "eyes.npz"
    with np.load(eye_npz, allow_pickle=False) as raw:
        points = np.asarray(raw["points_m"], dtype=np.float64)
        faces = np.asarray(raw["triangles"], dtype=np.int64)
        components = np.asarray(raw["vertex_component_ids"], dtype=np.int64)
    proxy_points, first, original_to_proxy = np.unique(
        points, axis=0, return_index=True, return_inverse=True
    )
    proxy_faces = original_to_proxy[faces]
    assert len(proxy_points) <= len(points)
    assert np.array_equal(proxy_points[original_to_proxy], points)
    assert not np.any(
        (proxy_faces[:, 0] == proxy_faces[:, 1])
        | (proxy_faces[:, 1] == proxy_faces[:, 2])
        | (proxy_faces[:, 2] == proxy_faces[:, 0])
    )
    proxy_components = components[first]
    assert np.array_equal(proxy_components[original_to_proxy], components)
    source = pv.PolyData(
        proxy_points,
        np.column_stack((np.full(len(proxy_faces), 3), proxy_faces)),
    )
    source.point_data["SourceEyeComponent"] = proxy_components
    source.save(output_dir / "eye-containment-proxy.vtp")
    mapping_path = output_dir / "eye-containment-proxy-mapping.npz"
    np.savez_compressed(
        mapping_path,
        original_to_proxy_ids=original_to_proxy.astype("<i8"),
        proxy_first_source_vertex_ids=first.astype("<i8"),
        proxy_triangles=proxy_faces.astype("<i8"),
    )
    assert source.is_manifold
    assert source.n_open_edges == 0
    labels = np.asarray(source.point_data["SourceEyeComponent"])
    result = []
    for component in np.unique(labels):
        part = (
            source.extract_points(labels == component, adjacent_cells=True)
            .extract_surface(algorithm=None)
            .triangulate()
        )
        assert part.is_manifold
        assert part.n_open_edges == 0
        result.append(part)
    return result, {
        "source_eyes_npz": _record(eye_npz),
        "proxy": _record(output_dir / "eye-containment-proxy.vtp"),
        "mapping": _record(mapping_path),
        "source_vertices": len(points),
        "proxy_vertices": len(proxy_points),
        "source_triangles": len(faces),
        "proxy_triangles": len(proxy_faces),
        "method": "exact bitwise-coordinate welding only; raw source eye collider remains unchanged",
    }


def main(cfg: Config) -> None:  # noqa: PLR0915 - one independent endpoint receipt
    torch.set_default_dtype(torch.float64)
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    archive_sources(cfg.output_dir)
    # Reconstructing the hash-bound collision model requires a live Warp runtime.
    load_script("68-run-simple-skin-forward.py").configure_cuda()
    summary_path = cfg.run_dir / "summary.json"
    protocol_path = cfg.run_dir / "protocol.json"
    summary = json.loads(summary_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    checkpoint = summary["checkpoint"]
    checkpoint_path = Path(checkpoint["path"])
    assert sha256(checkpoint_path) == checkpoint["sha256"]
    with np.load(checkpoint_path, allow_pickle=False) as data:
        u = np.asarray(data["displacement_m"], dtype=np.float64)
    assert np.isfinite(u).all()
    neutral = FrozenNeutral.load()
    eye_receipt = protocol["geometry"]["eyes"]
    assert eye_receipt["eyes_npz_sha256"] == sha256(cfg.eyes_dir / "eyes.npz")
    assert eye_receipt["source_sha256"] == sha256(Path(eye_receipt["source_path"]))
    physics, _ = build_eye_collision_physics(neutral, cfg.eyes_dir)
    geometry = physics.full_skull.geometry
    assert u.shape == (geometry.fem_node_count, 3)
    fixed_error = float(np.abs(u[geometry.fixed_global_ids]).max())
    fixed_limit = float(
        8 * np.finfo(np.float64).eps * np.abs(geometry.fem_reference_points_m).max()
    )
    assert fixed_error <= fixed_limit
    full_u = physics.full_skull.extend_seed(torch.as_tensor(u), torch.zeros(6))
    eye_u = full_u[physics.full_skull.eye_global_ids].detach().cpu().numpy()
    assert float(np.abs(eye_u).max()) == 0.0
    collision = physics.runtime.forward.model.collision
    collision_state = collision.state_at(full_u)
    collision_positions = (
        (collision.vertices + full_u[collision.indices]).detach().cpu().numpy()
    )
    ipc_intersections = bool(
        ipctk.has_intersections(
            collision.collision_mesh, collision_positions, ipctk.LBVH()
        )
    )
    contact = collision.diagnostics(collision_state, full_u)
    weights = np.asarray(
        [
            collision_state.collisions[i].weight
            for i in range(len(collision_state.collisions))
        ]
    )

    original_protocol = json.loads(
        Path(neutral.manifest["sources"]["protocol"]["path"]).read_text()
    )
    prepared_record = original_protocol["inputs"]
    prepared = PreparedInputs.load(
        Path(prepared_record["prepared_npz"]),
        Path(prepared_record["prepared_manifest"]),
    )
    volume = pv.read(prepared.volume_path)
    reference = np.asarray(volume.points).copy()
    old_u = np.asarray(neutral.arrays["neutral_displacement_m"], dtype=np.float64)
    assert old_u.shape == u.shape
    cells = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    dm = reference[cells[:, 1:]] - reference[cells[:, :1]]
    ds = (reference + u)[cells[:, 1:]] - (reference + u)[cells[:, :1]]
    detf = np.linalg.det(ds) / np.linalg.det(dm)
    original_skin = pv.read(prepared.skin_path).triangulate()
    observation_ids = np.asarray(neutral.arrays["observation_node_ids"], dtype=np.int64)
    neutral_change = u - old_u
    observed_change = neutral_change[observation_ids]
    neutral_difference = {
        "observation_nodes": len(observation_ids),
        "observation_surface_unweighted_rms_mm": float(
            1000 * np.sqrt(np.mean(np.sum(observed_change**2, axis=1)))
        ),
        "observation_surface_max_mm": float(
            1000 * np.linalg.norm(observed_change, axis=1).max()
        ),
        "all_fem_nodes_max_mm": float(
            1000 * np.linalg.norm(neutral_change, axis=1).max()
        ),
        "definition": "difference between final eye-inclusive total displacement and prior adopted-neutral total displacement; observed surface statistic is unweighted",
    }
    skin_ids = np.asarray(original_skin.point_data["GlobalPointId"], dtype=np.int64)
    old_skin = original_skin.copy(deep=True)
    old_skin.points = reference[skin_ids] + old_u[skin_ids]
    final_skin = original_skin.copy(deep=True)
    final_skin.points = reference[skin_ids] + u[skin_ids]
    final_skin.point_data["displacement_from_constitutive_mm"] = 1000 * np.linalg.norm(
        u[skin_ids], axis=1
    )
    final_skin.point_data["change_from_adopted_neutral_mm"] = 1000 * np.linalg.norm(
        (u - old_u)[skin_ids], axis=1
    )
    deformed_volume = volume.copy(deep=True)
    deformed_volume.points = reference + u
    deformed_volume.point_data["EyeNeutralDisplacement_m"] = u
    deformed_volume.point_data["ChangeFromAdoptedNeutral_m"] = u - old_u
    deformed_volume.cell_data["PhysicalDetF"] = detf
    deformed_volume.save(cfg.output_dir / "eye-neutral-deformed-volume.vtu")
    final_skin.save(cfg.output_dir / "eye-neutral-deformed-skin.vtp")

    eyes = pv.read(cfg.eyes_dir / "eyes.vtp").triangulate()
    with np.load(
        Path(prepared_record["geometry"]["geometry"]["geometry_path"]),
        allow_pickle=False,
    ) as data:
        cranium = pv.PolyData(
            data["cranium_points_m"],
            np.column_stack(
                (np.full(len(data["cranium_faces"]), 3), data["cranium_faces"])
            ),
        )
        mandible = pv.PolyData(
            data["mandible_points_m"],
            np.column_stack(
                (np.full(len(data["mandible_faces"]), 3), data["mandible_faces"])
            ),
        )
    rigid = pv.MultiBlock(
        {"cranium_fixed": cranium, "mandible_fixed": mandible, "eyes_fixed": eyes}
    )
    rigid.save(cfg.output_dir / "fixed-rigid-obstacles.vtm")
    merged = cranium.merge(mandible, merge_points=False).merge(eyes, merge_points=False)
    merged.save(cfg.output_dir / "fixed-rigid-obstacles.vtp")

    soft = _pure_soft(volume, reference + u)
    source_pairs = {}
    for name, rigid_surface in {
        "cranium": cranium,
        "mandible": mandible,
        "eyes": eyes,
    }.items():
        pairs, _, midpoints, lengths = _collision_geometry(soft, rigid_surface)
        source_pairs[name] = {
            "pairs": len(pairs),
            "soft_triangles": len(np.unique(pairs[:, 0])) if len(pairs) else 0,
            "rigid_triangles": len(np.unique(pairs[:, 1])) if len(pairs) else 0,
            "segment_length_sum_m": float(lengths.sum()),
            "midpoints_m": midpoints,
        }
    proxy_components, proxy_provenance = _containment_proxy(
        cfg.eyes_dir, cfg.output_dir
    )
    eye_clearance = {}
    groups = np.asarray(volume.point_data["GroupId"], dtype=np.int32)
    names = [
        str(name) for name in np.asarray(volume.field_data["GroupName"]).reshape(-1)
    ]
    free_fem_ids = np.flatnonzero(
        (groups != names.index("Cranium")) & (groups != names.index("Mandible"))
    )
    for component, subset in enumerate(proxy_components):
        clearance = _signed_clearance(np.asarray(soft.points), subset)
        all_clearance = _signed_clearance(
            reference[free_fem_ids] + u[free_fem_ids], subset
        )
        eye_clearance[str(component)] = {
            "minimum_boundary_m": float(clearance.min()),
            "inside_soft_boundary_vertices": int(np.count_nonzero(clearance < 0)),
            "q01_boundary_m": float(np.quantile(clearance, 0.01)),
            "inside_nonrigid_fem_nodes": int(np.count_nonzero(all_clearance < 0)),
            "minimum_nonrigid_fem_node_m": float(all_clearance.min()),
        }
    assert not ipc_intersections
    assert all(value["pairs"] == 0 for value in source_pairs.values())
    assert all(
        value["inside_soft_boundary_vertices"] == 0 for value in eye_clearance.values()
    )

    assets: list[dict[str, str]] = []
    camera_front = _camera(final_skin, (0.0, 0.0, 2.7), 0.84)
    camera_side = _camera(final_skin, (2.7, 0.0, 0.0), 0.84)
    for view, camera in (("front", camera_front), ("side", camera_side)):
        plot = pv.Plotter(off_screen=True, shape=(1, 3), window_size=(1800, 750))
        plot.set_background("#f7f7f5")
        labels = (
            ("Original constitutive reference", original_skin, "#878787"),
            ("Previous neutral (no eye contact)", old_skin, "#4c8ca2"),
            ("Eye-inclusive forward", final_skin, "#d6755c"),
        )
        for index, (label, surface, color) in enumerate(labels):
            plot.subplot(0, index)
            plot.add_text(
                f"{label}\n{view} · true scale",
                position="upper_left",
                font_size=14,
                color="#202124",
            )
            plot.add_mesh(surface, color=color, smooth_shading=True)
            plot.camera_position = camera["position"]
            plot.camera.parallel_projection = True
            plot.camera.parallel_scale = camera["scale"]
            plot.reset_camera_clipping_range()
        filename = f"01-neutral-comparison-{view}.png"
        plot.show(screenshot=cfg.output_dir / filename, auto_close=True)
        assets.append(
            {
                "filename": filename,
                "caption": "Original constitutive reference, adopted neutral, and fixed-eye forward endpoint at the same true-scale camera.",
            }
        )
    eye_camera = _camera(eyes, (0.2, 0.0, 2.2), 1.15)
    for name, surface, scalar, title in (
        (
            "motion",
            final_skin,
            "displacement_from_constitutive_mm",
            "Displacement from constitutive reference (mm)",
        ),
        (
            "change",
            final_skin,
            "change_from_adopted_neutral_mm",
            "Change from adopted neutral (mm)",
        ),
    ):
        plot = pv.Plotter(off_screen=True, window_size=(1600, 1200))
        plot.set_background("#f7f7f5")
        plot.enable_anti_aliasing("ssaa")
        plot.add_mesh(eyes, color="#edf3f6", opacity=0.72, smooth_shading=True)
        plot.add_mesh(
            surface,
            scalars=scalar,
            cmap="viridis",
            smooth_shading=True,
            scalar_bar_args={
                "title": f"Periorbital endpoint\n{title}",
                "color": "#000000",
                "background_color": "#ffffff",
                "fill": True,
                "vertical": False,
                "position_x": 0.16,
                "position_y": 0.035,
                "width": 0.68,
                "height": 0.10,
                "title_font_size": 16,
                "label_font_size": 14,
                "n_labels": 5,
            },
        )
        scalar_bar = next(reversed(plot.scalar_bars.values()))
        scalar_bar.GetBackgroundProperty().SetColor(1.0, 1.0, 1.0)
        scalar_bar.GetTitleTextProperty().SetColor(0.0, 0.0, 0.0)
        scalar_bar.GetLabelTextProperty().SetColor(0.0, 0.0, 0.0)
        filename = f"02-periorbital-{name}.png"
        _save(plot, cfg.output_dir / filename, eye_camera)
        assets.append(
            {
                "filename": filename,
                "caption": "Both source eyes are fixed collision obstacles. The map is rendered on the final skin at true geometric scale.",
            }
        )
    plot = _plotter("Fixed full skull and both source eyes with final soft surface")
    plot.add_mesh(cranium, color="#ddc69d", opacity=0.40, smooth_shading=True)
    plot.add_mesh(mandible, color="#598999", opacity=0.45, smooth_shading=True)
    plot.add_mesh(eyes, color="#e8edf0", opacity=0.85, smooth_shading=True)
    plot.add_mesh(final_skin, color="#d6755c", opacity=0.26, smooth_shading=True)
    filename = "03-full-rigid-context.png"
    _save(plot, cfg.output_dir / filename, camera_side)
    assets.append(
        {
            "filename": filename,
            "caption": "Complete fixed cranium, mandible, and exact registered source eyes used by the collision model.",
        }
    )
    trace = _trace(cfg.run_dir / "trace.jsonl")
    force_rows = [row for row in trace if "accepted_state_free_force_norm" in row]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    if force_rows:
        axes[0].semilogy(
            [row["step"] for row in force_rows],
            [row["accepted_state_free_force_norm"] * 1e6 for row in force_rows],
            color="#087d81",
            label="accepted-state samples",
        )
    target_n = summary["force_threshold"] * 1e6
    final_n = summary["final_free_force_norm"] * 1e6
    target_mn = summary["force_threshold"] * 1e9
    final_mn = summary["final_free_force_norm"] * 1e9
    axes[0].axhline(
        target_n,
        color="#ae614a",
        linestyle="--",
        label=f"target {target_mn:.6f} mN",
    )
    axes[0].scatter(
        [summary["accepted_steps"]],
        [final_n],
        color="#171717",
        zorder=3,
        label=f"verified terminal {final_mn:.6f} mN",
    )
    axes[0].legend(fontsize=7)
    axes[0].set(
        xlabel="Accepted PNCG step",
        ylabel="Free force (N)",
        title="Accepted-state free force",
    )
    axes[1].plot(
        [row["step"] for row in trace],
        [row["energy"] * 1e6 for row in trace],
        color="#087d81",
    )
    axes[1].set(xlabel="Accepted PNCG step", ylabel="Energy (J)", title="Total energy")
    axes[2].hist(detf, bins=150, color="#7da5ac", log=True)
    axes[2].axvline(1, color="#555555", linewidth=0.8)
    axes[2].set(
        xlabel="Physical det(F)",
        ylabel="Tet count (log)",
        title=f"J: {detf.min():.4g} to {detf.max():.4g}",
    )
    for axis in axes:
        axis.xaxis.set_major_locator(MaxNLocator(6))
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.15)
    fig.savefig(cfg.output_dir / "00-solver-and-volume-trends.png", dpi=180)
    plt.close(fig)
    assets.insert(
        0,
        {
            "filename": "00-solver-and-volume-trends.png",
            "caption": "Accepted-state force, total energy, and endpoint Jacobian distribution. The force threshold is the prescribed equilibrium criterion; inversion count remains a separate diagnostic.",
        },
    )
    receipt = {
        "schema": "joint-eye-neutral-forward-independent-review-v1",
        "success": bool(summary["success"]),
        "run": {
            "summary": _record(summary_path),
            "protocol": _record(protocol_path),
            "checkpoint": checkpoint,
        },
        "frozen_neutral_manifest_sha256": sha256(neutral.directory / "manifest.json"),
        "eyes_manifest": _record(cfg.eyes_dir / "manifest.json"),
        "fixed_fem_displacement_max_m": fixed_error,
        "fixed_fem_roundoff_limit_m": fixed_limit,
        "fixed_eye_displacement_max_m": float(np.abs(eye_u).max()),
        "ipc_soft_rigid_intersections": ipc_intersections,
        "independent_triangle_intersections": {
            name: {key: value for key, value in result.items() if key != "midpoints_m"}
            for name, result in source_pairs.items()
        },
        "eye_containment_proxy": proxy_provenance,
        "eye_containment": eye_clearance,
        "nonrigid_fem_nodes_classified": len(free_fem_ids),
        "difference_from_previous_neutral": neutral_difference,
        "contact": contact,
        "negative_contact_weights": int(np.count_nonzero(weights < 0)),
        "detF_min": float(detf.min()),
        "detF_max": float(detf.max()),
        "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
        "assets": assets,
        "paraview": {
            "deformed_volume": _record(
                cfg.output_dir / "eye-neutral-deformed-volume.vtu"
            ),
            "deformed_skin": _record(cfg.output_dir / "eye-neutral-deformed-skin.vtp"),
            "fixed_rigid_obstacles": _record(
                cfg.output_dir / "fixed-rigid-obstacles.vtm"
            ),
        },
        "scope": "Independent saved-endpoint audit. It separately checks the IPC mesh, VTK soft-versus-each-rigid triangle intersections, and point containment in each watertight eye. All nonrigid FEM nodes are classified separately; interior inclusion is reported, because triangle clearance is the contact condition. It does not establish anatomical correctness or global mechanical stability.",
    }
    write_json(cfg.output_dir / "summary.json", receipt)
    cherries.log_output(cfg.output_dir)
    cherries.log_metrics(
        {
            "review/intersections": int(ipc_intersections),
            "review/inverted_tetrahedra": receipt["inverted_tetrahedra"],
            "review/detF_min": receipt["detF_min"],
            "review/final_force_N": summary["final_free_force_norm"] * 1e6,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
