"""Render and export exact saved face endpoint geometry for static and browser 3-D review.

A manifest supplies real reference and endpoint VTU files.  Points are interpreted only
according to ``point_convention``: ``positions_are_state`` (default) means a VTU's
points are already that saved state; ``rest_plus_displacement`` adds its named point
array to its rest points.  No tissue target is fabricated from a skin target.
"""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "data" / "50-face-comparisons"
PNG_SIZE = (1800, 1200)
DEFAULT_FPS = 30
MATERIAL_COLORS = ["#267a72", "#806800", "#604f9b"]
MATERIAL_LEGEND = [
    ("fat", MATERIAL_COLORS[0]),
    ("muscle", MATERIAL_COLORS[1]),
    ("aponeurosis", MATERIAL_COLORS[2]),
]


class ContractError(ValueError):
    """Raised when supplied saved geometry cannot support an exact visualization."""


def digest(path: Path) -> dict[str, str | int]:
    """Return a content identity for a source or generated artifact."""
    hasher = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1 << 20), b""):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def resolve(value: str, manifest: Path) -> Path:
    """Resolve manifest-relative paths without silently accepting missing inputs."""
    path = Path(value)
    path = path if path.is_absolute() else manifest.parent / path
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def explicit_frames(history: dict[str, Any], manifest: Path) -> list[Path]:
    """Read an explicit VTU list or VTK series; never infer missing time states."""
    if "frames" in history:
        raw = history["frames"]
        if not isinstance(raw, list) or not raw:
            raise ContractError("history.frames must be a nonempty ordered list")
        return [resolve(str(value), manifest) for value in raw]
    series_path = resolve(str(history["series"]), manifest)
    document = json.loads(series_path.read_text())
    entries = document.get("files")
    if not isinstance(entries, list) or not entries:
        raise ContractError(f"no explicit files in {series_path}")
    frames: list[Path] = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else None
        if not isinstance(name, str):
            raise ContractError(f"invalid series entry in {series_path}")
        frames.append(resolve(name, series_path))
    return frames


def state_grid(
    path: Path, convention: str, displacement_array: str
) -> pv.UnstructuredGrid:
    """Read an actual state, applying only an explicitly declared displacement array."""
    grid = pv.read(path)
    if not isinstance(grid, pv.UnstructuredGrid):
        raise ContractError(f"expected unstructured VTU grid: {path}")
    if convention == "positions_are_state":
        return grid
    if convention != "rest_plus_displacement":
        raise ContractError(f"unknown point convention: {convention}")
    if displacement_array not in grid.point_data:
        raise ContractError(
            f"missing point displacement array {displacement_array!r}: {path}"
        )
    displacement = np.asarray(grid.point_data[displacement_array], dtype=float)
    if displacement.shape != grid.points.shape or not np.isfinite(displacement).all():
        raise ContractError(f"invalid displacement array: {path}")
    result = grid.copy(deep=True)
    result.points = np.asarray(result.points, dtype=float) + displacement
    return result


def topology(
    reference: pv.UnstructuredGrid, endpoint: pv.UnstructuredGrid, label: str
) -> None:
    """Reject nonmatching endpoints; a same-name comparison must share real topology."""
    if reference.n_points != endpoint.n_points or reference.n_cells != endpoint.n_cells:
        raise ContractError(f"{label}: reference/endpoint topology differs")
    if not np.array_equal(
        reference.celltypes, endpoint.celltypes
    ) or not np.array_equal(reference.cells, endpoint.cells):
        raise ContractError(f"{label}: reference/endpoint connectivity differs")


def union_bounds(
    grids: list[pv.UnstructuredGrid],
) -> tuple[float, float, float, float, float, float]:
    """Return a shared physical bounding box for all comparison panels."""
    bounds = np.array([grid.bounds for grid in grids], dtype=float)
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
        float(bounds[:, 4].min()),
        float(bounds[:, 5].max()),
    )


def face_normal(reference: pv.UnstructuredGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return saved face center and mean outward normal from genuine IsFace points."""
    surface = skin(reference).compute_normals(point_normals=True, cell_normals=False)
    if "IsFace" not in surface.point_data:
        message = "face camera requires an IsFace point selection"
        raise ContractError(message)
    mask = np.asarray(surface.point_data["IsFace"], dtype=bool)
    if not mask.any():
        message = "face camera IsFace selection is empty"
        raise ContractError(message)
    center = np.asarray(surface.points)[mask].mean(axis=0)
    normal = np.asarray(surface.point_data["Normals"])[mask].mean(axis=0)
    normal /= np.linalg.norm(normal)
    return center, normal


def camera(
    bounds: tuple[float, float, float, float, float, float],
    reference: pv.UnstructuredGrid,
) -> dict[str, list[float] | float]:
    """Build an upright, frontal physical camera from the face normal on a Y-up mesh."""
    lo = np.asarray((bounds[0], bounds[2], bounds[4]))
    hi = np.asarray((bounds[1], bounds[3], bounds[5]))
    center = (lo + hi) / 2
    _face_center, normal = face_normal(reference)
    vertical_span = bounds[3] - bounds[2]
    scale = 0.62 * vertical_span
    return {
        "position": (center + normal * scale * 3).tolist(),
        "focal_point": center.tolist(),
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": scale,
    }


def mouth_camera(reference: pv.UnstructuredGrid) -> dict[str, list[float] | float]:
    """Frame all actual lip vertices upright from their mean outward normal."""
    surface = skin(reference).compute_normals(point_normals=True, cell_normals=False)
    if "IsLip" not in surface.point_data:
        message = "mouth closeup requires an IsLip point selection"
        raise ContractError(message)
    mask = np.asarray(surface.point_data["IsLip"], dtype=bool)
    if not mask.any():
        message = "mouth closeup IsLip selection is empty"
        raise ContractError(message)
    lips = np.asarray(surface.points)[mask]
    extent = lips.max(axis=0) - lips.min(axis=0)
    # The saved IsLip selector reaches toward the nose.  Frame the lower 25% of
    # its Y span so this is a mouth-and-chin inspection rather than a face crop.
    center = lips.mean(axis=0)
    center[1] = lips.min(axis=0)[1] + 0.25 * extent[1]
    normal = np.asarray(surface.point_data["Normals"])[mask].mean(axis=0)
    normal /= np.linalg.norm(normal)
    scale = 0.72 * max(float(extent[1]), float(extent[2]))
    return {
        "position": (center + normal * scale * 3).tolist(),
        "focal_point": center.tolist(),
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": scale,
    }


def three_quarter_camera(
    base: dict[str, list[float] | float], reference: pv.UnstructuredGrid
) -> dict[str, list[float] | float]:
    """Add an optional upright 3/4 camera with the same physical scale."""
    center, normal = face_normal(reference)
    up = np.asarray((0.0, 1.0, 0.0))
    side = np.cross(up, normal)
    side /= np.linalg.norm(side)
    scale = float(base["parallel_scale"])
    direction = normal + 0.55 * side
    direction /= np.linalg.norm(direction)
    return {
        "position": (center + direction * scale * 3).tolist(),
        "focal_point": center.tolist(),
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": scale,
    }


def apply_camera(plotter: pv.Plotter, spec: dict[str, list[float] | float]) -> None:
    plotter.set_background("#fbfaf7")
    plotter.enable_parallel_projection()
    plotter.camera.position = spec["position"]
    plotter.camera.focal_point = spec["focal_point"]
    plotter.camera.up = spec["view_up"]
    plotter.camera.parallel_scale = spec["parallel_scale"]


def skin(grid: pv.UnstructuredGrid) -> pv.PolyData:
    """Use the actual exterior surface; no fitted skin surface is introduced."""
    return grid.extract_surface().triangulate()


def selected_face_skin(grid: pv.UnstructuredGrid, selector: str) -> pv.PolyData:
    """Return actual exterior triangles whose vertices carry the face selector."""
    surface = skin(grid)
    if selector not in surface.point_data:
        raise ContractError(f"missing face selector {selector!r}")
    flags = np.asarray(surface.point_data[selector], dtype=bool)
    cells = np.asarray(surface.faces).reshape(-1, 4)[:, 1:]
    selected = np.flatnonzero(np.all(flags[cells], axis=1))
    if not len(selected):
        raise ContractError(
            f"face selector {selector!r} selected no exterior triangles"
        )
    return surface.extract_cells(selected).extract_surface().triangulate()


def target_skin(
    path: Path,
    displacement_array: str,
    selector: str,
    valid_selector: str | None = None,
) -> tuple[pv.PolyData, dict[str, int]]:
    """Make target-only face skin; never make target tissue geometry."""
    grid = pv.read(path)
    if not isinstance(grid, pv.UnstructuredGrid):
        raise ContractError(f"expected unstructured target VTU: {path}")
    if displacement_array not in grid.point_data:
        raise ContractError(
            f"missing target displacement {displacement_array!r}: {path}"
        )
    displacement = np.asarray(grid.point_data[displacement_array], dtype=float)
    selector_mask = np.asarray(grid.point_data[selector], dtype=bool)
    if displacement.shape != grid.points.shape:
        raise ContractError(f"invalid target displacement on selected skin: {path}")
    valid_mask = selector_mask
    if valid_selector is not None:
        if valid_selector not in grid.point_data:
            raise ContractError(
                f"missing target validity selector {valid_selector!r}: {path}"
            )
        valid_mask = selector_mask & np.asarray(
            grid.point_data[valid_selector], dtype=bool
        )
    if not np.isfinite(displacement[valid_mask]).all():
        raise ContractError(
            f"invalid target displacement on valid selected skin: {path}"
        )
    moved = grid.copy(deep=True)
    points = np.asarray(moved.points, dtype=float).copy()
    points[valid_mask] += displacement[valid_mask]
    moved.points = points
    return selected_face_skin(moved, selector), {
        "selected_points": int(selector_mask.sum()),
        "displaced_points": int(valid_mask.sum()),
        "retained_rest_points": int((selector_mask & ~valid_mask).sum()),
    }


def categorical_colors(
    surface: pv.PolyData, material_array: str | None
) -> tuple[str | None, dict[str, Any]]:
    """Attach discrete material classes when a genuine cell-data array exists."""
    if material_array == "DominantMaterialPhase":
        names = ("FatFraction", "MuscleFraction", "AponeurosisFraction")
        if not all(name in surface.cell_data for name in names):
            return None, {"material_array": material_array, "available": False}
        fractions = np.stack(
            [np.asarray(surface.cell_data[name], dtype=float) for name in names], axis=1
        )
        surface["_MaterialClass"] = np.argmax(fractions, axis=1).astype(np.uint8)
        return "_MaterialClass", {
            "material_array": material_array,
            "available": True,
            "values": ["fat", "muscle", "aponeurosis"],
            "definition": "largest saved tet volume fraction",
        }
    if material_array is None or material_array not in surface.cell_data:
        return None, {"material_array": None, "available": False}
    source = np.asarray(surface.cell_data[material_array])
    if source.ndim != 1 or not np.issubdtype(source.dtype, np.number):
        return None, {"material_array": material_array, "available": False}
    values = np.unique(source)
    if len(values) > 12:
        return None, {
            "material_array": material_array,
            "available": False,
            "reason": "more than 12 categories",
        }
    labels = np.searchsorted(values, source).astype(np.int32)
    surface["_MaterialClass"] = labels
    return "_MaterialClass", {
        "material_array": material_array,
        "available": True,
        "values": values.tolist(),
    }


def fixed_rest_cohort(
    reference: pv.UnstructuredGrid, specification: dict[str, Any]
) -> dict[str, Any]:
    """Select a rest-defined tet half-space that remains identical in every state."""
    culprit = int(specification["culprit_cell_id"])
    if culprit < 0 or culprit >= reference.n_cells:
        raise ContractError(f"culprit cell outside fixture: {culprit}")
    normal = np.asarray(specification["normal"], dtype=float)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not normal.any():
        raise ContractError(
            "fixed_rest_cohort.normal must be a nonzero finite 3-vector"
        )
    normal /= np.linalg.norm(normal)
    centers = reference.cell_centers().points
    origin = centers[culprit]
    selected = np.flatnonzero((centers - origin) @ normal <= 0)
    if culprit not in selected or not len(selected):
        raise ContractError("rest cohort did not retain the culprit tet")
    return {
        "cell_ids": selected,
        "culprit_cell_id": culprit,
        "plane_origin": origin.tolist(),
        "plane_normal": normal.tolist(),
    }


def cohort_camera(cohort: dict[str, Any]) -> dict[str, list[float] | float]:
    """Frame the retained rest cohort at its culprit plane for a readable comparison."""
    origin = np.asarray(cohort["plane_origin"], dtype=float)
    normal = np.asarray(cohort["plane_normal"], dtype=float)
    scale = 0.025
    return {
        "position": (origin + normal * scale * 3).tolist(),
        "focal_point": origin.tolist(),
        "view_up": [0.0, 1.0, 0.0],
        "parallel_scale": scale,
    }


def render_pair(
    reference: pv.UnstructuredGrid,
    endpoint: pv.UnstructuredGrid,
    spec: dict[str, list[float] | float],
    output: Path,
    title: str,
    material_array: str | None = None,
    cutaway: dict[str, Any] | None = None,
    fixed_cohort: dict[str, Any] | None = None,
    *,
    show_edges: bool = False,
) -> None:
    """Render actual rest and endpoint geometry with a common physical camera."""
    plot = pv.Plotter(
        shape=(1, 2), off_screen=True, window_size=PNG_SIZE, lighting="three lights"
    )
    material_legend = False
    for index, (_label, grid) in enumerate(
        (("Reference", reference), ("Saved endpoint", endpoint))
    ):
        plot.subplot(0, index)
        apply_camera(plot, spec)
        displayed: pv.DataSet = grid
        if fixed_cohort is not None:
            displayed = grid.extract_cells(fixed_cohort["cell_ids"])
        elif cutaway is not None:
            displayed = grid.clip(
                normal=cutaway["normal"], origin=cutaway["origin"], invert=False
            )
        surface = skin(
            displayed.cast_to_unstructured_grid()
            if isinstance(displayed, pv.PolyData)
            else displayed
        )
        scalar, _metadata = categorical_colors(surface, material_array)
        if scalar is None:
            plot.add_mesh(
                surface,
                color="#d9a486",
                smooth_shading=True,
                show_edges=show_edges,
                edge_color="#48504d",
                line_width=0.35,
            )
        else:
            plot.add_mesh(
                surface,
                scalars=scalar,
                categories=True,
                cmap=MATERIAL_COLORS,
                smooth_shading=True,
                show_edges=show_edges,
                edge_color="#48504d",
                line_width=0.35,
                show_scalar_bar=False,
            )
            material_legend = True
        if fixed_cohort is not None:
            fixed = np.asarray(displayed.point_data["IsFixed"], dtype=bool)
            if fixed.any():
                plot.add_points(
                    displayed.points[fixed],
                    color="#126f7c",
                    point_size=7,
                    render_points_as_spheres=True,
                )
            marker = grid.cell_centers().points[fixed_cohort["culprit_cell_id"]]
            plot.add_points(
                marker[None, :],
                color="#d65368",
                point_size=14,
                render_points_as_spheres=True,
            )
        plot.add_axes(line_width=2)
    image = plot.screenshot(return_img=True)
    plot.close()
    caption_height = 125
    result = Image.new(
        "RGB", (image.shape[1], image.shape[0] + caption_height), "#fbfaf7"
    )
    result.paste(Image.fromarray(image), (0, 0))
    caption = ImageDraw.Draw(result)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 23)
    detail_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    caption.text((24, image.shape[0] + 9), title, fill="#172321", font=title_font)
    cohort_note = (
        "Fixed-rest tet cohort cutaway; teal=fixed nodes, pink=cell 662949; visualization only"
        if fixed_cohort is not None
        else ""
    )
    caption.text(
        (24, image.shape[0] + 44),
        "Reference (left) vs saved endpoint (right) · actual saved geometry · true scale",
        fill="#172321",
        font=detail_font,
    )
    if cohort_note:
        caption.text(
            (24, image.shape[0] + 76), cohort_note, fill="#172321", font=detail_font
        )
    if material_legend:
        legend_y = image.shape[0] + (102 if cohort_note else 76)
        x = 24
        for label, color in MATERIAL_LEGEND:
            caption.rectangle((x, legend_y + 3, x + 17, legend_y + 20), fill=color)
            caption.text((x + 25, legend_y), label, fill="#172321", font=detail_font)
            x += 25 + int(caption.textlength(label, font=detail_font)) + 24
    result.save(output)


def render_target_skin(
    reference: pv.UnstructuredGrid,
    target: pv.PolyData,
    spec: dict[str, list[float] | float],
    output: Path,
    title: str,
) -> None:
    """Render target skin over rest face evidence with a separate caption band."""
    plot = pv.Plotter(
        off_screen=True, window_size=(1200, 1200), lighting="three lights"
    )
    apply_camera(plot, spec)
    plot.add_mesh(
        selected_face_skin(reference, "IsFace"),
        style="wireframe",
        color="#737b78",
        line_width=0.5,
        opacity=0.5,
    )
    plot.add_mesh(target, color="#d65368", smooth_shading=True, opacity=0.9)
    plot.add_axes(line_width=2)
    image = plot.screenshot(return_img=True)
    plot.close()
    caption_height = 112
    result = Image.new(
        "RGB", (image.shape[1], image.shape[0] + caption_height), "#fbfaf7"
    )
    result.paste(Image.fromarray(image), (0, 0))
    caption = ImageDraw.Draw(result)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    detail_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    caption.text((24, image.shape[0] + 12), title, fill="#172321", font=title_font)
    caption.text(
        (24, image.shape[0] + 50),
        "Rest-face wireframe + target skin only · no target tissue deformation",
        fill="#172321",
        font=detail_font,
    )
    result.save(output)


def arrays_for_browser(
    surface: pv.PolyData, material_array: str | None
) -> dict[str, Any]:
    """Export triangle positions/normals and optional material class for browser-only viewing."""
    triangles = surface.triangulate()
    faces = np.asarray(triangles.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ContractError("surface export contains nontriangles after triangulation")
    normalled = triangles.compute_normals(
        point_normals=True, cell_normals=False, inplace=False
    )
    record: dict[str, Any] = {
        "positions": np.asarray(normalled.points, dtype=np.float32)
        .reshape(-1)
        .tolist(),
        "normals": np.asarray(normalled.point_data["Normals"], dtype=np.float32)
        .reshape(-1)
        .tolist(),
        "indices": faces[:, 1:].astype(np.uint32).reshape(-1).tolist(),
    }
    scalar, details = categorical_colors(triangles, material_array)
    record["material"] = details
    if scalar is not None:
        record["material_classes"] = np.asarray(
            triangles.cell_data[scalar], dtype=np.uint8
        ).tolist()
    return record


def write_json(path: Path, value: Any) -> None:
    """Write compact deterministic JSON for a browser-fetched geometry asset."""
    path.write_text(json.dumps(value, separators=(",", ":")))


def write_geometry_assets(
    output: Path, cases: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Split state data from a small viewer manifest and deduplicate shared arrays."""
    geometry_dir = output / "geometry"
    geometry_dir.mkdir(exist_ok=True)
    topology_files: dict[str, str] = {}
    state_files: dict[str, str] = {}
    manifest_cases: list[dict[str, Any]] = []
    for case in cases:

        def externalize(state: dict[str, Any]) -> str:
            topology = {
                key: state[key]
                for key in ("indices", "material", "material_classes")
                if key in state
            }
            topology_key = json.dumps(topology, separators=(",", ":"))
            topology_file = topology_files.get(topology_key)
            if topology_file is None:
                topology_file = f"topology-{len(topology_files):04d}.json"
                write_json(geometry_dir / topology_file, topology)
                topology_files[topology_key] = topology_file
            payload = {
                "positions": state["positions"],
                "normals": state["normals"],
                "topology": topology_file,
            }
            payload_key = json.dumps(payload, separators=(",", ":"))
            state_file = state_files.get(payload_key)
            if state_file is None:
                state_file = f"state-{len(state_files):04d}.json"
                write_json(geometry_dir / state_file, payload)
                state_files[payload_key] = state_file
            return f"geometry/{state_file}"

        state_pointers: dict[str, str] = {}
        for name, state in case["states"].items():
            state_pointers[name] = externalize(state)
        history = [
            {
                "step": item["step"],
                "file": externalize(item["state"]),
                **(
                    {"cutaway_file": externalize(item["cutaway_state"])}
                    if "cutaway_state" in item
                    else {}
                ),
            }
            for item in case.get("history_states", [])
        ]
        manifest_cases.append(
            {
                "id": case["id"],
                "label": case["label"],
                "states": state_pointers,
                "history": history,
                "receipt": case["receipt"],
            }
        )
    return manifest_cases


def fiber_assets(reference: pv.UnstructuredGrid, output: Path) -> dict[str, Any] | None:
    """Export a deterministic, unoriented line subsample from saved cell vectors."""
    name = "ActivationFiber"
    if name not in reference.cell_data:
        return None
    vectors = np.asarray(reference.cell_data[name], dtype=float)
    regions = np.asarray(
        reference.cell_data.get("MuscleId", -np.ones(reference.n_cells))
    )
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(vectors).all(axis=1) & (lengths > 1e-12)
    if not valid.any():
        return None
    centers = reference.cell_centers().points
    volume = np.asarray(reference.cell_data.get("Volume", np.ones(reference.n_cells)))
    half_length = float(np.median(np.cbrt(volume[valid])) * 0.8)

    def segments(indices: np.ndarray, limit: int) -> list[float]:
        chosen = indices[:: max(1, int(np.ceil(len(indices) / limit)))]
        unit = vectors[chosen] / lengths[chosen, None]
        ends = np.stack(
            (
                centers[chosen] - half_length * unit,
                centers[chosen] + half_length * unit,
            ),
            axis=1,
        )
        return ends.astype(np.float32).reshape(-1).tolist()

    records: dict[str, list[float]] = {"all": segments(np.flatnonzero(valid), 1600)}
    for region in np.unique(regions[valid]):
        indices = np.flatnonzero(valid & (regions == region))
        records[str(int(region))] = segments(indices, 80)
    path = output / "geometry" / "fixture-activation-fibers.json"
    write_json(
        path,
        {
            "label": "Fixture ActivationFiber: geometry-estimated, unoriented",
            "source_array": name,
            "orientation": "unoriented line geometry; vector sign is not displayed",
            "sampling": "deterministic sorted cell-index stride; at most 1600 all-region or 80 per MuscleId",
            "half_length": half_length,
            "regions": records,
        },
    )
    return {"file": "geometry/fixture-activation-fibers.json", "regions": list(records)}


def write_viewer(
    output: Path,
    title: str,
    cases: list[dict[str, Any]],
    cameras: dict[str, dict[str, list[float] | float]],
    fiber_reference: pv.UnstructuredGrid,
) -> Path:
    """Write a lazy no-build Three.js viewer for exact saved surface triangles."""
    manifest_cases = write_geometry_assets(output, cases)
    fibers = fiber_assets(fiber_reference, output)
    write_json(
        output / "viewer-manifest.json",
        {"title": title, "cases": manifest_cases, "cameras": cameras, "fibers": fibers},
    )
    html_path = output / "viewer.html"
    html_path.write_text("""<!doctype html>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Saved face geometry viewer</title>
<style>html,body{margin:0;height:100%;background:#f7f5ef;color:#182422;font:15px system-ui}body{height:100dvh;display:flex;flex-direction:column;overflow:hidden}header{flex:0 0 auto;padding:.75rem 1rem;background:#173b3a;color:#fff;display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}select,label,button{font:inherit}button{cursor:pointer}.controls{display:flex;gap:.3rem}.hint{font-size:.82rem;color:#c8e0dc}#view{width:100%;flex:1 1 auto;min-height:0;display:block}</style>
<header><strong id="title">Saved geometry</strong><label>case <select id="case"></select></label><label>state <select id="state"><option value="reference">full exterior rest</option><option value="endpoint" selected>full exterior endpoint</option><option value="history">optimization step</option><option value="target_skin">target skin only</option></select></label><label>step <input id="step" type="range" min="0" max="0" value="0" disabled> <output id="stepLabel">—</output></label><label><input id="cutaway" type="checkbox"> fixed-rest cohort cutaway</label><label><input id="material" type="checkbox" checked> material</label><label><input id="targetOverlay" type="checkbox"> target overlay</label><label><input id="fiber" type="checkbox"> prepared geometry-estimated fibers</label><label>muscle region <select id="fiberRegion"><option value="all">all sampled</option></select></label><span class="controls"><button data-camera="front">front</button><button data-camera="three_quarter">3/4</button><button data-camera="mouth">mouth</button><button data-camera="front">reset</button></span><span class="hint" id="hint">Orbit: drag · zoom: wheel · saved exterior triangles</span></header>
<canvas id="view"></canvas>
<script type="module">
import * as THREE from './vendor/three.module.js';
import {OrbitControls} from './vendor/OrbitControls.js';
const data=await (await fetch('viewer-manifest.json')).json(), canvas=document.querySelector('#view'), select=document.querySelector('#case'), state=document.querySelector('#state'), step=document.querySelector('#step'), stepLabel=document.querySelector('#stepLabel'), cutaway=document.querySelector('#cutaway'), material=document.querySelector('#material'), targetOverlay=document.querySelector('#targetOverlay'), fiber=document.querySelector('#fiber'), fiberRegion=document.querySelector('#fiberRegion'), hint=document.querySelector('#hint');
document.querySelector('#title').textContent=data.title;
for(const c of data.cases)select.add(new Option(c.label,c.id));
if(data.fibers)for(const r of data.fibers.regions.filter(x=>x!=='all'))fiberRegion.add(new Option('MuscleId '+r,r));else fiber.parentElement.hidden=true;
const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setClearColor(0xf7f5ef); const scene=new THREE.Scene(); scene.add(new THREE.HemisphereLight(0xffffff,0x334455,2)); const light=new THREE.DirectionalLight(0xffffff,2); light.position.set(4,-5,6);scene.add(light);
const camera=new THREE.PerspectiveCamera(36,1,.001,1e9), cache=new Map(); const controls=new OrbitControls(camera,canvas); controls.enableDamping=true; let mesh,overlay,fiberLines,loadNumber=0;
const palette=[0xd9a486,0xa54b59,0xd8b17d,0x4f8f8b,0x9b8dc2,0x77a866,0xe0a458,0x7095c8];
function dispose(x){x.geometry.dispose();if(Array.isArray(x.material))x.material.forEach(m=>m.dispose());else x.material.dispose()}
async function json(path){if(!cache.has(path))cache.set(path,fetch(path).then(r=>{if(!r.ok)throw Error(path+' '+r.status);return r.json()}));return cache.get(path)}
async function geometry(path){const s=await json(path),t=await json('geometry/'+s.topology);return {...t,...s}}
function makeMesh(m,overlay=false){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(m.positions,3));g.setAttribute('normal',new THREE.Float32BufferAttribute(m.normals,3));g.setIndex(m.indices);let mats;if(!overlay&&material.checked&&m.material_classes){const groups=[];let last=-1,start=0;for(let i=0;i<m.material_classes.length;i++){const k=m.material_classes[i];if(i&&k!==last){groups.push([start,i-start,last]);start=i}last=k}groups.push([start,m.material_classes.length-start,last]);for(const [a,n,k] of groups)g.addGroup(3*a,3*n,k);mats=palette.map(color=>new THREE.MeshStandardMaterial({color,roughness:.8,metalness:0}))}else mats=new THREE.MeshStandardMaterial({color:overlay?0xd65368:0xd9a486,roughness:.85,transparent:overlay,opacity:overlay?.55:1,depthWrite:!overlay});return new THREE.Mesh(g,mats)}
function preset(name){const p=data.cameras[name];camera.position.fromArray(p.position);controls.target.fromArray(p.focal_point);camera.up.fromArray(p.view_up);camera.near=Math.max(+p.parallel_scale/1000,1e-6);camera.far=+p.parallel_scale*100;camera.updateProjectionMatrix();controls.update()}
function selectedCase(){return data.cases.find(x=>x.id===select.value)}
function status(message){hint.textContent=message}
function syncHistory(){const c=selectedCase(),h=c.history||[],historyOption=state.querySelector('option[value="history"]'),targetOption=state.querySelector('option[value="target_skin"]');historyOption.disabled=!h.length;targetOption.disabled=!c.states.target_skin;if((state.value==='history'&&!h.length)||(state.value==='target_skin'&&!c.states.target_skin))state.value='endpoint';step.max=Math.max(h.length-1,0);step.disabled=state.value!=='history'||!h.length;stepLabel.textContent=h.length?'optimization step '+h[+step.value].step:'—'}
async function load(){const ticket=++loadNumber,c=selectedCase(),history=c.history||[];syncHistory();let pointer;if(state.value==='history'){const frame=history[+step.value];if(!frame){state.value='endpoint';return load()}pointer=cutaway.checked&&frame.cutaway_file?frame.cutaway_file:frame.file}else{pointer=c.states[state.value];if(cutaway.checked&&state.value!=='target_skin')pointer=c.states[state.value+'_cutaway']||pointer}if(!pointer){status('Selected saved geometry is unavailable');return}status('Loading saved geometry…');try{const mainPromise=geometry(pointer),overlayPointer=targetOverlay.checked&&state.value!=='target_skin'?c.states.target_skin:null,overlayPromise=overlayPointer?geometry(overlayPointer):null;const main=await mainPromise,overlayData=overlayPromise?await overlayPromise:null;if(ticket!==loadNumber)return;const nextMesh=makeMesh(main),nextOverlay=overlayData?makeMesh(overlayData,true):null;if(ticket!==loadNumber){dispose(nextMesh);if(nextOverlay)dispose(nextOverlay);return}if(mesh){scene.remove(mesh);dispose(mesh)}if(overlay){scene.remove(overlay);dispose(overlay)}mesh=nextMesh;overlay=nextOverlay;scene.add(mesh);if(overlay)scene.add(overlay);status('Orbit: drag · zoom: wheel · saved exterior triangles')}catch(error){if(ticket===loadNumber)status('Geometry load failed: '+error.message)}}
let fiberLoadNumber=0;
async function updateFibers(){const ticket=++fiberLoadNumber;if(fiberLines){scene.remove(fiberLines);dispose(fiberLines);fiberLines=null}if(!fiber.checked||!data.fibers)return;status('Loading estimated fibers…');try{const d=await json(data.fibers.file),p=d.regions[fiberRegion.value];if(ticket!==fiberLoadNumber||!fiber.checked||!p)return;const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(p,3));const next=new THREE.LineSegments(g,new THREE.LineBasicMaterial({color:0x167a85,transparent:true,opacity:.72}));if(ticket!==fiberLoadNumber){dispose(next);return}fiberLines=next;scene.add(fiberLines);status(d.label+' · '+d.orientation)}catch(error){if(ticket===fiberLoadNumber)status('Fiber load failed: '+error.message)}}
function resize(){renderer.setSize(canvas.clientWidth,canvas.clientHeight,false);camera.aspect=canvas.clientWidth/canvas.clientHeight;camera.updateProjectionMatrix()} new ResizeObserver(resize).observe(canvas);select.onchange=()=>{if(!(selectedCase().history||[]).length)state.value='endpoint';syncHistory();load()};state.onchange=()=>{syncHistory();load()};step.oninput=()=>{syncHistory();if(state.value==='history')load()};cutaway.onchange=()=>load();material.onchange=()=>load();targetOverlay.onchange=()=>load();fiber.onchange=()=>updateFibers();fiberRegion.onchange=()=>updateFibers();document.querySelectorAll('[data-camera]').forEach(b=>b.onclick=()=>preset(b.dataset.camera));preset('front');syncHistory();load();(function animate(){requestAnimationFrame(animate);controls.update();renderer.render(scene,camera)})();
</script>""")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    document = json.loads(manifest.read_text())
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ContractError("manifest requires a nonempty cases list")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    shutil.copytree(ROOT / "site" / "vendor", output / "vendor")
    rendered_cases: list[dict[str, Any]] = []
    all_grids: list[pv.UnstructuredGrid] = []
    loaded: list[
        tuple[dict[str, Any], pv.UnstructuredGrid, pv.UnstructuredGrid, Path, Path]
    ] = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise ContractError("case must be an object")
        identity, label = raw.get("id"), raw.get("label")
        if (
            not isinstance(identity, str)
            or not identity
            or not isinstance(label, str)
            or not label
        ):
            raise ContractError("case needs nonempty id and label")
        convention = raw.get("point_convention", "positions_are_state")
        displacement = raw.get("displacement_array", "Displacement")
        reference_path = resolve(str(raw["reference_vtu"]), manifest)
        endpoint_path = resolve(str(raw["endpoint_vtu"]), manifest)
        reference = state_grid(reference_path, convention, displacement)
        endpoint = state_grid(endpoint_path, convention, displacement)
        topology(reference, endpoint, identity)
        loaded.append((raw, reference, endpoint, reference_path, endpoint_path))
        all_grids.extend((reference, endpoint))
    shared_bounds = union_bounds(all_grids)
    shared_camera = camera(shared_bounds, loaded[0][1])
    closeup_camera = mouth_camera(loaded[0][1])
    three_quarter = three_quarter_camera(shared_camera, loaded[0][1])
    for raw, reference, endpoint, reference_path, endpoint_path in loaded:
        identity = raw["id"]
        case_out = output / identity
        case_out.mkdir()
        material_array = raw.get("material_array")
        cutaway = raw.get("cutaway_plane")
        cohort_spec = raw.get("fixed_rest_cohort")
        if cohort_spec is not None and not isinstance(cohort_spec, dict):
            raise ContractError(f"{identity}: fixed_rest_cohort must be an object")
        fixed_cohort = (
            fixed_rest_cohort(reference, cohort_spec)
            if cohort_spec is not None
            else None
        )
        if cutaway is not None and (
            not isinstance(cutaway, dict) or set(cutaway) != {"origin", "normal"}
        ):
            raise ContractError(f"{identity}: cutaway_plane requires origin and normal")
        render_pair(
            reference,
            endpoint,
            shared_camera,
            case_out / "full-head.png",
            raw["label"],
            material_array,
        )
        render_pair(
            reference,
            endpoint,
            closeup_camera,
            case_out / "mouth-closeup.png",
            raw["label"],
            material_array,
        )
        render_pair(
            reference,
            endpoint,
            three_quarter,
            case_out / "three-quarter.png",
            raw["label"],
            material_array,
        )
        if cutaway is not None:
            render_pair(
                reference,
                endpoint,
                shared_camera,
                case_out / "material-cutaway.png",
                raw["label"],
                material_array,
                cutaway,
            )
        if fixed_cohort is not None:
            render_pair(
                reference,
                endpoint,
                shared_camera,
                case_out / "fixed-rest-cohort-cutaway.png",
                raw["label"],
                material_array,
                fixed_cohort=fixed_cohort,
            )
            render_pair(
                reference,
                endpoint,
                cohort_camera(fixed_cohort),
                case_out / "fixed-rest-cohort-closeup.png",
                raw["label"],
                material_array,
                fixed_cohort=fixed_cohort,
                show_edges=True,
            )
        # Physical rest/endpoint use the full exterior boundary. Target remains an
        # IsFace-only surface because the target does not define physical tissue state.
        states = {
            "reference": arrays_for_browser(skin(reference), material_array),
            "endpoint": arrays_for_browser(skin(endpoint), material_array),
        }
        if cutaway is not None:
            for name, grid in (("reference", reference), ("endpoint", endpoint)):
                clipped = grid.clip(
                    normal=cutaway["normal"], origin=cutaway["origin"], invert=False
                )
                clipped_surface = skin(clipped)
                if clipped_surface.n_points == 0:
                    message = f"{identity}: cutaway plane has no actual geometry"
                    raise ContractError(message)
                states[f"{name}_cutaway"] = arrays_for_browser(
                    clipped_surface, material_array
                )
        if fixed_cohort is not None:
            for name, grid in (("reference", reference), ("endpoint", endpoint)):
                states[f"{name}_cutaway"] = arrays_for_browser(
                    skin(grid.extract_cells(fixed_cohort["cell_ids"])), material_array
                )
        target_receipt: dict[str, Any] | None = None
        target_spec = raw.get("target_skin")
        if target_spec is not None:
            if not isinstance(target_spec, dict):
                raise ContractError(f"{identity}: target_skin must be an object")
            target_path = resolve(str(target_spec["vtu"]), manifest)
            selector = str(target_spec.get("selector", "IsFace"))
            displacement = str(
                target_spec.get("displacement_array", "TargetDisplacement")
            )
            valid_selector = target_spec.get("valid_selector")
            target, target_counts = target_skin(
                target_path, displacement, selector, valid_selector
            )
            render_target_skin(
                reference,
                target,
                shared_camera,
                case_out / "target-skin-full-head.png",
                raw["label"],
            )
            render_target_skin(
                reference,
                target,
                closeup_camera,
                case_out / "target-skin-mouth-closeup.png",
                raw["label"],
            )
            render_target_skin(
                reference,
                target,
                three_quarter,
                case_out / "target-skin-three-quarter.png",
                raw["label"],
            )
            states["target_skin"] = arrays_for_browser(target, None)
            target_receipt = {
                "source": digest(target_path),
                "selector": selector,
                "displacement_array": displacement,
                "valid_selector": valid_selector,
                "point_counts": target_counts,
                "scope": "target skin only; no target interior physics claim",
            }
        history_receipt: dict[str, Any] | None = None
        history_states: list[dict[str, Any]] = []
        if raw.get("history") is not None:
            frames = explicit_frames(raw["history"], manifest)
            history_dir = case_out / "history-frames"
            history_dir.mkdir()
            requested_steps = raw["history"].get("steps", list(range(len(frames))))
            if not isinstance(requested_steps, list) or len(requested_steps) != len(
                frames
            ):
                raise ContractError(
                    f"{identity}: history.steps must match history frames"
                )
            for index, (frame_path, step) in enumerate(
                zip(frames, requested_steps, strict=True)
            ):
                frame = state_grid(
                    frame_path,
                    raw.get("point_convention", "positions_are_state"),
                    raw.get("displacement_array", "Displacement"),
                )
                topology(reference, frame, f"{identity} history {index}")
                render_pair(
                    reference,
                    frame,
                    shared_camera,
                    history_dir / f"frame-{index:04d}.png",
                    raw["label"],
                    material_array,
                )
                state = arrays_for_browser(skin(frame), material_array)
                item: dict[str, Any] = {"step": int(step), "state": state}
                if fixed_cohort is not None:
                    item["cutaway_state"] = arrays_for_browser(
                        skin(frame.extract_cells(fixed_cohort["cell_ids"])),
                        material_array,
                    )
                history_states.append(item)
            fps = int(raw["history"].get("fps", DEFAULT_FPS))
            movie = case_out / "evolution.mp4"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(fps),
                    "-i",
                    str(history_dir / "frame-%04d.png"),
                    "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                    "-c:v",
                    "libx264",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(movie),
                ],
                check=True,
            )
            history_receipt = {
                "frame_count": len(frames),
                "fps": fps,
                "frames": [digest(frame) for frame in frames],
                "video": digest(movie),
                "interpolation": "none; one frame per supplied saved state",
            }
        receipt = {
            "case": identity,
            "label": raw["label"],
            "point_convention": raw.get("point_convention", "positions_are_state"),
            "sources": {
                "reference": digest(reference_path),
                "endpoint": digest(endpoint_path),
            },
            "topology": {"points": reference.n_points, "cells": reference.n_cells},
            "shared_bounds": shared_bounds,
            "full_head_camera": shared_camera,
            "mouth_closeup_camera": closeup_camera,
            "cutaway_plane": cutaway,
            "fixed_rest_cohort": (
                {
                    "culprit_cell_id": fixed_cohort["culprit_cell_id"],
                    "plane_origin": fixed_cohort["plane_origin"],
                    "plane_normal": fixed_cohort["plane_normal"],
                    "tet_count": len(fixed_cohort["cell_ids"]),
                    "definition": "rest-cell centroid half-space; identical tet IDs in every state",
                }
                if fixed_cohort is not None
                else None
            ),
            "history": history_receipt,
            "target_skin": target_receipt,
            "browser_surface": "full exterior boundary for physical rest/endpoint; IsFace only for target skin",
            "static_surface": "full exterior boundary for endpoint panels; IsFace only for target overlays",
            "three_quarter_camera": three_quarter,
            "no_deformation_exaggeration": True,
            "outputs": {path.name: digest(path) for path in case_out.glob("*.png")},
        }
        (case_out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
        rendered_cases.append(
            {
                "id": identity,
                "label": raw["label"],
                "states": states,
                "history_states": history_states,
                "receipt": str((case_out / "receipt.json").relative_to(output)),
            }
        )
    write_viewer(
        output,
        str(document.get("title", "Saved face geometry")),
        rendered_cases,
        {
            "front": shared_camera,
            "three_quarter": three_quarter,
            "mouth": closeup_camera,
        },
        loaded[0][1],
    )
    shutil.copy2(manifest, output / "manifest.json")


if __name__ == "__main__":
    main()
