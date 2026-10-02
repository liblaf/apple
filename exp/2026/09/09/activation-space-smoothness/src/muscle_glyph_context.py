"""Reference-muscle surfaces and silhouettes for activation-glyph views."""

# ruff: noqa: EM101, EM102, PERF401, TRY003

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyvista as pv
from vtkmodules.vtkRenderingCore import vtkCoordinate


@dataclass(frozen=True)
class MuscleRegionSurface:
    """One activation region extracted independently from the volume mesh."""

    control_id: int
    muscle_id: int
    name: str
    color_rgb: tuple[int, int, int]
    surface: pv.PolyData


@dataclass(frozen=True)
class MuscleRegionContext:
    """Per-region surfaces plus one labeled combined surface for export."""

    regions: tuple[MuscleRegionSurface, ...]
    combined: pv.PolyData


@dataclass(frozen=True)
class RegionVisibility:
    """Visibility evidence; label image rows follow screenshot top-to-bottom order."""

    global_cell_ids: np.ndarray
    control_ids: np.ndarray
    mask: np.ndarray
    front_control_ids: np.ndarray
    projected_pixel_xy: np.ndarray
    projected_inside: np.ndarray
    front_label_image: np.ndarray
    window_size: tuple[int, int]
    front_surface_control_ids: tuple[int, ...]
    retained_control_ids: tuple[int, ...]
    fully_occluded_control_ids: tuple[int, ...]
    no_retained_control_ids: tuple[int, ...]
    retained_count: int
    retained_boundary_source_count: int
    retained_interior_count: int
    projected_inside_count: int
    projected_background_count: int
    occluded_by_other_region_count: int


def _muted_color(control_id: int) -> tuple[int, int, int]:
    hue = (0.11 + control_id * 0.6180339887498949) % 1.0
    rgb = colorsys.hsv_to_rgb(hue, 0.42, 0.82)
    return tuple(round(255 * channel) for channel in rgb)


def build_muscle_region_context(volume: pv.UnstructuredGrid) -> MuscleRegionContext:
    """Extract every active region separately, retaining shared interfaces."""
    active = np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    control = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)
    muscle = np.asarray(volume.cell_data["MuscleId"], dtype=np.int64)
    region_names = np.asarray(volume.field_data["ActivationRegionName"]).astype(str)
    region_muscle_ids = np.asarray(
        volume.field_data["ActivationRegionMuscleId"], dtype=np.int64
    )
    expected = np.arange(len(region_names), dtype=np.int64)
    if not np.array_equal(np.unique(control[active]), expected):
        raise ValueError(
            "active ActivationControlId values are not contiguous region indices"
        )
    if len(region_muscle_ids) != len(region_names):
        raise ValueError("activation-region name and MuscleId tables differ in length")

    source = volume.copy(deep=False)
    source.cell_data["SourceTetraGlobalCellId"] = np.arange(
        volume.n_cells, dtype=np.int64
    )
    regions: list[MuscleRegionSurface] = []
    for control_id, name in enumerate(region_names):
        cell_ids = np.flatnonzero(active & (control == control_id))
        muscle_ids = np.unique(muscle[cell_ids])
        if len(cell_ids) == 0 or not np.array_equal(
            muscle_ids, region_muscle_ids[control_id : control_id + 1]
        ):
            raise ValueError(
                f"invalid MuscleId mapping for activation region {control_id}"
            )
        surface = source.extract_cells(cell_ids).extract_surface()
        source_tetra_ids = np.asarray(
            surface.cell_data["SourceTetraGlobalCellId"], dtype=np.int64
        ).copy()
        surface.clear_data()
        color = _muted_color(control_id)
        surface.cell_data["ActivationControlId"] = np.full(
            surface.n_cells, control_id, dtype=np.int64
        )
        surface.cell_data["MuscleId"] = np.full(
            surface.n_cells, muscle_ids[0], dtype=np.int64
        )
        surface.cell_data["SourceTetraGlobalCellId"] = source_tetra_ids
        surface.cell_data["RegionColorRGB"] = np.tile(
            np.asarray(color, dtype=np.uint8), (surface.n_cells, 1)
        )
        surface.field_data["ActivationRegionName"] = np.asarray([name])
        regions.append(
            MuscleRegionSurface(control_id, int(muscle_ids[0]), name, color, surface)
        )

    combined = pv.merge([region.surface for region in regions], merge_points=False)
    if not isinstance(combined, pv.PolyData) or combined.n_cells != sum(
        region.surface.n_cells for region in regions
    ):
        raise AssertionError(
            "combining region surfaces lost shared-interface triangles"
        )
    combined.field_data["ActivationRegionName"] = region_names
    combined.field_data["ActivationRegionMuscleId"] = region_muscle_ids
    combined.field_data["RegionColorRGBTable"] = np.asarray(
        [region.color_rgb for region in regions], dtype=np.uint8
    )
    return MuscleRegionContext(tuple(regions), combined)


def save_muscle_region_context(context: MuscleRegionContext, path: Path) -> Path:
    """Save a labeled VTP while retaining duplicate shared-region interfaces."""
    if path.suffix.lower() != ".vtp":
        raise ValueError("muscle-region context must be saved as .vtp")
    path.parent.mkdir(parents=True, exist_ok=True)
    context.combined.save(path, binary=True)
    return path


def set_parallel_camera(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    """Apply the frozen parallel camera before constructing silhouettes."""
    plotter.enable_parallel_projection()
    plotter.camera.position = camera["position"]
    plotter.camera.focal_point = camera["focal_point"]
    plotter.camera.up = camera["view_up"]
    plotter.camera.parallel_scale = camera["parallel_scale"]
    plotter.reset_camera_clipping_range()


def _id_rgb(control_ids: np.ndarray) -> np.ndarray:
    code = np.asarray(control_ids, dtype=np.int64) + 1
    return np.column_stack((code & 255, (code >> 8) & 255, (code >> 16) & 255)).astype(
        np.uint8
    )


def visible_region_mask(
    context: MuscleRegionContext,
    centroids: np.ndarray,
    global_cell_ids: np.ndarray,
    control_ids: np.ndarray,
    camera: dict[str, Any],
    *,
    window_size: tuple[int, int] = (1800, 1800),
) -> RegionVisibility:
    """Keep centroids whose front rasterized muscle label matches their label."""
    centroids = np.asarray(centroids, dtype=np.float64)
    global_cell_ids = np.asarray(global_cell_ids, dtype=np.int64)
    control_ids = np.asarray(control_ids, dtype=np.int64)
    if (
        centroids.shape != (len(global_cell_ids), 3)
        or control_ids.shape != global_cell_ids.shape
    ):
        raise ValueError("centroid, global-cell-ID, and control-ID shapes disagree")
    expected = np.arange(len(context.regions), dtype=np.int64)
    if not np.array_equal(np.unique(control_ids), expected):
        raise ValueError("centroid controls do not contain every context region")

    id_surface = context.combined.copy(deep=False)
    surface_control_ids = np.asarray(
        id_surface.cell_data["ActivationControlId"], dtype=np.int64
    )
    id_surface.cell_data["VisibilityIdRGB"] = _id_rgb(surface_control_ids)
    width, height = window_size
    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.ren_win.SetMultiSamples(0)
    plotter.set_background((0, 0, 0))
    plotter.add_mesh(
        id_surface,
        scalars="VisibilityIdRGB",
        rgb=True,
        preference="cell",
        lighting=False,
        smooth_shading=False,
        show_scalar_bar=False,
        opacity=1.0,
    )
    set_parallel_camera(plotter, camera)
    image = np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8)
    coordinate = vtkCoordinate()
    coordinate.SetCoordinateSystemToWorld()
    pixels = np.empty((len(centroids), 2), dtype=np.int64)
    for index, point in enumerate(centroids):
        coordinate.SetValue(*point)
        pixels[index] = coordinate.GetComputedDisplayValue(plotter.renderer)
    plotter.close()

    codes = (
        image[..., 0].astype(np.int64)
        + (image[..., 1].astype(np.int64) << 8)
        + (image[..., 2].astype(np.int64) << 16)
    )
    valid_codes = np.arange(len(context.regions) + 1, dtype=np.int64)
    if not np.all(np.isin(np.unique(codes), valid_codes)):
        raise AssertionError("visibility buffer contains blended or unknown region IDs")
    x, y = pixels[:, 0], pixels[:, 1]
    inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
    front_codes = np.zeros(len(centroids), dtype=np.int64)
    front_codes[inside] = codes[height - 1 - y[inside], x[inside]]
    front_control_ids = front_codes - 1
    mask = inside & (front_control_ids == control_ids)

    pixel_region_ids = np.unique(codes[codes > 0] - 1)
    retained_region_ids = np.unique(control_ids[mask])
    fully_occluded = np.setdiff1d(expected, pixel_region_ids, assume_unique=True)
    no_retained = np.setdiff1d(expected, retained_region_ids, assume_unique=True)
    boundary_source_ids = np.unique(
        np.asarray(
            context.combined.cell_data["SourceTetraGlobalCellId"], dtype=np.int64
        )
    )
    boundary = np.isin(global_cell_ids, boundary_source_ids)
    background = inside & (front_codes == 0)
    other_region = inside & (front_codes > 0) & (front_control_ids != control_ids)
    return RegionVisibility(
        global_cell_ids=global_cell_ids.copy(),
        control_ids=control_ids.copy(),
        mask=mask,
        front_control_ids=front_control_ids.astype(np.int16),
        projected_pixel_xy=pixels.astype(np.int32),
        projected_inside=inside,
        front_label_image=(codes - 1).astype(np.int16),
        window_size=window_size,
        front_surface_control_ids=tuple(int(value) for value in pixel_region_ids),
        retained_control_ids=tuple(int(value) for value in retained_region_ids),
        fully_occluded_control_ids=tuple(int(value) for value in fully_occluded),
        no_retained_control_ids=tuple(int(value) for value in no_retained),
        retained_count=int(np.count_nonzero(mask)),
        retained_boundary_source_count=int(np.count_nonzero(mask & boundary)),
        retained_interior_count=int(np.count_nonzero(mask & ~boundary)),
        projected_inside_count=int(np.count_nonzero(inside)),
        projected_background_count=int(np.count_nonzero(background)),
        occluded_by_other_region_count=int(np.count_nonzero(other_region)),
    )


def save_region_visibility(visibility: RegionVisibility, path: Path) -> Path:
    """Save the complete raster/projection evidence needed to replay the mask."""
    if path.suffix.lower() != ".npz":
        raise ValueError("region visibility must be saved as .npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        global_cell_ids=visibility.global_cell_ids,
        control_ids=visibility.control_ids,
        mask=visibility.mask,
        front_control_ids=visibility.front_control_ids,
        projected_pixel_xy=visibility.projected_pixel_xy,
        projected_inside=visibility.projected_inside,
        front_label_image=visibility.front_label_image,
        window_size=np.asarray(visibility.window_size, dtype=np.int64),
        front_surface_control_ids=np.asarray(
            visibility.front_surface_control_ids, dtype=np.int64
        ),
        retained_control_ids=np.asarray(
            visibility.retained_control_ids, dtype=np.int64
        ),
        fully_occluded_control_ids=np.asarray(
            visibility.fully_occluded_control_ids, dtype=np.int64
        ),
        no_retained_control_ids=np.asarray(
            visibility.no_retained_control_ids, dtype=np.int64
        ),
        retained_count=np.asarray(visibility.retained_count, dtype=np.int64),
        retained_boundary_source_count=np.asarray(
            visibility.retained_boundary_source_count, dtype=np.int64
        ),
        retained_interior_count=np.asarray(
            visibility.retained_interior_count, dtype=np.int64
        ),
        projected_inside_count=np.asarray(
            visibility.projected_inside_count, dtype=np.int64
        ),
        projected_background_count=np.asarray(
            visibility.projected_background_count, dtype=np.int64
        ),
        occluded_by_other_region_count=np.asarray(
            visibility.occluded_by_other_region_count, dtype=np.int64
        ),
    )
    return path


def add_muscle_context(
    plotter: pv.Plotter,
    context: MuscleRegionContext,
    camera: dict[str, Any],
    *,
    anatomy: bool = False,
    visible_control_ids: tuple[int, ...] | None = None,
) -> list[pv.Actor]:
    """Add region surfaces and camera-dependent per-region silhouettes."""
    set_parallel_camera(plotter, camera)
    actors: list[pv.Actor] = []
    regions = context.regions
    combined = context.combined
    if visible_control_ids is not None:
        visible = np.asarray(visible_control_ids, dtype=np.int64)
        regions = tuple(
            region for region in regions if region.control_id in visible_control_ids
        )
        combined = combined.extract_cells(
            np.isin(combined.cell_data["ActivationControlId"], visible)
        )
    if anatomy:
        actors.append(
            plotter.add_mesh(
                combined,
                scalars="RegionColorRGB",
                rgb=True,
                preference="cell",
                opacity=1.0,
                lighting=True,
                smooth_shading=False,
                show_scalar_bar=False,
            )
        )
        silhouette_color, silhouette_opacity, silhouette_width = "#182028", 0.85, 1.0
    else:
        actors.append(
            plotter.add_mesh(
                combined,
                color="#b5bec6",
                opacity=0.07,
                lighting=False,
                smooth_shading=False,
                show_scalar_bar=False,
            )
        )
        silhouette_color, silhouette_opacity, silhouette_width = "#3f454d", 0.85, 1.2
    for region in regions:
        actors.append(
            plotter.add_silhouette(
                region.surface,
                color=silhouette_color,
                opacity=silhouette_opacity,
                line_width=silhouette_width,
            )
        )
    return actors
