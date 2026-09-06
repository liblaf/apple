"""Render the revised h=.10 pork comparison from exact saved states only.

This ParaView-only program makes one PNG per saved state and a 30 FPS movie
for every available history.  It deliberately leaves tiles, movies, and
shared-control squares absent when their requested source history was never
saved; it never substitutes a different target height, Poisson ratio, or
activation protocol.  No solver module is imported or evaluated.
"""

from __future__ import annotations

# ruff: noqa: EM101, EM102, TRY003
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import paraview.simple as pvs
from paraview import vtk
from PIL import Image, ImageDraw, ImageFont


def legacy_module() -> Any:
    """Load stable shared material/camera/encoding helpers without a solver."""
    path = Path(__file__).with_name("60-render-focused-h020-materials.py")
    spec = importlib.util.spec_from_file_location("focused_h020_renderer", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R = legacy_module()
FPS = R.FPS
IMAGE_SIZE = R.IMAGE_SIZE
FAT_RGB = R.FAT_RGB
MUSCLE_RGB = R.MUSCLE_RGB
EDGE_RGB = R.EDGE_RGB


@dataclass(frozen=True)
class Source:
    case_name: str
    provenance: str
    path: Path
    summary: dict[str, Any]
    manifest: tuple[dict[str, Any], ...]
    output_name: str


@dataclass(frozen=True)
class Missing:
    case_name: str
    reason: str
    output_name: str


H010_CASES = (
    ("h010-direct-nu49", "nu=.49 independent", "direct-nu49"),
    ("h010-direct-nu35", "nu=.35 independent", "direct-nu35"),
    ("h010-shared-nu49", "nu=.49 shared", "shared-nu49"),
    (
        "h010-shared-release-nu49",
        "nu=.49 shared then independent",
        "shared-release-nu49",
    ),
)


def json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def digest(path: Path) -> dict[str, Any]:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            hasher.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hasher.hexdigest(),
    }


def sha256(path: Path) -> str:
    return str(digest(path)["sha256"])


def saved_source(root: Path) -> Source:
    """Accept only the original h=.10 direct, nu=.49 saved history.

    Its legacy summary predates the structured ``case`` receipt.  The strict
    name/height/resolution/energy/loss/DoF checks below bind it to the original
    controlled h=.10 independent run.  Its checked-in run configuration fixes
    that study's ``NU = 0.49``; the legacy VTU did not retain a Poisson array.
    """
    case_dir = root / "height-high"
    summary_path = case_dir / "summary.json"
    series_path = case_dir / "history.vtu.series"
    final_path = case_dir / "final.vtu"
    if any(not path.is_file() for path in (summary_path, series_path, final_path)):
        raise FileNotFoundError(f"incomplete h=.10 source: {case_dir}")
    summary = json_object(summary_path)
    expected = {
        "name": "height-high",
        "height": 0.1,
        "energy": "stable",
        "loss": "l2",
        "resolution": [100, 10],
        "activation_dofs": 1200,
        "n_muscle_triangles": 400,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"not the exact h=.10 direct baseline: {summary_path}")
    series = json_object(series_path)
    files = series.get("files")
    if not isinstance(files, list) or len(files) != summary.get("evaluations"):
        raise ValueError(f"invalid saved-state series: {series_path}")
    manifest: list[dict[str, Any]] = []
    for step, item in enumerate(files):
        expected_name = f"frames/step-{step:04d}.vtu"
        if (
            not isinstance(item, dict)
            or item.get("name") != expected_name
            or item.get("time") != float(step)
        ):
            raise ValueError(f"nonconsecutive saved state {step}: {series_path}")
        frame = case_dir / expected_name
        if not frame.is_file():
            raise FileNotFoundError(frame)
        manifest.append({"step": step, "time": float(step), **digest(frame)})
    if sha256(final_path) != manifest[-1]["sha256"]:
        raise ValueError(f"final.vtu must be the final saved state: {case_dir}")
    return Source(
        "h010-direct-nu49",
        "legacy controlled saved run",
        series_path.resolve(),
        summary,
        tuple(manifest),
        "direct-nu49",
    )


def h020_shared(root: Path) -> Source:
    source = R.exact_source(root, "h020-shared", "canonical")
    return Source(
        source.case_name,
        source.provenance,
        source.path,
        source.summary,
        source.manifest,
        "h020-shared-nu49",
    )


def loss_source(root: Path, name: str) -> Source:
    """Load one exact h=.05 loss-control saved-state history."""
    case_dir = root / name
    summary_path = case_dir / "summary.json"
    series_path = case_dir / "history.vtu.series"
    final_path = case_dir / "final.vtu"
    if any(not path.is_file() for path in (summary_path, series_path, final_path)):
        raise FileNotFoundError(f"incomplete loss-control source: {case_dir}")
    summary = json_object(summary_path)
    expected_loss = {"baseline": "l2", "loss-l1": "l1", "loss-linf": "linf"}[name]
    expected = {
        "name": name,
        "height": 0.05,
        "energy": "stable",
        "loss": expected_loss,
        "resolution": [100, 10],
        "activation_dofs": 1200,
        "n_muscle_triangles": 400,
    }
    if any(summary.get(key) != value for key, value in expected.items()):
        raise ValueError(f"not the exact h=.05 {expected_loss} control: {summary_path}")
    series = json_object(series_path)
    files = series.get("files")
    if not isinstance(files, list) or len(files) != summary.get("evaluations"):
        raise ValueError(f"invalid saved-state series: {series_path}")
    manifest: list[dict[str, Any]] = []
    for step, item in enumerate(files):
        expected_name = f"frames/step-{step:04d}.vtu"
        if (
            not isinstance(item, dict)
            or item.get("name") != expected_name
            or item.get("time") != float(step)
        ):
            raise ValueError(f"nonconsecutive saved state {step}: {series_path}")
        frame = case_dir / expected_name
        if not frame.is_file():
            raise FileNotFoundError(frame)
        manifest.append({"step": step, "time": float(step), **digest(frame)})
    if sha256(final_path) != manifest[-1]["sha256"]:
        raise ValueError(f"final.vtu must be the final saved state: {case_dir}")
    return Source(
        f"h005-{expected_loss}",
        "controlled loss-type saved run",
        series_path.resolve(),
        summary,
        tuple(manifest),
        f"loss-{expected_loss}",
    )


def source_times(reader: Any) -> list[float]:
    reader.UpdatePipeline()
    values = [float(value) for value in pvs.GetTimeKeeper().TimestepValues]
    return values or [0.0]


def arrays(reader: Any, association: str, time: float) -> set[str]:
    reader.UpdatePipeline(time)
    info = reader.GetDataInformation()
    data = (
        info.GetPointDataInformation()
        if association == "POINTS"
        else info.GetCellDataInformation()
    )
    return {
        data.GetArrayInformation(index).GetName()
        for index in range(data.GetNumberOfArrays())
    }


def assert_material_arrays(reader: Any, time: float) -> None:
    points, cells = arrays(reader, "POINTS", time), arrays(reader, "CELLS", time)
    if {"Displacement", "TargetDisplacement"} - points or "MuscleMask" not in cells:
        raise KeyError("saved state lacks render geometry/material arrays")


def combined_time_annotation(reader: Any, view: Any) -> None:
    """The only animation annotation is pipeline-driven, avoiding static Text."""
    annotation = pvs.AnnotateTimeFilter(
        registrationName="optimization step", Input=reader
    )
    annotation.Format = "saved optimization step = {time:.0f}"
    display = pvs.Show(annotation, view, "TextSourceRepresentation")
    display.WindowLocation, display.FontSize, display.Bold = "Upper Left Corner", 19, 1
    display.Color = [0.96, 0.97, 0.98]


def animated_scene(reader: Any, view: Any, bounds: tuple[float, ...]) -> None:
    R.material(reader, view, 0.0, 0.5, FAT_RGB, "fat")
    R.material(reader, view, 0.5, 1.0, MUSCLE_RGB, "muscle")
    R.target(reader, view)
    assert_material_arrays(reader, 0.0)
    combined_time_annotation(reader, view)
    R.configure(view, bounds)


def render_history(
    source: Source, output_root: Path, bounds: tuple[float, ...]
) -> dict[str, Any]:
    R.reset()
    output = output_root / source.output_name
    output.mkdir()
    reader = pvs.OpenDataFile(str(source.path), registrationName=source.case_name)
    times = source_times(reader)
    if times != [row["time"] for row in source.manifest]:
        raise ValueError(f"ParaView/source time mismatch: {source.path}")
    for time in times:
        assert_material_arrays(reader, time)
    view, layout = (
        pvs.CreateView("RenderView"),
        pvs.CreateLayout(name=f"{source.case_name} evolution"),
    )
    layout.SetSize(*IMAGE_SIZE)
    if not layout.AssignView(0, view):
        raise RuntimeError("could not assign animation view")
    animated_scene(reader, view, bounds)
    animation = pvs.GetAnimationScene()
    animation.UpdateAnimationUsingDataTimeSteps()
    animation.PlayMode = "Snap To TimeSteps"
    frames_dir = output / "frames"
    frames_dir.mkdir()
    pvs.SaveAnimation(
        str(frames_dir / "frame.png"),
        layout,
        animation,
        FrameWindow=[0, len(times) - 1],
        SuffixFormat="_%05d",
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) != len(source.manifest) or not all(
        frame.stat().st_size > 20_000 for frame in frames
    ):
        raise ValueError(f"invalid PNG sequence: {frames_dir}")
    png_manifest = tuple(
        {"step": step, **digest(frame)} for step, frame in enumerate(frames)
    )
    if len({row["sha256"] for row in png_manifest}) != len(png_manifest):
        raise ValueError(f"duplicate rendered frames: {frames_dir}")
    final = output / "final-shape.png"
    shutil.copy2(frames[-1], final)
    if sha256(final) != png_manifest[-1]["sha256"]:
        raise ValueError("final still must byte-match final frame")
    video = R.encode(frames, output / "evolution.mp4")
    receipt = {
        "status": "ok",
        "case": source.case_name,
        "provenance": source.provenance,
        "source": digest(source.path),
        "source_manifest": source.manifest,
        "png_manifest": png_manifest,
        "one_saved_state_per_png": True,
        "no_interpolation_or_duplication": True,
        "animation_annotation": "AnnotateTimeFilter only; no static Text source",
        "rendering": {
            "filled_material_polygons": True,
            "material_mask": "MuscleMask",
            "fat_rgb": FAT_RGB,
            "muscle_rgb": MUSCLE_RGB,
            "thin_charcoal_triangle_edges": True,
            "metric_coloring": False,
            "metric_scalar_bars": False,
        },
        "camera": {"shared_union_bounds": bounds, **R.camera(bounds)},
        "video": video,
    }
    R.write_json(output / "render-receipt.json", receipt)
    return receipt


def split_six(layout: Any) -> list[int]:
    layout.SplitHorizontal(0, 1 / 3)
    left, right = (
        int(layout.SMProxy.GetFirstChild(0)),
        int(layout.SMProxy.GetSecondChild(0)),
    )
    layout.SplitHorizontal(right, 0.5)
    slots: list[int] = []
    for column in (
        left,
        int(layout.SMProxy.GetFirstChild(right)),
        int(layout.SMProxy.GetSecondChild(right)),
    ):
        layout.SplitVertical(column, 0.5)
        slots.extend(
            (
                int(layout.SMProxy.GetFirstChild(column)),
                int(layout.SMProxy.GetSecondChild(column)),
            )
        )
    return slots


def static_text(view: Any, value: str, position: str, size: int = 16) -> None:
    source = pvs.Text(registrationName=value[:40])
    source.Text = value
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation, display.FontSize, display.Bold = position, size, 1
    display.Color = [0.96, 0.97, 0.98]


def overlay_exact_label(
    path: Path, value: str, position: tuple[int, int] = (25, 20)
) -> None:
    """Add title pixels after rendering, bypassing ParaView TextSource."""
    image = Image.open(path).convert("RGB")
    font = ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", 28)
    draw = ImageDraw.Draw(image)
    draw.text(
        position,
        value,
        fill=(245, 247, 250),
        font=font,
        stroke_width=1,
        stroke_fill=(0, 0, 0),
    )
    image.save(path)


def final_comparison(
    h010: Source, shared020: Source, output: Path, bounds: tuple[float, ...]
) -> dict[str, Any]:
    """Six slots show the five requested comparisons and one evidence key."""
    R.reset()
    layout = pvs.CreateLayout(name="requested h=.10 and h=.20 final comparison")
    layout.SetSize(*IMAGE_SIZE)
    slots = split_six(layout)
    tiles: tuple[Source | Missing, ...] = (
        h010,
        Missing(
            "h010-direct-nu35",
            "NO exact saved h=.10, nu=.35 independent run",
            "direct-nu35",
        ),
        Missing(
            "h010-shared-nu49",
            "NO exact saved h=.10 shared-activation run",
            "shared-nu49",
        ),
        Missing(
            "h010-shared-release-nu49",
            "NO exact saved h=.10 shared-to-independent continuation",
            "shared-release-nu49",
        ),
        shared020,
        Missing(
            "legend",
            "gold: fat | red: muscle | pink: target\nfilled triangles + thin edges\nno metric colors",
            "legend",
        ),
    )
    labels = (
        "h=.10 | nu=.49 | independent",
        "h=.10 | nu=.35 | independent",
        "h=.10 | nu=.49 | shared",
        "h=.10 | nu=.49 | shared → independent",
        "h=.20 | nu=.49 | shared",
        "evidence key",
    )
    emitted: list[str] = []
    for slot, tile, label in zip(slots, tiles, labels, strict=True):
        view = pvs.CreateView("RenderView")
        if not layout.AssignView(slot, view):
            raise RuntimeError("comparison layout assignment failed")
        R.configure(view, bounds)
        static_text(view, label, "Upper Left Corner", 15)
        if isinstance(tile, Source):
            reader = pvs.OpenDataFile(
                tile.manifest[-1]["path"], registrationName=f"final {tile.case_name}"
            )
            reader.UpdatePipeline()
            R.material(reader, view, 0.0, 0.5, FAT_RGB, f"fat {tile.case_name}")
            R.material(reader, view, 0.5, 1.0, MUSCLE_RGB, f"muscle {tile.case_name}")
            R.target(reader, view)
            emitted.append(tile.case_name)
        else:
            static_text(view, tile.reason, "Lower Left Corner", 14)
            emitted.append(f"{tile.case_name}: {tile.reason}")
    pvs.RenderAllViews()
    pvs.SaveScreenshot(
        str(output),
        layout,
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    if output.stat().st_size <= 20_000:
        raise ValueError("empty final comparison")
    return {
        "path": output.name,
        **digest(output),
        "tiles": emitted,
        "shared_union_camera": {"bounds": bounds, **R.camera(bounds)},
    }


def loss_final_comparison(
    items: tuple[Source, Source, Source], output: Path, bounds: tuple[float, ...]
) -> dict[str, Any]:
    """Render the exact h=.05 L1/L2/Linf controls with material colors only."""
    R.reset()
    layout = pvs.CreateLayout(name="h=.05 loss controls final comparison")
    layout.SetSize(*IMAGE_SIZE)
    layout.SplitHorizontal(0, 1 / 3)
    left, right = (
        int(layout.SMProxy.GetFirstChild(0)),
        int(layout.SMProxy.GetSecondChild(0)),
    )
    layout.SplitHorizontal(right, 0.5)
    slots = (
        left,
        int(layout.SMProxy.GetFirstChild(right)),
        int(layout.SMProxy.GetSecondChild(right)),
    )
    for slot, item in zip(slots, items, strict=True):
        view = pvs.CreateView("RenderView")
        if not layout.AssignView(slot, view):
            raise RuntimeError("loss comparison layout assignment failed")
        R.configure(view, bounds)
        static_text(
            view,
            f"h=.05 | {item.summary['loss'].upper()} loss",
            "Upper Left Corner",
            18,
        )
        reader = pvs.OpenDataFile(item.path, registrationName=f"final {item.case_name}")
        reader.UpdatePipeline()
        assert_material_arrays(reader, 0.0)
        R.material(reader, view, 0.0, 0.5, FAT_RGB, f"fat {item.case_name}")
        R.material(reader, view, 0.5, 1.0, MUSCLE_RGB, f"muscle {item.case_name}")
        R.target(reader, view)
    pvs.RenderAllViews()
    pvs.SaveScreenshot(
        str(output),
        layout,
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    if output.stat().st_size <= 20_000:
        raise ValueError("empty loss final comparison")
    return {
        "path": output.name,
        **digest(output),
        "cases": [item.case_name for item in items],
        "camera": {"bounds": bounds, **R.camera(bounds)},
    }


def final_shape(
    source: Source, output: Path, bounds: tuple[float, ...]
) -> dict[str, Any]:
    """Write one uncomposited, material-only final shape PNG."""
    R.reset()
    view, layout = pvs.CreateView("RenderView"), pvs.CreateLayout(name=source.case_name)
    layout.SetSize(*IMAGE_SIZE)
    if not layout.AssignView(0, view):
        raise RuntimeError("final-shape layout assignment failed")
    R.configure(view, bounds)
    reader = pvs.OpenDataFile(
        str(source.manifest[-1]["path"]), registrationName=f"final {source.case_name}"
    )
    reader.UpdatePipeline()
    assert_material_arrays(reader, 0.0)
    R.material(reader, view, 0.0, 0.5, FAT_RGB, f"fat {source.case_name}")
    R.material(reader, view, 0.5, 1.0, MUSCLE_RGB, f"muscle {source.case_name}")
    R.target(reader, view)
    pvs.SaveScreenshot(
        str(output),
        layout,
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    overlay_exact_label(output, source.case_name)
    if output.stat().st_size <= 20_000:
        raise ValueError(f"empty final shape: {output}")
    return {
        "path": output.name,
        **digest(output),
        "case": source.case_name,
        "camera": {"bounds": bounds, **R.camera(bounds)},
    }


def safe_show_polydata(
    polydata: vtk.vtkPolyData, view: Any, color: tuple[float, float, float], name: str
) -> None:
    """ParaView 6.1 rejects ColorBy(None) for a trivial producer."""
    producer = pvs.TrivialProducer(registrationName=name)
    producer.GetClientSideObject().SetOutput(polydata)
    display = pvs.Show(producer, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.ColorArrayName = [None, ""]
    display.DiffuseColor, display.AmbientColor, display.EdgeColor = (
        color,
        color,
        EDGE_RGB,
    )
    display.LineWidth, display.Opacity, display.Ambient, display.Diffuse = (
        1.2,
        0.85,
        0.35,
        0.65,
    )


def shared_square_configure(view: Any, bounds: tuple[float, ...]) -> None:
    """Fit a full square in one half of the two-column square layout."""
    xmin, xmax, ymin, ymax, _zmin, _zmax = bounds
    center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
    width, height = xmax - xmin, ymax - ymin
    view.Background = [0.035, 0.043, 0.055]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.CameraParallelProjection = 1
    view.CameraPosition = [center_x, center_y, max(width, height, 1.0) * 3]
    view.CameraFocalPoint = [center_x, center_y, 0.0]
    view.CameraViewUp = [0.0, 1.0, 0.0]
    # A half-width view has aspect ratio .9, not IMAGE_SIZE's 1.8.
    view.CameraParallelScale = 0.64 * max(height, width / 0.9, 1.0e-6)


def missing_square(output: Path) -> dict[str, Any]:
    R.reset()
    view, layout = (
        pvs.CreateView("RenderView"),
        pvs.CreateLayout(name="h=.10 shared activation unavailable"),
    )
    layout.SetSize(*IMAGE_SIZE)
    if not layout.AssignView(0, view):
        raise RuntimeError("could not assign missing shared-square view")
    R.configure(view, (0.0, 1.0, 0.0, 1.0, 0.0, 0.0))
    static_text(view, "h=.10 shared activation square", "Upper Left Corner", 23)
    static_text(
        view,
        "NO exact saved h=.10 shared-activation history\nNo deformed square is fabricated.",
        "Lower Left Corner",
        20,
    )
    pvs.SaveScreenshot(
        str(output),
        layout,
        ImageResolution=list(IMAGE_SIZE),
        FontScaling="Do not scale fonts",
    )
    return {
        "path": output.name,
        **digest(output),
        "status": "missing_exact_source",
        "reason": "h=.10 shared control was not saved",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h010-root", type=Path, required=True)
    parser.add_argument("--h020-canonical-root", type=Path, required=True)
    parser.add_argument("--loss-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--square-only", action="store_true")
    parser.add_argument("--video-case", type=str)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if (
        output.exists()
        and any(output.iterdir())
        and not (args.square_only or args.video_case is not None)
    ):
        raise FileExistsError(f"output root must be empty: {output}")
    if args.square_only:
        shared020 = h020_shared(args.h020_canonical_root.resolve())
        R.show_polydata = safe_show_polydata
        R.configure = shared_square_configure
        R.FAT_RGB = (0.93, 0.36, 0.36)
        square_path = output / "h020-shared-activation-square.png"
        h020_square = R.render_shared_square(shared020, square_path)
        overlay_exact_label(square_path, "h=.20 shared muscle activation", (640, 52))
        h020_square = {**h020_square, **digest(square_path)}
        receipt_path = output / "render-receipt.json"
        receipt = R.json_object(receipt_path)
        receipt["shared_activation_squares"]["h020"] = h020_square
        receipt["renderer_source"] = digest(Path(__file__))
        R.write_json(receipt_path, receipt)
        return
    h010 = saved_source(args.h010_root.resolve())
    shared020 = h020_shared(args.h020_canonical_root.resolve())
    losses = tuple(
        loss_source(args.loss_root.resolve(), name)
        for name in ("baseline", "loss-l1", "loss-linf")
    )
    available = {item.case_name: item for item in (h010, shared020, *losses)}
    if args.video_case is not None:
        if args.video_case not in available:
            raise ValueError(
                f"video case must be one of {sorted(available)}, got {args.video_case}"
            )
        source = available[args.video_case]
        output.mkdir(parents=True, exist_ok=True)
        receipt = render_history(source, output, R.scan_union_camera((source,)))
        R.write_json(
            output / f"{source.output_name}-video-receipt.json",
            {
                "status": "ok",
                "video_case": source.case_name,
                "physics_runs": 0,
                "one_saved_state_per_video_frame": True,
                "history": receipt,
                "renderer_source": digest(Path(__file__)),
            },
        )
        return
    bounds = R.scan_union_camera((h010, shared020))
    loss_bounds = R.scan_union_camera(losses)
    output.mkdir(parents=True, exist_ok=True)
    final_shapes = [
        final_shape(h010, output / "h010-direct-nu49-final-shape.png", bounds),
        final_shape(shared020, output / "h020-shared-nu49-final-shape.png", bounds),
        *[
            final_shape(item, output / f"{item.case_name}-final-shape.png", loss_bounds)
            for item in losses
        ],
    ]
    missing_h010_square = missing_square(output / "h010-shared-activation-square.png")
    R.show_polydata = safe_show_polydata
    R.configure = shared_square_configure
    # Both panels depict muscle under one shared control; the light red rest
    # square is deliberately not fat-colored.
    R.FAT_RGB = (0.93, 0.36, 0.36)
    h020_square = R.render_shared_square(
        shared020, output / "h020-shared-activation-square.png"
    )
    overlay_exact_label(
        output / "h020-shared-activation-square.png",
        "h=.20 shared muscle activation",
        (640, 52),
    )
    h020_square = {
        **h020_square,
        **digest(output / "h020-shared-activation-square.png"),
    }
    histories = (
        []
        if args.static_only
        else [render_history(source, output, bounds) for source in (h010, shared020)]
    )
    R.write_json(
        output / "render-receipt.json",
        {
            "status": "ok",
            "static_only": args.static_only,
            "requested_cases": [name for name, _label, _output_name in H010_CASES]
            + ["h020-shared-nu49"],
            "available_exact_cases": [h010.case_name, shared020.case_name],
            "missing_exact_cases": [
                "h010-direct-nu35",
                "h010-shared-nu49",
                "h010-shared-release-nu49",
            ],
            "loss_control_cases": [item.case_name for item in losses],
            "fps": FPS,
            "one_saved_state_per_video_frame": True,
            "shared_union_camera": {"bounds": bounds, **R.camera(bounds)},
            "render_contract": {
                "material_only": True,
                "filled_framework": True,
                "material_mask": "MuscleMask",
                "metric_scalar_coloring": False,
                "metric_scalar_bars": False,
                "fat_rgb": FAT_RGB,
                "muscle_rgb": MUSCLE_RGB,
                "triangle_edges": "thin charcoal",
                "animation_text": "AnnotateTimeFilter only",
            },
            "histories": histories,
            "uncomposited_final_shapes": final_shapes,
            "shared_activation_squares": {
                "h010": missing_h010_square,
                "h020": h020_square,
            },
            "renderer_source": digest(Path(__file__)),
        },
    )


if __name__ == "__main__":
    main()
