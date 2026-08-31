from __future__ import annotations

# Executed by ParaView's pvpython, not by the project interpreter.
# ruff: noqa: C901, EM101, EM102, FBT003, PLR0912, PLR0915, SLF001, TRY003
import argparse
import hashlib
import json
import shutil
import struct
import subprocess
from pathlib import Path
from typing import Any

import paraview.simple as pvs

EXPECTED_PARAVIEW_VERSION = "6.1.1"
EXPECTED_SCHEMA_VERSION = 1
EXPECTED_DESIGN = "unreachable-layered-plane-strain-strip-inverse"
CASE_ORDER = ("baseline-per-cell", "smoothed-per-cell", "shared-muscle")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value} in {path}")

    payload = json.loads(
        path.read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG: {path}")
    return struct.unpack(">II", header[16:24])


def paraview_version() -> str:
    from paraview import servermanager

    manager = servermanager.vtkSMProxyManager
    return ".".join(
        str(value)
        for value in (
            manager.GetVersionMajor(),
            manager.GetVersionMinor(),
            manager.GetVersionPatch(),
        )
    )


def validate_contract(path: Path) -> dict[str, Any]:
    contract = read_json(path)
    if contract.get("schema_version") != EXPECTED_SCHEMA_VERSION:
        raise ValueError("ParaView contract schema changed")
    if contract.get("design") != EXPECTED_DESIGN:
        raise ValueError("ParaView contract design changed")
    if contract.get("complete") is not True:
        raise ValueError("numerical experiment is incomplete")
    if contract.get("required_paraview_version") != EXPECTED_PARAVIEW_VERSION:
        raise ValueError("ParaView version contract changed")
    cases = contract.get("cases")
    if (
        not isinstance(cases, list)
        or tuple(case["name"] for case in cases) != CASE_ORDER
    ):
        raise ValueError("case order changed")
    return contract


def checked_input(root: Path, relative: str, expected_hash: str) -> Path:
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise ValueError(f"input escapes numerical output root: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    if sha256(path) != expected_hash:
        raise ValueError(f"input identity changed: {path}")
    return path


def configure_camera(view: Any, contract: dict[str, Any]) -> None:
    camera = contract["camera"]
    view.CameraPosition = [float(value) for value in camera["position"]]
    view.CameraFocalPoint = [float(value) for value in camera["focal_point"]]
    view.CameraViewUp = [float(value) for value in camera["view_up"]]
    view.CenterOfRotation = [float(value) for value in camera["focal_point"]]
    view.CameraParallelProjection = 1
    view.CameraParallelScale = float(camera["parallel_scale"])


def configure_view(view: Any, contract: dict[str, Any]) -> None:
    view.Background = [0.035, 0.043, 0.055]
    view.UseColorPaletteForBackground = 0
    view.OrientationAxesVisibility = 0
    view.UseColorPaletteForBackground = 0
    configure_camera(view, contract)


def split_even(layout: Any, location: int, count: int) -> list[int]:
    if count == 1:
        return [location]
    layout.SplitVertical(location, 1.0 / count)
    first = int(layout.SMProxy.GetFirstChild(location))
    second = int(layout.SMProxy.GetSecondChild(location))
    if first < 0 or second < 0:
        raise RuntimeError("ParaView layout split failed")
    return [first, *split_even(layout, second, count - 1)]


def add_text(
    view: Any,
    name: str,
    text: str,
    *,
    location: str = "Upper Left Corner",
    font_size: int = 17,
) -> None:
    source = pvs.Text(registrationName=name)
    source.Text = text
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = location
    display.FontSize = font_size
    display.Color = [0.96, 0.97, 0.98]
    display.Bold = 1


def add_time_annotation(reader: Any, view: Any, name: str) -> None:
    source = pvs.AnnotateTimeFilter(registrationName=name, Input=reader)
    source.Format = "inverse step = {time:.0f}"
    display = pvs.Show(source, view, "TextSourceRepresentation")
    display.WindowLocation = "Upper Right Corner"
    display.FontSize = 18
    display.Color = [0.96, 0.97, 0.98]
    display.Bold = 1


def show_line(
    source: Any,
    view: Any,
    color: tuple[float, float, float],
    width: float,
) -> None:
    display = pvs.Show(source, view, "GeometryRepresentation")
    display.Representation = "Surface"
    display.ColorArrayName = [None, ""]
    display.DiffuseColor = list(color)
    display.AmbientColor = list(color)
    display.LineWidth = width
    display.RenderLinesAsTubes = 1


def show_deformed(
    reader: Any,
    view: Any,
    *,
    color_array: str,
    association: str,
    scalar_range: tuple[float, float],
    preset: str,
    scalar_title: str,
    show_bar: bool,
) -> tuple[Any, Any]:
    warp = pvs.WarpByVector(registrationName=f"Warp by {color_array}", Input=reader)
    warp.Vectors = ["POINTS", "Displacement"]
    warp.ScaleFactor = 1.0
    display = pvs.Show(warp, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.12, 0.13, 0.16]
    display.LineWidth = 0.28
    display.Ambient = 0.24
    display.Diffuse = 0.74
    display.Specular = 0.12
    display.SpecularPower = 18.0
    pvs.ColorBy(display, (association, color_array))
    lookup = pvs.GetColorTransferFunction(color_array)
    lookup.ApplyPreset(preset, True)
    lookup.RescaleTransferFunction(*scalar_range)
    if show_bar:
        display.SetScalarBarVisibility(view, True)
        scalar_bar = pvs.GetScalarBar(lookup, view)
        scalar_bar.Title = scalar_title
        scalar_bar.ComponentTitle = (
            "model length" if color_array == "DisplacementY" else ""
        )
        scalar_bar.Orientation = "Horizontal"
        scalar_bar.WindowLocation = "Lower Right Corner"
        scalar_bar.ScalarBarLength = 0.32
        scalar_bar.ScalarBarThickness = 14
        scalar_bar.TitleColor = [0.96, 0.97, 0.98]
        scalar_bar.LabelColor = [0.96, 0.97, 0.98]
        scalar_bar.TitleFontSize = 13
        scalar_bar.LabelFontSize = 11
    return warp, display


def save_screenshot(
    path: Path,
    layout: Any,
    resolution: tuple[int, int],
    *,
    font_scaling: str = "Scale fonts proportionally",
) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pvs.SaveScreenshot(
        str(temporary),
        layout,
        ImageResolution=list(resolution),
        TransparentBackground=0,
        FontScaling=font_scaling,
    )
    if png_size(temporary) != resolution or temporary.stat().st_size < 20_000:
        raise ValueError(f"invalid ParaView screenshot: {temporary}")
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
        "width": resolution[0],
        "height": resolution[1],
    }


def save_state(path: Path) -> dict[str, Any]:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    pvs.SaveState(str(temporary))
    head = temporary.read_text(encoding="utf-8", errors="strict")[:2048]
    if "ServerManagerState" not in head:
        raise ValueError(f"invalid ParaView state: {temporary}")
    temporary.replace(path)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def open_series(path: Path, name: str) -> Any:
    reader = pvs.OpenDataFile(str(path), registrationName=name)
    reader.PointArrayStatus = [
        "Displacement",
        "DisplacementY",
        "TargetDisplacement",
        "TargetMask",
        "FixedMask",
    ]
    reader.CellArrayStatus = [
        "MaterialId",
        "YoungModulusMPa",
        "PoissonRatio",
        "ActivationXX",
        "ActivationYY",
        "ActivationXYEngineering",
        "ActivationNorm",
        "MuscleMask",
        "SMASMask",
    ]
    reader.UpdatePipeline()
    return reader


def render_evolution(
    contract: dict[str, Any],
    numerical_root: Path,
    output_dir: Path,
    rest_path: Path,
    target_path: Path,
    *,
    fps: int,
    discard_frames: bool,
) -> dict[str, Any]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()
    baseline = contract["cases"][0]
    series_path = checked_input(
        numerical_root, baseline["series"], baseline["series_sha256"]
    )
    reader = open_series(series_path, "Baseline inverse history")
    rest = pvs.OpenDataFile(str(rest_path), registrationName="Rest top")
    target = pvs.OpenDataFile(str(target_path), registrationName="Uniform target top")

    shape_view = pvs.CreateView("RenderView")
    activation_view = pvs.CreateView("RenderView")
    configure_view(shape_view, contract)
    configure_view(activation_view, contract)
    layout = pvs.CreateLayout(name="Layered 2D inverse evolution")
    locations = split_even(layout, 0, 2)
    if not layout.AssignView(locations[0], shape_view):
        raise RuntimeError("failed to assign shape view")
    if not layout.AssignView(locations[1], activation_view):
        raise RuntimeError("failed to assign activation view")
    resolution = tuple(int(value) for value in contract["image_resolution"])
    layout.SetSize(*resolution)

    displacement_range = tuple(
        float(value) for value in contract["displacement_y_range"]
    )
    activation_range = tuple(
        float(value) for value in contract["activation_norm_range"]
    )
    show_deformed(
        reader,
        shape_view,
        color_array="DisplacementY",
        association="POINTS",
        scalar_range=displacement_range,
        preset="Cool to Warm",
        scalar_title="vertical displacement",
        show_bar=True,
    )
    show_deformed(
        reader,
        activation_view,
        color_array="ActivationNorm",
        association="CELLS",
        scalar_range=activation_range,
        preset="Viridis",
        scalar_title="activation norm",
        show_bar=True,
    )
    for view in (shape_view, activation_view):
        show_line(rest, view, (0.82, 0.84, 0.87), 1.6)
        show_line(target, view, (0.90, 0.28, 0.62), 3.0)
    add_text(
        shape_view,
        "Shape title",
        "Deformed strip: color = vertical displacement\nMagenta = uniform +0.1 target; gray = rest top",
    )
    add_text(
        activation_view,
        "Activation title",
        "Inverse controls on the same deformed strip\nColor = active-strain tensor norm; no control regularizer",
    )
    add_time_annotation(reader, shape_view, "Shape inverse step")
    add_time_annotation(reader, activation_view, "Activation inverse step")

    scene = pvs.GetAnimationScene()
    scene.UpdateAnimationUsingDataTimeSteps()
    scene.PlayMode = "Snap To TimeSteps"
    time_values = [float(value) for value in pvs.GetTimeKeeper().TimestepValues]
    if len(time_values) < 3:
        raise ValueError("expected at least three inverse-history time steps")
    expected_time_values = [
        float(value) for value in baseline.get("selected_steps", time_values)
    ]
    if time_values != expected_time_values:
        raise ValueError("ParaView time steps do not match the numerical contract")
    history_sampling = str(baseline.get("history_sampling", "selected-checkpoints"))
    step_by_step = history_sampling == "every-step"
    if step_by_step and time_values != [
        float(step) for step in range(int(baseline["steps"]) + 1)
    ]:
        raise ValueError("step-by-step history must contain every consecutive step")
    output_stem = "evolution-step-by-step" if step_by_step else "evolution"
    pvs.Render(shape_view)
    pvs.Render(activation_view)
    configure_camera(shape_view, contract)
    configure_camera(activation_view, contract)
    pvs.Render(shape_view)
    pvs.Render(activation_view)

    frames_dir = output_dir / f"{output_stem}-frames"
    frames_dir.mkdir(parents=False, exist_ok=False)
    pvs.SaveAnimation(
        str(frames_dir / f"{output_stem}.png"),
        layout,
        scene,
        FrameWindow=[0, len(time_values) - 1],
        SuffixFormat="_{:03d}",
        ImageResolution=list(resolution),
    )
    frames = sorted(frames_dir.glob(f"{output_stem}_*.png"))
    if len(frames) != len(time_values):
        raise ValueError(
            f"ParaView wrote {len(frames)} frames for {len(time_values)} time steps"
        )
    for frame in frames:
        if png_size(frame) != resolution or frame.stat().st_size < 40_000:
            raise ValueError(f"invalid ParaView animation frame: {frame}")

    stills: dict[str, Any] = {}
    for label, index in (
        ("initial", 0),
        ("middle", len(time_values) // 2),
        ("final", len(time_values) - 1),
    ):
        still = output_dir / f"{output_stem}-{label}.png"
        shutil.copy2(frames[index], still)
        if png_size(still) != resolution or still.stat().st_size < 40_000:
            raise ValueError(f"invalid copied ParaView frame: {still}")
        stills[label] = {
            "path": str(still.resolve()),
            "source_frame": str(frames[index].resolve()),
            "time": time_values[index],
            "size_bytes": still.stat().st_size,
            "sha256": sha256(still),
            "width": resolution[0],
            "height": resolution[1],
        }

    state = save_state(output_dir / f"{output_stem}.pvsm")
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("ffmpeg and ffprobe are required to encode ParaView frames")
    video = output_dir / f"{output_stem}.mp4"
    subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / f"{output_stem}_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(video),
        ],
        check=True,
    )
    probe = json.loads(
        subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,pix_fmt,width,height,nb_frames,r_frame_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                str(video),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    stream = probe["streams"][0]
    if (
        stream["codec_name"] != "h264"
        or stream["pix_fmt"] != "yuv420p"
        or int(stream["width"]) != resolution[0]
        or int(stream["height"]) != resolution[1]
        or int(stream["nb_frames"]) != len(frames)
    ):
        raise ValueError(f"unexpected encoded video metadata: {probe}")
    frames_receipt = {
        "directory": str(frames_dir.resolve()),
        "retained": not discard_frames,
        "first_sha256": sha256(frames[0]),
        "last_sha256": sha256(frames[-1]),
    }
    if discard_frames:
        for frame in frames:
            frame.unlink()
        frames_dir.rmdir()
    return {
        "series": str(series_path.resolve()),
        "history_sampling": history_sampling,
        "step_by_step": step_by_step,
        "time_values": time_values,
        "frame_count": len(frames),
        "frames": frames_receipt,
        "stills": stills,
        "state": state,
        "video": {
            "path": str(video.resolve()),
            "size_bytes": video.stat().st_size,
            "sha256": sha256(video),
            "ffprobe": probe,
        },
    }


def render_comparison(
    contract: dict[str, Any],
    numerical_root: Path,
    output_dir: Path,
    rest_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()
    target = pvs.OpenDataFile(str(target_path), registrationName="Uniform target top")
    rest = pvs.OpenDataFile(str(rest_path), registrationName="Rest top")
    layout = pvs.CreateLayout(name="Bump-mechanism final comparison")
    locations = split_even(layout, 0, len(contract["cases"]))
    resolution = tuple(int(value) for value in contract["comparison_resolution"])
    layout.SetSize(*resolution)
    displacement_range = tuple(
        float(value) for value in contract["displacement_y_range"]
    )
    inputs: list[dict[str, Any]] = []

    for location, case in zip(locations, contract["cases"], strict=True):
        frame_path = checked_input(
            numerical_root, case["best_frame"], case["best_frame_sha256"]
        )
        reader = open_series(frame_path, f"{case['name']} best state")
        view = pvs.CreateView("RenderView")
        configure_view(view, contract)
        if not layout.AssignView(location, view):
            raise RuntimeError(f"failed to assign comparison view for {case['name']}")
        show_deformed(
            reader,
            view,
            color_array="DisplacementY",
            association="POINTS",
            scalar_range=displacement_range,
            preset="Cool to Warm",
            scalar_title="vertical displacement",
            show_bar=False,
        )
        show_line(rest, view, (0.82, 0.84, 0.87), 1.5)
        show_line(target, view, (0.90, 0.28, 0.62), 2.8)
        metrics = case["best"]
        add_text(
            view,
            f"{case['name']} metrics",
            (
                f"{case['label']}  |  best step {int(case['best_step'])}\n"
                f"target RMS = {float(metrics['error_rms_fraction_of_target']):.2%}  |  "
                f"top range = {float(metrics['top_y_range']):.5f}\n"
                f"high-pass RMS = {float(metrics['top_y_highpass_rms']):.6f}  |  "
                f"second-difference RMS = {float(metrics['top_y_second_difference_rms']):.6f}  |  "
                f"control jump RMS = {float(metrics['activation_neighbor_jump_rms']):.4f}"
            ),
            font_size=11,
        )
        pvs.Render(view)
        configure_camera(view, contract)
        pvs.Render(view)
        inputs.append(
            {
                "name": case["name"],
                "path": str(frame_path.resolve()),
                "sha256": sha256(frame_path),
            }
        )

    screenshot = save_screenshot(
        output_dir / "final-comparison.png",
        layout,
        resolution,
        font_scaling="Do not scale fonts",
    )
    state = save_state(output_dir / "final-comparison.pvsm")
    return {"inputs": inputs, "screenshot": screenshot, "state": state}


def render_setup(
    contract: dict[str, Any],
    numerical_root: Path,
    output_dir: Path,
    rest_path: Path,
    target_path: Path,
) -> dict[str, Any]:
    pvs.ResetSession()
    pvs._DisableFirstRenderCameraReset()
    baseline = contract["cases"][0]
    frame_path = checked_input(
        numerical_root, baseline["best_frame"], baseline["best_frame_sha256"]
    )
    reader = open_series(frame_path, "Layered strip material setup")
    rest = pvs.OpenDataFile(str(rest_path), registrationName="Rest top")
    target = pvs.OpenDataFile(str(target_path), registrationName="Uniform target top")
    view = pvs.CreateView("RenderView")
    configure_view(view, contract)
    layout = pvs.CreateLayout(name="Layered 2D experiment setup")
    if not layout.AssignView(0, view):
        raise RuntimeError("failed to assign setup view")
    resolution = (1600, 560)
    layout.SetSize(*resolution)
    display = pvs.Show(reader, view, "GeometryRepresentation")
    display.Representation = "Surface With Edges"
    display.EdgeColor = [0.12, 0.13, 0.16]
    display.LineWidth = 0.32
    display.Ambient = 0.25
    display.Diffuse = 0.75
    pvs.ColorBy(display, ("CELLS", "MaterialId"))
    lookup = pvs.GetColorTransferFunction("MaterialId")
    lookup.InterpretValuesAsCategories = 1
    lookup.Annotations = ["0", "fat", "1", "SMAS", "2", "muscle"]
    lookup.IndexedColors = [
        0.88,
        0.80,
        0.67,
        0.34,
        0.62,
        0.80,
        0.86,
        0.28,
        0.22,
    ]
    display.SetScalarBarVisibility(view, True)
    scalar_bar = pvs.GetScalarBar(lookup, view)
    scalar_bar.Title = "material"
    scalar_bar.ComponentTitle = ""
    scalar_bar.Orientation = "Horizontal"
    scalar_bar.WindowLocation = "Lower Right Corner"
    scalar_bar.ScalarBarLength = 0.28
    scalar_bar.ScalarBarThickness = 16
    scalar_bar.TitleColor = [0.96, 0.97, 0.98]
    scalar_bar.LabelColor = [0.96, 0.97, 0.98]
    scalar_bar.TitleFontSize = 14
    scalar_bar.LabelFontSize = 12
    show_line(rest, view, (0.82, 0.84, 0.87), 1.8)
    show_line(target, view, (0.90, 0.28, 0.62), 3.0)
    add_text(
        view,
        "Experiment setup",
        (
            "No-skin layered plane-strain strip: fat / stiff SMAS / local muscle\n"
            "bottom + both sides fixed; every free top point targets (0, +0.1)\n"
            "Magenta = target top; gray = rest top"
        ),
        font_size=16,
    )
    pvs.Render(view)
    configure_camera(view, contract)
    pvs.Render(view)
    screenshot = save_screenshot(output_dir / "setup-materials.png", layout, resolution)
    state = save_state(output_dir / "setup-materials.pvsm")
    return {
        "input": {
            "path": str(frame_path.resolve()),
            "sha256": sha256(frame_path),
        },
        "screenshot": screenshot,
        "state": state,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--discard-frames", action="store_true")
    args = parser.parse_args()
    if args.fps < 1:
        raise ValueError("fps must be positive")
    version = paraview_version()
    if version != EXPECTED_PARAVIEW_VERSION:
        raise RuntimeError(
            f"requires ParaView {EXPECTED_PARAVIEW_VERSION}, found {version}"
        )
    contract_path = args.contract.resolve()
    contract = validate_contract(contract_path)
    numerical_root = contract_path.parent.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        if any(output_dir.iterdir()):
            raise FileExistsError(f"output directory must be empty: {output_dir}")
    else:
        output_dir.mkdir(parents=True)
    rest_path = checked_input(
        numerical_root, contract["rest_top"], contract["rest_top_sha256"]
    )
    target_path = checked_input(
        numerical_root, contract["target_top"], contract["target_top_sha256"]
    )
    setup = render_setup(contract, numerical_root, output_dir, rest_path, target_path)
    evolution = render_evolution(
        contract,
        numerical_root,
        output_dir,
        rest_path,
        target_path,
        fps=args.fps,
        discard_frames=args.discard_frames,
    )
    comparison = render_comparison(
        contract, numerical_root, output_dir, rest_path, target_path
    )
    receipt = {
        "schema_version": EXPECTED_SCHEMA_VERSION,
        "design": EXPECTED_DESIGN,
        "complete": True,
        "status": "ok",
        "paraview_version": version,
        "native_paraview_rendering": True,
        "video_note": "ffmpeg encodes only PNG frames rendered by ParaView",
        "contract": {
            "path": str(contract_path),
            "sha256": sha256(contract_path),
        },
        "setup": setup,
        "evolution": evolution,
        "comparison": comparison,
    }
    write_json(output_dir / "render-receipt.json", receipt)


if __name__ == "__main__":
    main()
