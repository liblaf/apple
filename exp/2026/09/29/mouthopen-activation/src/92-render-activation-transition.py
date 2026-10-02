# Copyright (c) 2026 liblaf
# ruff: noqa: C901, E402, EM101, EM102, PLR0912, PLR0915, RUF001, TRY003
"""Render a re-equilibrated Smile-to-MouthOpen activation transition."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
STRESS_SRC = ROOT / "exp/2026/09/21/stress-activation-loss/src"
CONTEXT_SRC = ROOT / "exp/2026/09/14/dominant-activation-ablation/src"
sys.path.extend((str(STRESS_SRC), str(CONTEXT_SRC)))

from experiment import Profile
from shape_scene import CRANIUM_PATH, EYES_PATH, MANDIBLE_PATH

spec = importlib.util.spec_from_file_location(
    "mouthopen_transition_render_helpers",
    STRESS_SRC / "52-render-four-stage-figures.py",
)
assert spec is not None
assert spec.loader is not None
FIG = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FIG
spec.loader.exec_module(FIG)

LOG = logging.getLogger(__name__)
FPS = 30
TRANSITION_STATES = 121
HOLD_FRAMES_EACH_END = 30
VIDEO_SIZE = (1920, 1080)
PANEL_WINDOW = FIG.WINDOW
PANEL_DISPLAY_SIZE = (960, 780)
PANEL_ORIGIN = ((0, 145), (960, 145))
PAGE_BACKGROUND = "#101315"
AMOUNT_CAP = 60.0
FONT_DIR = Path("/usr/share/fonts/TTF")
KEYFRAME_INDICES = (0, 30, 60, 90, 120)


class Config(cherries.BaseConfig):
    source: Path = Path("91-smile-mouthopen-transition-003")
    fixture: Path = Path("30-pruned-fixture")
    prepared: Path = Path("10-mandible/prepared.npz")
    mouthopen: Path = Path("70-mouthopen-four-stage/rankone_learned/last.npz")
    output: Path = Path("92-smile-mouthopen-transition-render")
    fps: int = FPS
    hold_frames_each_end: int = HOLD_FRAMES_EACH_END


def record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "bytes": path.stat().st_size,
    }


def verify_record(item: dict[str, Any], label: str) -> Path:
    path = Path(item["path"])
    if record(path)["sha256"] != item["sha256"]:
        raise ValueError(f"{label} receipt mismatch: {path}")
    return path


def font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont:
    suffix = "-Bold" if bold else ""
    return ImageFont.truetype(str(FONT_DIR / f"DejaVuSans{suffix}.ttf"), size)


def rigid(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (
        (points - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
    )


def expected_alpha(index: int, count: int) -> float:
    return float((1 - np.cos(np.pi * index / (count - 1))) / 2)


def make_camera(
    points: np.ndarray,
    frame_paths: list[Path],
    mandible_points: np.ndarray,
    pivot: np.ndarray,
    cranium: pv.PolyData,
    eyes: pv.PolyData,
) -> dict[str, Any]:
    toward = np.array([0.65, 0.03, 1.0], dtype=np.float64)
    toward /= np.linalg.norm(toward)
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    basis = np.column_stack((right, up, toward))
    low, high = np.full(3, np.inf), np.full(3, -np.inf)

    def include(value: np.ndarray) -> None:
        nonlocal low, high
        projected = np.asarray(value) @ basis
        low = np.minimum(low, projected.min(axis=0))
        high = np.maximum(high, projected.max(axis=0))

    include(np.asarray(cranium.points))
    include(np.asarray(eyes.points))
    for frame_path in frame_paths:
        with np.load(frame_path, allow_pickle=False) as data:
            u = data["u"]
            pose = data["pose"]
            include(points + u)
            include(rigid(mandible_points, pivot, pose))
    center = basis @ ((low + high) / 2)
    half = (high - low) / 2
    return {
        "position": (center + 0.6 * toward).tolist(),
        "focal_point": center.tolist(),
        "view_up": up.tolist(),
        "parallel_scale": float(
            1.055 * max(half[1], half[0] / (PANEL_WINDOW[0] / PANEL_WINDOW[1]))
        ),
        "projected_bounds_m": [low.tolist(), high.tolist()],
        "window_size": list(PANEL_WINDOW),
        "policy": "one orthographic camera fitted to every deformed tetmesh vertex and posed jaw over every re-equilibrated frame",
    }


def verify_transition(
    source: Path,
    fixture: Path,
    prepared_path: Path,
    mouthopen_path: Path,
    mesh70_path: Path,
) -> tuple[
    dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]
]:
    summary_path = source / "summary.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("schema") != "activation-expression-transition-v1":
        raise ValueError("unexpected transition schema")
    if summary.get("status") != "completed":
        raise RuntimeError(f"transition run is not complete: {summary.get('status')}")
    frame_entries = summary["frames"]
    if len(frame_entries) != TRANSITION_STATES:
        raise ValueError(f"expected {TRANSITION_STATES} equilibrium states")
    for item in summary["inputs"].values():
        verify_record(item, "transition input")
    for name, item in summary["outputs"].items():
        verify_record(item, f"transition output {name}")
    source_manifest_path = source / "source-manifest.json"
    source_manifest = json.loads(source_manifest_path.read_text())
    for module, item in source_manifest.items():
        verify_record(item, f"frozen transition source {module}")
        if "source" in item:
            verify_record(
                {"path": item["source"], "sha256": item["sha256"]},
                f"live source {module}",
            )
    np.testing.assert_array_equal(
        np.asarray(summary["mapping"]["remaining_active_cells"]),
        np.asarray(288172),
    )

    with (
        np.load(source / "mesh.npz", allow_pickle=False) as z,
        np.load(mesh70_path, allow_pickle=False) as reference,
    ):
        mesh = {key: z[key].copy() for key in z.files}
        for key in ("rest_points", "tets", "active_ids", "skin_ids", "triangles"):
            np.testing.assert_array_equal(mesh[key], reference[key])
    with np.load(source / "endpoints.npz", allow_pickle=False) as z:
        endpoints = {key: z[key].copy() for key in z.files}
    with np.load(mouthopen_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(endpoints["S_mouthopen"], z["S"])
    with np.load(prepared_path, allow_pickle=False) as z:
        np.testing.assert_array_equal(endpoints["pose_mouthopen"], z["pose"])
        np.testing.assert_array_equal(endpoints["pivot"], z["pivot"])

    fixture_volume = pv.read(fixture / "volume.vtu")
    fixed = np.asarray(fixture_volume.point_data["IsFixed"], dtype=bool)
    mandible_group = list(fixture_volume.field_data["GroupName"]).index("Mandible")
    jaw_fixed = fixed & (
        np.asarray(fixture_volume.point_data["GroupId"], dtype=np.int64)
        == mandible_group
    )
    if fixed.shape != (len(mesh["rest_points"]),):
        raise ValueError("fixture fixed-point mask does not match the saved mesh")
    if not jaw_fixed.any():
        raise ValueError("fixture has no fixed Mandible points")

    if endpoints["S_smile"].shape != endpoints["S_mouthopen"].shape:
        raise ValueError("Smile and MouthOpen endpoint tensors have different shapes")
    if endpoints["S_smile"].shape != (len(mesh["active_ids"]), 3, 3):
        raise ValueError("endpoint tensors do not match the active-cell mesh")
    for s in (endpoints["S_smile"], endpoints["S_mouthopen"]):
        np.testing.assert_allclose(s, np.swapaxes(s, 1, 2), rtol=0, atol=1e-12)

    frame_paths = []
    for index, entry in enumerate(frame_entries):
        if int(entry["index"]) != index:
            raise ValueError("transition frame indices are out of order")
        alpha = expected_alpha(index, len(frame_entries))
        np.testing.assert_allclose(float(entry["alpha"]), alpha, rtol=0, atol=2e-15)
        path = verify_record(entry["checkpoint"], f"transition frame {index}")
        frame_paths.append(path)
        with np.load(path, allow_pickle=False) as z:
            if set(z.files) != {"u", "alpha", "pose"}:
                raise ValueError(
                    f"frame {index} has unexpected stored fields {z.files}"
                )
            if z["u"].shape != mesh["rest_points"].shape:
                raise ValueError(
                    f"frame {index} displacement shape does not match mesh"
                )
            np.testing.assert_allclose(z["alpha"], alpha, rtol=0, atol=2e-15)
            np.testing.assert_allclose(
                z["pose"], alpha * endpoints["pose_mouthopen"], rtol=0, atol=2e-15
            )
            if not np.isfinite(z["u"]).all():
                raise ValueError(f"frame {index} contains nonfinite displacement")
            expected_jaw = (
                rigid(
                    mesh["rest_points"][jaw_fixed],
                    endpoints["pivot"],
                    z["pose"],
                )
                - mesh["rest_points"][jaw_fixed]
            )
            np.testing.assert_allclose(
                z["u"][jaw_fixed], expected_jaw, rtol=0, atol=1e-9
            )
            np.testing.assert_allclose(z["u"][fixed & ~jaw_fixed], 0, rtol=0, atol=1e-9)
        diag = entry["diagnostics"]
        if not diag.get("solver_valid"):
            raise ValueError(f"frame {index} lacks a valid equilibrium receipt")
        if float(diag["accepted_force_norm"]) > float(summary["config"]["force_atol"]):
            raise ValueError(f"frame {index} exceeds the saved force tolerance")
    if float(frame_entries[0]["alpha"]) != 0 or float(frame_entries[-1]["alpha"]) != 1:
        raise ValueError("transition endpoints do not span alpha 0 to 1")
    return summary, mesh, endpoints, frame_entries


def render_panel_images(
    *,
    frame_index: int,
    frame_path: Path,
    alpha: float,
    summary: dict[str, Any],
    mesh: dict[str, np.ndarray],
    endpoints: dict[str, np.ndarray],
    volume: pv.UnstructuredGrid,
    boundary: pv.PolyData,
    boundary_ids: np.ndarray,
    active_ids: np.ndarray,
    control_ids: np.ndarray,
    inverse_dm: np.ndarray,
    camera: dict[str, Any],
    scene: SimpleNamespace,
    mandible_rest: pv.PolyData,
    output: Path,
) -> tuple[Image.Image, dict[str, Any]]:
    with np.load(frame_path, allow_pickle=False) as archive:
        u = archive["u"]
        pose = archive["pose"]
    points = mesh["rest_points"]
    tets = mesh["tets"]
    deformed = points + u
    surface = boundary.copy(deep=True)
    surface.points = deformed[boundary_ids]
    if not np.allclose(surface.points, deformed[boundary_ids], rtol=0, atol=0):
        raise AssertionError("render boundary differs from the saved X+u surface")

    shape_path = output / "temporary-shape.png"
    jaw = mandible_rest.copy(deep=True)
    jaw.points = rigid(np.asarray(mandible_rest.points), endpoints["pivot"], pose)
    shape_scene = SimpleNamespace(
        bones={"cranium": scene.bones["cranium"], "mandible": jaw}, eyes=scene.eyes
    )
    # Draw the pose-specific jaw; the common helper receives this temporary scene.
    projection = FIG.render_shape(shape_scene, surface, camera, shape_path)

    active_tets = deformed[tets[active_ids]]
    deformation_gradient = (active_tets[:, 1:] - active_tets[:, :1]).transpose(
        0, 2, 1
    ) @ inverse_dm
    centers = active_tets.mean(axis=1)
    strain = (1.0 - alpha) * endpoints["S_smile"] + alpha * endpoints["S_mouthopen"]
    b = strain + np.eye(3)
    glyph = FIG.principal_glyphs(b, deformation_gradient, centers)
    positive = glyph.eligible & (glyph.signed_display_percent > 0)
    amplitudes = np.sqrt(1 + glyph.eigenvalues_z[positive]) - 1
    max_amplitude = float(amplitudes.max()) if len(amplitudes) else 0.0
    if max_amplitude > AMOUNT_CAP + 1e-10:
        raise ValueError(f"frame {frame_index} amplitude exceeds the shared scale")

    volume_copy = volume.copy(deep=True)
    volume_copy.points = deformed
    context = FIG.build_muscle_region_context(volume_copy)
    visibility = FIG.visible_region_mask(
        context,
        glyph.centers,
        active_ids,
        control_ids,
        camera,
        window_size=FIG.ACTIVATION_WINDOW,
    )
    mask = visibility.mask & positive
    activation_path = output / "temporary-activation.png"
    if mask.any():
        activation_projection = FIG.render_activation(
            surface, glyph, mask, camera, activation_path
        )
    else:
        plotter = FIG.new_plotter(FIG.ACTIVATION_WINDOW)
        plotter.add_mesh(
            surface, color=FIG.SHAPE_COLOR, opacity=FIG.ACTIVATION_CONTEXT_OPACITY
        )
        activation_projection = FIG.capture(plotter, camera, activation_path)
    np.testing.assert_allclose(projection, activation_projection, rtol=0, atol=1e-12)

    left = Image.open(shape_path).convert("RGB")
    right = Image.open(activation_path).convert("RGB")
    left.thumbnail(PANEL_DISPLAY_SIZE, Image.Resampling.LANCZOS)
    right.thumbnail(PANEL_DISPLAY_SIZE, Image.Resampling.LANCZOS)
    frame = Image.new("RGB", VIDEO_SIZE, PAGE_BACKGROUND)
    draw = ImageDraw.Draw(frame)
    draw.text(
        (960, 25),
        "Smile → MouthOpen activation transition",
        font=font(34, bold=True),
        fill="white",
        anchor="mt",
    )
    draw.text(
        (480, 105),
        "Re-equilibrated tetmesh shape",
        font=font(23, bold=True),
        fill="#e6e6e6",
        anchor="mt",
    )
    draw.text(
        (1440, 105),
        "Principal activation mode",
        font=font(23, bold=True),
        fill="#e6e6e6",
        anchor="mt",
    )
    draw.text(
        (1885, 73),
        f"Frame {frame_index + 1}/{TRANSITION_STATES} · α {alpha:.3f}",
        font=font(20),
        fill="#d9d9d9",
        anchor="rt",
    )
    frame.paste(left, (PANEL_ORIGIN[0][0], PANEL_ORIGIN[0][1]))
    frame.paste(right, (PANEL_ORIGIN[1][0], PANEL_ORIGIN[1][1]))
    left.close()
    right.close()
    draw_transition_legend(frame, draw)
    frame_record = {
        "index": frame_index,
        "alpha": alpha,
        "checkpoint": str(frame_path.resolve()),
        "visible_positive_modes": int(mask.sum()),
        "eligible_positive_modes": int(positive.sum()),
        "positive_principal_amplitude_max": max_amplitude,
        "full_S_blend_formula": "(1-alpha)*S_smile + alpha*S_mouthopen",
        "surface_points": surface.n_points,
        "surface_triangles": surface.n_cells,
        "solver": summary["frames"][frame_index]["diagnostics"],
    }
    shape_path.unlink()
    activation_path.unlink()
    del glyph, context, volume_copy, visibility, surface
    return frame, frame_record


def draw_transition_legend(frame: Image.Image, draw: ImageDraw.ImageDraw) -> None:
    draw.text(
        (35, 940),
        "Principal amplitude a = σmax(B) - 1",
        font=font(19, bold=True),
        fill="white",
    )
    left, top, width, height = 35, 975, 620, 18
    ramp = FIG.contraction_colormap()(np.linspace(0, 1, width), bytes=True)[:, :3]
    ramp_image = Image.fromarray(np.tile(ramp[None], (height, 1, 1)))
    frame.paste(ramp_image, (left, top))
    ramp_image.close()
    draw.text((left, 998), "0", font=font(15), fill="#dfdfdf")
    draw.text((left + width, 998), "60", font=font(15), fill="#dfdfdf", anchor="ra")
    draw.text(
        (690, 947),
        "Color: log(1+a)/log(61) · length: 4.5 mm × log(1+a)/log(61)",
        font=font(18),
        fill="#dfdfdf",
    )
    draw.text(
        (690, 979),
        "Per-frame equilibrium; contact off. Inversions/intersections are diagnostics, not validation.",
        font=font(17),
        fill="#bdbdbd",
    )
    draw.text(
        (690, 1010),
        "Activation tensors and prescribed jaw pose transition together.",
        font=font(17),
        fill="#bdbdbd",
    )


def ffprobe(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise FileNotFoundError("ffprobe is required to verify the encoded animation")
    result = subprocess.run(
        [
            executable,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def main(cfg: Config) -> None:
    if cfg.fps != FPS or cfg.hold_frames_each_end != HOLD_FRAMES_EACH_END:
        raise ValueError("timing is fixed at 30 fps with 30 endpoint frames")
    source = cherries.input(cfg.source)
    fixture = cherries.input(cfg.fixture)
    prepared_path = cherries.input(cfg.prepared)
    mouthopen_path = cherries.input(cfg.mouthopen)
    mesh70_path = cherries.input(GROUP / "data/70-mouthopen-four-stage/mesh.npz")
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    summary, mesh, endpoints, frame_entries = verify_transition(
        source, fixture, prepared_path, mouthopen_path, mesh70_path
    )
    frame_paths = [Path(entry["checkpoint"]["path"]) for entry in frame_entries]
    volume = pv.read(fixture / "volume.vtu")
    points = mesh["rest_points"]
    tets = mesh["tets"]
    active_ids = mesh["active_ids"]
    if len(points) != 227900 or len(tets) != 1144268 or len(active_ids) != 288172:
        raise ValueError(
            "transition mesh counts differ from the certified MouthOpen fixture"
        )
    boundary = volume.extract_surface(algorithm=None)
    boundary_ids = np.asarray(
        boundary.point_data["vtkOriginalPointIds"], dtype=np.int64
    ).copy()
    boundary_cell_ids = np.asarray(
        boundary.cell_data["vtkOriginalCellIds"], dtype=np.int64
    ).copy()
    boundary_faces = np.asarray(boundary.faces).reshape(-1, 4).copy()
    if len(boundary_ids) != 63282 or len(boundary_faces) != 126648:
        raise ValueError("complete tetrahedral boundary counts changed")
    if not np.all(boundary_faces[:, 0] == 3):
        raise ValueError("boundary contains non-triangle cells")
    np.testing.assert_array_equal(boundary.points, points[boundary_ids])
    face_point_ids = boundary_ids[boundary_faces[:, 1:]]
    parent_vertices = tets[boundary_cell_ids]
    membership = np.any(
        face_point_ids[:, :, None] == parent_vertices[:, None, :], axis=2
    )
    if not membership.all() or not np.all(membership.sum(axis=1) == 3):
        raise ValueError(
            "extracted boundary face does not map to exactly one parent tetrahedron face"
        )
    boundary.clear_data()
    topology_path = output / "full-boundary-topology.npz"
    np.savez_compressed(
        topology_path,
        point_ids=boundary_ids,
        triangles=boundary_faces[:, 1:],
        parent_tet_ids=boundary_cell_ids,
    )
    with np.load(prepared_path, allow_pickle=False) as z:
        pivot = z["pivot"].copy()
    control_ids = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        active_ids
    ]
    inverse_dm = np.linalg.inv(
        (points[tets[active_ids]][:, 1:] - points[tets[active_ids]][:, :1]).transpose(
            0, 2, 1
        )
    )
    cranium = pv.read(CRANIUM_PATH)
    eyes = pv.read(EYES_PATH)
    mandible = pv.read(MANDIBLE_PATH)
    camera = make_camera(
        points,
        frame_paths,
        np.asarray(mandible.points),
        pivot,
        cranium,
        eyes,
    )
    scene = SimpleNamespace(bones={"cranium": cranium, "mandible": mandible}, eyes=eyes)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to encode the animation")
    video = output / "smile-to-mouthopen-transition.mp4"
    ffmpeg_command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        f"{VIDEO_SIZE[0]}x{VIDEO_SIZE[1]}",
        "-r",
        str(cfg.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video),
    ]
    ffmpeg_log = output / "ffmpeg-stderr.log"
    contact_images: dict[int, Image.Image] = {}
    frame_receipts: list[dict[str, Any]] = []
    last_frame_image: Image.Image | None = None
    with ffmpeg_log.open("wb") as err:
        process = subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE, stderr=err)
        assert process.stdin is not None

        def write_frame(image: Image.Image) -> None:
            process.stdin.write(
                np.asarray(image.convert("RGB"), dtype=np.uint8).tobytes()
            )

        try:
            for index, (entry, frame_path) in enumerate(
                zip(frame_entries, frame_paths, strict=True)
            ):
                alpha = float(entry["alpha"])
                frame_image, receipt = render_panel_images(
                    frame_index=index,
                    frame_path=frame_path,
                    alpha=alpha,
                    summary=summary,
                    mesh=mesh,
                    endpoints=endpoints,
                    volume=volume,
                    boundary=boundary,
                    boundary_ids=boundary_ids,
                    active_ids=active_ids,
                    control_ids=control_ids,
                    inverse_dm=inverse_dm,
                    camera=camera,
                    scene=scene,
                    mandible_rest=mandible,
                    output=output,
                )
                if index == 0:
                    for _ in range(cfg.hold_frames_each_end - 1):
                        write_frame(frame_image)
                write_frame(frame_image)
                if index in KEYFRAME_INDICES:
                    contact_images[index] = frame_image.copy()
                if index == len(frame_entries) - 1:
                    last_frame_image = frame_image.copy()
                frame_receipts.append(receipt)
                LOG.info(
                    "Rendered transition frame %d/%d at alpha %.5f",
                    index + 1,
                    len(frame_entries),
                    alpha,
                )
                frame_image.close()
            if last_frame_image is None:
                raise RuntimeError("no terminal transition frame was rendered")
            for _ in range(cfg.hold_frames_each_end - 1):
                write_frame(last_frame_image)
        finally:
            process.stdin.close()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"ffmpeg exited with {returncode}; see {ffmpeg_log}")
    if last_frame_image is not None:
        last_frame_image.close()

    contact_sheet = output / "keyframe-contact-sheet.png"
    sheet = Image.new("RGB", VIDEO_SIZE, PAGE_BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (960, 18),
        "Smile → MouthOpen: re-equilibrated keyframes",
        font=font(30, bold=True),
        fill="white",
        anchor="mt",
    )
    tile_w, tile_h = 640, 360
    for n, index in enumerate(KEYFRAME_INDICES):
        col, row = n % 3, n // 3
        image = contact_images[index].resize((tile_w, tile_h), Image.Resampling.LANCZOS)
        x, y = col * tile_w, 70 + row * tile_h
        sheet.paste(image, (x, y))
        alpha = frame_receipts[index]["alpha"]
        ImageDraw.Draw(sheet).text(
            (x + 12, y + 10),
            f"Frame {index:03d} · α={alpha:.2f}",
            font=font(21, bold=True),
            fill="white",
            stroke_width=2,
            stroke_fill="#101315",
        )
        image.close()
    sheet.save(contact_sheet)
    sheet.close()
    for image in contact_images.values():
        image.close()

    probe = ffprobe(video)
    streams = [
        stream for stream in probe["streams"] if stream.get("codec_type") == "video"
    ]
    if len(streams) != 1:
        raise ValueError("encoded MP4 does not contain exactly one video stream")
    stream = streams[0]
    if (int(stream["width"]), int(stream["height"])) != VIDEO_SIZE:
        raise ValueError("encoded video dimensions differ from the requested 1920x1080")
    total_video_frames = TRANSITION_STATES + 2 * (cfg.hold_frames_each_end - 1)
    if int(stream["nb_frames"]) != total_video_frames:
        raise ValueError("encoded video frame count differs from the declared timing")
    if stream["r_frame_rate"] != f"{cfg.fps}/1":
        raise ValueError("encoded video frame rate differs from the requested rate")
    probed_duration = float(probe["format"]["duration"])
    expected_duration = total_video_frames / cfg.fps
    if abs(probed_duration - expected_duration) > 1e-3:
        raise ValueError("encoded video duration differs from its frame count and rate")
    ffprobe_path = output / "ffprobe.json"
    ffprobe_path.write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")

    inputs = {
        str(path.resolve()): record(path)
        for path in (
            source / "summary.json",
            source / "source-manifest.json",
            source / "mesh.npz",
            source / "endpoints.npz",
            fixture / "summary.json",
            fixture / "volume.vtu",
            fixture / "skin.vtp",
            prepared_path,
            mouthopen_path,
            mesh70_path,
        )
    }
    inputs.update({str(path.resolve()): record(path) for path in frame_paths})
    source_paths = [
        Path(__file__),
        GROUP / "src/90-render-full-tetmesh.py",
        STRESS_SRC / "shape_scene.py",
        STRESS_SRC / "activation_scene.py",
        STRESS_SRC / "52-render-four-stage-figures.py",
        CONTEXT_SRC / "muscle_glyph_context.py",
        CRANIUM_PATH,
        MANDIBLE_PATH,
        EYES_PATH,
    ]
    source_receipts = [record(path) for path in source_paths]
    manifest = {
        "schema": "smile-mouthopen-transition-render-v1",
        "status": "complete",
        "source_summary": record(source / "summary.json"),
        "source_manifest": record(source / "source-manifest.json"),
        "frame_count": len(frame_paths),
        "video": record(video),
        "video_contact_sheet": record(contact_sheet),
        "ffmpeg_command": ffmpeg_command,
        "ffprobe": record(ffprobe_path),
        "video_metadata": probe,
        "timing": {
            "fps": cfg.fps,
            "equilibrium_states": TRANSITION_STATES,
            "transition_span_seconds": (TRANSITION_STATES - 1) / cfg.fps,
            "total_endpoint_frames_each_including_transition_endpoint": cfg.hold_frames_each_end,
            "extra_endpoint_frames_each": cfg.hold_frames_each_end - 1,
            "total_video_frames": total_video_frames,
            "encoded_duration_seconds": probe["format"].get("duration"),
            "interpolation": summary["interpolation"],
        },
        "tetmesh_boundary": {
            "volume_points": len(points),
            "tetrahedra": len(tets),
            "active_tetrahedra": len(active_ids),
            "boundary_points": len(boundary_ids),
            "boundary_triangles": len(boundary_faces),
            "fit_face_filter_applied": False,
            "parent_tet_face_mapping_verified": True,
            "topology": record(topology_path),
        },
        "activation_display": {
            "field": "S(alpha)=(1-alpha)S_smile+alpha*S_mouthopen; B=I+S; recomputed from endpoints for each frame",
            "mode": "largest unique positive principal eigenmode of B B.T - I; may be rank-2 during the full-tensor blend",
            "shared_amplitude_cap": AMOUNT_CAP,
            "line_length_max_m": 0.0045,
            "frames": frame_receipts,
        },
        "inputs": inputs,
        "sources": source_receipts,
        "camera": camera,
        "panels_share_projection": True,
        "contact_off_exploratory_render": True,
        "mechanical_validity_claim": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    (output / "source.py").write_text(Path(__file__).read_text())
    cherries.log_metrics(
        {
            "transition_states": len(frame_paths),
            "movie_frames": total_video_frames,
            "equilibrium_states": TRANSITION_STATES,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
