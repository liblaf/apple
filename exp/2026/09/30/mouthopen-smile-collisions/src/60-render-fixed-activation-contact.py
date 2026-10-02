# ruff: noqa: C901, E402, EM101, EM102, PLR0912, PLR0915, RUF001, TRY003
"""Render the saved-activation transition with fixed-reference rigid contact."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import shutil
import subprocess
import sys
from datetime import datetime
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
PARENT = ROOT / "exp/2026/09/29/mouthopen-activation"
STRESS_SRC = ROOT / "exp/2026/09/21/stress-activation-loss/src"
CONTEXT_SRC = ROOT / "exp/2026/09/14/dominant-activation-ablation/src"
sys.path.extend((str(STRESS_SRC), str(CONTEXT_SRC)))

from experiment import Profile
from shape_scene import CRANIUM_PATH, EYES_PATH, MANDIBLE_PATH

spec = importlib.util.spec_from_file_location(
    "fixed_activation_contact_render_helpers",
    STRESS_SRC / "52-render-four-stage-figures.py",
)
assert spec is not None
assert spec.loader is not None
FIG = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FIG
spec.loader.exec_module(FIG)

LOG = logging.getLogger(__name__)
FPS = 30
STATE_COUNT = 121
HOLD_FRAMES_EACH_END = 30
PAGE_SIZE = (1920, 1080)
PANEL_SIZE = (960, 780)
PANEL_ORIGIN = ((0, 145), (960, 145))
BACKGROUND = "#101315"
AMOUNT_CAP = 60.0
FONT_DIR = Path("/usr/share/fonts/TTF")
KEYFRAMES = (0, 30, 60, 90, 120)


class Config(cherries.BaseConfig):
    source: Path = Path("52-fixed-activation-contact")
    fixture: Path = Path("50-fixed-reference/fixture")
    canonical: Path = PARENT / "data/91-smile-mouthopen-transition-003"
    output: Path = Path("60-fixed-activation-contact-render")
    diagnostic_only: bool = False
    diagnostic_frame_index: int | None = None
    fps: int = FPS


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


def beta_at(index: int, count: int) -> float:
    return float((1 - np.cos(np.pi * index / (count - 1))) / 2)


def verify_manifest(source: Path) -> None:
    manifest_path = source / "source-manifest.json"
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for name, item in manifest.items():
        verify_record(item, f"frozen solver source {name}")
        if item.get("source"):
            verify_record(
                {"path": item["source"], "sha256": item["sha256"]},
                f"live source {name}",
            )


def choose_states(
    summary: dict[str, Any],
    *,
    diagnostic_only: bool,
    diagnostic_frame_index: int | None = None,
) -> list[dict[str, Any]]:
    if summary.get("schema") != "fixed-reference-activation-transition-v1":
        raise ValueError("unexpected fixed-reference transition schema")
    if diagnostic_frame_index is not None and not diagnostic_only:
        raise ValueError("--diagnostic-frame-index requires --diagnostic-only")
    if not diagnostic_only:
        if summary.get("status") != "completed":
            raise RuntimeError(
                f"run is not complete: {summary.get('status')}; pass --diagnostic-only for accepted snapshots"
            )
        entries = summary.get("frames", [])
        if len(entries) != STATE_COUNT:
            raise ValueError(
                f"expected {STATE_COUNT} equilibrated frames; found {len(entries)}"
            )
        selected = entries
        for index, entry in enumerate(selected):
            if int(entry["index"]) != index:
                raise ValueError("frame indexes are not contiguous")
            expected = beta_at(index, STATE_COUNT)
            np.testing.assert_allclose(
                float(entry["beta"]), expected, rtol=0, atol=2e-15
            )
    elif diagnostic_frame_index is not None:
        if not diagnostic_only:
            raise ValueError("--diagnostic-frame-index requires --diagnostic-only")
        frames = summary.get("frames", [])
        if diagnostic_frame_index < 0 or diagnostic_frame_index >= len(frames):
            raise IndexError(
                f"diagnostic frame index {diagnostic_frame_index} is outside {len(frames)} saved frames"
            )
        entry = frames[diagnostic_frame_index]
        if int(entry["index"]) != diagnostic_frame_index:
            raise ValueError("requested diagnostic frame index does not match its row")
        selected = [entry]
    else:
        candidates = summary.get("accepted", [])
        if not candidates:
            raise ValueError("no accepted solver states are available for diagnostics")
        selected = (
            candidates if len(candidates) == 1 else [candidates[0], candidates[-1]]
        )

    states: list[dict[str, Any]] = []
    for ordinal, entry in enumerate(selected):
        item = entry["checkpoint"]
        path = verify_record(item, f"accepted state {ordinal}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"u", "u_full", "alpha", "pose"}
            if not required.issubset(archive.files):
                raise ValueError(
                    f"checkpoint missing {sorted(required - set(archive.files))}: {path}"
                )
            u, u_full = archive["u"].copy(), archive["u_full"].copy()
            alpha = float(archive["alpha"])
            pose = archive["pose"].copy()
            beta = float(archive["beta"]) if "beta" in archive.files else 1.0 - alpha
        if not np.isfinite(u).all() or not np.isfinite(u_full).all():
            raise ValueError(f"checkpoint displacement is nonfinite: {path}")
        if not diagnostic_only:
            np.testing.assert_allclose(
                alpha, 1.0 - float(entry["beta"]), rtol=0, atol=2e-15
            )
        states.append(
            {
                "entry": entry,
                "path": path,
                "u": u,
                "u_full": u_full,
                "alpha": alpha,
                "beta": beta,
                "pose": pose,
                "phase": str(entry.get("phase", "transition")),
                "initialization_fraction": float(entry.get("fraction", 0.0)),
            }
        )
    return states


def contact_gate(entry: dict[str, Any]) -> dict[str, Any]:
    diagnostics = entry.get("diagnostics", {})
    if not diagnostics.get("solver_valid", False):
        raise ValueError("selected state lacks a passed force gate")
    contact = diagnostics.get("contact", {})
    if not contact.get("contact_valid", False):
        raise ValueError("selected state lacks a passed declared contact gate")
    scope_ok = contact.get("scoped_boundary_no_intersections")
    if scope_ok is None:
        scope_ok = contact.get("scoped_no_intersections")
    if scope_ok is not True:
        raise ValueError("selected state has no passing scoped contact receipt")
    return contact


def camera_for(
    scene: Any,
    points: np.ndarray,
    states: list[dict[str, Any]],
    mandible: pv.PolyData,
    pivot: np.ndarray,
) -> dict[str, Any]:
    toward = np.array([0.65, 0.03, 1.0], dtype=np.float64)
    toward /= np.linalg.norm(toward)
    right = np.cross([0.0, 1.0, 0.0], toward)
    right /= np.linalg.norm(right)
    up = np.cross(toward, right)
    basis = np.column_stack((right, up, toward))
    point_sets = [np.asarray(x.points) for x in scene.bones.values()] + [
        np.asarray(scene.eyes.points)
    ]
    point_sets.extend(points + item["u"] for item in states)
    point_sets.extend(
        rigid(np.asarray(mandible.points), pivot, item["pose"]) for item in states
    )
    projected = np.vstack(point_sets) @ basis
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = basis @ ((low + high) / 2)
    half = (high - low) / 2
    return {
        "position": (center + 0.6 * toward).tolist(),
        "focal_point": center.tolist(),
        "view_up": up.tolist(),
        "parallel_scale": float(
            1.055 * max(half[1], half[0] / (FIG.WINDOW[0] / FIG.WINDOW[1]))
        ),
        "projected_bounds_m": [low.tolist(), high.tolist()],
        "policy": "shared orthographic camera fitted to every selected deformed boundary and posed source anatomy",
    }


def render_state(
    *,
    state: dict[str, Any],
    state_number: int,
    state_count: int,
    complete: bool,
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
    scene: Any,
    mandible: pv.PolyData,
    output: Path,
    captured_at: str,
) -> tuple[Image.Image, dict[str, Any]]:
    points, tets = mesh["rest_points"], mesh["tets"]
    u, pose, beta = state["u"], state["pose"], state["beta"]
    if u.shape != points.shape:
        raise ValueError(
            "physical displacement shape differs from repaired reference mesh"
        )
    if state["u_full"].shape[0] < len(points) or not np.array_equal(
        state["u_full"][: len(points)], u
    ):
        raise ValueError(
            "extended checkpoint does not preserve the physical displacement"
        )
    deformed = points + u
    surface = boundary.copy(deep=True)
    surface.points = deformed[boundary_ids]
    rendered_surface_points, rendered_surface_triangles = (
        surface.n_points,
        surface.n_cells,
    )
    shape_path, activation_path = (
        output / "shape-panel.png",
        output / "activation-panel.png",
    )
    jaw = mandible.copy(deep=True)
    jaw.points = rigid(np.asarray(mandible.points), endpoints["pivot"], pose)
    shape_scene = SimpleNamespace(
        bones={"cranium": scene.bones["cranium"], "mandible": jaw}, eyes=scene.eyes
    )
    projection = FIG.render_shape(shape_scene, surface, camera, shape_path)

    active_tets = deformed[tets[active_ids]]
    deformation_gradient = (active_tets[:, 1:] - active_tets[:, :1]).transpose(
        0, 2, 1
    ) @ inverse_dm
    centers = active_tets.mean(axis=1)
    if state["phase"] == "neutral":
        strain = np.zeros_like(endpoints["S_mouthopen"])
    elif state["phase"] == "initialization":
        strain = state["initialization_fraction"] * endpoints["S_mouthopen"]
    else:
        strain = (1.0 - beta) * endpoints["S_mouthopen"] + beta * endpoints["S_smile"]
    glyph = FIG.principal_glyphs(strain + np.eye(3), deformation_gradient, centers)
    positive = glyph.eligible & (glyph.signed_display_percent > 0)
    amplitudes = np.sqrt(1 + glyph.eigenvalues_z[positive]) - 1
    maximum = float(amplitudes.max()) if len(amplitudes) else 0.0
    if maximum > AMOUNT_CAP + 1e-10:
        raise ValueError(
            f"state principal amplitude exceeds shared display cap: {maximum}"
        )
    deformed_volume = volume.copy(deep=True)
    deformed_volume.points = deformed
    context = FIG.build_muscle_region_context(deformed_volume)
    visibility = FIG.visible_region_mask(
        context,
        glyph.centers,
        active_ids,
        control_ids,
        camera,
        window_size=FIG.ACTIVATION_WINDOW,
    )
    mask = visibility.mask & positive
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
    left.thumbnail(PANEL_SIZE, Image.Resampling.LANCZOS)
    right.thumbnail(PANEL_SIZE, Image.Resampling.LANCZOS)
    page = Image.new("RGB", PAGE_SIZE, BACKGROUND)
    draw = ImageDraw.Draw(page)
    if complete:
        title = "MouthOpen → Smile · saved-activation contact run"
    elif state["phase"] == "transition" and abs(beta) <= 1e-15:
        title = "MouthOpen endpoint · saved-activation contact state"
    elif state["phase"] == "neutral":
        title = "Repaired reference · neutral equilibrium"
    elif state["phase"] == "initialization":
        title = "Repaired reference · MouthOpen initialization"
    else:
        title = "Repaired reference · partial activation transition"
    draw.text((960, 25), title, font=font(33, bold=True), fill="white", anchor="mt")
    progress = f"Frame {state_number + 1}/{state_count} · Smile blend β {beta:.3f}"
    if not complete:
        if state["phase"] == "neutral":
            progress = (
                f"PARTIAL · {summary.get('status')} · neutral state · {captured_at}"
            )
        elif state["phase"] == "initialization":
            progress = (
                "PARTIAL · "
                f"{summary.get('status')} · MouthOpen {100 * state['initialization_fraction']:.1f}% · {captured_at}"
            )
        else:
            progress = f"PARTIAL · {summary.get('status')} · Smile β {beta:.3f} · {captured_at}"
    draw.text((1885, 73), progress, font=font(19), fill="#d9d9d9", anchor="rt")
    draw.text(
        (480, 105),
        "Re-equilibrated complete tetmesh boundary",
        font=font(22, bold=True),
        fill="#e6e6e6",
        anchor="mt",
    )
    draw.text(
        (1440, 105),
        (
            "Principal activation mode · full S blend"
            if complete or state["phase"] == "transition"
            else (
                "Neutral activation · S = 0"
                if state["phase"] == "neutral"
                else "Prescribed activation · scaled MouthOpen S"
            )
        ),
        font=font(22, bold=True),
        fill="#e6e6e6",
        anchor="mt",
    )
    page.paste(left, PANEL_ORIGIN[0])
    page.paste(right, PANEL_ORIGIN[1])
    draw.text(
        (35, 940),
        "Principal amplitude a = σmax(B) − 1",
        font=font(18, bold=True),
        fill="white",
    )
    ramp_width, ramp_height = 620, 18
    ramp = FIG.contraction_colormap()(np.linspace(0, 1, ramp_width), bytes=True)[:, :3]
    page.paste(Image.fromarray(np.tile(ramp[None], (ramp_height, 1, 1))), (35, 974))
    draw.text((35, 996), "0", font=font(14), fill="#dfdfdf")
    draw.text((655, 996), "60", font=font(14), fill="#dfdfdf", anchor="ra")
    draw.text(
        (690, 947),
        "Color: log(1+a)/log(61) · length: 4.5 mm × log(1+a)/log(61)",
        font=font(17),
        fill="#dfdfdf",
    )
    diagnostics = state["entry"].get("diagnostics", {})
    contact = diagnostics.get("contact", {})
    force_n = float(diagnostics.get("accepted_force_norm", float("nan"))) * 1e6
    inverted = int(diagnostics.get("inverted_cells", -1))
    gap_m = contact.get("minimum_active_distance_m")
    gap_text = (
        "no active IPC pair"
        if gap_m is None
        else f"minimum active gap {1e6 * float(gap_m):.3g} μm"
    )
    draw.text(
        (690, 976),
        f"Contact: soft tissue against cranium, jaw, and eyes · {gap_text}",
        font=font(16),
        fill="#bdbdbd",
    )
    draw.text(
        (690, 1001),
        f"Force residual {force_n:.3g} N / 0.01 N gate · inverted tets {inverted:,}",
        font=font(16),
        fill="#d9d9d9",
    )
    draw.text(
        (690, 1026),
        "Prescribed saved activation; displacement re-equilibrates. Exploratory, not mechanically validated.",
        font=font(15),
        fill="#bdbdbd",
    )
    left.close()
    right.close()
    for native in (
        shape_path.with_name("shape-panel-native.png"),
        activation_path.with_name("activation-panel-native.png"),
    ):
        native.unlink(missing_ok=True)
    surface.clear_data()
    jaw.clear_data()
    shape_path.unlink()
    activation_path.unlink()
    receipt = {
        "state_number": state_number,
        "beta_smile": beta,
        "alpha_mouthopen": state["alpha"],
        "checkpoint": record(state["path"]),
        "force_norm": state["entry"].get("diagnostics", {}).get("accepted_force_norm"),
        "inverted_cells": state["entry"].get("diagnostics", {}).get("inverted_cells"),
        "scoped_contact_intersections": contact.get(
            "scoped_has_intersections", contact.get("scoped_boundary_has_intersections")
        ),
        "minimum_active_distance_m": gap_m,
        "active_contact_count": contact.get("active_contact_count"),
        "contact_valid": contact.get("contact_valid"),
        "visible_positive_modes": int(mask.sum()),
        "eligible_positive_modes": int(positive.sum()),
        "maximum_principal_amplitude": maximum,
        "surface_points": rendered_surface_points,
        "surface_triangles": rendered_surface_triangles,
    }
    del glyph, context, deformed_volume, visibility
    return page, receipt


def ffprobe(path: Path) -> dict[str, Any]:
    executable = shutil.which("ffprobe")
    if executable is None:
        raise FileNotFoundError("ffprobe is required to validate the MP4")
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
    if cfg.fps != FPS:
        raise ValueError("movie rate is fixed at 30 fps")
    source = cherries.input(cfg.source)
    fixture_dir = cherries.input(cfg.fixture)
    canonical = cherries.input(cfg.canonical)
    summary = json.loads((source / "summary.json").read_text())
    verify_manifest(source)
    if not np.isclose(float(summary["config"]["force_atol"]), 1e-8, rtol=0, atol=0):
        raise ValueError(
            "fixed-reference force gate differs from requested 1e-8 MPa m²"
        )
    states = choose_states(
        summary,
        diagnostic_only=cfg.diagnostic_only,
        diagnostic_frame_index=cfg.diagnostic_frame_index,
    )
    captured_at = datetime.now().astimezone().isoformat(timespec="seconds")
    for state in states:
        contact_gate(state["entry"])
    if (
        summary.get("collision_scope")
        != "pure-soft FEM boundary against complete source cranium, mandible and eyes; bonded mixed attachment faces excluded; tissue self-contact and rigid-rigid contact not enabled"
    ):
        # The exact sentence may be revised in the solver summary; require the stated scope semantically.
        scope = str(summary.get("collision_scope", "")).lower()
        if not all(token in scope for token in ("soft", "cranium", "mandible", "eyes")):
            raise ValueError(
                f"unexpected contact scope: {summary.get('collision_scope')}"
            )
    if (
        "no activation optimization"
        not in str(summary.get("activation_policy", "")).lower()
    ):
        raise ValueError("run does not certify prescribed saved activation")

    with np.load(source / "mesh.npz", allow_pickle=False) as z:
        mesh = {key: z[key].copy() for key in z.files}
    with np.load(source / "endpoints.npz", allow_pickle=False) as z:
        endpoints = {key: z[key].copy() for key in z.files}
    with np.load(source / "rigid-geometry.npz", allow_pickle=False) as z:
        geometry = {key: z[key].copy() for key in z.files}
    volume = pv.read(fixture_dir / "volume.vtu")
    np.testing.assert_array_equal(np.asarray(volume.points), mesh["rest_points"])
    np.testing.assert_array_equal(
        np.asarray(volume.cells).reshape(-1, 5)[:, 1:], mesh["tets"]
    )
    np.testing.assert_array_equal(
        volume.cell_data["ActivationMask"].astype(bool).nonzero()[0], mesh["active_ids"]
    )
    np.testing.assert_array_equal(
        geometry["points"][: len(mesh["rest_points"])], mesh["rest_points"]
    )
    required_geometry = {
        "source_points",
        "rigid_triangles",
        "rigid_mandible_mask",
        "cranium_ids",
        "mandible_ids",
        "eye_ids",
        "contact_indices",
        "contact_faces",
    }
    if not required_geometry.issubset(geometry):
        raise ValueError(
            f"rigid geometry lacks fields: {sorted(required_geometry - set(geometry))}"
        )
    np.testing.assert_array_equal(
        geometry["source_points"], geometry["points"][len(mesh["rest_points"]) :]
    )
    np.testing.assert_array_equal(
        geometry["rigid_mandible_mask"].shape,
        (len(geometry["points"]) - len(mesh["rest_points"]),),
    )
    np.testing.assert_array_equal(
        np.sort(
            np.concatenate(
                (geometry["cranium_ids"], geometry["mandible_ids"], geometry["eye_ids"])
            )
        ),
        np.arange(len(geometry["source_points"])),
    )
    if geometry["rigid_triangles"].min() < 0 or geometry[
        "rigid_triangles"
    ].max() >= len(geometry["source_points"]):
        raise ValueError("rigid triangle index is outside appended source points")
    if endpoints["S_smile"].shape != endpoints["S_mouthopen"].shape or endpoints[
        "S_smile"
    ].shape != (len(mesh["active_ids"]), 3, 3):
        raise ValueError(
            "saved endpoint tensor shapes differ from repaired active-cell map"
        )
    with np.load(
        canonical / "endpoints.npz", allow_pickle=False
    ) as canonical_endpoints:
        np.testing.assert_array_equal(
            endpoints["S_smile"], canonical_endpoints["S_smile"]
        )
        np.testing.assert_array_equal(
            endpoints["S_mouthopen"], canonical_endpoints["S_mouthopen"]
        )
    np.testing.assert_allclose(
        endpoints["S_smile"], endpoints["S_smile"].swapaxes(1, 2), rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        endpoints["S_mouthopen"],
        endpoints["S_mouthopen"].swapaxes(1, 2),
        rtol=0,
        atol=1e-12,
    )
    with np.load(
        canonical / "endpoints.npz", allow_pickle=False
    ) as canonical_endpoints:
        for key in ("pose_mouthopen", "pivot"):
            np.testing.assert_array_equal(endpoints[key], canonical_endpoints[key])
    if len(mesh["rest_points"]) != 227900 or len(mesh["tets"]) != 1144268:
        raise ValueError("repaired-reference tetmesh counts changed")

    manifest_output = cherries.output(cfg.output / "manifest.json", mkdir=True)
    output = manifest_output.parent
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"renderer output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    boundary = volume.extract_surface(algorithm=None)
    boundary_ids = np.asarray(
        boundary.point_data["vtkOriginalPointIds"], dtype=np.int64
    ).copy()
    boundary_cell_ids = np.asarray(
        boundary.cell_data["vtkOriginalCellIds"], dtype=np.int64
    ).copy()
    faces = np.asarray(boundary.faces).reshape(-1, 4)
    if not np.all(faces[:, 0] == 3):
        raise ValueError("extracted full volume boundary contains non-triangle faces")
    face_points = boundary_ids[faces[:, 1:]]
    parents = mesh["tets"][boundary_cell_ids]
    membership = np.any(face_points[:, :, None] == parents[:, None, :], axis=2)
    if not membership.all() or not np.all(membership.sum(axis=1) == 3):
        raise ValueError("boundary triangles do not map to their parent tetrahedra")
    topology_path = output / "full-boundary-topology.npz"
    np.savez_compressed(
        topology_path,
        point_ids=boundary_ids,
        triangles=faces[:, 1:],
        parent_tet_ids=boundary_cell_ids,
    )
    control_ids = np.asarray(volume.cell_data["ActivationControlId"], dtype=np.int64)[
        mesh["active_ids"]
    ]
    points, tets, active_ids = mesh["rest_points"], mesh["tets"], mesh["active_ids"]
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    group_names = [str(x) for x in np.asarray(volume.field_data["GroupName"]).ravel()]
    group_id = np.asarray(volume.point_data["GroupId"], dtype=np.int64)
    jaw_fixed = fixed & (group_id == group_names.index("Mandible"))
    for state in states:
        if state["u_full"].shape != geometry["points"].shape:
            raise ValueError(
                "extended checkpoint shape differs from rigid contact geometry"
            )
        expected_jaw_u = (
            rigid(points[jaw_fixed], endpoints["pivot"], state["pose"])
            - points[jaw_fixed]
        )
        np.testing.assert_allclose(
            state["u"][jaw_fixed], expected_jaw_u, rtol=0, atol=1e-9
        )
        np.testing.assert_allclose(state["u"][fixed & ~jaw_fixed], 0, rtol=0, atol=1e-9)
        appended_expected = np.zeros_like(geometry["source_points"])
        mask = geometry["rigid_mandible_mask"]
        appended_expected[mask] = (
            rigid(geometry["source_points"][mask], endpoints["pivot"], state["pose"])
            - geometry["source_points"][mask]
        )
        np.testing.assert_allclose(
            state["u_full"][len(points) :], appended_expected, rtol=0, atol=1e-9
        )
    inverse_dm = np.linalg.inv(
        (points[tets[active_ids]][:, 1:] - points[tets[active_ids]][:, :1]).transpose(
            0, 2, 1
        )
    )
    cranium, eyes, mandible = (
        pv.read(CRANIUM_PATH),
        pv.read(EYES_PATH),
        pv.read(MANDIBLE_PATH),
    )
    np.testing.assert_array_equal(
        geometry["source_points"][geometry["cranium_ids"]], cranium.points
    )
    np.testing.assert_array_equal(
        geometry["source_points"][geometry["mandible_ids"]], mandible.points
    )
    np.testing.assert_array_equal(
        geometry["source_points"][geometry["eye_ids"]], eyes.points
    )
    np.testing.assert_array_equal(
        geometry["rigid_mandible_mask"],
        np.isin(np.arange(len(geometry["source_points"])), geometry["mandible_ids"]),
    )
    scene = SimpleNamespace(bones={"cranium": cranium, "mandible": mandible}, eyes=eyes)
    camera = camera_for(scene, points, states, mandible, endpoints["pivot"])

    frame_receipts: list[dict[str, Any]] = []
    keyframes: dict[int, Image.Image] = {}
    if cfg.diagnostic_only:
        for i, state in enumerate(states):
            page, receipt = render_state(
                state=state,
                state_number=i,
                state_count=len(states),
                complete=False,
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
                mandible=mandible,
                output=output,
                captured_at=captured_at,
            )
            image_path = output / f"accepted-state-{i:02d}.png"
            page.save(image_path)
            receipt["image"] = record(image_path)
            frame_receipts.append(receipt)
            page.close()
        total_video_frames = 0
    else:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise FileNotFoundError("ffmpeg is required to encode the animation")
        video = output / "mouthopen-to-smile-fixed-activation-contact.mp4"
        command = [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s:v",
            f"{PAGE_SIZE[0]}x{PAGE_SIZE[1]}",
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
        total_video_frames = STATE_COUNT + 2 * (HOLD_FRAMES_EACH_END - 1)
        with (output / "ffmpeg-stderr.log").open("wb") as error_log:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=error_log)
            assert process.stdin is not None
            try:
                last: Image.Image | None = None
                for i, state in enumerate(states):
                    page, receipt = render_state(
                        state=state,
                        state_number=i,
                        state_count=len(states),
                        complete=True,
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
                        mandible=mandible,
                        output=output,
                        captured_at=captured_at,
                    )
                    if i == 0:
                        for _ in range(HOLD_FRAMES_EACH_END - 1):
                            process.stdin.write(
                                np.asarray(page, dtype=np.uint8).tobytes()
                            )
                    process.stdin.write(np.asarray(page, dtype=np.uint8).tobytes())
                    if i in KEYFRAMES:
                        keyframes[i] = page.copy()
                    if last is not None:
                        last.close()
                    last = page.copy() if i == len(states) - 1 else None
                    frame_receipts.append(receipt)
                    page.close()
                    LOG.info(
                        "Rendered fixed-activation state %d/%d (beta %.4f)",
                        i + 1,
                        len(states),
                        state["beta"],
                    )
                assert last is not None
                for _ in range(HOLD_FRAMES_EACH_END - 1):
                    process.stdin.write(np.asarray(last, dtype=np.uint8).tobytes())
                last.close()
            finally:
                process.stdin.close()
            if process.wait() != 0:
                raise RuntimeError("ffmpeg failed; inspect ffmpeg-stderr.log")
        probe = ffprobe(video)
        stream = next(x for x in probe["streams"] if x.get("codec_type") == "video")
        if (
            (int(stream["width"]), int(stream["height"])) != PAGE_SIZE
            or int(stream["nb_frames"]) != total_video_frames
            or stream["r_frame_rate"] != f"{cfg.fps}/1"
        ):
            raise ValueError("encoded video dimensions, frame count, or rate mismatch")
        (output / "ffprobe.json").write_text(
            json.dumps(probe, indent=2, sort_keys=True) + "\n"
        )
        sheet = Image.new("RGB", PAGE_SIZE, BACKGROUND)
        draw = ImageDraw.Draw(sheet)
        draw.text(
            (960, 20),
            "MouthOpen → Smile · fixed-activation contact keyframes",
            font=font(30, bold=True),
            fill="white",
            anchor="mt",
        )
        tile_size = (640, 360)
        for n, idx in enumerate(KEYFRAMES):
            tile = keyframes[idx].resize(tile_size, Image.Resampling.LANCZOS)
            x, y = (n % 3) * 640, 70 + (n // 3) * 360
            sheet.paste(tile, (x, y))
            ImageDraw.Draw(sheet).text(
                (x + 10, y + 8),
                f"Frame {idx:03d} · β={frame_receipts[idx]['beta_smile']:.2f}",
                font=font(20, bold=True),
                fill="white",
                stroke_width=2,
                stroke_fill=BACKGROUND,
            )
            tile.close()
        sheet.save(output / "keyframe-contact-sheet.png")
        sheet.close()
        for image in keyframes.values():
            image.close()

    manifest = {
        "schema": "fixed-reference-activation-contact-render-v1",
        "status": "diagnostic-snapshots" if cfg.diagnostic_only else "complete",
        "source_summary": record(source / "summary.json"),
        "source_mesh": record(source / "mesh.npz"),
        "source_endpoints": record(source / "endpoints.npz"),
        "rigid_geometry": record(source / "rigid-geometry.npz"),
        "fixture_volume": record(fixture_dir / "volume.vtu"),
        "frame_count": len(states),
        "video_frames": total_video_frames,
        "fps": cfg.fps,
        "rendered_at": captured_at,
        "contact_scope": summary["collision_scope"],
        "force_tolerance_mpa_m2": summary["config"]["force_atol"],
        "activation": "S(beta)=(1-beta)S_mouthopen+beta*S_smile; no activation optimization",
        "shape_boundary": {
            "point_count": len(boundary_ids),
            "triangle_count": len(faces),
            "parent_tet_face_mapping_verified": True,
            "topology": record(topology_path),
        },
        "camera": camera,
        "state_receipts": frame_receipts,
        "inputs": {
            str(path.resolve()): record(path)
            for path in (
                source / "summary.json",
                source / "source-manifest.json",
                source / "mesh.npz",
                source / "endpoints.npz",
                source / "rigid-geometry.npz",
                fixture_dir / "volume.vtu",
                fixture_dir / "skin.vtp",
                canonical / "endpoints.npz",
                CRANIUM_PATH,
                MANDIBLE_PATH,
                EYES_PATH,
            )
        },
        "mechanical_validity_claim": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report_name = (
        f"../docs/{cfg.output.name}.md"
        if cfg.diagnostic_only
        else "../docs/60-fixed-activation-contact.md"
    )
    report_path = cherries.output(report_name, mkdir=True)
    diagnostic_report = ""
    if cfg.diagnostic_only:
        row = frame_receipts[0]
        gap_um = row["minimum_active_distance_m"]
        diagnostic_report = (
            f"- Selected state: β={row['beta_smile']:.3f}, "
            f"force={row['force_norm'] * 1e6:.6g} N, "
            f"minimum active gap={1e6 * gap_um:.6g} μm, "
            f"active contacts={row['active_contact_count']}, "
            f"inverted tetrahedra={row['inverted_cells']}\n"
        )
    report_path.write_text(
        "# Fixed-activation MouthOpen-to-Smile contact render\n\n"
        f"The renderer produced {'accepted-state diagnostic snapshots' if cfg.diagnostic_only else 'a 121-state re-equilibrated animation'} from `{source}`. "
        "The endpoint activation tensors are prescribed from the saved MouthOpen and Smile states; the run does not optimize activation. "
        "Each displayed state must pass the saved force and declared soft-tissue-to-bone/eye contact gates. "
        "The contact scope excludes soft-soft and rigid-rigid pairs, and this render does not establish mechanical validity.\n\n"
        f"- Status: `{summary.get('status')}`\n- Force tolerance: `{summary['config']['force_atol']:.3g} MPa m²`\n"
        f"- Selected states: {len(states)}\n- Full tetmesh boundary: {len(boundary_ids):,} vertices, {len(faces):,} triangles\n"
        f"{diagnostic_report}"
        f"- Manifest: [`{output / 'manifest.json'}`]({output / 'manifest.json'})\n"
    )
    cherries.log_metrics(
        {
            "render/states": len(states),
            "render/diagnostic_only": float(cfg.diagnostic_only),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
