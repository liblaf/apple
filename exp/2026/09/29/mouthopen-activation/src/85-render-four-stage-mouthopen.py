# ruff: noqa: C901, E402, EM101, EM102, PLR0912, PLR0915, RUF001, TRY003
"""Render four MouthOpen activation parameterizations from saved checkpoints."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
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
    "mouthopen_four_stage_helpers", STRESS_SRC / "52-render-four-stage-figures.py"
)
assert spec is not None
assert spec.loader is not None
FIG = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = FIG
spec.loader.exec_module(FIG)

STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")
LOG = logging.getLogger(__name__)
HEADINGS = {
    "symmetric6": "Symmetric, unrestricted",
    "psd6": "PSD, contraction only",
    "rankone_fixed": "Rank one, fixed axis",
    "rankone_learned": "Rank one, learned axis",
}
DOFS = {"symmetric6": 6, "psd6": 6, "rankone_fixed": 1, "rankone_learned": 3}
PAGE_LAYOUT = (5120, 2880)
PAGE = (10240, 5760)
WINDOW = FIG.WINDOW
FONT_DIR = Path("/usr/share/fonts/TTF")
BACKGROUND = "#101315"
AMOUNT_CAP = 60.0


class Config(cherries.BaseConfig):
    source: Path = Path("70-mouthopen-four-stage")
    fixture: Path = Path("30-pruned-fixture")
    prepared: Path = Path("10-mandible/prepared.npz")
    output: Path = Path("85-mouthopen-four-stage")


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
    return ImageFont.truetype(str(FONT_DIR / f"DejaVuSans{suffix}.ttf"), size * 2)


def point(x: float, y: float) -> tuple[int, int]:
    return round(2 * x), round(2 * y)


def rigid(points: np.ndarray, pivot: np.ndarray, pose: np.ndarray) -> np.ndarray:
    return (
        (points - pivot) @ Rotation.from_rotvec(pose[:3]).as_matrix().T
        + pivot
        + pose[3:]
    )


def surface_metrics(
    mesh: dict[str, np.ndarray], displacement: np.ndarray
) -> dict[str, float]:
    ids = mesh["skin_ids"]
    triangles = mesh["triangles"]
    weights = mesh["skin_vertex_weights"]
    reference = mesh["rest_points"][ids]
    target = reference + mesh["target_displacement_skin"]
    actual = reference + displacement[ids]
    error = actual - target
    fit = 1000 * np.sqrt(np.sum(weights * np.sum(error * error, axis=1)))

    def normals(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        triangle_points = points[triangles]
        cross = np.cross(
            triangle_points[:, 1] - triangle_points[:, 0],
            triangle_points[:, 2] - triangle_points[:, 0],
        )
        area = np.linalg.norm(cross, axis=1) / 2
        return cross / (2 * area[:, None]), area

    _, area = normals(reference)
    actual_normals, _ = normals(actual)
    target_normals, _ = normals(target)
    cosine = np.clip(np.einsum("ij,ij->i", actual_normals, target_normals), -1, 1)
    normal = np.rad2deg(np.sqrt(np.dot(area, np.arccos(cosine) ** 2) / area.sum()))
    return {"fit_rms_mm": float(fit), "normal_angle_rms_deg": float(normal)}


def load_chain(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol_path = source / "protocol.json"
    status_path = source / "chain-status.json"
    protocol = json.loads(protocol_path.read_text())
    status = json.loads(status_path.read_text())
    if protocol.get("schema") != "mouthopen-four-stage-chain-v1":
        raise ValueError("unexpected four-stage protocol schema")
    if tuple(status.get("stage_sequence", ())) != STAGES:
        raise ValueError("chain stage sequence does not match the renderer")
    if status.get("status") == "running":
        raise RuntimeError("refusing to render while the chain is still running")
    for name, receipt in protocol["inputs"].items():
        verify_record(receipt, f"chain input {name}")
    verify_record(protocol["mesh"], "chain mesh")
    source_manifest_receipt = protocol["source_manifest"]
    verify_record(source_manifest_receipt, "chain source manifest")
    source_manifest = json.loads(Path(source_manifest_receipt["path"]).read_text())
    source_modules = []
    for module, item in source_manifest.items():
        path = verify_record(item, f"frozen source {module}")
        if "source" in item:
            verify_record(
                {"path": item["source"], "sha256": item["sha256"]},
                f"source origin {module}",
            )
        source_modules.append({"module": module, **record(path)})
    return protocol, {
        "protocol": record(protocol_path),
        "status": record(status_path),
        "source_manifest": source_modules,
    }


def load_states(source: Path, chain_status: dict[str, Any]) -> list[dict[str, Any]]:
    states = []
    for index, mode in enumerate(STAGES):
        folder = source / mode
        summary_path = folder / "summary.json"
        checkpoint_path = folder / "last.npz"
        initialization_path = folder / "initialization.npz"
        if not summary_path.is_file():
            states.append(
                {
                    "mode": mode,
                    "index": index,
                    "summary": {
                        "status": "not_run",
                        "attempted_updates": 0,
                        "optimizer_updates": 0,
                        "skipped_updates": 0,
                    },
                    "state": None,
                    "metrics": {},
                    "gradient": {"status": "unavailable"},
                    "terminal": False,
                    "status": "not_run",
                    "checkpoint": None,
                    "summary_receipt": None,
                    "gradient_receipt": None,
                    "chain_entry_status": chain_status.get("stages", {})
                    .get(mode, {})
                    .get("status", "not_run"),
                }
            )
            continue
        summary = json.loads(summary_path.read_text())
        if summary.get("mode") != mode or summary.get("activation_model") != "strain":
            raise ValueError(f"stage {mode} summary mode/model mismatch")
        if checkpoint_path.is_file():
            checkpoint_receipt = summary.get("final_checkpoint")
            if checkpoint_receipt is not None:
                verify_record(checkpoint_receipt, f"{mode} final checkpoint")
            with np.load(checkpoint_path, allow_pickle=False) as archive:
                required = {
                    "u",
                    "B",
                    "S",
                    "mode",
                    "step",
                    "activation_model",
                    "solver_valid",
                }
                if not required.issubset(archive.files):
                    raise ValueError(
                        f"{mode} checkpoint missing keys: {sorted(required - set(archive.files))}"
                    )
                state = {key: archive[key].copy() for key in archive.files}
            checkpoint_source = checkpoint_path
        elif initialization_path.is_file():
            with np.load(initialization_path, allow_pickle=False) as archive:
                state = {key: archive[key].copy() for key in archive.files}
            state["u"] = state.pop("u_seed")
            state["step"] = np.asarray(-1)
            state["activation_model"] = np.asarray("strain")
            state["solver_valid"] = np.asarray(0, dtype=bool)
            checkpoint_source = initialization_path
        else:
            states.append(
                {
                    "mode": mode,
                    "index": index,
                    "summary": summary,
                    "state": None,
                    "metrics": summary.get("last_metrics") or {},
                    "gradient": {"status": "unavailable"},
                    "terminal": False,
                    "status": "no_saved_state",
                    "checkpoint": None,
                    "summary_receipt": record(summary_path),
                    "gradient_receipt": None,
                    "chain_entry_status": chain_status.get("stages", {})
                    .get(mode, {})
                    .get("status"),
                }
            )
            continue
        if str(state["mode"]) != mode or str(state["activation_model"]) != "strain":
            raise ValueError(f"{mode} checkpoint mode/model mismatch")
        step = int(state["step"])
        metrics = summary.get("last_metrics") or {}
        if step >= 0 and int(metrics.get("attempt", -1)) != step:
            raise ValueError(f"{mode} checkpoint step does not match last metrics")
        if step >= 0 and int(metrics.get("optimizer_updates", -1)) != int(
            summary["optimizer_updates"]
        ):
            raise ValueError(f"{mode} optimizer count does not match last metrics")
        if int(summary["attempted_updates"]) != int(summary["optimizer_updates"]) + int(
            summary["skipped_updates"]
        ):
            raise ValueError(f"{mode} attempt/update/skip counts do not balance")
        np.testing.assert_allclose(state["B"], state["S"] + np.eye(3), rtol=0, atol=0)
        np.testing.assert_allclose(
            state["S"], np.swapaxes(state["S"], -1, -2), rtol=0, atol=1e-12
        )
        if state["u"].ndim != 2 or state["u"].shape[1] != 3:
            raise ValueError(
                f"{mode} displacement has invalid shape {state['u'].shape}"
            )
        if state["B"].ndim != 3 or state["B"].shape[1:] != (3, 3):
            raise ValueError(
                f"{mode} activation tensor has invalid shape {state['B'].shape}"
            )
        gradient_path = folder / "gradient-balance.json"
        gradient = (
            json.loads(gradient_path.read_text())
            if gradient_path.is_file()
            else {"status": "unavailable"}
        )
        if gradient.get("status") == "available":
            verify_record(gradient["checkpoint"], f"{mode} gradient checkpoint")
            verify_record(gradient["components"], f"{mode} gradient components")
        chain_entry = chain_status.get("stages", {}).get(mode, {})
        terminal_stage = (
            summary.get("status") == "completed_attempt_budget"
            and int(summary["attempted_updates"]) == 200
        )
        states.append(
            {
                "mode": mode,
                "index": index,
                "summary": summary,
                "state": state,
                "metrics": metrics,
                "gradient": gradient,
                "terminal": terminal_stage,
                "status": (
                    "initialization_snapshot"
                    if int(state["step"]) < 0
                    else summary.get("status", "unknown")
                ),
                "checkpoint": record(checkpoint_source),
                "summary_receipt": record(summary_path),
                "gradient_receipt": record(gradient_path)
                if gradient_path.is_file()
                else None,
                "chain_entry_status": chain_entry.get("status"),
            }
        )
    return states


def compose(
    output: Path,
    states: list[dict[str, Any]],
    panel_receipts: list[dict[str, Any]],
    *,
    chain_status: str,
    full_pose_fixed_error_m: list[float],
) -> tuple[Path, Path]:
    page = Image.new("RGB", PAGE, BACKGROUND)
    draw = ImageDraw.Draw(page)
    title = font(82, bold=True)
    subtitle = font(36)
    heading = font(54, bold=True)
    draw.text(
        point(PAGE_LAYOUT[0] / 2, 30),
        "MouthOpen: four activation parameterizations",
        font=title,
        fill="white",
        anchor="mt",
    )
    draw.text(
        point(PAGE_LAYOUT[0] / 2, 150),
        "Full prescribed jaw pose · deformed shape (top) · principal activation (bottom)",
        font=subtitle,
        fill="#d9d9d9",
        anchor="mt",
    )
    for index, state in enumerate(states):
        mode = state["mode"]
        x = 64 + index * 1248
        center = x + 1230 / 2
        draw.multiline_text(
            point(center, 250),
            HEADINGS[mode],
            font=heading,
            fill="white",
            anchor="ma",
            align="center",
            spacing=5,
        )
        summary = state["summary"]
        attempted = int(summary.get("attempted_updates", 0))
        accepted = int(summary.get("optimizer_updates", 0))
        skips = int(summary.get("skipped_updates", 0))
        phase = (
            f"{DOFS[mode]} DoF · {attempted}/200 attempts · {accepted} updates · {skips} skips"
            if state["terminal"]
            else f"SNAPSHOT · {attempted}/200 attempts · {accepted} updates · {summary.get('status', 'unknown')}"
        )
        draw.text(
            point(center, 390),
            phase,
            font=font(31),
            fill="#dfdfdf" if state["terminal"] else "#f1cc83",
            anchor="mt",
        )
        metrics = state["metrics"]
        fit_value = metrics.get("fit_rms_mm")
        normal_value = metrics.get("normal_angle_rms_deg")
        metric_label = (
            f"Fit {float(fit_value):.3f} mm · normal {float(normal_value):.2f}°"
            if fit_value is not None and normal_value is not None
            else "Fit/normal: n/a · no accepted state"
        )
        draw.text(
            point(center, 445), metric_label, font=font(31), fill="white", anchor="mt"
        )
        balance = state["gradient"]
        ratio = (
            balance.get("smoothness_to_l2_gradient_ratio")
            if balance.get("status") == "available"
            else None
        )
        ratio_text = "n/a" if ratio is None else f"{float(ratio):.3g}"
        draw.text(
            point(center, 495),
            f"Full-S smoothness/L2 grad ratio {ratio_text}",
            font=font(27),
            fill="#bdc7cb",
            anchor="mt",
        )
        for row, y in (("shape", 550), ("activation", 1576)):
            panel_index = 2 * index + (row == "activation")
            panel = panel_receipts[panel_index]["path"]
            with Image.open(panel) as image:
                if image.size != WINDOW:
                    raise ValueError(
                        f"unexpected panel dimensions {image.size}: {panel}"
                    )
                page.paste(image.convert("RGB"), point(x, y))

    draw.text(
        point(64, 2605),
        "Principal activation amplitude a (dimensionless)",
        font=font(36),
        fill="white",
    )
    colorbar_x, colorbar_y, colorbar_w = 1700, 2613, 1700
    ramp = FIG.contraction_colormap()(np.linspace(0, 1, colorbar_w * 2), bytes=True)[
        :, :3
    ]
    page.paste(
        Image.fromarray(np.tile(ramp[None, :, :], (80, 1, 1))),
        point(colorbar_x, colorbar_y),
    )
    for value in (0, 1, 3, 10, 30, 60):
        fraction = np.log1p(value) / np.log1p(AMOUNT_CAP)
        x = colorbar_x + float(fraction) * colorbar_w
        draw.line(
            (point(x, colorbar_y + 40), point(x, colorbar_y + 50)),
            fill="#dfdfdf",
            width=3,
        )
        draw.text(
            point(x, colorbar_y + 53),
            f"{value:g}",
            font=font(29),
            fill="#dfdfdf",
            anchor="mt",
        )
    draw.text(point(4140, 2612), "Red: contraction-like", font=font(33), fill="#df6255")
    draw.text(
        point(64, 2680),
        "Color: log(1 + a); a = σmax(B) - 1 · line length: 4.5 mm x log(1 + a) / log(61)",
        font=font(30),
        fill="#dfdfdf",
        anchor="lt",
    )
    draw.text(
        point(64, 2750),
        f"Chain status: {chain_status} · all activation tensors drive their saved deformed shape; glyphs display only the largest positive principal mode.",
        font=font(27),
        fill="#bdbdbd",
    )
    draw.text(
        point(64, 2803),
        "Full prescribed jaw pose verified on fixed nodes (max errors by stage, mm): "
        + ", ".join(
            "n/a" if value is None else f"{value * 1000:.3g}"
            for value in full_pose_fixed_error_m
        )
        + " · finite inversions/intersections remain diagnostics, not validation.",
        font=font(24),
        fill="#bdbdbd",
    )
    figure = output / "mouthopen-four-stage.png"
    preview = output / "mouthopen-four-stage-preview.png"
    page.save(figure, dpi=(300, 300))
    page.resize((1920, 1080), Image.Resampling.LANCZOS).save(preview)
    page.close()
    return figure, preview


def blank_panel(path: Path, label: str, camera: dict[str, Any]) -> np.ndarray:
    plotter = FIG.new_plotter(WINDOW)
    projection = FIG.capture(plotter, camera, path)
    with Image.open(path) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        point(WINDOW[0] / 2, WINDOW[1] / 2),
        label,
        font=font(42, bold=True),
        fill="white",
        anchor="mm",
    )
    canvas.save(path)
    canvas.close()
    return projection


def main(cfg: Config) -> None:
    source = cherries.input(cfg.source)
    fixture = cherries.input(cfg.fixture)
    prepared_path = cherries.input(cfg.prepared)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    panels = output / "panels"
    panels.mkdir()

    _protocol, chain_receipts = load_chain(source)
    chain_status = json.loads((source / "chain-status.json").read_text())
    states = load_states(source, chain_status)
    with np.load(source / "mesh.npz", allow_pickle=False) as archive:
        mesh = {key: archive[key].copy() for key in archive.files}
    required_mesh = {
        "rest_points",
        "tets",
        "active_ids",
        "skin_ids",
        "triangles",
        "target_displacement_skin",
        "skin_vertex_weights",
    }
    if not required_mesh.issubset(mesh):
        raise ValueError(
            f"chain mesh missing keys: {sorted(required_mesh - set(mesh))}"
        )
    volume = pv.read(fixture / "volume.vtu")
    skin_mesh = pv.read(fixture / "skin.vtp")
    points = np.asarray(volume.points, dtype=np.float64)
    tets = np.asarray(volume.cells).reshape(-1, 5)[:, 1:].astype(np.int64)
    skin_ids = np.asarray(skin_mesh.point_data["GlobalPointId"], dtype=np.int64)
    skin_faces = np.asarray(skin_mesh.faces).copy()
    active_ids = np.flatnonzero(
        np.asarray(volume.cell_data["ActivationMask"], dtype=bool)
    )
    np.testing.assert_array_equal(mesh["rest_points"], points)
    np.testing.assert_array_equal(mesh["tets"], tets)
    np.testing.assert_array_equal(mesh["skin_ids"], skin_ids)
    np.testing.assert_array_equal(mesh["active_ids"], active_ids)
    np.testing.assert_array_equal(mesh["triangles"], skin_faces.reshape(-1, 4)[:, 1:])
    for state in states:
        if state["state"] is None:
            continue
        if state["state"]["u"].shape != points.shape:
            raise ValueError(f"{state['mode']} displacement does not match shared mesh")
        if state["state"]["B"].shape != (len(active_ids), 3, 3):
            raise ValueError(
                f"{state['mode']} tensor field does not match active cells"
            )
        state["metrics"] = surface_metrics(mesh, state["state"]["u"])
        saved_metrics = state["summary"].get("last_metrics") or {}
        for key in ("fit_rms_mm", "normal_angle_rms_deg"):
            if key in saved_metrics:
                np.testing.assert_allclose(
                    state["metrics"][key], saved_metrics[key], rtol=2e-8, atol=1e-10
                )

    with np.load(prepared_path, allow_pickle=False) as archive:
        prepared = {key: archive[key].copy() for key in archive.files}
    pose = np.asarray(prepared["pose"], dtype=np.float64)
    pivot = np.asarray(prepared["pivot"], dtype=np.float64)
    cranium = pv.read(CRANIUM_PATH)
    eyes = pv.read(EYES_PATH)
    mandible = pv.read(MANDIBLE_PATH)
    jaw = mandible.copy(deep=True)
    jaw.points = rigid(np.asarray(mandible.points), pivot, pose)
    group_names = [str(name) for name in volume.field_data["GroupName"]]
    fixed = np.asarray(volume.point_data["IsFixed"], dtype=bool)
    mandible_group = np.asarray(
        volume.point_data["GroupId"], dtype=np.int64
    ) == group_names.index("Mandible")
    jaw_fixed = fixed & mandible_group
    rotation = Rotation.from_rotvec(pose[:3]).as_matrix()
    expected = np.zeros_like(points)
    expected[jaw_fixed] = (
        (points[jaw_fixed] - pivot) @ rotation.T + pivot + pose[3:] - points[jaw_fixed]
    )
    pose_errors = []
    for state in states:
        if state["state"] is None:
            pose_errors.append(None)
            continue
        displacement = np.asarray(state["state"]["u"], dtype=np.float64)
        np.testing.assert_allclose(
            displacement[fixed & ~jaw_fixed], 0, rtol=0, atol=1e-12
        )
        error = float(np.max(np.abs(displacement[jaw_fixed] - expected[jaw_fixed])))
        np.testing.assert_allclose(
            displacement[jaw_fixed], expected[jaw_fixed], rtol=0, atol=1e-12
        )
        pose_errors.append(error)

    skin_states = [
        points[skin_ids] + state["state"]["u"][skin_ids]
        for state in states
        if state["state"] is not None
    ]
    if not skin_states:
        skin_states = [points[skin_ids] + mesh["target_displacement_skin"]]
    # Fit one camera to all four deformed skins and the static anatomy.
    projected_toward = np.array([0.65, 0.03, 1.0], dtype=np.float64)
    projected_toward /= np.linalg.norm(projected_toward)
    right = np.cross([0.0, 1.0, 0.0], projected_toward)
    right /= np.linalg.norm(right)
    up = np.cross(projected_toward, right)
    basis = np.column_stack((right, up, projected_toward))
    projected = (
        np.vstack(
            [
                np.asarray(cranium.points),
                np.asarray(jaw.points),
                np.asarray(eyes.points),
                *skin_states,
            ]
        )
        @ basis
    )
    low, high = projected.min(axis=0), projected.max(axis=0)
    center = basis @ ((low + high) / 2)
    half = (high - low) / 2
    camera = {
        "position": (center + 0.6 * projected_toward).tolist(),
        "focal_point": center.tolist(),
        "view_up": up.tolist(),
        "parallel_scale": float(
            1.055 * max(half[1], half[0] / (WINDOW[0] / WINDOW[1]))
        ),
        "projected_bounds_m": [low.tolist(), high.tolist()],
        "window_size": list(WINDOW),
        "policy": "single orthographic camera fitted to all four full deformed skins and static anatomy",
    }
    scene = SimpleNamespace(bones={"cranium": cranium, "mandible": jaw}, eyes=eyes)
    control_ids = np.asarray(volume.cell_data["ActivationControlId"])[active_ids]
    skin_tris = mesh["triangles"]
    skin_faces = np.column_stack((np.full(len(skin_tris), 3), skin_tris)).ravel()
    inverse_dm = np.linalg.inv(
        (points[tets[active_ids]][:, 1:] - points[tets[active_ids]][:, :1]).transpose(
            0, 2, 1
        )
    )
    projections = []
    panel_receipts = []
    glyph_receipts = []
    ratio_by_stage: dict[str, float | None] = {}
    for state in states:
        mode = state["mode"]
        if state["state"] is None:
            for row in ("shape", "activation"):
                path = panels / f"{mode}-{row}.png"
                projection = blank_panel(
                    path,
                    "Not run" if state["status"] == "not_run" else "No saved state",
                    camera,
                )
                projections.append(projection)
                panel_receipts.append(
                    {
                        "path": str(path.resolve()),
                        **record(path),
                        "mode": mode,
                        "row": row,
                        "status": state["status"],
                    }
                )
            ratio_by_stage[mode] = None
            glyph_receipts.append(
                {
                    "mode": mode,
                    "status": state["status"],
                    "eligible_positive_modes": 0,
                    "visible_positive_modes": 0,
                }
            )
            continue
        displacement = state["state"]["u"]
        deformed = points + displacement
        skin = pv.PolyData(deformed[skin_ids], skin_faces)
        shape_path = panels / f"{mode}-shape.png"
        shape_projection = FIG.render_shape(scene, skin, camera, shape_path)
        projections.append(shape_projection)
        active_tet = deformed[tets[active_ids]]
        deformation_gradient = (active_tet[:, 1:] - active_tet[:, :1]).transpose(
            0, 2, 1
        ) @ inverse_dm
        glyph = FIG.principal_glyphs(
            state["state"]["B"], deformation_gradient, active_tet.mean(axis=1)
        )
        positive = glyph.eligible & (glyph.signed_display_percent > 0)
        amplitude = np.sqrt(1 + glyph.eigenvalues_z[positive]) - 1
        max_amplitude = float(amplitude.max()) if len(amplitude) else 0.0
        if max_amplitude > AMOUNT_CAP + 1e-10:
            raise ValueError(
                f"{mode} principal amplitude {max_amplitude} exceeds shared 60 scale"
            )
        volume_copy = volume.copy(deep=True)
        volume_copy.points = deformed
        region_context = FIG.build_muscle_region_context(volume_copy)
        visibility = FIG.visible_region_mask(
            region_context,
            glyph.centers,
            active_ids,
            control_ids,
            camera,
            window_size=FIG.ACTIVATION_WINDOW,
        )
        mask = visibility.mask & positive
        activation_path = panels / f"{mode}-activation.png"
        if mask.any():
            activation_projection = FIG.render_activation(
                skin, glyph, mask, camera, activation_path
            )
        else:
            plotter = FIG.new_plotter(FIG.ACTIVATION_WINDOW)
            plotter.add_mesh(
                skin, color=FIG.SHAPE_COLOR, opacity=FIG.ACTIVATION_CONTEXT_OPACITY
            )
            activation_projection = FIG.capture(plotter, camera, activation_path)
        projections.append(activation_projection)
        if not np.allclose(shape_projection, activation_projection, rtol=0, atol=1e-12):
            raise ValueError(f"shape and activation projections differ for {mode}")
        panel_receipts.extend(
            [
                {"path": str(shape_path.resolve()), **record(shape_path)},
                {"path": str(activation_path.resolve()), **record(activation_path)},
            ]
        )
        balance = state["gradient"]
        ratio = (
            balance.get("smoothness_to_l2_gradient_ratio")
            if balance.get("status") == "available"
            else None
        )
        ratio_by_stage[mode] = None if ratio is None else float(ratio)
        glyph_receipts.append(
            {
                "mode": mode,
                **glyph.receipt,
                "eligible_positive_modes": int(positive.sum()),
                "visible_positive_modes": int(mask.sum()),
                "positive_principal_amplitude_max": max_amplitude,
                "color_formula": "log1p(a)/log1p(60), a=sqrt(1+z)-1",
                "length_formula": "4.5mm*log1p(a)/log1p(60)",
                "scale_cap": AMOUNT_CAP,
                "active_control_visibility": visibility.retained_count,
            }
        )
        LOG.info(
            "Rendered %s, update %s, %d visible modes",
            mode,
            state["state"]["step"],
            int(mask.sum()),
        )
        del volume_copy, region_context, visibility, glyph, skin

    for projection in projections:
        np.testing.assert_allclose(projection, projections[0], rtol=0, atol=1e-12)
    figure, preview = compose(
        output,
        states,
        panel_receipts,
        chain_status=str(chain_status["status"]),
        full_pose_fixed_error_m=pose_errors,
    )
    input_paths = [
        source / "protocol.json",
        source / "chain-status.json",
        source / "mesh.npz",
        source / "source-manifest.json",
        fixture / "summary.json",
        fixture / "volume.vtu",
        fixture / "skin.vtp",
        prepared_path,
    ]
    input_paths.extend(
        path
        for state in states
        for path in (
            source / state["mode"] / "summary.json",
            source / state["mode"] / "last.npz",
            source / state["mode"] / "gradient-balance.json",
        )
        if path.is_file()
    )
    source_paths = [
        Path(__file__),
        STRESS_SRC / "shape_scene.py",
        STRESS_SRC / "activation_scene.py",
        STRESS_SRC / "52-render-four-stage-figures.py",
        CONTEXT_SRC / "muscle_glyph_context.py",
        CRANIUM_PATH,
        MANDIBLE_PATH,
        EYES_PATH,
    ]
    manifest = {
        "schema": "mouthopen-four-stage-render-v1",
        "chain_status": chain_status["status"],
        "all_stages_terminal_200_attempts": all(state["terminal"] for state in states),
        "stages": [
            {
                "mode": state["mode"],
                "dof": DOFS[state["mode"]],
                "heading": HEADINGS[state["mode"]],
                "status": state["status"],
                "attempted_updates": state["summary"]["attempted_updates"],
                "optimizer_updates": state["summary"]["optimizer_updates"],
                "skipped_updates": state["summary"]["skipped_updates"],
                "checkpoint": state["checkpoint"],
                "summary": state["summary_receipt"],
                "gradient_balance": state["gradient_receipt"],
                "gradient_ratio": ratio_by_stage[state["mode"]],
                "metrics": state["metrics"],
            }
            for state in states
        ],
        "chain_receipts": chain_receipts,
        "protocol": record(source / "protocol.json"),
        "mesh": record(source / "mesh.npz"),
        "jaw_pose": {
            "source": record(prepared_path),
            "pose": pose.tolist(),
            "pivot": pivot.tolist(),
            "max_fixed_node_displacement_error_m": pose_errors,
            "full_pose_applied_to_rendered_mandible": True,
        },
        "camera": camera,
        "screen_projection_identical_across_all_shape_and_activation_panels": True,
        "activation_display": {
            "full_B_field_drives_saved_geometry": True,
            "only_largest_positive_principal_mode_displayed": True,
            "shared_amplitude_scale_max": AMOUNT_CAP,
            "line_length_max_m": 0.0045,
            "glyphs": glyph_receipts,
        },
        "panels": panel_receipts,
        "inputs": {str(path.resolve()): record(path) for path in input_paths},
        "sources": [record(path) for path in source_paths],
        "figure": record(figure),
        "preview": record(preview),
        "physical_validity_claim": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n"
    )
    (output / "source.py").write_text(Path(__file__).read_text())
    cherries.log_metrics(
        {
            "completed_stages": sum(state["terminal"] for state in states),
            "sampled_direction_review": 0,
            **{
                f"{mode}/fit_rms_mm": states[i]["metrics"]["fit_rms_mm"]
                for i, mode in enumerate(STAGES)
                if states[i]["metrics"].get("fit_rms_mm") is not None
            },
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
