# ruff: noqa: C901, EM101, EM102, PLW0603, TRY003
"""Render saved continuation states with the frozen regional camera receipt."""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

mpl.use("Agg")
import matplotlib.pyplot as plt
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
FIXTURE = (
    ROOT
    / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture/volume.vtu"
)
FROZEN_REGIONS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions"
CAMERA_RECEIPT = FROZEN_REGIONS / "summary.json"
TOPOLOGY = FROZEN_REGIONS / "rest-skin.vtp"
CANONICAL_STEP200 = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/20-baseline/final.npz"
)
BACKGROUND = "#242c36"
SECTION_Y = (2.170, 2.180, 2.190)
SECTION_X = (1.414, 1.460)
SECTION_Z_MIN = 0.040
COLORS = {"reference": "#87929f", "current": "#c53d3d", "target": "#1b78a5"}
DONE = False
LOG = logging.getLogger(__name__)


class ProfileCometNoCommit(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(plugins.Comet(run=run, disabled=False))
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Local(run=run))
        run.plugins.register(plugins.Logging(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    state: Path
    reference_state: Path = CANONICAL_STEP200
    output_dir: Path = cherries.output("30-render-comparison", mkdir=True)
    resolution: int = 1000


def record(path: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def load_state(path: Path, rest: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    with np.load(path, allow_pickle=False) as saved:
        required = {"step", "solver_valid", "rest_points", "u"}
        if not required <= set(saved.files):
            raise ValueError(
                f"state {path} lacks {sorted(required - set(saved.files))}"
            )
        if not bool(saved["solver_valid"]):
            raise ValueError(f"state {path} is not solver-valid")
        stored_rest = np.asarray(saved["rest_points"], dtype=np.float64)
        u = np.asarray(saved["u"], dtype=np.float64)
        if not np.array_equal(stored_rest, rest):
            raise ValueError(f"state {path} does not use the fixture rest geometry")
        if u.shape != rest.shape or not np.isfinite(u).all():
            raise ValueError(f"state {path} has invalid displacement geometry")
        return rest + u, {"step": int(saved["step"]), "solver_valid": True}


def surface(base: pv.PolyData, points: np.ndarray, ids: np.ndarray) -> pv.PolyData:
    result = base.copy(deep=True)
    result.clear_data()
    result.points = points[ids].copy()
    result.point_data["VolumePointIndex"] = ids
    return result


def render(
    surfaces: list[pv.PolyData],
    view: dict[str, object],
    labels: list[str],
    path: Path,
    resolution: int,
) -> dict[str, object]:
    camera = view["camera"]
    assert isinstance(camera, dict)
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    # Exact unchanged regional-comparison raking-light construction.
    key = focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward)
    fill = focus + 0.3 * backward
    plotter = pv.Plotter(
        shape=(1, 3),
        off_screen=True,
        window_size=(3 * resolution, resolution),
        lighting="none",
        border=False,
    )
    for index, mesh in enumerate(surfaces):
        plotter.subplot(0, index)
        actor = plotter.add_mesh(
            mesh,
            color="#eeeeea",
            smooth_shading=False,
            ambient=0.20,
            diffuse=0.80,
            specular=0.0,
        )
        assert actor.GetProperty().GetInterpolation() == 0
        plotter.add_light(
            pv.Light(
                position=key,
                focal_point=focus,
                intensity=0.85,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )
        plotter.add_light(
            pv.Light(
                position=fill,
                focal_point=focus,
                intensity=0.20,
                light_type="scene light",
                positional=False,
            ),
            only_active=True,
        )
        assert len(plotter.renderer.lights) == 2
        plotter.enable_parallel_projection()
        plotter.camera.position = camera["position"]
        plotter.camera.focal_point = camera["focal_point"]
        plotter.camera.up = camera["view_up"]
        plotter.camera.parallel_scale = camera["parallel_scale"]
        plotter.reset_camera_clipping_range()
        plotter.set_background(BACKGROUND)
    raster = plotter.screenshot(return_img=True)
    plotter.close()
    image = Image.new("RGB", (3 * resolution, resolution + 130), BACKGROUND)
    image.paste(Image.fromarray(raster).convert("RGB"), (0, 90))
    draw = ImageDraw.Draw(image)
    title = ImageFont.truetype("DejaVuSans-Bold.ttf", 28)
    font = ImageFont.truetype("DejaVuSans.ttf", 21)
    for index, label in enumerate(labels):
        draw.text((index * resolution + 20, 12), label, fill="white", font=title)
        draw.text(
            (index * resolution + 20, 51), str(view["label"]), fill="#c7d0da", font=font
        )
        draw.text(
            (index * resolution + 20, resolution + 99),
            "Same fixed camera and lights | flat shading | actual geometry",
            fill="#c7d0da",
            font=font,
        )
    image.save(path)
    return {
        "file": record(path),
        "camera": camera,
        "key_light_position": key.tolist(),
        "fill_light_position": fill.tolist(),
        "flat_shading": True,
        "size": list(image.size),
    }


def triangle_segments(
    points: np.ndarray, triangles: np.ndarray, y: float
) -> np.ndarray:
    """Return the exact linear edge intersections used by the prior closeups."""
    segments: list[np.ndarray] = []
    for triangle in points[triangles]:
        signed = triangle[:, 1] - y
        hits: list[np.ndarray] = []
        for left, right in ((0, 1), (1, 2), (2, 0)):
            a, b = triangle[left], triangle[right]
            sa, sb = signed[left], signed[right]
            if sa == 0.0 and sb == 0.0:
                continue
            if sa == 0.0:
                hits.append(a)
            elif sb == 0.0:
                hits.append(b)
            elif (sa < 0.0) != (sb < 0.0):
                hits.append(a + (-sa / (sb - sa)) * (b - a))
        unique: list[np.ndarray] = []
        for hit in hits:
            if not any(np.allclose(hit, old, rtol=0, atol=1e-13) for old in unique):
                unique.append(hit)
        if len(unique) == 2:
            segment = np.stack(unique)
            if (
                segment[:, 0].max() >= SECTION_X[0]
                and segment[:, 0].min() <= SECTION_X[1]
                and segment[:, 2].max() >= SECTION_Z_MIN
            ):
                segments.append(segment)
    return np.asarray(segments, dtype=np.float64)


def render_sections(
    surfaces: dict[str, pv.PolyData], triangles: np.ndarray, output: Path
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    figure, axes = plt.subplots(
        len(SECTION_Y), 1, figsize=(8, 9), constrained_layout=True
    )
    for axis, y in zip(axes, SECTION_Y, strict=True):
        for name, mesh in surfaces.items():
            segments = triangle_segments(np.asarray(mesh.points), triangles, y)
            for index, segment in enumerate(segments):
                axis.plot(
                    segment[:, 0],
                    segment[:, 2],
                    color=COLORS[name],
                    linewidth=0.65,
                    label=name if index == 0 else None,
                )
                for endpoint, point in enumerate(segment):
                    rows.append(
                        {
                            "y_m": y,
                            "surface": name,
                            "segment": index,
                            "endpoint": endpoint,
                            "x_m": point[0],
                            "z_m": point[2],
                        }
                    )
        axis.set_title(f"Exact horizontal triangle sections at y = {y:.3f} m")
        axis.set_xlim(*SECTION_X)
        axis.set_ylim(SECTION_Z_MIN, 0.115)
        axis.set_aspect("equal", adjustable="box")
        axis.set_ylabel("z (m)")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("x (m)")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    figure.suptitle(
        "Right nasolabial surface sections | exact triangle intersections, no smoothing"
    )
    png = output / "nasolabial-horizontal-sections.png"
    csv_path = output / "nasolabial-horizontal-segments.csv"
    figure.savefig(png, dpi=220)
    plt.close(figure)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("y_m", "surface", "segment", "endpoint", "x_m", "z_m")
        )
        writer.writeheader()
        writer.writerows(rows)
    return {
        "method": "Per-triangle affine intersections with fixed y planes; x/z display filtering matches closeups diagnostics and does not join or smooth segments.",
        "planes_y_m": SECTION_Y,
        "x_range_m": SECTION_X,
        "z_min_m": SECTION_Z_MIN,
        "segment_count": len(rows) // 2,
        "outputs": {png.name: record(png), csv_path.name: record(csv_path)},
    }


def main(cfg: Config) -> None:
    global DONE
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), out
    for path in (FIXTURE, CAMERA_RECEIPT, TOPOLOGY, cfg.state, cfg.reference_state):
        if not path.is_file():
            raise FileNotFoundError(path)
    receipt = json.loads(CAMERA_RECEIPT.read_text())
    if not (
        receipt["same_topology"]
        and not receipt["geometry_smoothing"]
        and receipt["deformation_scale"] == 1
    ):
        raise ValueError("frozen camera receipt has incompatible geometry settings")
    views = receipt["views"]
    if len(views) != 5:
        raise ValueError("frozen camera receipt must contain exactly five views")
    fixture = pv.read(FIXTURE)
    rest = np.asarray(fixture.points, dtype=np.float64)
    target_u = np.asarray(fixture.point_data["Smile"], dtype=np.float64)
    base = pv.read(TOPOLOGY)
    ids = np.asarray(base.point_data["VolumePointIndex"], dtype=np.int64)
    faces = np.asarray(base.faces, dtype=np.int64).reshape(-1, 4)
    if base.n_points != 15_299 or base.n_cells != 29_899 or np.any(faces[:, 0] != 3):
        raise ValueError("frozen display topology changed")
    if (
        array_digest(ids) != receipt["point_ids_sha256"]
        or array_digest(base.faces) != receipt["faces_sha256"]
    ):
        raise ValueError("frozen display topology hash differs from regional receipt")
    if not np.array_equal(np.asarray(base.points, dtype=np.float64), rest[ids]):
        raise ValueError("frozen display rest geometry no longer maps to fixture")
    reference_points, reference_meta = load_state(cfg.reference_state, rest)
    current_points, current_meta = load_state(cfg.state, rest)
    if not np.isfinite(target_u[ids]).all():
        raise ValueError("display topology includes a non-finite Smile target")
    target_points = rest + target_u
    surfaces = {
        "reference": surface(base, reference_points, ids),
        "current": surface(base, current_points, ids),
        "target": surface(base, target_points, ids),
    }
    for name, mesh in surfaces.items():
        if not np.array_equal(mesh.faces, base.faces) or not np.array_equal(
            np.asarray(mesh.point_data["VolumePointIndex"]), ids
        ):
            raise ValueError(f"{name} topology differs from frozen display topology")
    reference_current_positions_equal = bool(
        np.array_equal(reference_points[ids], current_points[ids])
    )
    labels = [
        f"Reference state | step {reference_meta['step']}",
        f"Current state | step {current_meta['step']}",
        "Fixture target | Smile",
    ]
    figures = {}
    ordered = [surfaces["reference"], surfaces["current"], surfaces["target"]]
    for view in views:
        figures[view["id"]] = render(
            ordered, view, labels, out / f"{view['id']}.png", cfg.resolution
        )
        LOG.info("Rendered %s", view["id"])
    sections = render_sections(surfaces, faces[:, 1:], out)
    summary = {
        "status": "completed_saved_state_render_postprocess",
        "scope": "No forward solve, adjoint solve, optimizer update, smoothing, interpolation, deformation scaling, or full-mesh render.",
        "inputs": {
            str(path): record(path)
            for path in (
                FIXTURE,
                CAMERA_RECEIPT,
                TOPOLOGY,
                cfg.reference_state,
                cfg.state,
            )
        },
        "source": record(Path(__file__)),
        "camera_source": {
            "path": str(CAMERA_RECEIPT),
            "sha256": record(CAMERA_RECEIPT)["sha256"],
            "unchanged_five_views": views,
        },
        "states": {
            "reference": {**reference_meta, "label": labels[0]},
            "current": {**current_meta, "label": labels[1]},
            "target": {"field": "Smile", "label": labels[2]},
        },
        "native_geometry_equality": {
            "fixture_rest_equals_frozen_topology_points": True,
            "reference_state_rest_equals_fixture": True,
            "current_state_rest_equals_fixture": True,
            "display_point_ids_sha256": array_digest(ids),
            "faces_sha256": array_digest(base.faces),
            "same_ids_and_faces_for_reference_current_target": True,
            "reference_current_display_positions_equal": reference_current_positions_equal,
            "display_points": base.n_points,
            "display_triangles": base.n_cells,
        },
        "rendering": {
            "flat_shading": True,
            "geometry_smoothing": False,
            "deformation_scale": 1,
            "fixed_lights": "exact regional-comparison raking key/fill construction",
            "figures": figures,
        },
        "sections": sections,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    for path in out.iterdir():
        cherries.log_output(path)
    DONE = True


if __name__ == "__main__":
    cherries.main(
        main, profile="debug" if os.environ.get("DEBUG") else ProfileCometNoCommit
    )
    if not DONE:
        raise SystemExit(1)
