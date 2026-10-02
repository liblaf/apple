# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Render immutable face endpoints with the established saved-geometry exporter."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
EARLIER_EXPERIMENT = EXPERIMENT.parent / "face-actuation-diagnosis"
FROZEN_RENDERER = EARLIER_EXPERIMENT / "src/50-render-diagnosis.py"
RENDER_PROTOCOL = EXPERIMENT / "docs/50-render-protocol.md"
DEFAULT_MANIFEST = EXPERIMENT / "docs/50-render-manifest.json"
EXPECTED_CASE_IDS = (
    "raw6",
    "psd",
    "psd-smooth",
    "psd-smooth-rank",
)
HISTORY_STEPS = (0, 16, 32, 48, 64)
STATIC_COLOR = "#d9a486"
STATIC_BACKGROUND = "#fbfaf7"
STATIC_SIZE = (3000, 720)
COMPLETED = False


class Config(cherries.BaseConfig):
    """Immutable comparison manifest and Cherries-managed render output."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    manifest: Path = DEFAULT_MANIFEST
    output_dir: Path = cherries.output("50-face-comparison", mkdir=True)


def sha256(path: Path) -> str:
    """Return a streaming SHA-256 receipt."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe a source or artifact by immutable content."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def write_json(path: Path, value: Any) -> None:
    """Atomically write strict JSON."""
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_script(name: str, path: Path) -> ModuleType:
    """Load the frozen earlier renderer without running its CLI."""
    if not path.is_file():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        spec.loader.exec_module(module)
    return module


def resolve(manifest_path: Path, value: str) -> Path:
    """Resolve one explicit manifest-relative immutable file."""
    path = Path(value)
    path = path if path.is_absolute() else manifest_path.parent / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if "latest" in path.name.lower():
        raise ValueError(f"moving latest file is forbidden: {path}")
    return path


def read_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read the fixed four-arm render contract."""
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema_version") != 1:
        raise ValueError("render manifest must use schema version 1")
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise TypeError("render manifest cases must be a list")
    ids = tuple(row.get("id") for row in cases if isinstance(row, dict))
    if ids != EXPECTED_CASE_IDS:
        raise ValueError(f"case order must be exactly {EXPECTED_CASE_IDS}")
    if document.get("history_steps") != list(HISTORY_STEPS):
        raise ValueError(f"history_steps must be exactly {list(HISTORY_STEPS)}")
    if document.get("static_material") != "uniform-skin":
        raise ValueError("static_material must be uniform-skin")
    if document.get("viewer_material") != "DominantMaterialPhase":
        raise ValueError("viewer_material must be DominantMaterialPhase")
    return document, cases


def load_grid(renderer: ModuleType, path: Path) -> pv.UnstructuredGrid:
    """Load an unchanged saved state whose points are already the state."""
    return renderer.state_grid(path, "positions_are_state", "Displacement")


def reference_grid(renderer: ModuleType, path: Path) -> pv.UnstructuredGrid:
    """Use the saved step-0 RestPosition as the common reference geometry."""
    reference = renderer.state_grid(path, "rest_position_array", "RestPosition")
    if "ActivationFiber" in reference.cell_data:
        raise ValueError("step-0 reference unexpectedly contains ActivationFiber")
    rest = np.asarray(reference.point_data["RestPosition"], dtype=np.float64)
    if not np.array_equal(np.asarray(reference.points, dtype=np.float64), rest):
        raise ValueError("reference points do not equal saved RestPosition")
    return reference


def validate_summary(path: Path, expected_step: int) -> dict[str, Any]:
    """Require a completed fixed-budget run at the declared actual endpoint."""
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("status") != "completed_fixed_budget":
        raise ValueError(f"run is not completed_fixed_budget: {path}")
    endpoint = summary.get("primary_endpoint")
    if not isinstance(endpoint, dict) or int(endpoint.get("step", -1)) != expected_step:
        raise ValueError(f"primary endpoint is not step {expected_step}: {path}")
    if endpoint.get("solver_valid") is not True:
        raise ValueError(f"primary endpoint is not solver-valid: {path}")
    return {
        "source": record(path),
        "status": summary["status"],
        "primary_step": int(endpoint["step"]),
        "solver_valid": bool(endpoint["solver_valid"]),
    }


def target_surface(
    renderer: ModuleType, reference_path: Path
) -> tuple[pv.PolyData, dict[str, Any]]:
    """Build only the exact target skin from the saved step-0 arrays."""
    surface, counts = renderer.target_skin(
        reference_path,
        "TargetDisplacement",
        "IsFace",
        rest_position_array="RestPosition",
    )
    if not np.isfinite(np.asarray(surface.points, dtype=np.float64)).all():
        raise ValueError("selected target surface contains non-finite points")
    excluded = int(counts["selected_points"] - surface.n_points)
    if excluded < 0:
        raise ValueError("selected target surface exceeds the full IsFace selection")
    return surface, {
        "source": record(reference_path),
        "displacement_array": "TargetDisplacement",
        "selector": "IsFace",
        "rest_position_array": "RestPosition",
        "counts": counts,
        "selected_exterior_points": int(surface.n_points),
        "selected_exterior_triangles": int(surface.n_cells),
        "tagged_points_excluded_from_exterior": excluded,
        "scope": "target skin only; no target interior tissue state",
    }


def camera_bounds(
    surfaces: list[pv.PolyData],
) -> tuple[float, float, float, float, float, float]:
    """Return one physical bound shared by all static front panels."""
    bounds = np.asarray([surface.bounds for surface in surfaces], dtype=np.float64)
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
        float(bounds[:, 4].min()),
        float(bounds[:, 5].max()),
    )


def add_centered_label(
    draw: ImageDraw.ImageDraw,
    label: str,
    center_x: int,
    top: int,
    width: int,
    font: ImageFont.FreeTypeFont,
) -> None:
    """Draw a compact, wrapped label below one static panel."""
    approximate_characters = max(12, width // 15)
    lines = textwrap.wrap(label, width=approximate_characters)[:2]
    for row, line in enumerate(lines):
        box = draw.textbbox((0, 0), line, font=font)
        draw.text(
            (center_x - (box[2] - box[0]) / 2, top + row * 25),
            line,
            fill="#172321",
            font=font,
        )


def render_static_grid(
    renderer: ModuleType,
    surfaces: list[pv.PolyData],
    labels: list[str],
    camera_spec: dict[str, Any],
    output_stem: Path,
    title: str,
) -> dict[str, Any]:
    """Render five uniform-color skins with one physical camera and true scale."""
    if len(surfaces) != 5 or len(labels) != 5:
        raise ValueError("static comparison requires four arms and one target")
    plotter = pv.Plotter(
        shape=(1, 5),
        off_screen=True,
        window_size=STATIC_SIZE,
        lighting="three lights",
    )
    for column, surface in enumerate(surfaces):
        plotter.subplot(0, column)
        renderer.apply_camera(plotter, camera_spec)
        plotter.add_mesh(
            surface,
            color=STATIC_COLOR,
            smooth_shading=True,
            show_edges=False,
        )
    image = plotter.screenshot(return_img=True)
    plotter.close()
    caption_height = 105
    composed = Image.new(
        "RGB", (image.shape[1], image.shape[0] + caption_height), STATIC_BACKGROUND
    )
    composed.paste(Image.fromarray(image), (0, 0))
    draw = ImageDraw.Draw(composed)
    title_font = ImageFont.truetype("DejaVuSans.ttf", 22)
    label_font = ImageFont.truetype("DejaVuSans.ttf", 17)
    draw.text((24, image.shape[0] + 8), title, fill="#172321", font=title_font)
    column_width = image.shape[1] / 5
    for column, label in enumerate(labels):
        add_centered_label(
            draw,
            label,
            int((column + 0.5) * column_width),
            image.shape[0] + 45,
            int(column_width) - 24,
            label_font,
        )
    png = output_stem.with_suffix(".png")
    pdf = output_stem.with_suffix(".pdf")
    composed.save(png)
    composed.save(pdf, "PDF", resolution=180.0)
    return {
        "png": record(png),
        "pdf": record(pdf),
        "camera": camera_spec,
        "surface": "actual saved IsFace exterior triangles; target is target skin only",
        "material": f"uniform {STATIC_COLOR}",
        "deformation_scale": 1.0,
    }


def add_viewer_theme_controls(path: Path) -> None:
    """Add system/light/dark controls without changing the frozen geometry exporter."""
    html = path.read_text(encoding="utf-8")
    old_style = (
        "<style>html,body{margin:0;height:100%;background:#f7f5ef;color:#182422;"
        "font:15px system-ui}body{height:100dvh;display:flex;flex-direction:column;"
        "overflow:hidden}header{flex:0 0 auto;padding:.75rem 1rem;background:#173b3a;"
        "color:#fff;display:flex;gap:.8rem;align-items:center;flex-wrap:wrap}"
    )
    new_style = (
        "<style>:root{color-scheme:light;--page:#f7f5ef;--text:#182422;"
        "--header:#173b3a;--hint:#c8e0dc}:root[data-theme=dark]{color-scheme:dark;"
        "--page:#101716;--text:#edf4f1;--header:#102f2e;--hint:#bad4cf}"
        "@media(prefers-color-scheme:dark){:root:not([data-theme]){color-scheme:dark;"
        "--page:#101716;--text:#edf4f1;--header:#102f2e;--hint:#bad4cf}}"
        "html,body{margin:0;height:100%;background:var(--page);color:var(--text);"
        "font:15px system-ui}body{height:100dvh;display:flex;flex-direction:column;"
        "overflow:hidden}header{flex:0 0 auto;padding:.75rem 1rem;"
        "background:var(--header);color:#fff;display:flex;gap:.8rem;"
        "align-items:center;flex-wrap:wrap}"
    )
    if html.count(old_style) != 1:
        raise ValueError("frozen viewer style anchor changed")
    html = html.replace(old_style, new_style)
    html = html.replace(
        ".hint{font-size:.82rem;color:#c8e0dc}",
        ".hint{font-size:.82rem;color:var(--hint)}",
    )
    header_anchor = '<header><strong id="title">Saved geometry</strong>'
    header_replacement = (
        '<header><strong id="title">Saved geometry</strong>'
        '<label>theme <select id="theme"><option value="system">system</option>'
        '<option value="light">light</option><option value="dark">dark</option>'
        "</select></label>"
    )
    if html.count(header_anchor) != 1:
        raise ValueError("frozen viewer header anchor changed")
    html = html.replace(header_anchor, header_replacement)
    cutaway_control = (
        '<label><input id="cutaway" type="checkbox"> fixed-rest cohort cutaway</label>'
    )
    if html.count(cutaway_control) != 1:
        raise ValueError("frozen viewer cutaway anchor changed")
    html = html.replace(cutaway_control, f"<label hidden>{cutaway_control[7:]}")
    fiber_fallback = "else fiber.parentElement.hidden=true;"
    if html.count(fiber_fallback) != 1:
        raise ValueError("frozen viewer fiber fallback anchor changed")
    html = html.replace(
        fiber_fallback,
        "else{fiber.parentElement.hidden=true;fiberRegion.parentElement.hidden=true}",
    )
    renderer_anchor = (
        "const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); "
        "renderer.setPixelRatio(Math.min(devicePixelRatio,2)); "
        "renderer.setClearColor(0xf7f5ef);"
    )
    renderer_replacement = (
        "const requestedTheme=new URLSearchParams(location.search).get('theme'),"
        "theme=document.querySelector('#theme'),scheme=matchMedia('(prefers-color-scheme:dark)');"
        "if(requestedTheme==='light'||requestedTheme==='dark'){"
        "document.documentElement.dataset.theme=requestedTheme;theme.value=requestedTheme}"
        "const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); "
        "renderer.setPixelRatio(Math.min(devicePixelRatio,2));"
        "function applyTheme(){const selected=theme.value;"
        "if(selected==='system')delete document.documentElement.dataset.theme;"
        "else document.documentElement.dataset.theme=selected;"
        "const dark=selected==='dark'||(selected==='system'&&scheme.matches);"
        "renderer.setClearColor(dark?0x101716:0xf7f5ef)}"
        "theme.onchange=applyTheme;scheme.addEventListener('change',applyTheme);applyTheme();"
    )
    if html.count(renderer_anchor) != 1:
        raise ValueError("frozen viewer renderer anchor changed")
    path.write_text(
        html.replace(renderer_anchor, renderer_replacement), encoding="utf-8"
    )


def archive_sources(output: Path, manifest: Path) -> dict[str, Any]:
    """Archive the exact renderer and local wrapper/protocol used for this output."""
    sources = output / "sources"
    sources.mkdir()
    copied = {
        "frozen_renderer": (FROZEN_RENDERER, sources / FROZEN_RENDERER.name),
        "wrapper": (Path(__file__), sources / Path(__file__).name),
        "protocol": (RENDER_PROTOCOL, sources / RENDER_PROTOCOL.name),
        "manifest": (manifest, sources / manifest.name),
    }
    receipt: dict[str, Any] = {}
    for name, (source, destination) in copied.items():
        shutil.copy2(source, destination)
        if sha256(source) != sha256(destination):
            raise OSError(f"archived source hash mismatch: {source}")
        receipt[name] = {"source": record(source), "archived": record(destination)}
    return receipt


def main(cfg: Config) -> None:
    """Render the fixed four-arm comparison without running any physics."""
    global COMPLETED  # noqa: PLW0603
    output = cfg.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"choose an empty output directory: {output}")
    manifest_path = cfg.manifest.resolve()
    document, raw_cases = read_manifest(manifest_path)
    renderer = load_script("frozen_render_diagnosis_50", FROZEN_RENDERER)
    source_receipt = archive_sources(output, manifest_path)
    shutil.copytree(renderer.ROOT / "site/vendor", output / "vendor")

    reference_path = resolve(manifest_path, str(document["reference_vtu"]))
    reference = reference_grid(renderer, reference_path)
    target, target_receipt = target_surface(renderer, reference_path)
    cases: list[dict[str, Any]] = []
    endpoint_surfaces: list[pv.PolyData] = []
    labels: list[str] = []
    input_receipts: list[dict[str, Any]] = []
    all_grids: list[pv.UnstructuredGrid] = [reference]

    for raw in raw_cases:
        endpoint_path = resolve(manifest_path, str(raw["endpoint_vtu"]))
        summary_path = resolve(manifest_path, str(raw["summary_json"]))
        expected_step = int(raw.get("expected_final_step", 64))
        if expected_step != 64:
            raise ValueError("this frozen comparison requires actual final step 64")
        endpoint = load_grid(renderer, endpoint_path)
        renderer.topology(reference, endpoint, raw["id"])
        summary_receipt = validate_summary(summary_path, expected_step)
        history = raw.get("history")
        if not isinstance(history, list) or len(history) != len(HISTORY_STEPS):
            raise ValueError(f"{raw['id']}: history must list five explicit VTUs")
        history_steps = tuple(int(item.get("step", -1)) for item in history)
        if history_steps != HISTORY_STEPS:
            raise ValueError(f"{raw['id']}: history steps changed")
        history_states: list[dict[str, Any]] = []
        history_receipts: list[dict[str, Any]] = []
        for item in history:
            step = int(item["step"])
            frame_path = resolve(manifest_path, str(item["vtu"]))
            expected_name = f"step-{step:04d}.vtu"
            if (
                frame_path.parent != endpoint_path.parent
                or frame_path.name != expected_name
            ):
                raise ValueError(
                    f"{raw['id']}: step {step} is not {endpoint_path.parent / expected_name}"
                )
            paired_npz = frame_path.with_suffix(".npz")
            if not paired_npz.is_file():
                raise FileNotFoundError(paired_npz)
            with np.load(paired_npz) as saved:
                saved_step = int(saved["step"])
            if saved_step != step:
                raise ValueError(
                    f"{raw['id']}: paired NPZ says step {saved_step}, expected {step}"
                )
            frame = load_grid(renderer, frame_path)
            renderer.topology(reference, frame, f"{raw['id']} step {step}")
            if step == 0:
                frame_rest = np.asarray(
                    frame.point_data["RestPosition"], dtype=np.float64
                )
                if not np.array_equal(frame_rest, reference.points):
                    raise ValueError(f"{raw['id']}: step-0 RestPosition changed")
            if step == expected_step and not np.array_equal(
                frame.points, endpoint.points
            ):
                raise ValueError(f"{raw['id']}: step-64 and final geometry differ")
            history_states.append(
                {
                    "step": step,
                    "label": f"optimization step {step}",
                    "state": renderer.arrays_for_browser(
                        renderer.skin(frame), "DominantMaterialPhase"
                    ),
                }
            )
            history_receipts.append(
                {
                    "step": step,
                    "vtu": record(frame_path),
                    "paired_npz": record(paired_npz),
                }
            )
        states = {
            "reference": renderer.arrays_for_browser(
                renderer.skin(reference), "DominantMaterialPhase"
            ),
            "endpoint": renderer.arrays_for_browser(
                renderer.skin(endpoint), "DominantMaterialPhase"
            ),
            "target_skin": renderer.arrays_for_browser(target.copy(deep=True), None),
        }
        receipt_path = output / f"{raw['id']}-receipt.json"
        case_receipt = {
            "id": raw["id"],
            "label": raw["label"],
            "reference": record(reference_path),
            "endpoint": record(endpoint_path),
            "run_summary": summary_receipt,
            "history": history_receipts,
            "topology": {
                "points": int(reference.n_points),
                "tetrahedra": int(reference.n_cells),
            },
            "static_surface": "saved endpoint IsFace exterior triangles",
            "browser_surface": "full exterior boundary extracted from unchanged saved VTU",
            "viewer_material": {
                "name": "DominantMaterialPhase",
                "definition": "argmax of saved FatFraction, MuscleFraction, AponeurosisFraction",
            },
            "cutaway": None,
            "deformation_scale": 1.0,
        }
        write_json(receipt_path, case_receipt)
        cases.append(
            {
                "id": raw["id"],
                "label": raw["label"],
                "states": states,
                "history_states": history_states,
                "receipt": receipt_path.name,
            }
        )
        endpoint_surfaces.append(renderer.selected_face_skin(endpoint, "IsFace"))
        labels.append(raw["label"])
        input_receipts.append(case_receipt)
        all_grids.append(endpoint)

    endpoint_surfaces.append(target)
    labels.append(str(document.get("target_label", "Smile target")))
    viewer_bounds = renderer.union_bounds(all_grids)
    viewer_front_camera = renderer.camera(viewer_bounds, reference)
    static_front_camera = renderer.camera(camera_bounds(endpoint_surfaces), reference)
    mouth_camera = renderer.mouth_camera(reference)
    three_quarter_camera = renderer.three_quarter_camera(viewer_front_camera, reference)
    static = {
        "front": render_static_grid(
            renderer,
            endpoint_surfaces,
            labels,
            static_front_camera,
            output / "face-comparison-front",
            "Actual final-step skins and exact target · common front camera · true scale",
        ),
        "mouth": render_static_grid(
            renderer,
            endpoint_surfaces,
            labels,
            mouth_camera,
            output / "face-comparison-mouth",
            "Actual final-step mouth surfaces and exact target · common camera · true scale",
        ),
    }
    viewer_path = renderer.write_viewer(
        output,
        str(document.get("title", "Tensor active-stress face comparison")),
        cases,
        {
            "front": viewer_front_camera,
            "three_quarter": three_quarter_camera,
            "mouth": mouth_camera,
        },
        reference,
    )
    add_viewer_theme_controls(viewer_path)
    viewer_manifest = json.loads((output / "viewer-manifest.json").read_text())
    if viewer_manifest.get("fibers") is not None:
        raise ValueError("step-0 reference unexpectedly produced viewer fiber data")
    output_inventory = {
        str(path.relative_to(output)): {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in sorted(output.rglob("*"))
        if path.is_file() and path.name != "summary.json"
    }

    summary = {
        "schema_version": 1,
        "status": "completed",
        "scope": (
            "saved-geometry rendering only; no physics solve, interpolation, "
            "deformation exaggeration, cutaway, or target interior state"
        ),
        "inputs": {
            "manifest": record(manifest_path),
            "reference_step0": record(reference_path),
            "target": target_receipt,
            "cases": input_receipts,
        },
        "sources": source_receipt,
        "history_steps": list(HISTORY_STEPS),
        "static": static,
        "viewer": {
            "html": record(viewer_path),
            "manifest": record(output / "viewer-manifest.json"),
            "material_toggle": "DominantMaterialPhase inferred from saved fractions",
            "fiber_data": None,
            "cutaway": None,
            "theme": "system/light/dark selector; ?theme=light or ?theme=dark supported",
            "geometry": (
                "full exterior surfaces extracted from unchanged full-tetrahedron "
                "step and final VTUs"
            ),
        },
        "output_inventory": output_inventory,
    }
    summary_path = output / "summary.json"
    write_json(summary_path, summary)
    # ``output_dir`` is already queued as one Cherries-managed directory.
    # Re-logging its children collides with the Local plugin's copied tree.
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
