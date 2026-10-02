# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, TRY003
"""Render manifest-pinned saved continuation geometry without numerical work."""

from __future__ import annotations

import contextlib
import csv
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
EARLIER = EXPERIMENT.parent / "face-actuation-diagnosis"
FROZEN_RENDERER = EARLIER / "src/50-render-diagnosis.py"
DEFAULT_MANIFEST = EXPERIMENT / "docs/85-render-manifest.json"
DEFAULT_OUTPUT = EXPERIMENT / "data/85-continuation-render"
STATIC_COLOR = "#d9a486"
STATIC_BACKGROUND = "#fbfaf7"
COMPLETED = False


class Config(cherries.BaseConfig):
    """Completed-only render configuration."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    manifest: Path = DEFAULT_MANIFEST
    output_dir: Path | None = None


def sha256(path: Path) -> str:
    """Return a streaming SHA-256."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    """Describe a source or output artifact."""
    if not path.is_file():
        raise FileNotFoundError(path)
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
    """Load the old pure saved-geometry exporter quietly."""
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


def read_json(path: Path) -> dict[str, Any]:
    """Read an object-only JSON document."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"object required: {path}")
    return value


def pinned(manifest: Path, value: Any, suffix: str | None = None) -> Path:
    """Resolve and hash-check one manifest-pinned immutable input."""
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("path"), str)
        or not isinstance(value.get("sha256"), str)
    ):
        raise TypeError("pinned input requires path and sha256")
    path = (manifest.parent / value["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if suffix is not None and path.suffix != suffix:
        raise ValueError(f"expected {suffix}: {path}")
    if "latest" in path.name.lower():
        raise ValueError(f"moving latest artifact forbidden: {path}")
    if sha256(path) != value["sha256"]:
        raise ValueError(f"pinned hash differs: {path}")
    return path


def require_equal(actual: Any, expected: Any, label: str) -> None:
    """Reject an unexpected saved-state contract value."""
    if actual != expected:
        raise ValueError(f"{label} differs")


def validate_manifest(
    document: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate generic cases and panel membership before opening files."""
    require_equal(document.get("schema_version"), 1, "schema_version")
    if not isinstance(document.get("title"), str) or not isinstance(
        document.get("target_label"), str
    ):
        raise TypeError("title and target_label are required")
    cases, panels = document.get("cases"), document.get("static_panels")
    if (
        not isinstance(cases, list)
        or not cases
        or not isinstance(panels, list)
        or not panels
    ):
        raise ValueError("nonempty cases and static_panels are required")
    ids: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise TypeError("case must be an object")
        for key in ("id", "label", "kind", "endpoint", "history"):
            if key not in case:
                raise ValueError(f"case lacks {key}")
        if not isinstance(case["id"], str) or case["id"] in ids:
            raise ValueError("case IDs must be unique nonempty strings")
        ids.add(case["id"])
        if case["kind"] not in {"completed_endpoint", "parent_or_intermediate"}:
            raise ValueError(f"unsupported case kind: {case['kind']}")
        if (
            not isinstance(case["endpoint"], dict)
            or not isinstance(case["history"], list)
            or not case["history"]
        ):
            raise TypeError(f"case {case['id']} needs endpoint and nonempty history")
        if case["kind"] == "completed_endpoint":
            if not isinstance(case.get("source80_endpoint_id"), str):
                raise ValueError(f"case {case['id']} lacks source80_endpoint_id")
        else:
            evidence = case.get("evidence")
            if (
                not isinstance(evidence, dict)
                or not isinstance(evidence.get("reason"), str)
                or not evidence["reason"]
            ):
                raise ValueError(
                    f"case {case['id']} needs concrete original-state evidence reason"
                )
    for panel in panels:
        if (
            not isinstance(panel, dict)
            or not isinstance(panel.get("id"), str)
            or not isinstance(panel.get("title"), str)
        ):
            raise TypeError("panel needs id and title")
        selected = panel.get("case_ids")
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) > 5
            or len(set(selected)) != len(selected)
        ):
            raise ValueError("panel needs one to five unique case IDs")
        if not set(selected) <= ids:
            raise ValueError(f"panel {panel['id']} names unknown case")
    return cases, panels


def reference_grid(renderer: ModuleType, path: Path) -> pv.UnstructuredGrid:
    """Use only the saved PSD step-0 rest state as shared reference."""
    expected = (EXPERIMENT / "data/21-psd/step-0000.vtu").resolve()
    if path != expected:
        raise ValueError("shared reference must be data/21-psd/step-0000.vtu")
    grid = renderer.state_grid(path, "rest_position_array", "RestPosition")
    rest = np.asarray(grid.point_data["RestPosition"], dtype=np.float64)
    if not np.array_equal(np.asarray(grid.points, dtype=np.float64), rest):
        raise ValueError("reference points differ from saved RestPosition")
    if "ActivationFiber" in grid.cell_data:
        raise ValueError("reference unexpectedly carries fiber display data")
    return grid


def verify_state(
    renderer: ModuleType,
    reference: pv.UnstructuredGrid,
    npz_path: Path,
    vtu_path: Path,
    expected_step: int,
    label: str,
) -> pv.UnstructuredGrid:
    """Bind a saved NPZ Q/u state exactly to its actual VTU geometry."""
    with np.load(npz_path, allow_pickle=False) as saved:
        required = {"q", "u", "Q", "active_ids", "step", "solver_valid"}
        missing = required - set(saved.files)
        if missing:
            raise ValueError(f"{label} NPZ lacks {sorted(missing)}")
        if int(saved["step"]) != expected_step or not bool(saved["solver_valid"]):
            raise ValueError(f"{label} has wrong step or invalid solver state")
        q = np.asarray(saved["q"])
        u = np.asarray(saved["u"])
        stress = np.asarray(saved["Q"])
        active_ids = np.asarray(saved["active_ids"])
    if (
        q.shape != (288235, 6)
        or u.shape != reference.points.shape
        or stress.shape != (288235, 3, 3)
    ):
        raise ValueError(f"{label} NPZ array shape changed")
    if (
        not np.isfinite(q).all()
        or not np.isfinite(u).all()
        or not np.isfinite(stress).all()
    ):
        raise FloatingPointError(f"{label} NPZ contains nonfinite state")
    grid = renderer.state_grid(vtu_path, "positions_are_state", "Displacement")
    renderer.topology(reference, grid, label)
    rest = np.asarray(grid.point_data["RestPosition"], dtype=np.float64)
    if not np.array_equal(rest, np.asarray(reference.points, dtype=np.float64)):
        raise ValueError(f"{label} rest geometry differs from reference")
    if not np.allclose(np.asarray(grid.points) - rest, u, rtol=0.0, atol=3e-15):
        raise ValueError(f"{label} VTU points differ from saved u")
    active = np.asarray(grid.cell_data["ActivationMask"], dtype=bool)
    if not np.array_equal(np.flatnonzero(active), active_ids):
        raise ValueError(f"{label} active IDs differ")
    vtk_stress = np.asarray(
        grid.cell_data["ActiveStressMatrixMPa"], dtype=np.float64
    ).reshape(-1, 3, 3)
    if not np.array_equal(vtk_stress[active], stress) or not np.array_equal(
        vtk_stress[~active], np.zeros_like(vtk_stress[~active])
    ):
        raise ValueError(f"{label} VTU stress differs from saved Q")
    return grid


def trace_row(path: Path, step: int) -> dict[str, str]:
    """Return one original runner trace row at an explicit saved step."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    found = [row for row in rows if int(row.get("step", -1)) == step]
    if len(found) != 1 or found[0].get("solver_valid") not in {"True", "true", "1"}:
        raise ValueError(f"trace lacks one solver-valid row at step {step}: {path}")
    return found[0]


def require_trace_state(row: dict[str, Any], grid: pv.UnstructuredGrid) -> None:
    """Bind a solver-valid trace row to metrics recomputed from its saved VTU."""
    detf = np.asarray(grid.cell_data["DetF"], dtype=np.float64)
    checks = {
        "detF_min": float(detf.min()),
        "detF_max": float(detf.max()),
    }
    for key, expected in checks.items():
        if not np.isclose(float(row[key]), expected, rtol=1e-12, atol=1e-12):
            raise ValueError(f"trace {key} differs from saved VTU state")
    if int(row["inverted_tetrahedra"]) != int(np.count_nonzero(detf <= 0)):
        raise ValueError("trace inversion count differs from saved VTU state")


def typed_trace_row(summary_row: dict[str, Any], row: dict[str, str]) -> dict[str, Any]:
    """Type one CSV row with the completed summary's endpoint schema."""
    if row.keys() != summary_row.keys():
        raise ValueError("summary and trace endpoint schemas differ")
    typed: dict[str, Any] = {}
    for key, expected in summary_row.items():
        raw = row[key]
        if expected is None:
            actual = None if raw == "" else float(raw)
        elif isinstance(expected, bool):
            if raw not in {"True", "true", "1", "False", "false", "0"}:
                raise ValueError(f"invalid trace Boolean: {key}")
            actual = raw in {"True", "true", "1"}
        elif isinstance(expected, int):
            actual = int(raw)
        elif isinstance(expected, float):
            actual = float(raw)
        else:
            actual = raw
        typed[key] = actual
    return typed


def verify_original_evidence(
    manifest: Path,
    case: dict[str, Any],
    endpoint_step: int,
    endpoint_npz: Path,
    endpoint_vtu: Path,
    grid: pv.UnstructuredGrid,
) -> dict[str, Any]:
    """Bind parent/intermediate geometry to its original summary and trace."""
    evidence = case["evidence"]
    summary_path = pinned(manifest, evidence.get("summary_json"), ".json")
    trace_path = pinned(manifest, evidence.get("trace_csv"), ".csv")
    run = summary_path.parent
    if (
        trace_path.parent != run
        or endpoint_npz.parent != run
        or endpoint_vtu.parent != run
    ):
        raise ValueError(
            "parent/intermediate evidence must belong to one run directory"
        )
    summary = read_json(summary_path)
    if summary.get("status") not in {
        "completed_fixed_budget",
        "completed_fixed_budget_continuation",
    }:
        raise ValueError(f"original evidence is not completed: {summary_path}")
    primary = summary.get("primary_endpoint")
    if not isinstance(primary, dict) or primary.get("solver_valid") is not True:
        raise ValueError("completed run lacks a solver-valid primary endpoint")
    row = typed_trace_row(primary, trace_row(trace_path, endpoint_step))
    require_trace_state(row, grid)
    if endpoint_step == int(primary["step"]):
        if endpoint_npz.name != "final.npz" or endpoint_vtu.name != "final.vtu":
            raise ValueError("final parent endpoint must use the run's final NPZ/VTU")
        require_equal(row, primary, "summary/trace primary endpoint")
    else:
        stem = f"step-{endpoint_step:04d}"
        if endpoint_npz.name != f"{stem}.npz" or endpoint_vtu.name != f"{stem}.vtu":
            raise ValueError("intermediate endpoint must use its exact named NPZ/VTU")
        initial_step = int(summary.get("initial_endpoint", {}).get("step", -1))
        if not initial_step < endpoint_step < int(primary["step"]):
            raise ValueError("intermediate endpoint lies outside the completed run")
    return {
        "reason": evidence["reason"],
        "summary": record(summary_path),
        "trace": record(trace_path),
        "trace_row": row,
        "run_directory": str(run.resolve()),
        "selected_step": endpoint_step,
        "endpoint_is_primary": endpoint_step == int(primary["step"]),
    }


def verify_source80(
    manifest: Path,
    document: dict[str, Any],
    case: dict[str, Any],
    endpoint_vtu: Path,
    endpoint_npz: Path,
) -> dict[str, Any]:
    """Require a named completed endpoint in the verified source80 receipt."""
    root = document.get("verified_comparison")
    if not isinstance(root, dict):
        raise TypeError("verified_comparison is required for completed endpoint cases")
    summary_path = pinned(manifest, root.get("summary_json"), ".json")
    summary = read_json(summary_path)
    if summary.get("status") != "completed_postprocessing":
        raise ValueError("source80 comparison is not completed")
    matches = [
        row
        for row in summary.get("endpoints", [])
        if row.get("id") == case["source80_endpoint_id"]
    ]
    if len(matches) != 1:
        raise ValueError(f"source80 endpoint is absent: {case['source80_endpoint_id']}")
    files = matches[0].get("files", {})
    for name, path in (("final.vtu", endpoint_vtu), ("final.npz", endpoint_npz)):
        if files.get(name, {}).get("sha256") != sha256(path):
            raise ValueError(f"source80 endpoint hash differs for {case['id']}: {name}")
    return {
        "summary": record(summary_path),
        "endpoint_id": case["source80_endpoint_id"],
    }


def target_skin(
    renderer: ModuleType, reference_path: Path
) -> tuple[pv.PolyData, dict[str, Any]]:
    """Create exact finite target skin only, never target volume tissue."""
    reference = pv.read(reference_path)
    selected = np.asarray(reference.point_data["IsFace"], dtype=bool)
    displacement = np.asarray(
        reference.point_data["TargetDisplacement"], dtype=np.float64
    )
    if not np.isfinite(displacement[selected]).all():
        raise ValueError("selected target displacement is nonfinite")
    skin, counts = renderer.target_skin(
        reference_path,
        "TargetDisplacement",
        "IsFace",
        rest_position_array="RestPosition",
    )
    if not np.isfinite(np.asarray(skin.points)).all():
        raise ValueError("target skin points are nonfinite")
    return skin, {
        "reference": record(reference_path),
        "selector": "IsFace",
        "selected_target_displacement_finite": True,
        "displacement": "TargetDisplacement",
        "counts": counts,
        "target_interior": None,
    }


def camera_bounds(
    surfaces: list[pv.PolyData],
) -> tuple[float, float, float, float, float, float]:
    """Build one shared physical bounds tuple for a static panel."""
    bounds = np.asarray([surface.bounds for surface in surfaces])
    return (
        float(bounds[:, 0].min()),
        float(bounds[:, 1].max()),
        float(bounds[:, 2].min()),
        float(bounds[:, 3].max()),
        float(bounds[:, 4].min()),
        float(bounds[:, 5].max()),
    )


def static_grid(
    renderer: ModuleType,
    surfaces: list[pv.PolyData],
    labels: list[str],
    camera: dict[str, Any],
    stem: Path,
    title: str,
) -> dict[str, Any]:
    """Render one-to-five actual skins plus target with true deformation scale."""
    count = len(surfaces)
    if count != len(labels) or not 2 <= count <= 6:
        raise ValueError("static grid supports one to five named cases plus target")
    plotter = pv.Plotter(
        shape=(1, count),
        off_screen=True,
        window_size=(600 * count, 720),
        lighting="three lights",
    )
    for column, surface in enumerate(surfaces):
        plotter.subplot(0, column)
        renderer.apply_camera(plotter, camera)
        plotter.add_mesh(
            surface, color=STATIC_COLOR, smooth_shading=True, show_edges=False
        )
    image = plotter.screenshot(return_img=True)
    plotter.close()
    caption_height = 105
    output = Image.new(
        "RGB", (image.shape[1], image.shape[0] + caption_height), STATIC_BACKGROUND
    )
    output.paste(Image.fromarray(image), (0, 0))
    draw = ImageDraw.Draw(output)
    title_font, label_font = (
        ImageFont.truetype("DejaVuSans.ttf", 22),
        ImageFont.truetype("DejaVuSans.ttf", 16),
    )
    draw.text((24, image.shape[0] + 8), title, fill="#172321", font=title_font)
    width = image.shape[1] / count
    for column, label in enumerate(labels):
        for row, line in enumerate(
            textwrap.wrap(label, width=max(12, int(width // 15)))[:2]
        ):
            box = draw.textbbox((0, 0), line, font=label_font)
            draw.text(
                (
                    (column + 0.5) * width - (box[2] - box[0]) / 2,
                    image.shape[0] + 45 + row * 24,
                ),
                line,
                fill="#172321",
                font=label_font,
            )
    png, pdf = stem.with_suffix(".png"), stem.with_suffix(".pdf")
    output.save(png)
    output.save(pdf, "PDF", resolution=180.0)
    return {
        "png": record(png),
        "pdf": record(pdf),
        "camera": camera,
        "material": f"uniform {STATIC_COLOR}",
        "deformation_scale": 1.0,
    }


def add_viewer_controls(path: Path) -> None:
    """Retain old exporter material toggle; add theme and hide unsupported tools."""
    html = path.read_text(encoding="utf-8")
    style = "<style>html,body{margin:0;height:100%;background:#f7f5ef;color:#182422;font:15px system-ui}"
    replacement = "<style>:root{color-scheme:light}:root[data-theme=dark]{color-scheme:dark}@media(prefers-color-scheme:dark){:root:not([data-theme]){color-scheme:dark}:root:not([data-theme]) body{background:#101716;color:#edf4f1}}html,body{margin:0;height:100%;background:#f7f5ef;color:#182422;font:15px system-ui}:root[data-theme=dark] body{background:#101716;color:#edf4f1}"
    if html.count(style) != 1:
        raise ValueError("old viewer style anchor changed")
    html = html.replace(style, replacement)
    header = '<header><strong id="title">Saved geometry</strong>'
    if html.count(header) != 1:
        raise ValueError("old viewer header anchor changed")
    html = html.replace(
        header,
        header
        + '<label>theme <select id="theme"><option value="system">system</option><option value="light">light</option><option value="dark">dark</option></select></label>',
    )
    for control in (
        '<label><input id="cutaway" type="checkbox"> fixed-rest cohort cutaway</label>',
        '<label><input id="fiber" type="checkbox"> prepared geometry-estimated fibers</label>',
        '<label>muscle region <select id="fiberRegion"><option value="all">all sampled</option></select></label>',
    ):
        if html.count(control) != 1:
            raise ValueError("old viewer unsupported-control anchor changed")
        html = html.replace(control, f"<span hidden>{control}</span>")
    anchor = "const renderer=new THREE.WebGLRenderer({canvas,antialias:true}); renderer.setPixelRatio(Math.min(devicePixelRatio,2)); renderer.setClearColor(0xf7f5ef);"
    themed = "const theme=document.querySelector('#theme'),prefersDark=matchMedia('(prefers-color-scheme:dark)');const renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));function applyTheme(){const value=theme.value;if(value==='system')delete document.documentElement.dataset.theme;else document.documentElement.dataset.theme=value;const dark=value==='dark'||(value==='system'&&prefersDark.matches);renderer.setClearColor(dark?0x101716:0xf7f5ef)}theme.onchange=applyTheme;prefersDark.addEventListener('change',applyTheme);applyTheme();"
    if html.count(anchor) != 1:
        raise ValueError("old viewer renderer anchor changed")
    path.write_text(html.replace(anchor, themed), encoding="utf-8")


def main(cfg: Config) -> None:
    """Render manifest-pinned saved meshes and browser geometry only."""
    global COMPLETED  # noqa: PLW0603
    manifest_path = cfg.manifest.resolve()
    document = read_json(manifest_path)
    raw_cases, panels = validate_manifest(document)
    output = (
        cfg.output_dir
        or (manifest_path.parent / document.get("output_dir", str(DEFAULT_OUTPUT)))
    ).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"output must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    resolved_config = {
        "manifest": str(manifest_path),
        "output_dir": str(output),
    }
    config_path = output / "config.json"
    write_json(config_path, resolved_config)
    source_dir = output / "sources"
    source_dir.mkdir()
    source_paths = {
        "wrapper": Path(__file__).resolve(),
        "protocol": (EXPERIMENT / "docs/85-render-protocol.md").resolve(),
        "frozen_renderer": FROZEN_RENDERER.resolve(),
    }
    source_archive: dict[str, Any] = {}
    for name, source in source_paths.items():
        destination = source_dir / f"{name}{source.suffix}"
        shutil.copy2(source, destination)
        source_archive[name] = {
            "source": record(source),
            "archived": record(destination),
        }
        require_equal(
            source_archive[name]["source"]["sha256"],
            source_archive[name]["archived"]["sha256"],
            f"archived source {name}",
        )
    cherries.log_input(manifest_path)
    renderer = load_script("continuation_old50", FROZEN_RENDERER)
    shutil.copytree(renderer.ROOT / "site/vendor", output / "vendor")
    reference_info = document.get("reference")
    if not isinstance(reference_info, dict):
        raise TypeError("reference is required")
    reference_path = pinned(manifest_path, reference_info.get("vtu"), ".vtu")
    reference_npz = pinned(manifest_path, reference_info.get("npz"), ".npz")
    reference = reference_grid(renderer, reference_path)
    verify_state(
        renderer, reference, reference_npz, reference_path, 0, "shared reference"
    )
    target, target_receipt = target_skin(renderer, reference_path)
    loaded: dict[str, dict[str, Any]] = {}
    viewer_cases: list[dict[str, Any]] = []
    all_grids = [reference]
    for raw in raw_cases:
        endpoint = raw["endpoint"]
        if not isinstance(endpoint, dict) or not isinstance(endpoint.get("step"), int):
            raise TypeError(f"case {raw['id']} endpoint needs integer step")
        endpoint_npz = pinned(manifest_path, endpoint.get("npz"), ".npz")
        endpoint_vtu = pinned(manifest_path, endpoint.get("vtu"), ".vtu")
        grid = verify_state(
            renderer, reference, endpoint_npz, endpoint_vtu, endpoint["step"], raw["id"]
        )
        proof = (
            verify_source80(manifest_path, document, raw, endpoint_vtu, endpoint_npz)
            if raw["kind"] == "completed_endpoint"
            else verify_original_evidence(
                manifest_path,
                raw,
                endpoint["step"],
                endpoint_npz,
                endpoint_vtu,
                grid,
            )
        )
        history_states, history_receipts, previous_step = [], [], -1
        for item in raw["history"]:
            if (
                not isinstance(item, dict)
                or not isinstance(item.get("step"), int)
                or item["step"] <= previous_step
            ):
                raise ValueError(
                    f"case {raw['id']} history must have increasing explicit steps"
                )
            previous_step = item["step"]
            frame_npz, frame_vtu = (
                pinned(manifest_path, item.get("npz"), ".npz"),
                pinned(manifest_path, item.get("vtu"), ".vtu"),
            )
            frame = verify_state(
                renderer,
                reference,
                frame_npz,
                frame_vtu,
                item["step"],
                f"{raw['id']} history {item['step']}",
            )
            history_states.append(
                {
                    "step": item["step"],
                    "label": f"saved global step {item['step']}",
                    "state": renderer.arrays_for_browser(
                        renderer.skin(frame), "DominantMaterialPhase"
                    ),
                }
            )
            history_receipts.append(
                {
                    "step": item["step"],
                    "npz": record(frame_npz),
                    "vtu": record(frame_vtu),
                }
            )
        if (
            history_receipts[-1]["step"] != endpoint["step"]
            or history_receipts[-1]["npz"]["sha256"] != sha256(endpoint_npz)
            or history_receipts[-1]["vtu"]["sha256"] != sha256(endpoint_vtu)
        ):
            raise ValueError(
                f"case {raw['id']} history must end at exact endpoint state"
            )
        receipt = {
            "id": raw["id"],
            "label": raw["label"],
            "kind": raw["kind"],
            "endpoint": {
                "step": endpoint["step"],
                "npz": record(endpoint_npz),
                "vtu": record(endpoint_vtu),
            },
            "proof": proof,
            "history": history_receipts,
            "reference": record(reference_path),
            "static_surface": "actual IsFace exterior triangles",
            "browser_surface": "full exterior boundary from unchanged VTU",
            "deformation_scale": 1.0,
            "fibers": None,
            "cutaway": None,
        }
        receipt_path = output / f"{raw['id']}-receipt.json"
        write_json(receipt_path, receipt)
        viewer_cases.append(
            {
                "id": raw["id"],
                "label": raw["label"],
                "states": {
                    "reference": renderer.arrays_for_browser(
                        renderer.skin(reference), "DominantMaterialPhase"
                    ),
                    "endpoint": renderer.arrays_for_browser(
                        renderer.skin(grid), "DominantMaterialPhase"
                    ),
                    "target_skin": renderer.arrays_for_browser(
                        target.copy(deep=True), None
                    ),
                },
                "history_states": history_states,
                "receipt": receipt_path.name,
            }
        )
        loaded[raw["id"]] = {
            "grid": grid,
            "face": renderer.selected_face_skin(grid, "IsFace"),
            "receipt": receipt,
        }
        all_grids.append(grid)
    viewer_front = renderer.camera(renderer.union_bounds(all_grids), reference)
    cameras = {
        "front": viewer_front,
        "three_quarter": renderer.three_quarter_camera(viewer_front, reference),
        "mouth": renderer.mouth_camera(reference),
    }
    static: dict[str, Any] = {}
    for panel in panels:
        faces = [loaded[item]["face"] for item in panel["case_ids"]] + [target]
        labels = [
            next(raw["label"] for raw in raw_cases if raw["id"] == item)
            for item in panel["case_ids"]
        ] + [document["target_label"]]
        static_camera = renderer.camera(camera_bounds(faces), reference)
        static[panel["id"]] = {
            "front": static_grid(
                renderer,
                faces,
                labels,
                static_camera,
                output / f"{panel['id']}-front",
                panel["title"] + " · common front camera · true scale",
            ),
            "mouth": static_grid(
                renderer,
                faces,
                labels,
                cameras["mouth"],
                output / f"{panel['id']}-mouth",
                panel["title"] + " · common mouth camera · true scale",
            ),
        }
    viewer = renderer.write_viewer(
        output, document["title"], viewer_cases, cameras, reference
    )
    add_viewer_controls(viewer)
    viewer_manifest = read_json(output / "viewer-manifest.json")
    if viewer_manifest.get("fibers") is not None:
        raise ValueError("continuation viewer must not export fibers")
    inventory = {
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
        "scope": "saved geometry only; no numerical run, interpolation, deformation exaggeration, cutaway, fibers, or target interior state",
        "wrapper": record(Path(__file__).resolve()),
        "resolved_config": {
            "values": resolved_config,
            "receipt": record(config_path),
        },
        "source_archive": source_archive,
        "inputs": {
            "manifest": record(manifest_path),
            "reference": {"vtu": record(reference_path), "npz": record(reference_npz)},
            "target": target_receipt,
            "cases": [loaded[key]["receipt"] for key in loaded],
        },
        "renderer": record(FROZEN_RENDERER),
        "static": static,
        "viewer": {
            "html": record(viewer),
            "manifest": record(output / "viewer-manifest.json"),
            "cameras": cameras,
            "material_toggle": "DominantMaterialPhase",
            "theme": "system/light/dark",
            "fibers": None,
            "cutaway": None,
        },
        "output_inventory": inventory,
        "output": {
            "path": str(output),
            "files_before_summary": len(inventory),
            "bytes_before_summary": sum(item["bytes"] for item in inventory.values()),
        },
    }
    write_json(output / "summary.json", summary)
    cherries.log_output(output / "summary.json")
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
