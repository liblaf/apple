"""Compare saved June and PSD geometry using white skin and a fixed crinkle cut."""

from __future__ import annotations

import hashlib
import importlib.util
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from experiment_profile import ProfileCometNoCommit
from PIL import Image, ImageDraw, ImageFont

from liblaf import cherries

ROOT = Path(__file__).resolve().parents[1]
DONE = False
BACKGROUND = "#303946"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    manifest: Path = ROOT / "docs/110-white-crinkle-manifest.json"
    output_dir: Path = ROOT / "data/110-white-crinkle"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def array_hash(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def render(
    r85, r50, surfaces, labels, camera, stem, title, *, muscle=False, edges=False
):
    """Render exact positions with one camera, material and light setup per panel."""
    plotter = pv.Plotter(
        shape=(1, len(surfaces)),
        off_screen=True,
        window_size=(750 * len(surfaces), 850),
        lighting="three lights",
        border=False,
    )
    for i, surface in enumerate(surfaces):
        plotter.subplot(0, i)
        plotter.add_mesh(
            surface,
            color="#e4a9a6" if muscle else "white",
            smooth_shading=not muscle,
            show_edges=muscle or edges,
            edge_color="#57515a" if muscle else "#737e8c",
            line_width=0.45,
            ambient=0.22,
            diffuse=0.75,
            specular=0.12,
            specular_power=20,
        )
        r50.apply_camera(plotter, camera)
        plotter.set_background(BACKGROUND)
    raster = plotter.screenshot(return_img=True)
    plotter.close()
    canvas = Image.new("RGB", (raster.shape[1], raster.shape[0] + 116), BACKGROUND)
    canvas.paste(Image.fromarray(raster), (0, 0))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("DejaVuSans.ttf", 23)
    small = ImageFont.truetype("DejaVuSans.ttf", 21)
    draw.text((24, raster.shape[0] + 8), title, fill="white", font=font)
    for i, label in enumerate(labels):
        for row, line in enumerate(label.split("\n")):
            draw.text(
                (750 * (i + 0.5), raster.shape[0] + 51 + 27 * row),
                line,
                anchor="mt",
                fill="#eff2f6",
                font=small,
            )
    for ext in ("png", "pdf"):
        canvas.save(stem.with_suffix("." + ext), resolution=180)
    return {
        "png": r85.record(stem.with_suffix(".png")),
        "pdf": r85.record(stem.with_suffix(".pdf")),
        "camera": camera,
        "material": "uniform muscle rose" if muscle else "uniform white",
        "edges": muscle or edges,
        "deformation_scale": 1.0,
        "geometry_smoothing": False,
    }


def main(cfg: Config):
    global DONE
    log = logging.getLogger(__name__)
    r85 = load("white85", ROOT / "src/85-render-continuations.py")
    manifest_path = cfg.manifest.resolve()
    doc = r85.read_json(manifest_path)
    out = cfg.output_dir.resolve()
    assert not out.exists(), out
    out.mkdir(parents=True)
    Image.init()
    sources = out / "sources"
    sources.mkdir()
    for name, binding in doc["sources"].items():
        source = r85.pinned(manifest_path, binding)
        shutil.copy2(source, sources / (name + source.suffix))
    shutil.copy2(manifest_path, out / "manifest.json")
    cherries.log_input(manifest_path)
    r50 = load("white50", r85.pinned(manifest_path, doc["sources"]["renderer50"]))
    old = load(
        "white11", r85.pinned(manifest_path, doc["sources"]["historical_export11"])
    )
    old_summary_path = r85.pinned(manifest_path, doc["historical_summary"])
    old_summary = r85.read_json(old_summary_path)
    assert old_summary["recorded_june_metrics"]["best_step"] == 194
    for value in old_summary["export_hashes"].values():
        r85.pinned(manifest_path, value)
    for key in ("historical_endpoint", "historical_target", "historical_summary"):
        r85.pinned(manifest_path, old_summary["source_hashes"][key])
    m107_path = r85.pinned(manifest_path, doc["current_render_manifest"])
    m107 = r85.read_json(m107_path)
    refpath = r85.pinned(m107_path, m107["reference"]["vtu"])
    ref = r85.reference_grid(r50, refpath)
    baseline_path = r85.pinned(manifest_path, old_summary["export_hashes"]["final_vtu"])
    baseline_npz = r85.pinned(manifest_path, old_summary["export_hashes"]["final_npz"])
    baseline = r50.state_grid(baseline_path, "positions_are_state", "Displacement")
    r50.topology(ref, baseline, "June baseline")
    with np.load(baseline_npz, allow_pickle=False) as saved:
        assert int(saved["step"]) == 194
        assert np.array_equal(saved["rest_points"], ref.points)
        assert np.allclose(baseline.points - ref.points, saved["u"], atol=3e-15, rtol=0)
        assert np.array_equal(
            saved["active_ids"], np.flatnonzero(ref.cell_data["ActivationMask"])
        )
        active = np.asarray(ref.cell_data["ActivationMask"], dtype=bool)
        assert np.array_equal(
            np.asarray(baseline.cell_data["ActivationInverseMatrix"]).reshape(-1, 3, 3)[
                active
            ],
            saved["Ainv"],
        )
        assert np.allclose(
            old.activation_matrix(saved["q"]), saved["Ainv"], atol=1e-15, rtol=0
        )
    current_case = next(x for x in m107["cases"] if x["id"] == "selected-final")
    current_path = r85.pinned(m107_path, current_case["endpoint"]["vtu"])
    current_npz = r85.pinned(m107_path, current_case["endpoint"]["npz"])
    current = r85.verify_state(r50, ref, current_npz, current_path, 1024, "current PSD")
    current_evidence = r85.verify_source80(
        m107_path, m107, current_case, current_path, current_npz
    )
    for state in (baseline, current):
        for key in (
            "RestPosition",
            "IsFace",
            "IsLip",
            "TargetDisplacement",
            "GlobalPointId",
        ):
            assert np.array_equal(
                ref.point_data[key], state.point_data[key], equal_nan=True
            ), key
        for key in ("ActivationMask", "MuscleId", "MuscleFraction"):
            assert np.array_equal(ref.cell_data[key], state.cell_data[key]), key
    log.info("Verified both saved states against one exact rest topology and target.")

    # The exact same exterior triangles are advected for both fits and the target.
    face = r50.selected_face_skin(ref, "IsFace")
    volume_ids = np.asarray(ref.point_data["GlobalPointId"])
    order = np.argsort(volume_ids)
    face_ids = order[
        np.searchsorted(volume_ids[order], face.point_data["GlobalPointId"])
    ]
    assert np.array_equal(volume_ids[face_ids], face.point_data["GlobalPointId"])
    skins = []
    for state in (baseline, current):
        mesh = face.copy(deep=True)
        mesh.points = state.points[face_ids].copy()
        skins.append(mesh)
    target = face.copy(deep=True)
    target.points = (
        ref.points[face_ids] + ref.point_data["TargetDisplacement"][face_ids]
    )
    assert np.isfinite(target.points).all()
    skins.append(target)
    labels = [
        "Bumpy baseline | June step 194\nFit RMS 0.6543 mm",
        "Current PSD | step 1024\nFit RMS 1.6102 mm",
        "Target skin\nObserved vertices only",
    ]
    front = r50.camera(r85.camera_bounds(skins), ref)
    mouth = r50.mouth_camera(ref)
    figures = {}
    for key, cam, edges in (
        ("skin-front", front, False),
        ("skin-mouth", mouth, False),
        ("skin-mouth-edges", mouth, True),
    ):
        figures[key] = render(
            r85,
            r50,
            skins,
            labels,
            cam,
            out / key,
            "White skin | shared camera and lighting | actual deformation",
            edges=edges,
        )
    log.info("Rendered three white skin comparisons.")

    tets = old.tetrahedra(ref)
    plane = doc["crinkle"]
    normal, origin = np.array(plane["normal"]), np.array(plane["origin"])
    signed = (ref.points - origin) @ normal
    halfspace = signed[tets].min(axis=1) <= 0
    muscle_mask = (np.asarray(ref.cell_data["MuscleId"]) > 0) & (
        np.asarray(ref.cell_data["MuscleFraction"]) > 0
    )
    assert np.array_equal(
        muscle_mask, np.asarray(ref.cell_data["ActivationMask"], dtype=bool)
    )
    cell_ids = np.flatnonzero(halfspace & muscle_mask)
    assert len(cell_ids) == plane["expected_muscle_tetrahedra"]
    assert int(halfspace.sum()) == plane["expected_all_tetrahedra"]
    np.save(out / "crinkle-source-cell-ids.npy", cell_ids)
    interiors = []
    for name, state in (("rest", ref), ("baseline", baseline), ("current", current)):
        selected = state.extract_cells(cell_ids)
        assert np.array_equal(selected.cell_data["vtkOriginalCellIds"], cell_ids)
        selected.cell_data["SourceCellId"] = cell_ids
        surface = selected.extract_surface(algorithm="dataset_surface").triangulate()
        # Keep the real tetrahedron faces, including folds and inversions.
        assert np.isfinite(surface.points).all()
        surface.save(out / f"muscle-{name}.vtp")
        interiors.append(surface)
    labels_muscle = [
        "Rest muscle geometry\nSame retained tetrahedra",
        "Bumpy baseline | June step 194\nWhole mesh: 142 inverted tets",
        "Current PSD | step 1024\nWhole mesh: 0 inverted tets",
    ]
    for key, camera_key in (
        ("muscle-crinkle", "camera"),
        ("muscle-mouth", "mouth_camera"),
    ):
        figures[key] = render(
            r85,
            r50,
            interiors,
            labels_muscle,
            plane[camera_key],
            out / key,
            "Interior muscle | fixed rest crinkle cut | whole tetrahedra with edges",
            muscle=True,
        )
    # A separate locator avoids obscuring the muscle detail with exterior skin.
    context = pv.Plotter(
        off_screen=True, window_size=(1200, 1000), lighting="three lights"
    )
    context.add_mesh(r50.skin(ref), color="white", opacity=0.25)
    context.add_mesh(interiors[0], color="#e4a9a6", smooth_shading=False)
    context.add_mesh(
        pv.Plane(center=origin, direction=normal, i_size=0.15, j_size=0.21),
        color="#69c5e2",
        opacity=0.28,
        show_edges=True,
    )
    locator_camera = dict(plane["camera"])
    locator_camera["position"] = (origin + np.array([0.30, 0.07, 0.27])).tolist()
    r50.apply_camera(context, locator_camera)
    context.set_background(BACKGROUND)
    context.add_text(
        "Rest-space cut locator\nx <= 1.407023565 m | whole intersecting tets",
        font_size=17,
        color="white",
    )
    raster = context.screenshot(return_img=True)
    context.close()
    img = Image.fromarray(raster).convert("RGB")
    img.save(out / "plane-context.png")
    img.save(out / "plane-context.pdf", resolution=180)
    figures["plane-context"] = {
        ext: r85.record(out / ("plane-context." + ext)) for ext in ("png", "pdf")
    }
    log.info("Rendered actual muscle cohort and clip-plane locator.")

    fixture_skin = pv.read(
        r85.pinned(manifest_path, old_summary["source_hashes"]["current_fixture_skin"])
    )
    top = np.flatnonzero(ref.point_data["IsFace"])
    weights = old.surface_weights(ref, fixture_skin, top)
    displacement_target = ref.point_data["TargetDisplacement"][top]
    cases = {}
    for name, state, path, checkpoint, step in (
        ("baseline", baseline, baseline_path, baseline_npz, 194),
        ("current", current, current_path, current_npz, 1024),
    ):
        u = state.points - ref.points
        determinant = old.detf(ref.points, state.points, tets)
        assert np.allclose(determinant, state.cell_data["DetF"], atol=3e-10, rtol=3e-10)
        metrics = {
            "fit_rms_mm": float(
                1000
                * np.sqrt(
                    np.sum(weights[:, None] * (u[top] - displacement_target) ** 2)
                )
            ),
            "motion_rms_mm": float(
                1000 * np.sqrt(np.sum(weights[:, None] * u[top] ** 2))
            ),
            "detF_min": float(determinant.min()),
            "detF_max": float(determinant.max()),
            "inverted_tetrahedra": int((determinant < 0).sum()),
            "crinkle_muscle_inverted_tetrahedra": int(
                (determinant[cell_ids] < 0).sum()
            ),
        }
        expected = (
            old_summary["common_area_weighted_metrics"]
            if name == "baseline"
            else doc["expected_current_metrics"]
        )
        for key in (
            "fit_rms_mm",
            "motion_rms_mm",
            "detF_min",
            "detF_max",
            "inverted_tetrahedra",
        ):
            assert np.isclose(metrics[key], expected[key], rtol=3e-10, atol=3e-10), (
                name,
                key,
            )
        cases[name] = {
            "label": "Bumpy no-skin baseline"
            if name == "baseline"
            else "Current PSD active stress",
            "step": step,
            "vtu": r85.record(path),
            "npz": r85.record(checkpoint),
            "metrics": metrics,
        }
    summary = {
        "schema_version": 1,
        "status": "completed_postprocessing",
        "figures": figures,
        "cases": cases,
        "manifest": r85.record(manifest_path),
        "source": r85.record(Path(__file__)),
        "reference": r85.record(refpath),
        "baseline_summary": r85.record(old_summary_path),
        "current_evidence": current_evidence,
        "crinkle": {
            **plane,
            "source_cell_ids": r85.record(out / "crinkle-source-cell-ids.npy"),
            "source_cell_ids_array_sha256": array_hash(cell_ids),
            "muscle_selector": "MuscleId > 0 and MuscleFraction > 0; equals saved ActivationMask",
            "muscle_ids": np.unique(ref.cell_data["MuscleId"][cell_ids]).tolist(),
            "rule": "min signed distance across the four rest vertices <= 0; retain entire cell",
            "same_source_cell_ids_all_states": True,
            "reclip_deformed_state": False,
        },
        "skin": {
            "selector": "all three exterior triangle vertices have IsFace",
            "points": face.n_points,
            "triangles": face.n_cells,
            "same_triangle_connectivity": True,
            "material": "white",
            "geometry_smoothing": False,
            "normal_shading": "smooth vertex normals only",
        },
        "scope": "Saved geometry only; no optimization, forward solve, repair, interpolation or exaggeration.",
        "limitations": [
            "Endpoints use different activation formulations and optimizer budgets; this is a visual comparison, not a matched causal experiment.",
            "Target skin is observed; no target interior geometry exists.",
            "Crinkle cut boundaries are tetrahedron faces, so their jagged outline is partly mesh discretization.",
        ],
    }
    r85.write_json(out / "summary.json", summary)
    cherries.log_output(out / "summary.json")
    DONE = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not DONE:
        raise SystemExit(1)
