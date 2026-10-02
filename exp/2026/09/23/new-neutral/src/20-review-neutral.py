"""Render a saved-state review for the new hybrid neutral forward result.

This is a receipt consumer: it reads the immutable frozen-neutral manifests and
the forward endpoint, but never constructs a forward problem or takes a step.
"""

# ruff: noqa: EM102, PLR0915, PT018, RUF005, SLF001, TRY003

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from html import escape
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.ticker import MaxNLocator

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT_SRC = ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"
sys.path.insert(0, str(JOINT_SRC))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402
from joint_data import _collision_geometry  # noqa: E402

OLD_REVIEW = ROOT / "exp/2026/09/22/neutral-newton/src/20-review-neutral.py"
_SPEC = importlib.util.spec_from_file_location(
    "saved_neutral_review_helpers", OLD_REVIEW
)
assert _SPEC is not None and _SPEC.loader is not None
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

WINDOW = (1600, 760)
FORCE_TO_NEWTONS = 1e6
RIGID_COLORS = {"cranium": "#e5d8bb", "mandible": "#cdb787", "eyes": "#91c9da"}


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-active-strain-001"
    output_dir: Path = GROUP / "data/review-active-strain-001"


def _record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def _verified(item: dict[str, Any], label: str) -> Path:
    path = Path(item["path"])
    assert path.is_file(), f"missing {label}: {path}"
    assert sha256(path) == item["sha256"], f"SHA-256 mismatch for {label}: {path}"
    return path


def _camera(meshes: list[pv.DataSet], view: str) -> dict[str, Any]:
    bounds = np.asarray([mesh.bounds for mesh in meshes])
    low, high = (
        np.minimum.reduce(bounds[:, ::2], axis=0),
        np.maximum.reduce(bounds[:, 1::2], axis=0),
    )
    center, span = (low + high) / 2, float(max(high - low))
    if view == "front":
        eye, horizontal = center + (0.0, 0.0, 2.7 * span), high[0] - low[0]
    elif view == "side":
        eye, horizontal = center + (2.7 * span, 0.0, 0.0), high[2] - low[2]
    else:
        raise ValueError(view)
    scale = 1.08 * max((high[1] - low[1]) / 2, horizontal / 4)
    return {
        "position": [eye.tolist(), center.tolist(), [0.0, 1.0, 0.0]],
        "scale": scale,
    }


def _comparison(
    output: Path,
    reference: pv.PolyData,
    solved: pv.PolyData,
    *,
    endpoint_status: str,
    material_label: str,
    rigid_meshes: dict[str, pv.PolyData],
    skin_opacity: float = 1.0,
) -> list[str]:
    names: list[str] = []
    for view in ("front", "side"):
        camera = _camera([reference, solved, *rigid_meshes.values()], view)
        plot = pv.Plotter(off_screen=True, shape=(1, 2), window_size=WINDOW)
        plot.set_background("#f7f7f5")
        plot.enable_depth_peeling(number_of_peels=100, occlusion_ratio=0)
        for column, (title, surface, color) in enumerate(
            (
                ("Constitutive reference", reference, "#737b86"),
                (
                    f"{material_label} endpoint · {endpoint_status}",
                    solved,
                    "#c75f42",
                ),
            )
        ):
            plot.subplot(0, column)
            plot.add_text(
                f"{title}\n{view} · identical true scale",
                position="upper_left",
                font_size=14,
                color="#202124",
            )
            for name, rigid in rigid_meshes.items():
                plot.add_mesh(rigid, color=RIGID_COLORS[name], smooth_shading=True)
            plot.add_mesh(
                surface, color=color, opacity=skin_opacity, smooth_shading=True
            )
            plot.camera_position = camera["position"]
            plot.camera.parallel_projection = True
            plot.camera.parallel_scale = camera["scale"]
            plot.reset_camera_clipping_range()
        name = f"reference-vs-solved-{view}.png"
        plot.show(screenshot=output / name, auto_close=True)
        names.append(name)
    return names


def _trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid trace JSON at line {number}") from error
        assert isinstance(row, dict)
        rows.append(row)
    return rows


def _force_plot(
    output: Path,
    trace: list[dict[str, Any]],
    result: dict[str, Any],
    threshold_raw: float,
) -> str:
    rows = [row for row in trace if isinstance(row.get("force"), int | float)]
    fig, axis = plt.subplots(figsize=(8.5, 4.2), constrained_layout=True)
    if rows:
        x = np.arange(1, len(rows) + 1)
        axis.semilogy(
            x,
            [FORCE_TO_NEWTONS * row["force"] for row in rows],
            marker="o",
            ms=3,
            color="#087d81",
        )
    axis.axhline(
        FORCE_TO_NEWTONS * result["terminal_force"],
        color="#c75f42",
        linestyle="--",
        label="terminal force",
    )
    axis.axhline(
        FORCE_TO_NEWTONS * threshold_raw,
        color="#555",
        linestyle=":",
        label="force threshold",
    )
    axis.set(
        xlabel="Recorded forward event",
        ylabel="Free-force norm (N)",
        title="Saved hybrid forward force trace",
    )
    axis.xaxis.set_major_locator(MaxNLocator(8))
    axis.grid(alpha=0.16)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(fontsize=8)
    name = "force-trace.png"
    fig.savefig(output / name, dpi=180)
    plt.close(fig)
    return name


def _active_assets(
    value: Any, trail: tuple[str, ...] = ()
) -> list[tuple[str, Path, str | None]]:
    result: list[tuple[str, Path, str | None]] = []
    if isinstance(value, dict):
        if isinstance(value.get("path"), str):
            label = "/".join(trail)
            if any(word in label.lower() for word in ("strain", "mapping", "valid")):
                result.append((label, Path(value["path"]), value.get("sha256")))
        for key, item in value.items():
            if key != "path":
                result.extend(_active_assets(item, (*trail, str(key))))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            result.extend(_active_assets(item, (*trail, str(index))))
    return result


def _copy_active_assets(
    output: Path, active_strain: dict[str, Any] | None
) -> list[dict[str, str]]:
    if active_strain is None:
        return []
    assets: list[dict[str, str]] = []
    destination = output / "material-evidence"
    for index, (label, source, expected_hash) in enumerate(
        _active_assets(active_strain)
    ):
        assert source.is_file(), source
        if expected_hash is not None:
            assert sha256(source) == expected_hash, source
        target = destination / f"{index:02d}-{source.name}"
        target.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, target)
        assets.append(
            {
                "label": label,
                "path": str(target.relative_to(output)),
                "sha256": sha256(target),
            }
        )
    return assets


def _copy_material_snapshot(
    output: Path, assets: list[dict[str, str]], path: Path
) -> None:
    if not path.is_file():
        return
    target = output / "material-evidence" / path.name
    if any(item["path"] == str(target.relative_to(output)) for item in assets):
        return
    target.parent.mkdir(exist_ok=True)
    shutil.copyfile(path, target)
    assets.append(
        {
            "label": path.stem.replace("-", " "),
            "path": str(target.relative_to(output)),
            "sha256": sha256(target),
        }
    )


def _inversion_view(
    output: Path, skin: pv.PolyData, volume: pv.UnstructuredGrid, detf: np.ndarray
) -> str | None:
    inverted = np.flatnonzero(detf <= 0)
    if not len(inverted):
        return None
    cells = volume.extract_cells(inverted).extract_surface(algorithm=None).triangulate()
    centers = np.asarray(volume.cell_centers().points)[inverted]
    markers = pv.PolyData(centers).glyph(
        geom=pv.Sphere(radius=0.0015), scale=False, orient=False
    )
    plot = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plot.set_background("#f7f7f5")
    plot.add_text(
        f"INVALID geometry: {len(inverted)} inverted tetrahedra\nred = inverted-cell boundaries and enlarged 1.5 mm center markers",
        position="upper_left",
        font_size=15,
        color="#7b0014",
    )
    plot.add_mesh(
        skin, color="#8a929c", style="wireframe", opacity=0.35, line_width=0.5
    )
    plot.add_mesh(cells, color="#c00024", opacity=0.55, show_edges=True, line_width=1.0)
    plot.add_mesh(markers, color="#c00024")
    camera = _camera([skin], "front")
    plot.camera_position = camera["position"]
    plot.camera.parallel_projection = True
    plot.camera.parallel_scale = camera["scale"]
    plot.reset_camera_clipping_range()
    name = "invalid-inverted-tetrahedra.png"
    plot.show(screenshot=output / name, auto_close=True)
    return name


def _index(output: Path, receipt: dict[str, Any]) -> None:
    result, geometry, collision = (
        receipt["result"],
        receipt["geometry"],
        receipt["collision"],
    )
    valid = bool(result["valid_forward"])
    label = receipt["state_label"]
    material = receipt["material_formulation"]
    reference_description = ""
    if receipt.get("reference_configuration") is not None:
        clearance = receipt["reference_clearance"]
        reference_description = (
            "<p>The constitutive reference was repaired to clear all enabled "
            f"soft-rigid pairs: minimum gap {clearance['minimum_distance_m'] * 1e6:.6f} µm "
            f"≥ d_hat = {clearance['dhat_m'] * 1e6:g} µm. "
            "The reference has zero inverted tetrahedra and starts with zero barrier force. "
            '<a href="reference-volume.vtu">Reference volume</a>, '
            '<a href="reference-skin.vtp">reference skin</a>, '
            '<a href="reference-rebase.json">reference receipt</a>, '
            '<a href="reference-clearance-audit.json">clearance audit</a>.</p>'
        )
    motion_link = (
        '<p id="motion-detail"><a href="motion/">Displacement map and 10x motion view</a> '
        "(visual magnification only).</p>"
        if (output / "motion/index.html").is_file()
        else ""
    )
    captions = {
        "reference-vs-solved-front.png": f"Constitutive reference and endpoint, front view at identical true scale. {label}.",
        "reference-vs-solved-side.png": f"Constitutive reference and endpoint, side view at identical true scale. {label}.",
        "force-trace.png": "Recorded hybrid forward free-force trace.",
        "invalid-inverted-tetrahedra.png": "Diagnostic view: inverted tetrahedra are red. This invalidates the physical geometry gate.",
    }
    images = "".join(
        f'<figure><a href="{escape(name)}"><img src="{escape(name)}" alt="{escape(captions[name])}"></a><figcaption>{escape(captions[name])}</figcaption></figure>'
        for name in receipt["images"]
    )
    material_links = (
        "".join(
            f'<li><a href="{escape(item["path"])}">{escape(item["label"])}</a></li>'
            for item in receipt["material_evidence"]
        )
        or "<li>No mapped strain, mapping, or validation file was recorded in the protocol.</li>"
    )
    status = "pass" if valid else "fail"
    html = f"""<!doctype html><html lang=\"en\"><meta charset=\"utf-8\"><title>New neutral review</title>
<style>body{{font:16px system-ui,sans-serif;margin:2rem auto;max-width:1200px;color:#202124;background:#f7f7f5}}.status{{padding:1rem;background:{"#d9f2e6" if valid else "#ffe2dc"};font-weight:700}}table{{border-collapse:collapse}}td,th{{padding:.45rem .7rem;border:1px solid #bbb;text-align:left}}img{{max-width:100%;height:auto}}figure{{margin:2rem 0}}code{{font-size:.9em}}pre{{overflow-x:auto;white-space:pre}}</style>
<h1>New neutral: {escape(material["label"])} hybrid forward</h1><p class=\"status {status}\">{label}</p>{motion_link}
{reference_description}
<p>Saved-state rendering only. No forward solve was run while creating this review.</p>
<p>{escape(material["description"])}</p><details><summary>Raw active-strain formulation receipt</summary><pre>{escape(material["protocol_json"])}</pre></details><h2>Mapped strain evidence</h2><ul>{material_links}</ul>
<table><tr><th>Solver status</th><td>{escape(str(result["success"]))}</td></tr><tr><th>Valid-forward gate</th><td>{escape(str(valid))}</td></tr><tr><th>Terminal force</th><td>{receipt["terminal_force_n"]:.6g} N</td></tr><tr><th>Force threshold</th><td>{receipt["force_threshold_n"]:.6g} N</td></tr><tr><th>Forward wall time</th><td>{result["forward_wall_seconds"]:.3f} s</td></tr><tr><th>Inverted tetrahedra</th><td>{geometry["inverted_tetrahedra"]}</td></tr><tr><th>Minimum det(F)</th><td>{geometry["detF_min"]:.6g}</td></tr><tr><th>Collision feasible</th><td>{escape(str(collision["state_feasible"]))}</td></tr><tr><th>Triangle intersections</th><td>{receipt["triangle_intersection_pairs"]}</td></tr></table>
<p>Downloads: <a href=\"neutral-volume.vtu\">solved volume</a>, <a href=\"neutral-skin.vtp\">solved skin</a>, <a href=\"receipt.json\">review receipt</a>, <a href=\"independent-audit.json\">independent audit</a>, <a href=\"forward-summary.json\">solver summary</a>, <a href=\"forward-protocol.json\">solver protocol</a>, <a href=\"forward-trace.jsonl\">solver trace</a>.</p>{images}</html>"""
    (output / "index.html").write_text(html)


def _endpoint_status(result: dict[str, Any], detf: np.ndarray) -> str:
    if result["valid_forward"]:
        return "valid forward endpoint"
    if np.any(detf <= 0):
        return "invalid geometry"
    if not result["success"]:
        return "unconverged force residual; geometry passed"
    return "contact validity failed"


def main(cfg: Config) -> None:
    run_dir, output = cfg.run_dir.resolve(), cfg.output_dir.resolve()
    assert not output.exists(), f"refusing to overwrite {output}"
    summary_path, protocol_path, endpoint_path, trace_path, audit_path = (
        run_dir / name
        for name in (
            "summary.json",
            "protocol.json",
            "endpoint.npz",
            "trace.jsonl",
            "independent-audit.json",
        )
    )
    assert all(
        path.is_file()
        for path in (summary_path, protocol_path, endpoint_path, trace_path, audit_path)
    ), "forward output incomplete"
    summary, protocol = (
        json.loads(summary_path.read_text()),
        json.loads(protocol_path.read_text()),
    )
    assert summary["protocol"] == protocol
    result = summary["result"]
    active_strain = protocol.get("materials", {}).get("active_strain")
    assert active_strain is None or isinstance(active_strain, dict)
    material_label = "Active-strain" if active_strain is not None else "Stored-material"
    material_formulation = {
        "label": material_label,
        "description": (
            "Active-strain formulation: B = I in the bulk; mapped skin prestretch comes from the recorded skin field."
            if active_strain is not None
            else "The saved protocol does not expose an active-strain formulation."
        ),
        "protocol_json": json.dumps(active_strain, indent=2, sort_keys=True)
        if active_strain is not None
        else "null",
    }
    force_threshold_raw = float(
        protocol["contact_stiffness_policy"]["effective_force_tolerance"]
    )
    force_threshold_n = FORCE_TO_NEWTONS * force_threshold_raw
    terminal_force_n = FORCE_TO_NEWTONS * float(result["terminal_force"])
    neutral_dir, eyes_dir = (
        Path(protocol["config"]["neutral_dir"]),
        Path(protocol["config"]["eyes_dir"]),
    )
    neutral_manifest = json.loads((neutral_dir / "manifest.json").read_text())
    assert (
        neutral_manifest["schema"] == "joint-frozen-neutral-v1"
        and neutral_manifest["success"]
    )
    state_path = _verified(neutral_manifest["artifacts"]["state.npz"], "frozen state")
    volume_path = _verified(
        neutral_manifest["sources"]["constitutive_volume"], "constitutive volume"
    )
    skin_path = _verified(
        neutral_manifest["sources"]["constitutive_skin"], "constitutive skin"
    )
    original_volume_path = volume_path
    reference_configuration = protocol.get("reference_configuration")
    if protocol["fixture"]["fem_reference_rebased"]:
        assert reference_configuration is not None
        volume_path = _verified(
            reference_configuration["constitutive_volume"],
            "repaired constitutive volume",
        )
        skin_path = _verified(
            reference_configuration["constitutive_skin"], "repaired constitutive skin"
        )
        _verified(reference_configuration["repair_receipt"], "reference repair receipt")
        repair_path = _verified(
            reference_configuration["repair"], "reference repair arrays"
        )
    else:
        assert reference_configuration is None
    geometry_path = _verified(neutral_manifest["sources"]["geometry"], "rigid geometry")
    eyes_manifest = json.loads((eyes_dir / "manifest.json").read_text())
    eyes_path = _verified(eyes_manifest["artifacts"]["eyes.vtp"], "eyes")
    with np.load(endpoint_path, allow_pickle=False) as archive:
        displacement = np.asarray(archive["displacement_m"], dtype=np.float64)
    volume, skin = pv.read(volume_path), pv.read(skin_path).triangulate()
    assert (
        displacement.shape == (volume.n_points, 3) and np.isfinite(displacement).all()
    )
    reference = np.asarray(volume.points).copy()
    with np.load(state_path, allow_pickle=False) as archive:
        adopted_points = np.asarray(archive["neutral_points_m"], dtype=np.float64)
        adopted_displacement = np.asarray(
            archive["neutral_displacement_m"], dtype=np.float64
        )
    original_reference = np.asarray(pv.read(original_volume_path).points)
    reference_roundoff_m = float(
        np.abs(original_reference - (adopted_points - adopted_displacement)).max()
    )
    assert reference_roundoff_m <= 5e-16
    if reference_configuration is not None:
        with np.load(repair_path, allow_pickle=False) as archive:
            assert np.array_equal(original_reference, archive["reference_points_m"])
            assert np.array_equal(reference, archive["repaired_points_m"])
    else:
        assert np.array_equal(reference, original_reference)
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    reference_skin, solved_skin = skin.copy(deep=True), skin.copy(deep=True)
    solved_skin.points = reference[ids] + displacement[ids]
    solved_skin.point_data["DisplacementMagnitudeMm"] = 1000 * np.linalg.norm(
        displacement[ids], axis=1
    )
    cells = np.asarray(volume.cells).reshape(-1, 5)[:, 1:]
    rest_edges, solved_edges = (
        reference[cells[:, 1:]] - reference[cells[:, :1]],
        (reference + displacement)[cells[:, 1:]]
        - (reference + displacement)[cells[:, :1]],
    )
    detf = np.linalg.det(solved_edges) / np.linalg.det(rest_edges)
    assert np.isfinite(detf).all()
    assert np.isclose(
        detf.min(), result["geometry"]["detF_min"], rtol=1e-12, atol=1e-12
    )
    assert np.isclose(
        detf.max(), result["geometry"]["detF_max"], rtol=1e-12, atol=1e-12
    )
    assert int(np.count_nonzero(detf <= 0)) == result["geometry"]["inverted_tetrahedra"]
    solved_volume = volume.copy(deep=True)
    solved_volume.points = reference + displacement
    solved_volume.cell_data["PhysicalDetF"] = detf
    soft = _HELPERS._pure_soft(volume, reference + displacement)
    cranium, mandible = _HELPERS._geometry(geometry_path)
    eyes = pv.read(eyes_path).triangulate()
    intersections = {}
    for name, rigid in {"cranium": cranium, "mandible": mandible, "eyes": eyes}.items():
        pairs, _, _, lengths = _collision_geometry(soft, rigid)
        intersections[name] = {
            "pairs": len(pairs),
            "segment_length_sum_m": float(lengths.sum()),
        }
    output.mkdir(parents=True)
    material_evidence = _copy_active_assets(output, active_strain)
    for name in (
        "active-strain-mapping.json",
        "active-strain-fields.npz",
        "active-strain-field-snapshot.npz",
        "active-strain-validation.json",
    ):
        _copy_material_snapshot(output, material_evidence, run_dir / name)
    for source, name in (
        (summary_path, "forward-summary.json"),
        (protocol_path, "forward-protocol.json"),
        (trace_path, "forward-trace.jsonl"),
        (audit_path, "independent-audit.json"),
    ):
        (output / name).write_bytes(source.read_bytes())
    solved_volume.save(output / "neutral-volume.vtu")
    solved_skin.save(output / "neutral-skin.vtp")
    endpoint_status = _endpoint_status(result, detf)
    images = _comparison(
        output,
        reference_skin,
        solved_skin,
        endpoint_status=endpoint_status,
        material_label=material_label,
        rigid_meshes={"cranium": cranium, "mandible": mandible, "eyes": eyes},
    )
    force_trace = _force_plot(output, _trace(trace_path), result, force_threshold_raw)
    inversion_view = _inversion_view(output, solved_skin, solved_volume, detf)
    if inversion_view is not None:
        images.append(inversion_view)
    images.append(force_trace)
    receipt = {
        "schema": "new-neutral-saved-state-review-v1",
        "state_label": "valid forward endpoint"
        if result["valid_forward"]
        else "diagnostic result; forward validity gate failed",
        "run": {
            name: _record(path)
            for name, path in {
                "summary": summary_path,
                "protocol": protocol_path,
                "endpoint": endpoint_path,
                "trace": trace_path,
                "independent_audit": audit_path,
            }.items()
        },
        "inputs": {
            "frozen_neutral_manifest": _record(neutral_dir / "manifest.json"),
            "rigid_eyes_manifest": _record(eyes_dir / "manifest.json"),
            "constitutive_volume": _record(volume_path),
            "constitutive_skin": _record(skin_path),
            "frozen_state": _record(state_path),
        },
        "reference_coordinate_contract": {
            "definition": "constitutive volume coordinates equal frozen neutral_points_m minus frozen neutral_displacement_m to floating-point roundoff",
            "maximum_difference_m": reference_roundoff_m,
        },
        "material_formulation": material_formulation,
        "material_evidence": material_evidence,
        "force_conversion": {"raw_units": "MPa*m^2", "newton_factor": FORCE_TO_NEWTONS},
        "terminal_force_n": terminal_force_n,
        "force_threshold_n": force_threshold_n,
        "result": result,
        "geometry": {
            "detF_min": float(detf.min()),
            "detF_max": float(detf.max()),
            "inverted_tetrahedra": int(np.count_nonzero(detf <= 0)),
            "skin_rms_mm": float(
                1000 * np.sqrt(np.mean(np.sum(displacement[ids] ** 2, axis=1)))
            ),
        },
        "collision": result["collision"],
        "independent_triangle_intersections": intersections,
        "triangle_intersection_pairs": sum(
            item["pairs"] for item in intersections.values()
        ),
        "images": images,
        "assets": [
            *images,
            "neutral-volume.vtu",
            "neutral-skin.vtp",
            "forward-summary.json",
            "forward-protocol.json",
            "forward-trace.jsonl",
            "independent-audit.json",
            *[item["path"] for item in material_evidence],
            "index.html",
        ],
        "scope": "CPU/offscreen saved-state review. It does not run a forward solve. The endpoint displacement is applied to the verified constitutive reference, never the adopted neutral mesh. Triangle-pair checks are independent diagnostic intersections against cranium, mandible, and rigid-eye surfaces; solver validity comes only from the saved forward receipt.",
    }
    receipt["state_label"] = endpoint_status
    if reference_configuration is not None:
        assert reference_configuration["schema"] == "reference-configuration-rebase-v1"
        assert (
            json.loads((run_dir / "reference-rebase.json").read_text())
            == reference_configuration
        )
        reference_audit_path = (
            Path(reference_configuration["repair"]["path"]).parent
            / "independent-audit.json"
        )
        reference_audit = json.loads(reference_audit_path.read_text())
        assert reference_audit["independent_ipc"]["meets_target"]
        assert reference_audit["tetrahedra"]["inverted_tetrahedra"] == 0
        assert reference_configuration["zero_state"]["contact_force_norm_n"] == 0
        receipt["reference_clearance"] = {
            "minimum_distance_m": reference_audit["independent_ipc"][
                "minimum_distance_lower_bound_m"
            ],
            "dhat_m": reference_audit["independent_ipc"]["target_dhat_m"],
            "audit": _record(reference_audit_path),
        }
        receipt["reference_configuration"] = reference_configuration
        receipt["reference_coordinate_contract"] = {
            "definition": "original constitutive coordinates match the historical frozen state; current constitutive coordinates match the repaired reference; endpoint displacement is relative to the repaired reference",
            "maximum_original_reference_difference_m": reference_roundoff_m,
        }
        for source, target in (
            (volume_path, "reference-volume.vtu"),
            (skin_path, "reference-skin.vtp"),
            (run_dir / "reference-rebase.json", "reference-rebase.json"),
            (reference_audit_path, "reference-clearance-audit.json"),
        ):
            shutil.copyfile(source, output / target)
            receipt["assets"].append(target)
    write_json(output / "receipt.json", receipt)
    _index(output, receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "review/valid_forward": float(result["valid_forward"]),
            "review/detF_min": receipt["geometry"]["detF_min"],
            "review/inverted_tetrahedra": receipt["geometry"]["inverted_tetrahedra"],
            "review/triangle_intersection_pairs": receipt[
                "triangle_intersection_pairs"
            ],
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
