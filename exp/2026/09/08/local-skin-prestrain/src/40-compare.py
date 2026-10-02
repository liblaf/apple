"""Compare saved local-skin-prestrain ablations without re-solving physics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pyvista as pv
from PIL import Image, ImageDraw, ImageFont

mpl.use("Agg")
import matplotlib.pyplot as plt

GROUP = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[6]
FIXTURE = ROOT / "exp/2026/09/07/face-actuation-diagnosis/data/12-historical-fixture"
CAMERAS = ROOT / "exp/2026/09/08/physical-volume-closeups/data/20-regions/summary.json"
CASES = ("no-skin", "skin-zero", "skin-local-1pct")
REGIONS = ("right_mouth_corner", "right_lateral_cheek", "right_lower_cheek_jaw")
DEFAULT_OUTPUT = GROUP / "data/40-comparison"
FIT_TOLERANCE_MM = 0.05
MOTION_TOLERANCE_MM = 0.05
BACKGROUND = "#242c36"


def record(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": digest}


def rows(path: Path) -> list[dict[str, Any]]:
    """Keep runner booleans as booleans and convert its numeric CSV cells."""

    def parse(value: str) -> Any:
        if value == "":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        return float(value)

    with path.open(newline="") as stream:
        result = [
            {key: parse(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    if not result:
        raise ValueError(f"empty trace: {path}")
    return result


def case_dir(stage: str, case: str) -> Path:
    return (
        GROUP / "data" / f"{'20-forward' if stage == 'forward' else '30-refit'}-{case}"
    )


def best_row(trace: list[dict[str, Any]]) -> dict[str, Any]:
    return min(trace, key=lambda row: float(row["objective_mm2"]))


def matched_row(
    trace: list[dict[str, Any]], reference: dict[str, Any]
) -> dict[str, Any]:
    candidates = []
    for row in trace:
        fit_delta = abs(float(row["fit_rms_mm"]) - float(reference["fit_rms_mm"]))
        motion_delta = abs(
            float(row["motion_rms_mm"]) - float(reference["motion_rms_mm"])
        )
        candidates.append((fit_delta, motion_delta, row))
    fit_delta, motion_delta, selected = min(
        candidates,
        key=lambda item: (
            (item[0] / FIT_TOLERANCE_MM) ** 2 + (item[1] / MOTION_TOLERANCE_MM) ** 2
        ),
    )
    return {
        "step": int(selected["step"]),
        "fit_rms_mm": selected["fit_rms_mm"],
        "motion_rms_mm": selected["motion_rms_mm"],
        "fit_delta_mm": fit_delta,
        "motion_delta_mm": motion_delta,
        "within_tolerance": fit_delta <= FIT_TOLERANCE_MM
        and motion_delta <= MOTION_TOLERANCE_MM,
        "selection": "minimum normalized fit-and-motion distance over saved evaluated states; no interpolation",
    }


def common_overlap(
    left: list[dict[str, Any]], right: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """Find an actual pair within both tolerances; no nearest-outside fallback."""
    feasible = []
    for a in left:
        for b in right:
            fit_delta = abs(float(a["fit_rms_mm"]) - float(b["fit_rms_mm"]))
            motion_delta = abs(float(a["motion_rms_mm"]) - float(b["motion_rms_mm"]))
            if fit_delta <= FIT_TOLERANCE_MM and motion_delta <= MOTION_TOLERANCE_MM:
                feasible.append(
                    (
                        0.5 * (float(a["fit_rms_mm"]) + float(b["fit_rms_mm"])),
                        fit_delta,
                        motion_delta,
                        a,
                        b,
                    )
                )
    if not feasible:
        return None
    _, fit_delta, motion_delta, a, b = min(
        feasible, key=lambda item: (item[0], item[1], item[2])
    )
    return {
        "selection": "lowest mean fit among actual pair states within both tolerances",
        "left": a,
        "right": b,
        "fit_delta_mm": fit_delta,
        "motion_delta_mm": motion_delta,
        "fit_tolerance_mm": FIT_TOLERANCE_MM,
        "motion_tolerance_mm": MOTION_TOLERANCE_MM,
    }


def load_surface(directory: Path, step: int, ids: np.ndarray) -> np.ndarray:
    path = directory / f"surface-{step:04d}.npz"
    with np.load(path, allow_pickle=False) as saved:
        if int(saved["step"]) != step or not np.array_equal(saved["point_ids"], ids):
            raise ValueError(f"surface receipt mismatch: {path}")
        u = np.asarray(saved["u"], dtype=np.float64)
    if u.shape != (len(ids), 3) or not np.isfinite(u).all():
        raise ValueError(f"invalid surface displacement: {path}")
    return u


def add_lights(plotter: pv.Plotter, camera: dict[str, Any]) -> None:
    focus = np.asarray(camera["focal_point"], dtype=np.float64)
    backward = np.asarray(camera["position"], dtype=np.float64) - focus
    backward /= np.linalg.norm(backward)
    right = np.cross(np.asarray(camera["view_up"], dtype=np.float64), backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    key = focus + 0.3 * (0.72 * right + 0.35 * up + 0.60 * backward)
    fill = focus + 0.3 * backward
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


def render_states(
    base: pv.PolyData,
    states: list[tuple[str, np.ndarray]],
    view: dict[str, Any],
    path: Path,
) -> None:
    camera = view["camera"]
    resolution = 650
    plotter = pv.Plotter(
        shape=(1, len(states)),
        off_screen=True,
        window_size=(resolution * len(states), resolution),
        lighting="none",
        border=False,
    )
    for column, (label, displacement) in enumerate(states):
        mesh = base.copy(deep=True)
        mesh.points = np.asarray(base.points) + displacement
        plotter.subplot(0, column)
        actor = plotter.add_mesh(
            mesh,
            color="#eeeeea",
            smooth_shading=False,
            ambient=0.20,
            diffuse=0.80,
            specular=0.0,
        )
        if actor.GetProperty().GetInterpolation() != 0:
            raise AssertionError("fixed visual requires flat shading")
        add_lights(plotter, camera)
        plotter.enable_parallel_projection()
        plotter.camera.position = camera["position"]
        plotter.camera.focal_point = camera["focal_point"]
        plotter.camera.up = camera["view_up"]
        plotter.camera.parallel_scale = camera["parallel_scale"]
        plotter.set_background(BACKGROUND)
        plotter.reset_camera_clipping_range()
    raster = plotter.screenshot(return_img=True)
    plotter.close()
    header = 42
    image = Image.new("RGB", (resolution * len(states), resolution + header), "white")
    image.paste(Image.fromarray(raster).convert("RGB"), (0, header))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("DejaVuSans.ttf", 16)
    for column, (label, _) in enumerate(states):
        draw.text((column * resolution + 12, 12), label, fill="#202833", font=font)
    image.save(path)


def triangle_segments(
    points: np.ndarray, triangles: np.ndarray, y: float
) -> np.ndarray:
    """Exact affine skin-triangle intersections at one frozen horizontal plane."""
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
        unique = [
            hit
            for index, hit in enumerate(hits)
            if not any(
                np.allclose(hit, other, rtol=0, atol=1e-13) for other in hits[:index]
            )
        ]
        if len(unique) == 2:
            segment = np.stack(unique)
            if (
                segment[:, 0].max() >= 1.414
                and segment[:, 0].min() <= 1.460
                and segment[:, 2].max() >= 0.040
            ):
                segments.append(segment)
    return np.asarray(segments, dtype=np.float64)


def render_nlf_sections(
    base: pv.PolyData, states: list[tuple[str, np.ndarray, str]], path: Path
) -> None:
    """Three prior frozen NLF planes; segments are neither joined nor smoothed."""
    planes = (2.170, 2.180, 2.190)
    triangles = np.asarray(base.faces).reshape(-1, 4)[:, 1:]
    fig, axes = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)
    for axis, y in zip(axes, planes, strict=True):
        for label, u, color in states:
            for index, segment in enumerate(
                triangle_segments(np.asarray(base.points) + u, triangles, y)
            ):
                axis.plot(
                    segment[:, 0],
                    segment[:, 2],
                    color=color,
                    linewidth=0.7,
                    label=label if index == 0 else None,
                )
        axis.set(
            title=f"Exact skin-triangle section at y = {y:.3f} m",
            xlim=(1.414, 1.460),
            ylim=(0.040, 0.115),
            ylabel="z (m)",
        )
        axis.set_aspect("equal", adjustable="box")
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("x (m)")
    axes[0].legend(loc="upper left", frameon=False, fontsize=7)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_metrics(traces: dict[str, dict[str, list[dict[str, Any]]]], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for stage, marker in (("forward", "o"), ("refit", None)):
        for case in CASES:
            trace = traces[stage][case]
            label = f"{case} | {stage}"
            axes[0].plot(
                [row["fit_rms_mm"] for row in trace],
                [row["roi_right_nose_to_mouth_fit_vector_rms_mm"] for row in trace],
                marker=marker,
                label=label,
            )
            axes[1].plot(
                [row["fit_rms_mm"] for row in trace],
                [row["motion_rms_mm"] for row in trace],
                marker=marker,
                label=label,
            )
    axes[0].set(
        xlabel="global fit RMS (mm)",
        ylabel="right NLF fit vector RMS (mm)",
        title="NLF residual versus fit",
    )
    axes[1].set(
        xlabel="global fit RMS (mm)",
        ylabel="motion RMS (mm)",
        title="Motion versus fit",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    fig.savefig(out / "metrics-vs-fit.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    for row, region in enumerate(REGIONS):
        for stage, marker in (("forward", "o"), ("refit", None)):
            for case in CASES:
                trace = traces[stage][case]
                x = [item["fit_rms_mm"] for item in trace]
                label = f"{case} | {stage}"
                axes[row, 0].plot(
                    x,
                    [
                        item[f"roughness_{region}_normal_residual_highpass_5mm_rms_mm"]
                        for item in trace
                    ],
                    marker=marker,
                    label=label,
                )
                axes[row, 1].plot(
                    x,
                    [
                        item[
                            f"roughness_{region}_normal_displacement_highpass_5mm_rms_mm"
                        ]
                        for item in trace
                    ],
                    marker=marker,
                    label=label,
                )
        axes[row, 0].set(
            title=f"{region}: normal residual HP",
            xlabel="global fit RMS (mm)",
            ylabel="5 mm HP RMS (mm)",
        )
        axes[row, 1].set(
            title=f"{region}: normal displacement HP",
            xlabel="global fit RMS (mm)",
            ylabel="5 mm HP RMS (mm)",
        )
        for axis in axes[row]:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=6)
    fig.savefig(out / "roughness-vs-fit.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite {out}")
    traces = {
        stage: {case: rows(case_dir(stage, case) / "trace.csv") for case in CASES}
        for stage in ("forward", "refit")
    }
    summaries = {
        stage: {
            case: json.loads((case_dir(stage, case) / "summary.json").read_text())
            for case in CASES
        }
        for stage in ("forward", "refit")
    }
    reference = best_row(traces["refit"]["skin-zero"])
    matches = {case: matched_row(traces["refit"][case], reference) for case in CASES}
    common_overlaps = {
        case: common_overlap(traces["refit"]["skin-zero"], traces["refit"][case])
        for case in ("no-skin", "skin-local-1pct")
    }
    plot_metrics(traces, out)

    volume = pv.read(FIXTURE / "volume.vtu")
    skin = pv.read(FIXTURE / "skin.vtp")
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target_u = np.asarray(volume.point_data["Smile"], dtype=np.float64)[ids]
    if not np.isfinite(target_u).all():
        raise ValueError("skin target contains non-finite values")
    receipt = json.loads(CAMERAS.read_text())
    views = receipt["views"]
    if len(views) != 5 or any(
        "id" not in view or "camera" not in view for view in views
    ):
        raise ValueError("frozen regional camera receipt schema changed")
    figures: dict[str, list[str]] = {}
    forward_states = [
        (f"{case} | initial forward", load_surface(case_dir("forward", case), 0, ids))
        for case in CASES
    ] + [("Target smile", target_u)]
    refit_step0_states = [
        (
            f"{case} | re-equilibrated step 0",
            load_surface(case_dir("refit", case), 0, ids),
        )
        for case in CASES
    ] + [("Target smile", target_u)]
    refit_states = [
        (
            f"{case} | refit best step {int(best_row(traces['refit'][case])['step'])}",
            load_surface(
                case_dir("refit", case),
                int(best_row(traces["refit"][case])["step"]),
                ids,
            ),
        )
        for case in CASES
    ] + [("Target smile", target_u)]
    section_states = {
        "initial-fixed-forward": [
            (case, load_surface(case_dir("forward", case), 0, ids), color)
            for case, color in zip(
                CASES, ("#87929f", "#c53d3d", "#7a4eab"), strict=True
            )
        ]
        + [("Target smile", target_u, "#1b78a5")],
        "reequilibrated-fixed-q": [
            (case, load_surface(case_dir("refit", case), 0, ids), color)
            for case, color in zip(
                CASES, ("#87929f", "#c53d3d", "#7a4eab"), strict=True
            )
        ]
        + [("Target smile", target_u, "#1b78a5")],
        "equal-budget-best-fit": [
            (
                case,
                load_surface(
                    case_dir("refit", case),
                    int(best_row(traces["refit"][case])["step"]),
                    ids,
                ),
                color,
            )
            for case, color in zip(
                CASES, ("#87929f", "#c53d3d", "#7a4eab"), strict=True
            )
        ]
        + [("Target smile", target_u, "#1b78a5")],
    }
    for kind, states in (
        ("initial-fixed-forward", forward_states),
        ("reequilibrated-fixed-q", refit_step0_states),
        ("equal-budget-best-fit", refit_states),
    ):
        render_nlf_sections(
            skin,
            section_states[kind],
            out / f"{kind}-nasolabial-horizontal-sections.png",
        )
        paths = [f"{kind}-nasolabial-horizontal-sections.png"]
        for view in views:
            path = out / f"{kind}-{view['id']}.png"
            render_states(skin, states, view, path)
            paths.append(path.name)
        figures[kind] = paths
    matched_figures: dict[str, list[str]] = {}
    overlap_figures: dict[str, list[str]] = {}
    reference_u = load_surface(
        case_dir("refit", "skin-zero"), int(reference["step"]), ids
    )
    for case, overlap in common_overlaps.items():
        if overlap is None:
            continue
        left, right = overlap["left"], overlap["right"]
        left_u = load_surface(case_dir("refit", "skin-zero"), int(left["step"]), ids)
        right_u = load_surface(case_dir("refit", case), int(right["step"]), ids)
        paths = []
        for view in views:
            path = out / f"common-overlap-skin-zero-{case}-{view['id']}.png"
            render_states(
                skin,
                [
                    (f"skin-zero | overlap step {int(left['step'])}", left_u),
                    (f"{case} | overlap step {int(right['step'])}", right_u),
                    ("Target smile", target_u),
                ],
                view,
                path,
            )
            paths.append(path.name)
        overlap_figures[case] = paths
    for case, match in matches.items():
        if case == "skin-zero" or not match["within_tolerance"]:
            continue
        current_u = load_surface(case_dir("refit", case), int(match["step"]), ids)
        paths = []
        for view in views:
            path = out / f"matched-{case}-{view['id']}.png"
            render_states(
                skin,
                [
                    (f"skin-zero | refit step {int(reference['step'])}", reference_u),
                    (f"{case} | matched step {int(match['step'])}", current_u),
                    ("Target smile", target_u),
                ],
                view,
                path,
            )
            paths.append(path.name)
        matched_figures[case] = paths
    selected = {
        "initial_fixed_forward": {case: traces["forward"][case][0] for case in CASES},
        "reequilibrated_fixed_q_step0": {
            case: traces["refit"][case][0] for case in CASES
        },
        "best_200": {case: best_row(traces["refit"][case]) for case in CASES},
        "matched_skin_zero": {
            case: next(
                row
                for row in traces["refit"][case]
                if int(row["step"]) == int(match["step"])
            )
            for case, match in matches.items()
        },
    }
    compact_columns = (
        "selection",
        "case",
        "step",
        "objective_mm2",
        "fit_rms_mm",
        "motion_rms_mm",
        "roi_right_nose_to_mouth_fit_vector_rms_mm",
        *(
            f"roughness_{region}_normal_residual_highpass_5mm_rms_mm"
            for region in REGIONS
        ),
        *(
            f"roughness_{region}_normal_displacement_highpass_5mm_rms_mm"
            for region in REGIONS
        ),
        "fit_delta_mm",
        "motion_delta_mm",
        "within_tolerance",
    )
    compact_rows = []
    for selection, by_case in selected.items():
        for case, metric in by_case.items():
            row = {name: metric.get(name) for name in compact_columns}
            row.update(selection=selection, case=case)
            if selection == "matched_skin_zero":
                row.update(
                    {
                        name: matches[case].get(name)
                        for name in (
                            "fit_delta_mm",
                            "motion_delta_mm",
                            "within_tolerance",
                        )
                    }
                )
            compact_rows.append(row)
    with (out / "summary-table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=compact_columns)
        writer.writeheader()
        writer.writerows(compact_rows)
    summary = {
        "status": "completed_saved_state_comparison",
        "scope": "Reads saved runner states only; no solve, update, smoothing, interpolation, or deformation scaling. Refit step 0 is a re-equilibrated fixed-q state, never an optimized state.",
        "source": record(Path(__file__)),
        "fixed_inputs": {
            "fixture_volume": record(FIXTURE / "volume.vtu"),
            "fixture_skin": record(FIXTURE / "skin.vtp"),
            "camera_receipt": record(CAMERAS),
        },
        "matching_reference": {
            "case": "skin-zero",
            "stage": "refit",
            "best_step": int(reference["step"]),
            "fit_rms_mm": reference["fit_rms_mm"],
            "motion_rms_mm": reference["motion_rms_mm"],
        },
        "matching_tolerances_mm": {
            "fit": FIT_TOLERANCE_MM,
            "motion": MOTION_TOLERANCE_MM,
        },
        "matched_refit_states": matches,
        "common_overlaps": common_overlaps,
        "selected_metric_rows": selected,
        "unavailable_matches": [
            case for case, match in matches.items() if not match["within_tolerance"]
        ],
        "unavailable_common_overlaps": [
            case for case, overlap in common_overlaps.items() if overlap is None
        ],
        "figures": {
            **figures,
            "matched": matched_figures,
            "common_overlap": overlap_figures,
        },
        "summary_table": record(out / "summary-table.csv"),
        "inputs": {
            f"{stage}/{case}": {
                "trace": record(case_dir(stage, case) / "trace.csv"),
                "summary": record(case_dir(stage, case) / "summary.json"),
            }
            for stage in ("forward", "refit")
            for case in CASES
        },
        "runner_status": {
            f"{stage}/{case}": summaries[stage][case]["status"]
            for stage in ("forward", "refit")
            for case in CASES
        },
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
