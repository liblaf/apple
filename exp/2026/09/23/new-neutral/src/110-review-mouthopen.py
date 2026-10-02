"""Render a saved MouthOpen inverse endpoint; never evaluates inverse or forward physics."""
# ruff: noqa: PT018

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/inverse-mouthopen-002"
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    overwrite: bool = False


def record(path: Path) -> dict[str, str]:
    assert path.is_file(), path
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict[str, Any]) -> Path:
    path = Path(item["path"])
    actual = record(path)
    assert actual["path"] == item["path"] and actual["sha256"] == item["sha256"]
    return path


def surface(
    points: np.ndarray, ids: np.ndarray, triangles: np.ndarray, u: np.ndarray
) -> pv.PolyData:
    assert points.ndim == 2 and points.shape[1] == 3
    assert len(ids) == len(points) and len(np.unique(ids)) == len(ids)
    assert (
        triangles.ndim == 2
        and triangles.shape[1] == 3
        and triangles.min() >= 0
        and triangles.max() < len(ids)
    )
    assert ids.max() < len(u)
    faces = np.column_stack((np.full(len(triangles), 3), triangles)).ravel()
    return pv.PolyData(points + u[ids], faces)


def camera(meshes: list[pv.PolyData], view: str):
    pts = np.concatenate([x.points for x in meshes])
    low, high = pts.min(0), pts.max(0)
    center = (low + high) / 2
    span = float(max(high - low))
    eye = (
        center
        + (np.array([0, 0, 2.7]) if view == "front" else np.array([2.7, 0, 0])) * span
    )
    return [eye.tolist(), center.tolist(), [0.0, 1.0, 0.0]], 0.60 * span


def render(
    out: Path,
    target: pv.PolyData,
    fit: pv.PolyData,
    rigid: dict[str, pv.PolyData],
    status: str,
    view: str,
) -> str:
    pos, scale = camera([target, fit, *rigid.values()], view)
    p = pv.Plotter(off_screen=True, shape=(1, 2), window_size=(1800, 900))
    p.set_background("#f7f7f5")
    for col, (title, skin, color) in enumerate(
        (
            ("MouthOpen target", target, "#737b86"),
            (f"Inverse fit · {status}", fit, "#c75f42"),
        )
    ):
        p.subplot(0, col)
        p.add_text(
            f"{title}\n{view} · identical true scale",
            position="upper_left",
            font_size=15,
            color="#202124",
        )
        for name, m in rigid.items():
            p.add_mesh(
                m,
                color="#9fc5e8" if name == "eyes" else "#e8dfc8",
                smooth_shading=True,
                opacity=0.82,
            )
        p.add_mesh(skin, color=color, smooth_shading=True)
        p.camera_position = pos
        p.camera.parallel_projection = True
        p.camera.parallel_scale = scale
        p.reset_camera_clipping_range()
    name = f"target-vs-fit-{view}.png"
    p.show(screenshot=out / name, auto_close=True)
    return name


def curves(out: Path, rows: list[dict]) -> str:
    it = np.array([r["iteration"] for r in rows])
    fit = np.array([r["fit_rms_mm"] for r in rows])
    force = np.array([r["force_norm_n"] for r in rows])
    thresh = np.array([r["force_threshold_n"] for r in rows])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    ax[0].plot(it, fit, color="#c75f42")
    ax[0].set(xlabel="Inverse iteration", ylabel="Fit RMS (mm)", title="Target fit")
    ax[0].grid(alpha=0.2)
    ax[1].semilogy(it, force, color="#087d81", label="force norm")
    ax[1].semilogy(it, thresh, color="#555", ls=":", label="threshold")
    ax[1].set(
        xlabel="Inverse iteration", ylabel="Force norm (N)", title="Forward residual"
    )
    ax[1].grid(alpha=0.2)
    ax[1].legend()
    for a in ax:
        a.spines[["top", "right"]].set_visible(False)
    name = "fit-force-curves.png"
    fig.savefig(out / name, dpi=180)
    plt.close(fig)
    return name


def main(cfg: Config) -> None:
    run, review = cfg.run_dir.resolve(), cfg.review_dir.resolve()
    out = review / "mouthopen"
    if out.exists():
        assert cfg.overwrite
        shutil.rmtree(out)
    protocol = json.loads((run / "protocol.json").read_text())
    summary = json.loads((run / "summary.json").read_text())
    chin_receipt_path, chin_preview_path = (
        run / "chin-estimate.json",
        run / "chin-patch-preview.png",
    )
    assert chin_receipt_path.is_file() and chin_preview_path.is_file()
    chin = json.loads(chin_receipt_path.read_text())
    assert chin["schema"] == "chin-pose-seed-v1"
    rendering_path = verified(protocol["rendering"]["archive"])
    blend_path = verified(protocol["sources"]["blendshapes"])
    assert protocol["expression_name"] == "MouthOpen"
    with np.load(run / "endpoint.npz", allow_pickle=False) as a:
        assert {
            "displacement_m",
            "activation_inv",
            "jaw_angle_rad",
            "active_cell_ids",
        } <= set(a.files)
        u = a["displacement_m"]
        jaw = float(a["jaw_angle_rad"])
        active = a["activation_inv"]
        active_ids = a["active_cell_ids"]
    with np.load(rendering_path, allow_pickle=False) as a:
        needed = {
            "full_reference_points_m",
            "skin_global_ids",
            "skin_triangles",
            "cranium_global_ids",
            "cranium_triangles",
            "mandible_global_ids",
            "mandible_triangles",
            "eye_global_ids",
            "eye_triangles",
        }
        assert set(a.files) == needed, a.files
        x = {k: a[k] for k in needed}
    assert u.shape == x["full_reference_points_m"].shape and np.isfinite(u).all()
    assert active.shape == (len(active_ids), 6)
    assert np.isclose(
        np.degrees(jaw), summary["final"]["jaw_angle_deg"], rtol=0, atol=1e-12
    )
    with np.load(blend_path, allow_pickle=False) as a:
        names = [str(v) for v in a["expression_names"]]
        i = names.index("MouthOpen")
        assert np.array_equal(a["skin_global_ids"], x["skin_global_ids"])
        target = pv.PolyData(
            a["target_points_m"][i],
            np.column_stack(
                (np.full(len(x["skin_triangles"]), 3), x["skin_triangles"])
            ).ravel(),
        )
    fit = surface(
        x["full_reference_points_m"][x["skin_global_ids"]],
        x["skin_global_ids"],
        x["skin_triangles"],
        u,
    )
    rigid = {
        "cranium": surface(
            x["full_reference_points_m"][x["cranium_global_ids"]],
            x["cranium_global_ids"],
            x["cranium_triangles"],
            u,
        ),
        "mandible": surface(
            x["full_reference_points_m"][x["mandible_global_ids"]],
            x["mandible_global_ids"],
            x["mandible_triangles"],
            u,
        ),
        "eyes": surface(
            x["full_reference_points_m"][x["eye_global_ids"]],
            x["eye_global_ids"],
            x["eye_triangles"],
            u,
        ),
    }
    rows = [
        json.loads(line) for line in (run / "progress.jsonl").read_text().splitlines()
    ]
    assert rows and rows[-1]["iteration"] == summary["final"]["iteration"]
    final = summary["final"]
    status = str(summary["status"])
    out.mkdir()
    shutil.copy2(chin_receipt_path, out / "chin-estimate.json")
    shutil.copy2(chin_preview_path, out / "chin-patch-preview.png")
    images = [render(out, target, fit, rigid, status, v) for v in ("front", "side")]
    curve = curves(out, rows)
    receipt = {
        "schema": "saved-mouthopen-inverse-review-v1",
        "run": {n: record(run / f"{n}.json") for n in ("protocol", "summary")},
        "endpoint": record(run / "endpoint.npz"),
        "rendering": record(rendering_path),
        "blendshapes": record(blend_path),
        "expression": "MouthOpen",
        "summary": summary,
        "jaw_angle_rad": jaw,
        "jaw_angle_deg": float(np.degrees(jaw)),
        "active_tetrahedra": len(active_ids),
        "chin_initialization": {
            "receipt": record(chin_receipt_path),
            "preview": record(chin_preview_path),
            "angle_deg": chin["angle_deg"],
            "residual_rms_m": chin["residual_rms_m"],
        },
        "images": [*images, curve],
        "scope": "Saved inverse endpoint review only; no forward or inverse physics was evaluated.",
    }
    write_json(out / "receipt.json", receipt)
    figs = "".join(
        f'<figure><img src="{n}" alt="MouthOpen target and inverse fit with moved rigid anatomy"></figure>'
        for n in images
    )
    (out / "index.html").write_text(
        f"""<!doctype html><meta charset="utf-8"><title>MouthOpen inverse</title><style>body{{font:16px system-ui;max-width:1500px;margin:2rem auto;padding:0 1rem}}img{{max-width:100%}}.status{{padding:1rem;background:#fff1c7}}</style><p><a href="../">Back to neutral review</a></p><h1>MouthOpen inverse fit</h1><p class="status">{status}. Inverse converged: {final["inverse_converged"]}; forward converged: {final["forward_converged"]}; geometry: {final["geometry"]["inverted_tetrahedra"]} inverted tetrahedra, min detF {final["geometry"]["detF_min"]:.6g}; contact valid: {final["contact_valid"]}.</p><p>Absolute jaw angle from neutral: {np.degrees(jaw):.3f}°. Bones and eyes use their saved endpoint displacement.</p>{figs}<figure><img src="{curve}" alt="Fit and force iteration curves"></figure><p><a href="receipt.json">Receipt</a> · <a href="../blendshapes/assets/blendshapes.npz">Target bundle</a></p>"""
    )
    cherries.log_output(out)
    cherries.log_metrics(
        {
            "mouthopen/final_fit_rms_mm": final["fit_rms_mm"],
            "mouthopen/final_force_n": final["force_norm_n"],
            "mouthopen/jaw_angle_deg": float(np.degrees(jaw)),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
