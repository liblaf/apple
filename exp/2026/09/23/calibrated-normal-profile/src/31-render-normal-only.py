"""Eight calibrated normal fits in the original shape/activation slide layout."""

# ruff: noqa: C901, PLR0912, PLR0915, RUF001

from __future__ import annotations

import json
from pathlib import Path

import calibrated_study as ns
import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import slide_geometry as helpers
from experiment import Profile

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("31-normal-only-slide")
    dpi: int = 480


def main(cfg: Config) -> None:
    verification = ns.GROUP / "data/20-verification/comparison.json"
    checked = json.loads(verification.read_text())
    assert checked["passed"]
    cherries.log_input(verification)
    cells = {cell["mode"]: cell for cell in checked["cells"]}
    mesh = ns.ph.build_mesh(100, 10)
    variants = ("smooth-off-normal", "smooth-on-normal")
    metrics = {
        (mode, variant): cells[mode]["endpoints"][variant]
        for mode in ns.MODES
        for variant in variants
    }
    records = {
        (mode, variant): helpers.geometry(mode, variant, metrics[mode, variant], mesh)
        for mode in ns.MODES
        for variant in variants
    }
    assert len(records) == 8
    maximum = max(float(np.abs(d["eigenvalues"]).max()) for d in records.values())
    norm = mpl.colors.Normalize(-maximum, maximum)
    bounds = np.concatenate([d["points"] for d in records.values()])
    limits = (
        min(-0.02, float(bounds[:, 0].min()) - 0.005),
        max(1.02, float(bounds[:, 0].max()) + 0.005),
        min(-0.01, float(bounds[:, 1].min()) - 0.005),
        max(0.31, float(bounds[:, 1].max()) + 0.005),
    )
    mpl.rcParams.update(
        {"font.family": "DejaVu Sans", "svg.fonttype": "path", "pdf.fonttype": 42}
    )
    fig, axes = plt.subplots(4, 4, figsize=(16, 9), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.15, right=0.981, bottom=0.185, top=0.785, wspace=0.065, hspace=0.43
    )
    panel_manifest = []
    for row, mode in enumerate(ns.MODES):
        for group, variant in enumerate(variants):
            data = records[mode, variant]
            metric = metrics[mode, variant]
            unstable = helpers.eigenvalue(metric) < 0
            shape, activation = axes[row, 2 * group : 2 * group + 2]
            colors = np.where(
                data["muscle"][:, None], [[0.85, 0.50, 0.45]], [[0.91, 0.85, 0.75]]
            )
            shape.add_collection(
                PolyCollection(
                    data["points"][data["tri"]],
                    facecolors=colors,
                    edgecolors="#72665b",
                    linewidths=0.14,
                )
            )
            activation.add_collection(
                PolyCollection(
                    data["points"][data["tri"]],
                    facecolors="#faf9f6",
                    edgecolors="#d8d4cc",
                    linewidths=0.10,
                )
            )
            activation.add_collection(
                LineCollection(
                    data["segments"],
                    array=data["eigenvalues"],
                    norm=norm,
                    cmap="RdBu_r",
                    linewidths=0.58,
                )
            )
            x = np.linspace(0, 1, 401)
            for ax in (shape, activation):
                ax.plot(x, 0.1 + 0.8 * x * (1 - x), "--", color="#292929", lw=1)
                ax.set(
                    xlim=limits[:2],
                    ylim=limits[2:],
                    xticks=[0, 0.25, 0.5, 0.75, 1],
                    yticks=[0, 0.1, 0.2, 0.3],
                )
                ax.set_aspect("equal", adjustable="box")
                ax.tick_params(labelsize=8.5, length=2.7, pad=2)
                for spine in ax.spines.values():
                    spine.set_color("#4e4e4e")
                    spine.set_linewidth(0.75)
                if row == 3:
                    ax.set_xlabel("deformed x", fontsize=10)
                warnings = []
                if unstable:
                    warnings.append("UNSTABLE")
                if metric["failure"]:
                    warnings.append("STOPPED")
                if warnings:
                    ax.text(
                        0.98,
                        0.94,
                        " · ".join(warnings),
                        transform=ax.transAxes,
                        ha="right",
                        va="top",
                        fontsize=8,
                        color="#a71930",
                        weight="bold",
                    )
                    for spine in ax.spines.values():
                        spine.set_color("#a71930")
                        spine.set_linewidth(1.8)
            shape.text(
                0.02,
                0.96,
                f"RMS {metric['fit_rms']:.4f}",
                transform=shape.transAxes,
                va="top",
                fontsize=9,
            )
            activation.text(
                0.02,
                0.96,
                f"normal {metric['normal_angle_rms_deg']:.1f}°",
                transform=activation.transAxes,
                va="top",
                fontsize=8.5,
            )
            if row == 0:
                shape.set_title("Final shape", fontsize=13, pad=10)
                activation.set_title("Activation", fontsize=13, pad=10)
            panel_manifest.append(
                {
                    "mode": mode,
                    "variant": variant,
                    "step": metric["step"],
                    "position_rms": metric["fit_rms"],
                    "normal_rms_deg": metric["normal_angle_rms_deg"],
                    "unstable": unstable,
                    "failure": metric["failure"],
                }
            )
        axes[row, 0].set_ylabel(
            helpers.LABELS[row],
            rotation=0,
            ha="right",
            va="center",
            fontsize=12,
            labelpad=15,
        )
    fig.suptitle(
        "Final shape and activation · L2 + normal loss", x=0.53, y=0.974, fontsize=25
    )
    fig.text(
        0.55,
        0.924,
        "h = 0.20   ·   0.02 position RMS ≈ 5° normal error   ·   dashed curve: target",
        ha="center",
        fontsize=12.5,
    )
    fig.text(
        0.55,
        0.885,
        r"$L=L_2+0.1051165\,\langle1-\cos\theta\rangle+\alpha h^2 R_B$",
        ha="center",
        fontsize=14,
    )
    fig.canvas.draw()
    for group, label in enumerate(("Smoothness off", "Smoothness on (α = 1)")):
        left = axes[0, 2 * group].get_position().x0
        right = axes[0, 2 * group + 1].get_position().x1
        fig.text(
            (left + right) / 2, 0.837, label, ha="center", fontsize=16, weight="bold"
        )
    cax = fig.add_axes([0.34, 0.099, 0.42, 0.017])
    bar = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r"),
        cax=cax,
        orientation="horizontal",
    )
    bar.set_label(
        "Signed activation λ(B − I) · thin equal-length axes show orientation",
        fontsize=11,
    )
    bar.ax.tick_params(labelsize=9, length=2.5)
    steps = sorted({m["step"] for m in metrics.values()})
    all_complete = steps == [1200]
    note = (
        "All eight fits: 1,200 updates from neutral initialization."
        if all_complete
        else "Last accepted states shown; stopped fits have their update labeled."
    )
    if any(p["unstable"] for p in panel_manifest):
        note += " Red frames: negative forward Hessian."
    note += " Fixed budget; convergence not established."
    fig.text(0.55, 0.025, note, ha="center", fontsize=9.4)
    fig.canvas.draw()
    for ax in axes.flat:
        t = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        np.testing.assert_allclose(t[1, 0] - t[0, 0], t[2, 1] - t[0, 1], atol=1e-8)
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    stem = "l2-normal-shape-activation-h200-16x9"
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            output / f"{stem}.{ext}", dpi=cfg.dpi, bbox_inches=None, facecolor="white"
        )
    fig.savefig(
        output / f"{stem}-preview.png", dpi=120, bbox_inches=None, facecolor="white"
    )
    plt.close(fig)
    manifest = {
        "pixel_size": [16 * cfg.dpi, 9 * cfg.dpi],
        "fits": 8,
        "panels": 16,
        "calibration": ns.calibration_check(),
        "limits": limits,
        "activation_color_range": [-maximum, maximum],
        "panels_data": panel_manifest,
        "render_source_sha256": {
            str(p): helpers.hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(helpers.__file__))
        },
        "input_verification_sha256": helpers.hashlib.sha256(
            verification.read_bytes()
        ).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
