"""Place three exact endpoint render panels beside their measured responses."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def main() -> None:
    out = ROOT / "data/90-field-diffusion-figure"
    out.mkdir(exist_ok=False)
    summary_path = ROOT / "data/80-forward-field-diffusion/summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "completed"
    inputs = [record(summary_path)]
    for view in ("full-head", "mouth-closeup"):
        fig, axes = plt.subplots(1, 3, figsize=(12, 6.9))
        fig.subplots_adjust(left=0.01, right=0.99, bottom=0.14, top=0.87, wspace=0.015)
        for axis, row, title in zip(
            axes,
            summary["cases"],
            ("Unchanged field", "Mild diffusion", "Strong diffusion"),
            strict=True,
        ):
            source = ROOT / "data/82-field-diffusion-viewer" / row["id"] / f"{view}.png"
            pixels = np.asarray(Image.open(source).convert("RGB"))
            assert pixels.shape[:2] == (1325, 1800)
            # src50 renders two equally sized 900x1200 viewports above its caption.
            # Select the entire right (equilibrium) viewport without changing geometry.
            axis.imshow(pixels[:1200, 900:1800])
            axis.set_axis_off()
            axis.set_title(title, fontsize=14, fontweight="bold", pad=12)
            axis.text(
                0.5,
                -0.035,
                f"Field R/R₀ = {row['field']['roughness_ratio']:.3f}\n"
                f"Motion {row['area_motion_rms_mm']:.3f} mm · Error {row['area_fit_rms_mm']:.3f} mm",
                transform=axis.transAxes,
                ha="center",
                va="top",
                fontsize=10,
            )
            inputs.append(record(source))
        fig.suptitle(
            "Forward response to activation-field diffusion",
            fontsize=17,
            fontweight="bold",
            x=0.02,
            ha="left",
            y=0.975,
        )
        fig.text(
            0.02,
            0.035,
            "Actual saved equilibrium geometry · identical camera and scale · no vertex smoothing",
            fontsize=10,
            color="#475451",
        )
        for extension in ("png", "pdf"):
            fig.savefig(out / f"{view}.{extension}", dpi=200)
        plt.close(fig)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "inputs": inputs,
                "script": record(Path(__file__)),
                "panel_selection": "Right 900x1200 endpoint viewport from each verified 1800x1325 src50 PNG; crop only, no geometry processing",
                "outputs": [
                    record(p)
                    for p in sorted(out.iterdir())
                    if p.suffix in {".png", ".pdf"}
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
