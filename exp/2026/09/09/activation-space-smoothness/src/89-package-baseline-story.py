"""Package verified saved-state figures for the chronological baseline story."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

from PIL import Image

GROUP = Path(__file__).resolve().parents[1]
OUTPUT = GROUP / "data/89-baseline-story"
SOURCE = (
    Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
    / "exp/2026/09/08/physical-volume-baseline/data/30-comparison"
)


def record(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name in (
        "skin-front.png",
        "skin-mouth.png",
        "muscle-mouth.png",
        "cell27306-shapes.png",
        "metrics.csv",
    ):
        source = SOURCE / name
        target = OUTPUT / name
        shutil.copyfile(source, target)
        assert record(source)["sha256"] == record(target)["sha256"]
        copied[name] = {"source": record(source), "output": record(target)}

    extracted = {}
    for source_name, target_name in (
        ("skin-front.png", "old-vs-psd-skin-front.png"),
        ("skin-mouth.png", "old-vs-psd-skin-mouth.png"),
        ("muscle-mouth.png", "old-vs-psd-muscle-mouth.png"),
    ):
        source = SOURCE / source_name
        image = Image.open(source)
        assert image.size == (2250, 960), image.size
        target = Image.new(image.mode, (1500, 960))
        target.paste(image.crop((0, 0, 750, 960)), (0, 0))
        target.paste(image.crop((1500, 0, 2250, 960)), (750, 0))
        output = OUTPUT / target_name
        target.save(output, optimize=True)
        extracted[target_name] = {
            "source": record(source),
            "output": record(output),
            "operation": "Exact outer-panel extraction: source pixels x=0:750 and x=1500:2250; no resampling or rendering",
            "order": ["Old baseline | best 194", "PSD active stress | step 1024"],
        }

    panel_names = ("old", "corrected", "psd")
    for source_name, stem in (
        ("skin-front.png", "skin-front"),
        ("skin-mouth.png", "skin-mouth"),
        ("muscle-mouth.png", "muscle-mouth"),
    ):
        source = SOURCE / source_name
        image = Image.open(source)
        for index, panel_name in enumerate(panel_names):
            output = OUTPUT / f"{panel_name}-{stem}.png"
            image.crop((750 * index, 0, 750 * (index + 1), 960)).save(
                output, optimize=True
            )
            extracted[output.name] = {
                "source": record(source),
                "output": record(output),
                "operation": f"Exact source-panel extraction: x={750 * index}:{750 * (index + 1)}; no resampling or rendering",
            }

    summary = {
        "status": "completed_saved_figure_packaging",
        "scope": "Copies and exact pixel crops only; no solve, fit, geometry processing, smoothing, or resampling",
        "three_way_order": [
            "Old baseline | best 194",
            "Corrected physical-volume baseline | best 200",
            "PSD active stress | step 1024",
        ],
        "copied": copied,
        "extracted": extracted,
        "historical_cell_27306": {
            "selection": "Pure-muscle mouth-region cell selected before the corrected rerun",
            "old_law_volume_argument": "det(F @ Ainv)",
            "det_F_Ainv": 1.0732667899287331,
            "physical_det_F": 2.662139246391875,
            "evidence": record(
                Path(os.environ["APPLE_HISTORICAL_WORKTREE"])
                / "exp/2026/09/08/physical-volume-baseline/data/10-physical-volume-validation.json"
            ),
        },
        "source_receipt": record(SOURCE / "summary.json"),
        "captions": {
            "old-vs-psd-skin-front.png": "Historical active-strain baseline versus PSD active stress, saved full-face surfaces; actual deformation scale 1.",
            "skin-front.png": "Historical baseline, corrected physical-volume baseline, and PSD active stress; saved full-face surfaces in that order.",
            "skin-mouth.png": "The same three saved endpoints at the mouth camera.",
            "muscle-mouth.png": "The same three saved endpoints on the fixed rest-selected mouth-region muscle cutaway; connectivity is unchanged across states.",
            "cell27306-shapes.png": "Pure-muscle mouth-region cell 27306 at the three saved endpoints; shared bounds and view, with centroid translation removed.",
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
