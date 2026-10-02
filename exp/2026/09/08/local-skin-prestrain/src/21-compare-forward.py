"""Render and tabulate the three completed fixed-control forward cases only."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pyvista as pv

GROUP = Path(__file__).resolve().parents[1]
OUT = GROUP / "data/21-forward-comparison"
spec = importlib.util.spec_from_file_location("comparison", GROUP / "src/40-compare.py")
if spec is None or spec.loader is None:
    raise RuntimeError("cannot load frozen comparison renderer")
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"refusing to overwrite {OUT}")
    summaries = {
        case: json.loads((c.case_dir("forward", case) / "summary.json").read_text())
        for case in c.CASES
    }
    if any(
        summary["status"] != "completed_fixed_activation_forward"
        for summary in summaries.values()
    ):
        raise ValueError("all three saved fixed-control forward runs must be complete")
    skin = pv.read(c.FIXTURE / "skin.vtp")
    volume = pv.read(c.FIXTURE / "volume.vtu")
    ids = np.asarray(skin.point_data["GlobalPointId"], dtype=np.int64)
    target = np.asarray(volume.point_data["Smile"], dtype=np.float64)[ids]
    states = [
        (f"{case} | fixed forward", c.load_surface(c.case_dir("forward", case), 0, ids))
        for case in c.CASES
    ] + [("Target smile", target)]
    receipt = json.loads(c.CAMERAS.read_text())
    figures = []
    for view in receipt["views"]:
        path = OUT / f"{view['id']}.png"
        c.render_states(skin, states, view, path)
        figures.append(c.record(path))
    section_states = [
        (case, c.load_surface(c.case_dir("forward", case), 0, ids), color)
        for case, color in zip(c.CASES, ("#87929f", "#c53d3d", "#7a4eab"), strict=True)
    ] + [("Target smile", target, "#1b78a5")]
    sections = OUT / "nasolabial-horizontal-sections.png"
    c.render_nlf_sections(skin, section_states, sections)
    metrics = [dict(case=case, **summaries[case]["last_metrics"]) for case in c.CASES]
    fields = list(metrics[0])
    with (OUT / "metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)
    summary = {
        "status": "completed_saved_fixed_control_forward_comparison",
        "scope": "Reads only three completed step-0 forward surfaces and fixture Smile target; no forward solve, re-fit state, interpolation, smoothing, or deformation scaling.",
        "source": c.record(Path(__file__)),
        "renderer": c.record(GROUP / "src/40-compare.py"),
        "inputs": {
            case: {
                "summary": c.record(c.case_dir("forward", case) / "summary.json"),
                "surface": c.record(c.case_dir("forward", case) / "surface-0000.npz"),
            }
            for case in c.CASES
        },
        "camera_receipt": c.record(c.CAMERAS),
        "metrics_csv": c.record(OUT / "metrics.csv"),
        "metrics": metrics,
        "figures": figures + [c.record(sections)],
    }
    (OUT / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
