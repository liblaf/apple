"""Render the continued endpoint with explicit forward nonconvergence labels."""

# ruff: noqa: RUF001, SLF001

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pydantic_settings as ps
from PIL import Image

from liblaf import cherries

aligned = importlib.import_module("80-render-aligned")
render = aligned.render
GROUP = Path(__file__).resolve().parents[1]
CONTINUED = GROUP / "data/120-inexact-continuation"
STEM = "aligned-shape-activation-h200-inexact-16x9"
LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("130-inexact-figures")
    dpi: int = 960


def continuation_case():
    summaries = json.loads((CONTINUED / "summary.json").read_text())
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["outer_budget_complete"]
    assert summary["completed_updates"] == 1200
    assert not summary["final_forward_converged"]
    folder = CONTINUED / summary["name"]
    with (folder / "trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    # The continuation trace also contains status strings and Boolean columns.
    numeric_fields = (
        "step",
        "normalized_loss",
        "objective_normalized",
        "roughness",
        "tensor_neighbor_rms",
        "min_J",
        "force_residual_inf",
    )
    trace = {
        key: np.asarray([float(row[key]) for row in rows]) for key in numeric_fields
    }
    np.testing.assert_array_equal(trace["step"], np.arange(262, 1201))
    assert all(row["forward_termination"] == "line_search_failed" for row in rows[1:])
    assert all(np.all(np.isfinite(value)) for value in trace.values())
    assert trace["normalized_loss"][-1] == summary["final"]["normalized_loss"]
    return render.Case(
        source=CONTINUED,
        summary=summary,
        trace=trace,
        history=render._load_npz(folder / "history.npz"),
        checkpoint=render._load_npz(folder / "checkpoint.npz"),
    )


def main(cfg: Config) -> None:
    assert cfg.dpi > 0
    original = render._select(
        render._load_cases([GROUP / "data/tune-w0", GROUP / "data/tune-w1"]), 1.0
    )
    continued = continuation_case()
    selected = [continued if case.name == continued.name else case for case in original]
    assert sum(case is continued for case in selected) == 1
    # Preserve the common scale across both target heights used in prior figures.
    maximum, glyph_length = render._activation_scale(selected)
    cases = [case for case in selected if np.isclose(case.height, aligned.HEIGHT)]
    assert len(cases) == 8
    out = cherries.output(cfg.output)
    out.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(aligned.__file__), Path(render.__file__)]
    for source in sources:
        shutil.copy2(source, out / source.name)
    for case in cases:
        geometry = render._activation_geometry(case)
        if case is not continued:
            with np.load(
                GROUP / "data/70-readable-glyphs" / f"{case.name}-glyphs.npz"
            ) as prior:
                for key, value in geometry.items():
                    np.testing.assert_array_equal(value, prior[key])
        np.savez_compressed(out / f"{case.name}-glyphs.npz", **geometry)
        points, _, _, top, _, _ = render._history_contract(case)
        x, y = render._target(points, top, aligned.HEIGHT)
        cloud = np.vstack([geometry["points"], np.column_stack([x, y])])
        assert np.all(cloud.min(axis=0) > [aligned.LIMITS[0], aligned.LIMITS[2]])
        assert np.all(cloud.max(axis=0) < [aligned.LIMITS[1], aligned.LIMITS[3]])
    record = aligned.aligned_figure(
        cases,
        (0.0, 1.0),
        out,
        maximum,
        glyph_length,
        STEM,
        slide_format=True,
        dpi=cfg.dpi,
        vector_export=True,
        failure_label="UNCONVERGED*",
        footnote=(
            "* Free / off: activation reached step 1200; shape stayed fixed after the failed solve at step 263.\n"
            "Final force residual 2.61×10⁻⁴ (tolerance 10⁻¹⁰). Other panels: forward equilibria at step 1200."
        ),
    )
    # A lightweight preview opens quickly; the linked PNG retains the full resolution.
    Image.MAX_IMAGE_PIXELS = 16 * 9 * cfg.dpi**2 + 1
    with Image.open(out / f"{STEM}.png") as full:
        assert full.size == (16 * cfg.dpi, 9 * cfg.dpi)
        full.resize((2560, 1440), Image.Resampling.LANCZOS).save(out / "preview.png")
    (out / "alignment-checks.json").write_text(
        json.dumps(
            {
                "height": aligned.HEIGHT,
                "cases": 8,
                "other_seven_geometries_match_previous_render_exactly": True,
                "continued_case": continued.name,
                "continued_case_forward_converged": False,
                "continued_case_source": str(CONTINUED),
                "global_color_range": [-maximum, maximum],
                "glyph_full_length": glyph_length,
                "glyph_linewidth_points": 0.45,
                "source_sha256": {
                    source.name: hashlib.sha256(source.read_bytes()).hexdigest()
                    for source in sources
                },
                "input_checkpoint_sha256": {
                    case.name: hashlib.sha256(
                        (case.source / case.name / "checkpoint.npz").read_bytes()
                    ).hexdigest()
                    for case in cases
                },
                "figure": record,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    (out / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Activation comparison after continuation</title><style>
body{{font:16px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1600px;padding:2rem;color:#24211f;background:#faf9f6}}
img{{display:block;width:100%;height:auto}} a{{color:#315f88}} .note{{border-left:4px solid #a71930;padding:1rem;background:#f0ece6}}
</style></head><body><h1>Activation comparison after 1,200 updates</h1>
<p class="note">Free activation with smoothness off continued through 938 failed forward solves.
Its activation kept updating, but its shape stayed fixed after step 263. This endpoint is UNCONVERGED;
its L2 value is evaluated at the last finite forward iterate. The other seven endpoints are unchanged.</p>
<p><a href="{STEM}.png">Full-resolution PNG ({16 * cfg.dpi:,} × {9 * cfg.dpi:,})</a> ·
<a href="{STEM}.svg">Vector SVG</a> · <a href="{STEM}.pdf">Vector PDF</a></p>
<a href="{STEM}.png"><img src="preview.png" alt="Aligned whole-domain final shape and deformed activation comparison"></a>
<p>Exact 16:9 canvas. Equal physical scales. Thin equal-length activation axes follow F n / ‖F n‖.</p>
</body></html>""")
    cherries.log_metrics(
        {
            "selected_cases": 8,
            "continued_unconverged_cases": 1,
            "alignment_checks_passed": True,
        }
    )
    LOG.info("Wrote annotated PNG, PDF, SVG and preview to %s", out)


if __name__ == "__main__":
    cherries.main(main, profile=render.ProfileFigures)
