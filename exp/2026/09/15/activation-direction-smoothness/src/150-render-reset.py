"""Render saved rest-reset continuation and compare its recorded diagnostics."""

# ruff: noqa: RUF001, SLF001

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pydantic_settings as ps
from PIL import Image

from liblaf import cherries

aligned = importlib.import_module("80-render-aligned")
render = aligned.render
plt = aligned.plt
GROUP = Path(__file__).resolve().parents[1]
RESET = GROUP / "data/140-reset-continuation"
NO_RESET = GROUP / "data/120-inexact-continuation"
STEM = "aligned-shape-activation-h200-reset-16x9"
LOG = logging.getLogger(__name__)
NUMERIC_FIELDS = (
    "step",
    "normalized_loss",
    "objective_normalized",
    "roughness",
    "tensor_neighbor_rms",
    "min_J",
    "force_residual_inf",
)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output: Path = Path("150-reset-figures")
    dpi: int = 960


def load_continuation(source: Path) -> tuple[Any, list[dict[str, str]], dict[str, Any]]:
    cherries.log_input(source / "summary.json")
    summaries = json.loads((source / "summary.json").read_text())
    assert len(summaries) == 1
    summary = summaries[0]
    assert summary["name"] == "h200-unconstrained-w0"
    assert summary["outer_budget_complete"]
    assert summary["completed_updates"] == 1200
    folder = source / summary["name"]
    with (folder / "trace.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    trace = {
        key: np.asarray([float(row[key]) for row in rows]) for key in NUMERIC_FIELDS
    }
    np.testing.assert_array_equal(
        trace["step"],
        np.arange(summary["start_step"], summary["completed_updates"] + 1),
    )
    assert all(np.all(np.isfinite(value)) for value in trace.values())
    assert trace["normalized_loss"][-1] == summary["final"]["normalized_loss"]
    assert trace["force_residual_inf"][-1] == summary["final"]["force_residual_inf"]
    failed_count = sum(row["forward_converged"].lower() != "true" for row in rows)
    assert failed_count == sum(
        count
        for status, count in summary["forward_status_counts"].items()
        if status != "converged"
    )
    assert (rows[-1]["forward_converged"].lower() == "true") == summary[
        "final_forward_converged"
    ]
    drawing_summary = dict(summary)
    drawing_summary["failure"] = (
        None if summary["final_forward_converged"] else summary["failure"]
    )
    assert summary["final_forward_converged"] or drawing_summary["failure"]
    case = render.Case(
        source=source,
        summary=drawing_summary,
        trace=trace,
        history=render._load_npz(folder / "history.npz"),
        checkpoint=render._load_npz(folder / "checkpoint.npz"),
    )
    for name in ("trace.csv", "history.npz", "checkpoint.npz"):
        cherries.log_input(folder / name)
    return case, rows, summary


def diagnostic_history(
    no_reset: Any, reset: Any, output: Path, tolerance: float
) -> list[str]:
    fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True, layout="constrained")
    for case, color, label in (
        (no_reset, "#75619c", "No reset (120)"),
        (reset, "#217d79", "Reset displacement to rest after failure (140)"),
    ):
        steps = case.trace["step"]
        for axis, key in zip(
            axes, ("normalized_loss", "force_residual_inf", "min_J"), strict=True
        ):
            values = case.trace[key]
            axis.plot(steps, values, color=color, lw=1.6, label=label)
            unstable = (
                case.summary["final_forward_converged"]
                and case.summary["final_smallest_hessian_eigenvalue"] < 0
            )
            axis.scatter(
                steps[-1],
                values[-1],
                facecolors="none" if unstable else color,
                edgecolors=color,
                marker="^" if unstable else ("X" if case.failure else "o"),
                s=36,
            )
            axis.grid(alpha=0.2)
    axes[0].set_ylabel("L2 / h² at recorded iterate")
    axes[1].set_ylabel("Force residual infinity norm")
    axes[2].set_ylabel("Minimum physical J")
    axes[1].set_yscale("log")
    axes[2].set_yscale("log")
    axes[1].axhline(
        tolerance, color="0.35", ls=":", label=f"forward tolerance {tolerance:g}"
    )
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].legend(frameon=False, fontsize=9)
    axes[2].set_xlabel(
        "Recorded Adam update (no tails or interpolation beyond saved records)"
    )
    fig.suptitle(
        "Free activation / smoothness off · saved continuation histories", fontsize=15
    )
    fig.supxlabel(
        "×: no-reset endpoint unconverged; △: reset endpoint is an unstable equilibrium. Nonconverged-record L2 is off-equilibrium.",
        fontsize=9,
    )
    files = []
    for suffix in ("png", "pdf", "svg"):
        name = f"reset-vs-no-reset-history.{suffix}"
        with aligned.mpl.rc_context({"svg.fonttype": "path"}):
            fig.savefig(output / name, dpi=220, bbox_inches="tight")
        files.append(name)
    plt.close(fig)
    return files


def main(cfg: Config) -> None:
    assert cfg.dpi == 960
    reset, _, summary = load_continuation(RESET)
    no_reset, _, _ = load_continuation(NO_RESET)
    assert isinstance(summary["forward_resets_used"], int)
    with (RESET / summary["name"] / "trace.csv").open(newline="") as stream:
        reset_rows = list(csv.DictReader(stream))
    assert summary["forward_resets_used"] == sum(
        row["forward_seed_was_reset"].lower() == "true" for row in reset_rows
    )
    failure_count = sum(
        count
        for status, count in summary["forward_status_counts"].items()
        if status != "converged"
    )
    protocol = json.loads((RESET / "protocol.json").read_text())
    tolerance = float(protocol["config"]["forward_tolerance"])
    converged = bool(summary["final_forward_converged"])
    eigenvalue = float(summary["final_smallest_hessian_eigenvalue"])
    unstable = converged and eigenvalue < 0
    status = "UNSTABLE" if unstable else ("CONVERGED" if converged else "UNCONVERGED")
    if unstable:
        reset.summary["failure"] = {
            "reason": "unstable forward equilibrium",
            "smallest_hessian_eigenvalue": eigenvalue,
        }
    footnote = (
        f"* Free / off: {summary['forward_resets_used']} rest-seeded solves after failure; {failure_count} failed solves; step {summary['completed_updates']} reached forward equilibrium.\n"
        f"Negative Hessian eigenvalue {eigenvalue:.6g} indicates instability; force residual {summary['final']['force_residual_inf']:.3g} (tolerance {tolerance:g}). Other seven panels unchanged."
    )
    if not converged:
        footnote = (
            f"* Free / off: step {summary['completed_updates']}; {summary['forward_resets_used']} rest-seeded solves; {failure_count} failed solves; UNCONVERGED.\n"
            f"Force residual {summary['final']['force_residual_inf']:.3g} (tolerance {tolerance:g}). Other seven panels unchanged."
        )
    elif not unstable:
        footnote = (
            f"* Free / off: {summary['forward_resets_used']} rest-seeded solves after failure; {failure_count} failed solves; final forward equilibrium reached.\n"
            f"Smallest Hessian eigenvalue {eigenvalue:.6g}; force residual {summary['final']['force_residual_inf']:.3g} (tolerance {tolerance:g}). Other seven panels unchanged."
        )
    original = render._select(
        render._load_cases([GROUP / "data/tune-w0", GROUP / "data/tune-w1"]), 1.0
    )
    selected = [reset if case.name == reset.name else case for case in original]
    assert sum(case is reset for case in selected) == 1
    maximum, glyph_length = render._activation_scale(selected)
    cases = [case for case in selected if np.isclose(case.height, aligned.HEIGHT)]
    assert len(cases) == 8
    output = cherries.output(cfg.output)
    output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), Path(aligned.__file__), Path(render.__file__)]
    for source in sources:
        shutil.copy2(source, output / source.name)
    for case in cases:
        geometry = render._activation_geometry(case)
        if case is not reset:
            with np.load(
                GROUP / "data/70-readable-glyphs" / f"{case.name}-glyphs.npz"
            ) as prior:
                for key, value in geometry.items():
                    np.testing.assert_array_equal(value, prior[key])
        np.savez_compressed(output / f"{case.name}-glyphs.npz", **geometry)
    figure = aligned.aligned_figure(
        cases,
        (0.0, 1.0),
        output,
        maximum,
        glyph_length,
        STEM,
        slide_format=True,
        dpi=cfg.dpi,
        vector_export=True,
        failure_label=f"{status}*",
        footnote=footnote,
    )
    Image.MAX_IMAGE_PIXELS = 16 * 9 * cfg.dpi**2 + 1
    with Image.open(output / f"{STEM}.png") as full:
        assert full.size == (15360, 8640)
        full.resize((2560, 1440), Image.Resampling.LANCZOS).save(output / "preview.png")
    diagnostic_files = diagnostic_history(no_reset, reset, output, tolerance)
    receipt = {
        "height": aligned.HEIGHT,
        "cases": 8,
        "other_seven_geometries_match_original_exactly": True,
        "reset_source": str(RESET),
        "reset_case": reset.name,
        "forward_resets_used": summary["forward_resets_used"],
        "failed_forward_solves": failure_count,
        "final_next_seed_reset": summary["final_next_seed_reset"],
        "final_forward_converged": converged,
        "final_smallest_hessian_eigenvalue": eigenvalue,
        "final_equilibrium_unstable": unstable,
        "final_force_residual_inf": summary["final"]["force_residual_inf"],
        "footnote": footnote,
        "global_color_range": [-maximum, maximum],
        "figure": figure,
        "diagnostic_files": diagnostic_files,
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
    }
    (output / "alignment-checks.json").write_text(
        json.dumps(receipt, indent=2, allow_nan=False) + "\n"
    )
    (output / "index.html").write_text(f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Rest-reset continuation figures</title>
<style>body{{font:16px/1.5 system-ui,sans-serif;max-width:1600px;margin:auto;padding:2rem;background:#faf9f6;color:#24211f}}img{{width:100%;height:auto}}.note{{padding:1rem;background:#f0ece6;border-left:4px solid #a71930}}</style></head><body>
<h1>Free / off continuation with displacement resets</h1><p class="note">{footnote.replace(chr(10), "<br>")}<br>Nonconverged records are off-equilibrium iterates. This is a numerical continuation diagnostic, with no inverse-convergence or anatomical claim.</p>
<p><a href="{STEM}.png">Full PNG (15,360 × 8,640)</a> · <a href="{STEM}.svg">All-vector SVG</a> · <a href="{STEM}.pdf">All-vector PDF</a></p>
<a href="{STEM}.png"><img src="preview.png" alt="Aligned four-model off/on reset endpoint comparison"></a>
<h2>Reset versus no-reset recorded histories</h2><p><a href="reset-vs-no-reset-history.pdf">PDF</a> · <a href="reset-vs-no-reset-history.svg">SVG</a></p><img src="reset-vs-no-reset-history.png" alt="Recorded L2, force residual and min J histories">
</body></html>""")
    cherries.log_metrics(
        {
            "selected_cases": 8,
            "rest_seeded_solves": summary["forward_resets_used"],
            "failed_forward_solves": failure_count,
            "final_forward_converged": converged,
            "alignment_checks_passed": True,
        }
    )
    LOG.info("Wrote reset endpoint and diagnostic history figures to %s", output)


if __name__ == "__main__":
    cherries.main(
        main, profile=None if os.environ.get("DEBUG") else render.ProfileFigures
    )
