"""Plot the saved full force history and its Newton continuation."""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
sys.path.insert(0, str(ROOT / "exp/2026/09/21/joint-activation-material-mandible/src"))
from joint_common import ProfileJoint, sha256, write_json  # noqa: E402

FORCE_TO_NEWTONS = 1e6


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-repaired-reference-005"
    review_dir: Path = GROUP / "data/review-repaired-reference-005"
    overwrite: bool = False


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict) -> Path:
    path = Path(item["path"]).resolve()
    assert path.is_file(), path
    assert sha256(path) == item["sha256"], path
    return path


def load_rows(path: Path, source: str) -> list[dict]:
    return [
        dict(json.loads(line), source=source) for line in path.read_text().splitlines()
    ]


def plot(
    output: Path,
    data: list[dict],
    handoff: int,
    restart: int,
    threshold: float,
    events: list[dict],
    inverted: int,
) -> None:
    force = np.array([row["force_n"] for row in data])
    steps = np.arange(len(data))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 6), width_ratios=(1.4, 1))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.22, top=0.79, wspace=0.27)
    fig.suptitle(
        "Neutral solve and continuation · force residual",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=20,
    )
    fig.text(
        0.075,
        0.89,
        f"{handoff} PNCG + {len(data) - handoff - 1} Newton steps · continuation starts at step {restart} (Newton {restart - handoff})",
        fontsize=11,
        color="#505865",
    )
    colors = ("#2563a6", "#ce651c", "#008477")
    ranges = (
        (0, handoff + 1, "PNCG"),
        (handoff, restart + 1, "Newton · reset shifts"),
        (restart, len(data), "Continuation · shift reuse"),
    )
    for panel, ax in enumerate(axes):
        offset = 0 if panel == 0 else handoff
        ax.set_yscale("log")
        ax.set_facecolor("#fafbfc")
        ax.grid(which="major", color="#dce1e7", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylabel("Free-force norm (N)")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        for (start, end, label), color in zip(ranges, colors, strict=True):
            if panel == 1 and start == 0:
                continue
            # Do not connect residuals across a change of physical barrier stiffness.
            kappa = np.array([row["stiffness_mpa"] for row in data[start:end]])
            cuts = [start, *(start + np.flatnonzero(np.diff(kappa)) + 1), end]
            for part, (left, right) in enumerate(itertools.pairwise(cuts)):
                ax.plot(
                    steps[left:right] - offset,
                    force[left:right],
                    lw=1.9,
                    color=color,
                    label=label if part == 0 else None,
                )
        ax.axhline(
            threshold,
            color="#333333",
            ls="--",
            lw=1.3,
            label=f"Tolerance: {threshold:g} N",
        )
        ax.axvline(restart - offset, color="#80549c", ls=":", lw=1.4)
        ax.scatter(
            [len(data) - 1 - offset], [force[-1]], s=28, color=colors[2], zorder=5
        )
        ax.set_ylim(bottom=threshold * 0.65)
        ax.set_xlabel(
            "Accepted solver step" if panel == 0 else "Newton step (0 = phase start)"
        )
        ax.set_title(
            "Full force history" if panel == 0 else "Newton phase",
            loc="left",
            fontsize=12,
        )
    axes[0].axvline(handoff, color="#8b949e", ls=":", lw=1)
    for event in events:
        index = event["accepted_step"]
        axes[0].axvline(index, color="#9164a4", ls="--", lw=1)
        axes[0].annotate(
            f"κ: {event['stiffness_before']:g} → {event['stiffness_after']:g} MPa",
            xy=(index, force[index]),
            xytext=(-10, -28),
            textcoords="offset points",
            ha="right",
            fontsize=9,
            color="#744287",
        )
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    axes[1].text(
        restart - handoff + 2,
        0.91,
        "Shift reuse",
        color="#80549c",
        fontsize=10,
        transform=axes[1].get_xaxis_transform(),
    )
    axes[1].annotate(
        f"Final: {force[-1]:.6f} N",
        xy=(len(data) - 1 - handoff, force[-1]),
        xytext=(-9, 18),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color=colors[2],
    )
    fig.text(
        0.075,
        0.10,
        f"Force convergence reached: {force[-1]:.8f} N ≤ {threshold:g} N. Geometry remains invalid: {inverted} inverted tetrahedra.",
        fontsize=10,
        color="#8c3030",
    )
    fig.text(
        0.075,
        0.05,
        "Logarithmic force axis · unsmoothed saved samples · barrier-stiffness changes can jump the residual · 1 MPa·m² = 10⁶ N",
        fontsize=9,
        color="#505865",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"force-curve.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main(cfg: Config) -> None:  # noqa: PLR0915
    run, review = cfg.run_dir.resolve(), cfg.review_dir.resolve()
    output = review / "force"
    assert cfg.overwrite or not output.exists(), output
    review_receipt = json.loads((review / "receipt.json").read_text())
    protocol_path = verified(review_receipt["run"]["protocol"])
    summary_path = verified(review_receipt["run"]["summary"])
    trace_path = verified(review_receipt["run"]["trace"])
    assert protocol_path == run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    continuation = protocol["continuation"]
    parent_dir = Path(continuation["parent_directory"]).resolve()
    parent_inputs = {
        key: verified(continuation[f"parent_{key}"])
        for key in ("protocol", "summary", "endpoint", "stiffness", "trace")
    }
    assert all(path.parent == parent_dir for path in parent_inputs.values())
    parent_protocol = json.loads(parent_inputs["protocol"].read_text())
    assert "continuation" not in parent_protocol
    parent_result = json.loads(parent_inputs["summary"].read_text())["result"]
    rows = load_rows(parent_inputs["trace"], "parent")
    current_rows = load_rows(trace_path, "continuation")
    initial = current_rows.pop(0)
    assert initial["kind"] == "initial"
    for key, terminal in (
        ("force", "terminal_force"),
        ("energy", "terminal_energy"),
        ("stiffness_mpa", "terminal_stiffness_mpa"),
    ):
        np.testing.assert_allclose(
            initial[key], parent_result[terminal], rtol=1e-12, atol=0
        )
    parent_samples = sum(row["kind"] in {"initial", "pncg", "newton"} for row in rows)
    restart = parent_samples - 1
    rows += current_rows
    data, event_indices = [], {}
    for row in rows:
        if row["kind"] not in {"initial", "pncg", "newton"}:
            assert row["kind"] == "stiffness_change"
            continue
        key = (row["source"], row["kind"], row["step"])
        assert key not in event_indices
        event_indices[key] = len(data)
        data.append(
            {
                "accepted_step": len(data),
                "source": row["source"],
                "phase": row["kind"],
                "phase_step": row["step"],
                "force_mpa_m2": row["force"],
                "force_n": FORCE_TO_NEWTONS * row["force"],
                "stiffness_mpa": row["stiffness_mpa"],
                "newton_shift": row.get("accepted_step", {}).get("shift"),
            }
        )
    events = [
        dict(
            row, accepted_step=event_indices[(row["source"], row["phase"], row["step"])]
        )
        for row in rows
        if row["kind"] == "stiffness_change"
    ]
    handoff = sum(row["phase"] == "pncg" for row in data)
    force = np.array([row["force_n"] for row in data])
    assert np.isfinite(force).all()
    assert np.all(force > 0)
    threshold_raw = protocol["contact_stiffness_policy"]["effective_force_tolerance"]
    assert (
        threshold_raw
        == parent_protocol["contact_stiffness_policy"]["effective_force_tolerance"]
    )
    threshold = FORCE_TO_NEWTONS * threshold_raw
    result = summary["result"]
    np.testing.assert_allclose(
        data[-1]["force_mpa_m2"], result["terminal_force"], rtol=1e-12, atol=0
    )
    assert result["success"]
    assert force[-1] <= threshold
    assert result["collision"]["state_feasible"]
    inverted = result["geometry"]["inverted_tetrahedra"]
    output.mkdir(exist_ok=cfg.overwrite)
    with (output / "force-samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    plot(output, data, handoff, restart, threshold, events, inverted)
    (
        output / "index.html"
    ).write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Neutral force curve</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem;color:#202124}}img{{width:100%;height:auto}}.status{{padding:1rem;background:#ffe2dc}}</style>
<p><a href="../">Back to neutral review</a> · <a href="../energy/">Energy curve</a></p><h1>Force convergence curve</h1>
<p>{len(data)} recorded states: {handoff} PNCG + {len(data) - handoff - 1} Newton steps. At step {restart}, continuation changes the Newton shift policy from reset to reuse. The original force tolerance stays {threshold:g} N.</p>
<a href="force-curve.png"><img src="force-curve.svg" alt="Full free-force residual history and Newton detail on logarithmic axes, with original tolerance and continuation marked"></a>
<p class="status">Force converged: {force[-1]:.9f} N ≤ {threshold:g} N. Geometry remains invalid: {inverted} inverted tetrahedra. Contact checks pass.</p>
<p>The spike at step 193 coincides with the adaptive barrier stiffness doubling. Residuals are shown at their recorded stiffness; no smoothing or solver rerun is used.</p>
<p>Downloads: <a href="force-curve.png">PNG</a> · <a href="force-curve.svg">SVG</a> · <a href="force-curve.pdf">PDF</a> · <a href="force-samples.csv">CSV samples</a> · <a href="receipt.json">Provenance</a>.</p></html>""")
    index = review / "index.html"
    original_index = record(index)
    html = index.read_text()
    if 'id="force-detail"' in html:
        assert cfg.overwrite
    else:
        html = html.replace(
            "</h1>",
            '</h1><p id="force-detail"><a href="force/">Full force curve and convergence threshold</a></p>',
            1,
        )
    html = html.replace('href="force-trace.png"', 'href="force/"').replace(
        'src="force-trace.png"', 'src="force/force-curve.png"'
    )
    html = html.replace(
        "Recorded hybrid forward free-force trace.",
        "Full force history with continuation and original convergence threshold.",
    )
    index.write_text(html)
    receipt = {
        "schema": "saved-neutral-force-curve-v1",
        "samples": len(data),
        "inputs": {
            "protocol": record(protocol_path),
            "summary": record(summary_path),
            "trace": record(trace_path),
            "review": record(review / "receipt.json"),
            "parent": {name: record(path) for name, path in parent_inputs.items()},
        },
        "pncg_steps": handoff,
        "newton_steps": len(data) - handoff - 1,
        "continuation_restart_step": restart,
        "continuation_newton_steps": len(data) - parent_samples,
        "duplicate_initial_dropped": True,
        "force_to_newtons": FORCE_TO_NEWTONS,
        "initial_force_n": force[0],
        "restart_force_n": force[restart],
        "terminal_force_n": force[-1],
        "threshold_n": threshold,
        "solver_converged": result["success"],
        "valid_forward": result["valid_forward"],
        "inverted_tetrahedra": inverted,
        "stiffness_events": events,
        "index_before": original_index,
        "index_after": record(index),
        "source": record(Path(__file__)),
        "assets": {
            path.name: record(path)
            for path in sorted(output.iterdir())
            if path.name != "receipt.json"
        },
    }
    write_json(output / "receipt.json", receipt)
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "force/terminal_n": force[-1],
            "force/threshold_n": threshold,
            "force/samples": len(data),
            "force/continuation_step": restart,
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
