"""Plot saved hybrid energy samples without evaluating or rerunning the solver."""

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

ENERGY_TO_JOULES = 1e6


class Config(cherries.BaseConfig):
    run_dir: Path = GROUP / "data/forward-repaired-reference-004"
    review_dir: Path = GROUP / "data/review-repaired-reference-004"


def record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path)}


def verified(item: dict) -> Path:
    path = Path(item["path"])
    assert path.is_file(), path
    assert sha256(path) == item["sha256"], path
    return path


def load_rows(path: Path, source: str) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows
    for row in rows:
        assert isinstance(row, dict)
        assert isinstance(row.get("kind"), str)
        row["_source"] = source
    return rows


def samples(rows: list[dict]) -> tuple[list[dict], int, list[dict]]:
    pncg = [r for r in rows if r["kind"] == "pncg"]
    count = len(pncg)
    result = []
    accepted_step = {}
    for row in rows:
        if row["kind"] not in {"initial", "pncg", "newton"}:
            continue
        key = (row["_source"], row["kind"], int(row.get("step", 0)))
        assert key not in accepted_step, key
        accepted_step[key] = len(result)
        result.append(
            {
                "accepted_step": len(result),
                "phase": row["kind"],
                "phase_step": row.get("step", 0),
                "source": row["_source"],
                "elapsed_seconds": row["elapsed_seconds"],
                "energy_j": ENERGY_TO_JOULES * row["energy"],
                "energy_mpa_m3": row["energy"],
                "stiffness_mpa": row["stiffness_mpa"],
            }
        )
    events = []
    for row in rows:
        if row["kind"] != "stiffness_change":
            continue
        phase = row["phase"]
        assert phase in {"pncg", "newton"}, phase
        key = (row["_source"], phase, int(row["step"]))
        assert key in accepted_step, key
        events.append(
            {
                **{key: value for key, value in row.items() if key != "_source"},
                "accepted_step": accepted_step[key],
                "source": row["_source"],
            }
        )
    assert [r["accepted_step"] for r in result] == list(range(len(result)))
    assert np.isfinite([r["energy_j"] for r in result]).all()
    return result, count, events


def plot(output: Path, data: list[dict], pncg_steps: int, events: list[dict]) -> None:
    energy = np.array([r["energy_j"] for r in data])
    steps = np.arange(len(data))
    newton_steps = len(data) - pncg_steps - 1
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.2), width_ratios=(1.5, 1))
    fig.subplots_adjust(left=0.075, right=0.98, top=0.79, bottom=0.22, wspace=0.28)
    fig.suptitle(
        "Neutral solve and continuation · total energy",
        x=0.075,
        y=0.97,
        ha="left",
        fontsize=19,
    )
    fig.text(
        0.075,
        0.89,
        f"Active strain · repaired reference · {pncg_steps} PNCG + {newton_steps} Newton steps · cumulative accepted steps",
        fontsize=11,
        color="#505865",
    )
    for ax in axes:
        ax.set_facecolor("#fafbfc")
        ax.grid(color="#dce1e7", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_ylabel("Total energy (J)")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax = axes[0]

    def segments(values: np.ndarray) -> list[tuple[int, int]]:
        cuts = [0, *(np.flatnonzero(np.diff(values) != 0) + 1), len(values)]
        return list(itertools.pairwise(cuts))

    pncg_energy = energy[: pncg_steps + 1]
    pncg_stiffness = np.array([r["stiffness_mpa"] for r in data[: pncg_steps + 1]])
    # Disconnect samples evaluated at different recorded barrier stiffnesses.
    for segment, (start, end) in enumerate(segments(pncg_stiffness)):
        ax.plot(
            steps[start:end],
            pncg_energy[start:end],
            color="#2563a6",
            lw=1.9,
            label="PNCG" if segment == 0 else None,
        )
    newton_energy = energy[pncg_steps:]
    newton_stiffness = np.array([r["stiffness_mpa"] for r in data[pncg_steps:]])
    for segment, (start, end) in enumerate(segments(newton_stiffness)):
        ax.plot(
            steps[pncg_steps + start : pncg_steps + end],
            newton_energy[start:end],
            color="#ce651c",
            lw=2.1,
            label="Newton-CG" if segment == 0 else None,
        )
    ax.axvline(pncg_steps, color="#8b949e", lw=1, ls=":")
    for event in events:
        step = event["accepted_step"]
        ax.axvline(step, color="#9164a4", lw=1.2, ls="--")
        ax.text(
            step + max(1, len(data) // 100),
            0.80,
            f"{event.get('phase', 'solver')} κ: {event['stiffness_before']:g} → {event['stiffness_after']:g} MPa",
            transform=ax.get_xaxis_transform(),
            fontsize=9,
            color="#744287",
        )
    ax.scatter(
        [0, len(data) - 1],
        energy[[0, -1]],
        color=["#2563a6", "#ce651c"],
        s=22,
        zorder=4,
    )
    ax.set_xlabel("Accepted solver step")
    ax.set_title("Full solve", loc="left", fontsize=12)
    ax.legend(frameon=False, loc="upper right")
    ax = axes[1]
    for start, end in segments(newton_stiffness):
        ax.plot(
            np.arange(start, end),
            newton_energy[start:end],
            color="#ce651c",
            lw=2,
        )
    ax.set_xlabel("Newton step (0 = phase start)")
    ax.set_title("Newton phase · enlarged energy scale", loc="left", fontsize=12)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.annotate(
        f"Final: {energy[-1]:.9f} J",
        xy=(newton_steps, energy[-1]),
        xytext=(-5, 10),
        textcoords="offset points",
        ha="right",
        fontsize=10,
        color="#99450d",
    )
    fig.text(
        0.075,
        0.09,
        "Recorded energy includes the adaptive IPC barrier; κ changes alter the objective.",
        fontsize=10,
        color="#505865",
    )
    fig.text(
        0.075,
        0.045,
        "Small energy changes do not establish force convergence. Conversion: 1 MPa·m³ = 10⁶ J.",
        fontsize=9,
        color="#505865",
    )
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(output / f"energy-curve.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main(cfg: Config) -> None:  # noqa: PLR0915
    run, review = cfg.run_dir.resolve(), cfg.review_dir.resolve()
    output = review / "energy"
    assert not output.exists(), output
    parent = json.loads((review / "receipt.json").read_text())
    trace_path, summary_path = run / "trace.jsonl", run / "summary.json"
    assert record(trace_path) == parent["run"]["trace"]
    assert record(summary_path) == parent["run"]["summary"]
    summary = json.loads(summary_path.read_text())
    inverted_tetrahedra = int(parent["geometry"]["inverted_tetrahedra"])
    geometry_valid = inverted_tetrahedra == 0
    valid_forward = bool(parent["result"]["valid_forward"])
    protocol_path = run / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    current_rows = load_rows(trace_path, "continuation")
    continuation = protocol.get("continuation")
    inputs = {"trace": record(trace_path), "summary": record(summary_path)}
    if continuation is None:
        rows = current_rows
        continuation_receipt = None
    else:
        assert isinstance(continuation["parent_directory"], str)
        parent_dir = Path(continuation["parent_directory"]).resolve()
        assert parent_dir.is_dir(), parent_dir
        parent_protocol = verified(continuation["parent_protocol"])
        parent_summary_path = verified(continuation["parent_summary"])
        parent_endpoint = verified(continuation["parent_endpoint"])
        verified(continuation["parent_stiffness"])
        assert parent_protocol == parent_dir / "protocol.json"
        assert parent_summary_path == parent_dir / "summary.json"
        assert parent_endpoint == parent_dir / "endpoint.npz"
        parent_trace = parent_dir / "trace.jsonl"
        assert parent_trace.is_file(), parent_trace
        assert record(parent_trace) == continuation["parent_trace"]
        parent_rows = load_rows(parent_trace, "parent")
        assert parent_rows[0]["kind"] == "initial"
        assert current_rows[0]["kind"] == "initial"
        duplicate_initial = current_rows.pop(0)
        parent_summary = json.loads(parent_summary_path.read_text())
        np.testing.assert_allclose(
            duplicate_initial["energy"],
            parent_summary["result"]["terminal_energy"],
            rtol=1e-12,
            atol=0,
        )
        np.testing.assert_allclose(
            duplicate_initial["stiffness_mpa"],
            parent_summary["result"]["terminal_stiffness_mpa"],
            rtol=1e-12,
            atol=0,
        )
        rows = parent_rows + current_rows
        continuation_receipt = {
            "parent_directory": str(parent_dir),
            "parent_protocol": continuation["parent_protocol"],
            "parent_summary": continuation["parent_summary"],
            "parent_endpoint": continuation["parent_endpoint"],
            "parent_stiffness": continuation["parent_stiffness"],
            "parent_trace": continuation["parent_trace"],
            "duplicate_continuation_initial_dropped": True,
        }
        inputs["protocol"] = record(protocol_path)
    data, pncg_steps, events = samples(rows)
    np.testing.assert_allclose(
        data[-1]["energy_mpa_m3"],
        summary["result"]["terminal_energy"],
        rtol=1e-12,
        atol=0,
    )
    output.mkdir()
    with (output / "energy-samples.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    plot(output, data, pncg_steps, events)
    receipt = {
        "schema": "saved-neutral-energy-curve-v1",
        "inputs": inputs,
        "continuation": continuation_receipt,
        "samples": len(data),
        "pncg_steps": pncg_steps,
        "newton_steps": len(data) - pncg_steps - 1,
        "initial_energy_j": data[0]["energy_j"],
        "terminal_energy_j": data[-1]["energy_j"],
        "raw_units": "MPa*m^3",
        "joule_conversion_factor": ENERGY_TO_JOULES,
        "stiffness_events": events,
        "semantics": "Recorded total material plus IPC energy at each accepted state and its recorded barrier stiffness. For a continuation, parent and continuation traces are concatenated after dropping the duplicate continuation initial sample; accepted-step indices are cumulative. Curves are disconnected across recorded stiffness changes. No solver rerun or interpolation.",
        "solver_converged": summary["result"]["success"],
        "valid_forward": valid_forward,
        "geometry_valid": geometry_valid,
        "inverted_tetrahedra": inverted_tetrahedra,
        "terminal_force_n": parent["terminal_force_n"],
        "force_threshold_n": parent["force_threshold_n"],
        "assets": {
            name: record(output / name)
            for name in (
                "energy-curve.png",
                "energy-curve.svg",
                "energy-curve.pdf",
                "energy-samples.csv",
            )
        },
    }
    write_json(output / "receipt.json", receipt)
    html = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Neutral energy curve</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1rem;color:#202124}}img{{width:100%;height:auto}}.status{{padding:1rem;background:#fff1c7}}</style>
<p><a href="../">Back to neutral review</a></p><h1>Energy curve</h1>
<p>{len(data)} recorded samples: {data[0]["energy_j"]:.9f} → {data[-1]["energy_j"]:.9f} J. The dashed marker shows the adaptive barrier-stiffness change.</p>
<a href="energy-curve.png"><img src="energy-curve.svg" alt="Total energy across the hybrid solve, with Newton phase detail and adaptive barrier stiffness marked"></a>
<p class="status">Force solver converged: {receipt["solver_converged"]}. Forward validity: {receipt["valid_forward"]}. Geometry valid: {receipt["geometry_valid"]}; inverted tetrahedra: {receipt["inverted_tetrahedra"]}. Final force {receipt["terminal_force_n"]:.6g} N; required {receipt["force_threshold_n"]:.6g} N.</p>
<p>Downloads: <a href="energy-curve.png">PNG</a> · <a href="energy-curve.svg">SVG</a> · <a href="energy-curve.pdf">PDF</a> · <a href="energy-samples.csv">CSV samples</a> · <a href="receipt.json">Provenance</a>.</p>
<p>Energy includes materials and the IPC barrier at each recorded accepted state and its recorded stiffness. Curves are disconnected when that stiffness changes. The data is read from the saved solve trace; no solve was rerun.</p></html>"""
    (output / "index.html").write_text(html)
    index = review / "index.html"
    text = index.read_text()
    assert 'id="energy-detail"' not in text
    index.write_text(
        text.replace(
            "</h1>",
            '</h1><p id="energy-detail"><a href="energy/">Energy curve and Newton detail</a></p>',
            1,
        )
    )
    cherries.log_output(output)
    cherries.log_metrics(
        {
            "energy/initial_j": data[0]["energy_j"],
            "energy/terminal_j": data[-1]["energy_j"],
            "energy/samples": len(data),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
