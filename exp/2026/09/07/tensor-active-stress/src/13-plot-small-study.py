"""Render a compact, unit-labelled comparison from final small-study receipts."""
# ruff: noqa: PLR0915

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pydantic_settings as ps
from experiment_profile import ProfileCometNoCommit

from liblaf import cherries

HERE = Path(__file__).resolve().parent.parent
COMPLETED = False


class Config(cherries.BaseConfig):
    """Paths for the receipt-derived post-hoc comparison."""

    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_summary: Path = HERE / "data/12-small-model-study-v4/summary.json"
    interpretation_correction: Path = (
        HERE / "data/12-small-model-study-v4/interpretation-correction.json"
    )
    output_dir: Path = HERE / "data/13-small-study-comparison-v5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main(cfg: Config) -> None:
    """Plot final receipts only; this script performs no new equilibrium solve."""
    global COMPLETED  # noqa: PLW0603
    out = cfg.output_dir
    out.mkdir(parents=True, exist_ok=False)
    cherries.log_input(cfg.input_summary)
    cherries.log_input(cfg.interpretation_correction)
    summary: dict[str, Any] = json.loads(cfg.input_summary.read_text())
    correction: dict[str, Any] = json.loads(cfg.interpretation_correction.read_text())
    assert summary["status"] == "completed"
    assert correction["status"] == "metadata_correction"
    assert correction["affected_artifact"]["sha256"] == sha256(cfg.input_summary)
    assert correction["numerical_results_unchanged"] is True
    source = Path(__file__)
    (out / source.name).write_bytes(source.read_bytes())
    # Preserve the final numerical receipt byte-for-byte.  Its stale prose is
    # superseded below by the copied, hash-bound interpretation correction.
    (out / "summary.json").write_bytes(cfg.input_summary.read_bytes())
    correction_copy = out / "interpretation-correction.json"
    correction_copy.write_bytes(cfg.interpretation_correction.read_bytes())

    methods = [
        ("Q_balance_Qref", "Qref balance"),
        ("Q_balance_10Qref", "10Qref balance"),
        ("active_strain", "active strain"),
        ("active_strain_10x_muscle_E", "active strain x10 E"),
        ("Q_cap_Qref", "Qref bounded search"),
        ("Q_cap_10Qref", "10Qref bounded search"),
    ]
    records = {(state["target"], state["id"]): state for state in summary["states"]}
    labels = [label for _, label in methods]
    compatible = [
        records[("uniform_diagonal_Fa", key)]["target_shape_rms"] for key, _ in methods
    ]
    incompatible = [
        records[("incompatible_shared_edge_length", key)]["target_shape_rms"]
        for key, _ in methods
    ]
    edge = summary["incompatible_shared_edge"]
    requested = edge["requested_shared_edge_lengths"]
    projected = sum(requested) / 2
    bound = edge["analytical_minimax_length_mismatch_bound"]

    fig, (errors, lengths) = plt.subplots(
        1, 2, figsize=(12.0, 5.2), layout="constrained"
    )
    y = np.arange(len(labels))
    errors.scatter(
        compatible, y, marker="o", color="#1769aa", label="uniform compatible target"
    )
    errors.scatter(
        incompatible,
        y,
        marker="s",
        color="#d95f02",
        label="nodal compromise for conflicting commands",
    )
    errors.set_xscale("log")
    errors.set_yticks(y, labels)
    errors.invert_yaxis()
    errors.grid(axis="x", alpha=0.25)
    errors.set_xlabel("nodal target-shape RMS [fixture length]")
    errors.set_title("Error to each feasible nodal target")
    errors.legend(fontsize=8, loc="upper right")

    lengths.scatter(
        [requested[0], requested[1]],
        [0, 1],
        s=70,
        color="#d95f02",
        label="per-cell requested length",
    )
    lengths.axvline(projected, color="#1769aa", lw=2, label="nodal compromise edge")
    lengths.axvspan(
        projected - bound,
        projected + bound,
        color="#1769aa",
        alpha=0.13,
        label="minimax mismatch bound",
    )
    lengths.set_yticks([0, 1], ["cell 0 request", "cell 1 request"])
    lengths.set_ylim(-0.6, 1.6)
    lengths.set_xlabel("shared-edge length [rest edge length]")
    lengths.set_title("Kinematic incompatibility of the requested target")
    lengths.grid(axis="x", alpha=0.25)
    lengths.legend(fontsize=8, loc="upper left")
    lengths.text(
        0.02,
        0.03,
        f"requests: {requested[0]:.3f}, {requested[1]:.3f};\nany one realized edge misses at least one by {bound:.3f}",
        transform=lengths.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.suptitle(
        "Two shared-face mixed-material tetrahedra: receipt-derived comparison",
        fontsize=13,
    )
    png = out / "contraction-and-target-error.png"
    fig.savefig(png, dpi=220)
    fig.savefig(png.with_suffix(".pdf"))
    plt.close(fig)

    manifest = {
        "status": "completed",
        "scope": "Post-hoc visualization only; no new equilibrium or fitting calculation.",
        "input_summary": {
            "path": str(cfg.input_summary),
            "sha256": sha256(cfg.input_summary),
            "copied_as": "summary.json",
            "preservation": "Exact numerical receipt copy; its cap_interpretation prose is superseded by interpretation_correction.",
        },
        "interpretation_correction": {
            "path": str(cfg.interpretation_correction),
            "sha256": sha256(cfg.interpretation_correction),
            "copied_as": correction_copy.name,
            "supersedes_input_summary_field": "cap_interpretation",
            "correct_face_pilot_cap_mpa": correction["authoritative_face_settings"][
                "stress_cap_mpa"
            ],
            "numerical_results_unchanged": correction["numerical_results_unchanged"],
        },
        "outputs": {path.name: sha256(path) for path in (png, png.with_suffix(".pdf"))},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for path in out.iterdir():
        if path.is_file():
            cherries.log_output(path)
    (out / "COMPLETED").write_text("completed\n")
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileCometNoCommit)
    if not COMPLETED:
        raise SystemExit(1)
