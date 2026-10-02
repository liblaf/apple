"""Render one hash-proven branch of neutral preparation history."""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import torch
from joint_common import GROUP, ProfileJoint, sha256, write_json
from joint_field_visuals import coefficient_summary

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    tip_checkpoint: Path = (
        GROUP
        / "data/neutral-convergence-010-contact-metric-bfgs-segment-003/checkpoint-0030.pt"
    )
    output_dir: Path = GROUP / "data/neutral-lineage-visuals"


@dataclass(frozen=True)
class FrozenFile:
    path: Path
    sha256: str
    size: int
    mtime_ns: int
    content: bytes


@dataclass
class Segment:
    checkpoint: dict[str, Any]
    checkpoint_file: FrozenFile
    parent_path: Path
    parent_sha256: str
    run_dir: Path
    trace_file: FrozenFile
    trace_total_rows: int
    rows: list[dict[str, Any]]
    solver: str
    outer_method: str
    regime: str


def freeze(path: Path) -> FrozenFile:
    path = path.resolve()
    before = path.stat()
    content = path.read_bytes()
    after = path.stat()
    assert (
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ) == (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ), f"source changed during read: {path}"
    assert len(content) == after.st_size
    return FrozenFile(
        path=path,
        sha256=hashlib.sha256(content).hexdigest(),
        size=after.st_size,
        mtime_ns=after.st_mtime_ns,
        content=content,
    )


def load_checkpoint(path: Path) -> tuple[dict[str, Any], FrozenFile]:
    source = freeze(path)
    value = torch.load(
        io.BytesIO(source.content), map_location="cpu", weights_only=False
    )
    assert value["schema"] == "joint-inverse-checkpoint-v1"
    assert value["stage"] == "neutral"
    return value, source


def resolve_recorded_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else GROUP / path


def method_labels(
    protocol: dict[str, Any], rows: list[dict[str, Any]]
) -> tuple[str, str, str]:
    forward_solver = protocol.get("forward_solver")
    if forward_solver is None:
        assert all(row["forward"].get("method") is None for row in rows)
        solver = "PNCG (legacy receipt)"
    else:
        assert forward_solver["method"] == "newton_cg"
        solver = "Newton-CG"
    outer = protocol.get("outer_optimizer")
    if outer is None:
        assert "spectral projected gradient" in protocol["optimizer"]
        outer_method = "SPG"
    else:
        assert outer["method"] == "bfgs"
        outer_method = "metric BFGS" if "metric_projection" in outer else "BFGS"
    return solver, outer_method, f"{solver} + {outer_method}"


def build_lineage(path: Path) -> tuple[list[Segment], dict[str, Any], FrozenFile]:
    checkpoint, checkpoint_file = load_checkpoint(path)
    segments_reversed: list[Segment] = []
    root_checkpoint: dict[str, Any] | None = None
    root_file: FrozenFile | None = None
    seen: set[str] = set()
    while True:
        assert checkpoint_file.sha256 not in seen, "checkpoint ancestry cycle"
        seen.add(checkpoint_file.sha256)
        protocol = checkpoint.get("protocol")
        if protocol is None or "initial_checkpoint_sha256" not in protocol:
            root_checkpoint = checkpoint
            root_file = checkpoint_file
            break
        config = protocol["config"]
        run_dir = resolve_recorded_path(config["output_dir"]).resolve()
        trace_file = freeze(run_dir / "trace.json")
        trace = json.loads(trace_file.content)
        assert isinstance(trace, list)
        assert trace
        endpoint = int(checkpoint["update"])
        rows = [row for row in trace if int(row["update"]) <= endpoint]
        assert rows
        assert [int(row["update"]) for row in rows] == list(range(endpoint + 1))
        terminal_row = rows[-1]
        assert int(terminal_row["update"]) == endpoint
        assert int(checkpoint["metrics"]["update"]) == endpoint
        assert terminal_row["objective"] == checkpoint["metrics"]["objective"]
        assert np.array_equal(
            np.asarray(terminal_row["shared"]),
            checkpoint["shared_coefficients"].detach().cpu().numpy(),
        )
        parent_path = resolve_recorded_path(config["initial_checkpoint"]).resolve()
        parent, parent_file = load_checkpoint(parent_path)
        parent_sha256 = protocol["initial_checkpoint_sha256"]
        assert parent_file.sha256 == parent_sha256, (
            parent_path,
            parent_file.sha256,
            parent_sha256,
        )
        solver, outer_method, regime = method_labels(protocol, rows)
        segments_reversed.append(
            Segment(
                checkpoint=checkpoint,
                checkpoint_file=checkpoint_file,
                parent_path=parent_path,
                parent_sha256=parent_sha256,
                run_dir=run_dir,
                trace_file=trace_file,
                trace_total_rows=len(trace),
                rows=rows,
                solver=solver,
                outer_method=outer_method,
                regime=regime,
            )
        )
        assert freeze(checkpoint_file.path).sha256 == checkpoint_file.sha256
        checkpoint = parent
        checkpoint_file = parent_file
    assert root_checkpoint is not None
    assert root_file is not None
    segments = list(reversed(segments_reversed))
    assert segments
    for previous, current in itertools.pairwise(segments):
        assert current.parent_sha256 == previous.checkpoint_file.sha256
    return segments, root_checkpoint, root_file


def combine_rows(
    segments: list[Segment],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    combined: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    elapsed_offset = 0.0
    update_offset = 0
    previous_regime: str | None = None
    previous_fraction: float | None = None
    previous_basis: str | None = None
    for index, segment in enumerate(segments):
        fraction = float(segment.checkpoint["protocol"]["skin_prestress_fraction"])
        basis = segment.checkpoint["materials"]["schema"]
        basis_transition = index > 0 and basis != previous_basis
        target_transition = index > 0 and fraction != previous_fraction
        selected = (
            segment.rows
            if index == 0 or target_transition or basis_transition
            else segment.rows[1:]
        )
        assert selected, segment.run_dir
        start = update_offset + int(selected[0]["update"])
        for row in selected:
            copied = dict(row)
            copied["lineage_cumulative_update"] = update_offset + int(row["update"])
            copied["lineage_cumulative_elapsed_seconds"] = elapsed_offset + float(
                row["elapsed_seconds"]
            )
            copied["lineage_segment"] = segment.run_dir.name
            copied["lineage_regime"] = segment.regime
            copied["lineage_prestress_fraction"] = fraction
            copied["lineage_field_summary"] = coefficient_summary(
                row["shared"], segment.checkpoint["materials"]
            )
            combined.append(copied)
        update_offset += int(segment.rows[-1]["update"])
        end = update_offset
        elapsed_offset += float(segment.rows[-1]["elapsed_seconds"])
        transition = segment.regime != previous_regime
        receipts.append(
            {
                "run_dir": str(segment.run_dir),
                "trace": {
                    "path": str(segment.trace_file.path),
                    "sha256": segment.trace_file.sha256,
                    "size": segment.trace_file.size,
                    "mtime_ns": segment.trace_file.mtime_ns,
                    "selected_local_updates": [
                        int(selected[0]["update"]),
                        int(selected[-1]["update"]),
                    ],
                    "excluded_tail_rows": segment.trace_total_rows - len(segment.rows),
                },
                "checkpoint": {
                    "path": str(segment.checkpoint_file.path),
                    "sha256": segment.checkpoint_file.sha256,
                    "size": segment.checkpoint_file.size,
                    "mtime_ns": segment.checkpoint_file.mtime_ns,
                    "local_update": int(segment.checkpoint["update"]),
                },
                "parent_checkpoint": {
                    "path": str(segment.parent_path),
                    "sha256": segment.parent_sha256,
                },
                "cumulative_update_range": [start, end],
                "segment_elapsed_seconds": float(segment.rows[-1]["elapsed_seconds"]),
                "solver": segment.solver,
                "outer_method": segment.outer_method,
                "regime": segment.regime,
                "method_transition": transition,
                "target_transition": target_transition,
                "basis_transition": basis_transition,
                "shared_field_schema": basis,
                "skin_prestress_fraction": fraction,
            }
        )
        previous_regime = segment.regime
        previous_fraction = fraction
        previous_basis = basis
    assert combined[0]["lineage_cumulative_update"] == 0
    assert all(
        right["lineage_cumulative_update"] - left["lineage_cumulative_update"] in (0, 1)
        for left, right in itertools.pairwise(combined)
    )
    return combined, receipts


def add_method_transitions(
    axis: plt.Axes, receipts: list[dict[str, Any]], *, annotate: bool
) -> None:
    display = {
        "PNCG (legacy receipt) + SPG": "PNCG + SPG",
        "Newton-CG + SPG": "Newton + SPG",
        "Newton-CG + BFGS": "Newton + BFGS",
        "Newton-CG + metric BFGS": "Newton + metric BFGS",
    }
    transitions = [
        item
        for item in receipts
        if item["method_transition"]
        or item["target_transition"]
        or item["basis_transition"]
    ]
    for index, item in enumerate(transitions):
        start = item["cumulative_update_range"][0]
        if start > 0:
            axis.axvline(start - 0.5, color="#77818b", linewidth=0.8, linestyle=":")
        if annotate:
            label = (
                "Spatial baseline"
                if item["basis_transition"]
                else f"{100 * item['skin_prestress_fraction']:g}% prestress"
                if item["target_transition"]
                else display[item["regime"]]
            )
            axis.text(
                0.02,
                0.98 - 0.065 * index,
                f"{int(start)}: {label}",
                color="#39434d",
                fontsize=7,
                ha="left",
                va="top",
                transform=axis.transAxes,
            )


def render(  # noqa: PLR0915 - one compact, consistently styled multipanel figure.
    output: Path,
    rows: list[dict[str, Any]],
    receipts: list[dict[str, Any]],
    materials: dict[str, Any],
) -> list[str]:
    updates = np.asarray([row["lineage_cumulative_update"] for row in rows])
    elapsed_hours = (
        np.asarray([row["lineage_cumulative_elapsed_seconds"] for row in rows]) / 3600
    )
    objective = np.asarray([row["objective"] for row in rows])
    projected_gradient = np.asarray([row["projected_gradient_inf"] for row in rows])
    assert np.isfinite(objective).all()
    assert (objective > 0).all()
    assert np.isfinite(projected_gradient).all()
    assert (projected_gradient > 0).all()
    field_summaries = [row["lineage_field_summary"] for row in rows]
    spectra_kpa = np.asarray(
        [row["principal_anchor_min_mean_max_kpa"] for row in field_summaries]
    )
    resultants_n_per_m = np.asarray(
        [row["skin_resultant_n_per_m"] for row in field_summaries]
    )
    stiffness = np.asarray(
        [row["skin_stiffness_multiplier"] for row in field_summaries]
    )
    spatial = any(row["spatial"] for row in field_summaries)
    metrics = [row["metrics"] for row in rows]
    pg_tolerance = float(rows[-1]["convergence"]["projected_gradient_inf_tolerance"])
    skin_bounds = materials["constraints"]["skin_multiplier_bounds"]
    colors = ("#007d85", "#c26724", "#6845a3", "#426488")
    plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 10})
    fig, axes = plt.subplots(4, 2, figsize=(10, 14), layout="constrained")

    axes[0, 0].plot(updates, objective, color=colors[0], linewidth=1.5)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Objective on the selected branch", loc="left")
    axes[0, 0].set_ylabel("Dimensionless loss")
    add_method_transitions(axes[0, 0], receipts, annotate=True)

    axes[0, 1].semilogy(updates, projected_gradient, color=colors[0], linewidth=1.5)
    axes[0, 1].axhline(
        pg_tolerance,
        color=colors[1],
        linestyle="--",
        label=f"Required ≤ {pg_tolerance:g}",
    )
    axes[0, 1].set_title("Projected-gradient stationarity", loc="left")
    axes[0, 1].set_ylabel("Unit-step gradient mapping, max")
    axes[0, 1].legend(fontsize=7)

    axes[1, 0].plot(
        updates,
        [row["surface_motion_rms_mm"] for row in metrics],
        color=colors[0],
        label="Skin-surface RMS",
    )
    axes[1, 0].plot(
        updates,
        [row["muscle_centroid_motion_rms_mm"] for row in metrics],
        color=colors[1],
        label="Muscle-centroid RMS",
    )
    axes[1, 0].axhline(
        0.25, color=colors[0], linestyle="--", label="Skin bound 0.25 mm"
    )
    axes[1, 0].axhline(
        0.5, color=colors[1], linestyle="--", label="Muscle bound 0.5 mm"
    )
    axes[1, 0].set_title("Neutral geometry budgets", loc="left")
    axes[1, 0].set_ylabel("Reference displacement (mm)")
    axes[1, 0].legend(fontsize=7, ncol=2)

    axes[1, 1].plot(updates, elapsed_hours, color=colors[3], linewidth=1.5)
    axes[1, 1].set_title("Accumulated runtime of selected trace slices", loc="left")
    axes[1, 1].set_ylabel("Cumulative recorded wall time (hours)")
    axes[1, 1].text(
        0.02,
        0.95,
        "Wall time is summed per selected segment.\n"
        "The x-axis remains cumulative accepted updates.",
        fontsize=7,
        va="top",
        transform=axes[1, 1].transAxes,
    )
    add_method_transitions(axes[1, 1], receipts, annotate=False)

    for index, (axis, tissue) in enumerate(
        zip(axes[2:].flat[:3], ("Fat", "Aponeurosis", "Muscle"), strict=True)
    ):
        for principal in range(3):
            axis.plot(
                updates,
                spectra_kpa[:, index, 1, principal],
                color=colors[principal],
                linewidth=1.1,
                label=f"Principal {principal + 1}" + (" mean" if spatial else ""),
            )
            if spatial:
                axis.fill_between(
                    updates,
                    spectra_kpa[:, index, 0, principal],
                    spectra_kpa[:, index, 2, principal],
                    color=colors[principal],
                    alpha=0.15,
                )
        axis.axhline(0, color="#77818b", linewidth=0.7)
        axis.set_title(f"{tissue}: signed shared baseline stress", loc="left")
        axis.set_ylabel("kPa; positive = tension")
        axis.legend(fontsize=7, ncol=3)

    skin_axis = axes[3, 1]
    skin_axis.plot(
        updates,
        resultants_n_per_m,
        color=colors[0],
        label="Prescribed resultant",
    )
    skin_axis.set_title("Skin prestress and stiffness", loc="left")
    skin_axis.set_ylabel("Membrane resultant (N/m)", color=colors[0])
    stiffness_axis = skin_axis.twinx()
    stiffness_axis.plot(
        updates, stiffness, color=colors[1], label="Stiffness multiplier"
    )
    for bound in skin_bounds:
        stiffness_axis.axhline(bound, color=colors[1], linestyle="--", linewidth=0.8)
    stiffness_axis.set_ylabel("Young-modulus multiplier", color=colors[1])
    lines = skin_axis.lines[:1] + stiffness_axis.lines[:1]
    skin_axis.legend(lines, [line.get_label() for line in lines], fontsize=7)

    for axis in axes.flat:
        axis.grid(alpha=0.2)
        axis.set_xlabel("Cumulative accepted neutral updates on selected lineage")
        add_method_transitions(axis, receipts, annotate=False)
    fig.suptitle(
        "Neutral preparation: exact selected checkpoint lineage\n"
        f"{int(updates[-1])} accepted updates · branch tip {receipts[-1]['checkpoint']['sha256'][:12]}"
        + (
            "\nStress lines: anchor means; shading: anchor ranges, not volume averages"
            if spatial
            else ""
        ),
        fontsize=13,
    )
    assets = []
    for extension in ("png", "pdf"):
        name = f"neutral-lineage.{extension}"
        fig.savefig(output / name, dpi=180)
        assets.append(name)
    plt.close(fig)
    return assets


def main(cfg: Config) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=False)
    segments, root_checkpoint, root_file = build_lineage(cfg.tip_checkpoint)
    rows, receipts = combine_rows(segments)
    assets = render(
        cfg.output_dir,
        rows,
        receipts,
        segments[-1].checkpoint["materials"],
    )
    selected_trace = {
        "schema": "joint-neutral-selected-lineage-trace-v1",
        "rows": rows,
    }
    selected_trace_path = cfg.output_dir / "selected-trace.json"
    write_json(selected_trace_path, selected_trace)
    summary = {
        "schema": "joint-neutral-lineage-visualization-v1",
        "success": True,
        "branch_tip": receipts[-1]["checkpoint"],
        "root_checkpoint": {
            "path": str(root_file.path),
            "sha256": root_file.sha256,
            "stage": root_checkpoint.get("stage"),
        },
        "lineage_proven": True,
        "segments": receipts,
        "selected_points": len(rows),
        "selected_accepted_updates": rows[-1]["lineage_cumulative_update"],
        "selected_trace": {
            "path": str(selected_trace_path.resolve()),
            "sha256": sha256(selected_trace_path),
        },
        "cumulative_recorded_wall_seconds": rows[-1][
            "lineage_cumulative_elapsed_seconds"
        ],
        "branch_policy": (
            "follow each checkpoint protocol initial_checkpoint path and exact SHA256; "
            "slice each run trace through that checkpoint update; omit child update "
            "zero only for an unchanged prestress target; retain target-change initial "
            "evaluations without counting them as accepted updates; exclude all later "
            "rows on abandoned branches"
        ),
        "skin_prestress_status": (
            "model continuation proxy, not a measurement on this anatomy"
        ),
        "assets": assets,
    }
    write_json(cfg.output_dir / "summary.json", summary)
    cherries.log_metrics(
        {
            "lineage/segments": len(receipts),
            "lineage/accepted_updates": rows[-1]["lineage_cumulative_update"],
            "lineage/objective": rows[-1]["objective"],
            "lineage/projected_gradient_inf": rows[-1]["projected_gradient_inf"],
        }
    )
    cherries.log_output(cfg.output_dir)
    LOG.info(
        "Rendered %d accepted updates across %d hash-proven segments",
        rows[-1]["lineage_cumulative_update"],
        len(receipts),
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
