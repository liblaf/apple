"""Render the matched fixed/adaptive IPC stiffness forward-solve receipts."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import re
import sys
from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FORWARD_PROBLEM = (
    "This is a static, quasistatic cold forward solve at the fixed activation from "
    "the historical Smile Adam update 16 and a 0 degree jaw. Every arm starts from "
    "the same saved prestressed neutral displacement, which is not zero. The energy "
    "contains bulk Stable Neo-Hookean fat, muscle, and aponeurosis; heterogeneous "
    "plane-stress Stable Neo-Hookean skin with frozen skin tangent prestress; additive "
    "muscle active stress; and physical IPC contact against complete bones and fixed "
    "eyes. Bulk baseline stress is zero. It contains no inertia, target-fitting, or "
    "regularization terms."
)
spec = importlib.util.spec_from_file_location(
    "adaptive_ipc_benchmark", HERE / "10-benchmark.py"
)
assert spec is not None
assert spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class Config(cherries.BaseConfig):
    run_dirs: str = (
        "data/adaptive-ipc-maxe-fixed-001,"
        "data/adaptive-ipc-maxe-adaptive-001,"
        "data/adaptive-ipc-maxe-final-fixed-001"
    )
    labels: str = "fixed kappa0,adaptive kappa,fixed final kappa"
    output_dir: Path = EXPERIMENT / "data/adaptive-ipc-report-001"
    document_path: Path = EXPERIMENT / "docs/62-adaptive-ipc.md"


def _paths(value: str) -> list[Path]:
    return [EXPERIMENT / item.strip() for item in value.split(",") if item.strip()]


def _values(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _receipt_logs(directory: Path) -> list[dict[str, str]]:
    """Best-effort Cherries command/Comet receipts retained in the workspace."""
    candidates = sorted(
        {
            *EXPERIMENT.glob(f"tmp/*{directory.name}*.log"),
            *EXPERIMENT.glob(f"logs/*{directory.name}*.log"),
        }
    )
    records: list[dict[str, str]] = []
    for path in candidates:
        text = path.read_text(errors="replace")
        commands = re.findall(r"cherries/cmd\s+: (.+)", text)
        urls = re.findall(r"cherries/comet/url\s+: (https://\S+)", text)
        names = re.findall(r"\sname\s+: (.+)", text)
        if commands or urls or names:
            records.append(
                {
                    "path": str(path),
                    "command": commands[-1] if commands else "",
                    "comet_url": urls[-1] if urls else "",
                    "name": names[-1] if names else "",
                }
            )
    return records


def _state_rows(trace: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in trace if row.get("kind") in {"initial", "pncg", "newton"}]


def _gap_rows(stiffness: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        row
        for row in stiffness.get("observations", [])
        if row.get("minimum_distance_m") is not None
    ]


def _arm(directory: Path, label: str) -> dict[str, Any]:
    for name in ("protocol.json", "summary.json", "stiffness.json", "trace.jsonl"):
        assert (directory / name).is_file(), directory / name
    protocol = _read_json(directory / "protocol.json")
    summary = _read_json(directory / "summary.json")
    result = summary["result"]
    trace = _trace(directory / "trace.jsonl")
    stiffness = _read_json(directory / "stiffness.json")
    states = _state_rows(trace)
    pncg_observations = [
        row for row in stiffness.get("observations", []) if row.get("phase") == "pncg"
    ]
    pncg_steps = max((int(row["step"]) for row in pncg_observations), default=0)
    newton_steps = sum(row.get("kind") == "newton" for row in states)
    collision = result.get("collision", {})
    return {
        "label": label,
        "directory": directory,
        "protocol": protocol,
        "result": result,
        "trace": trace,
        "states": states,
        "stiffness": stiffness,
        "gap_rows": _gap_rows(stiffness),
        "pncg_steps": pncg_steps,
        "newton_steps": newton_steps,
        "terminal_force": result.get("terminal_force"),
        "terminal_gap_m": collision.get("minimum_active_distance_m"),
        "inversions": collision.get(
            "inverted_tetrahedra", result.get("shape", {}).get("inverted_tetrahedra")
        ),
        "events": list(stiffness.get("events", [])),
        "logs": _receipt_logs(directory),
    }


def _assert_shared_inputs(arms: list[dict[str, Any]]) -> None:
    keys = (
        "checkpoint_sha256",
        "neutral_checkpoint_sha256",
        "activation_sha256",
        "jaw_sha256",
        "seed_displacement_sha256",
        "uv_lock_sha256",
    )
    reference = arms[0]["protocol"]
    for arm in arms[1:]:
        other = arm["protocol"]
        for key in keys:
            assert other[key] == reference[key], (key, arm["label"])
        assert other["coverage"] == reference["coverage"], arm["label"]
        assert other["newton"] == reference["newton"], arm["label"]
    reference_binding = _normalized_binding(arms[0]["directory"])
    for arm in arms[1:]:
        assert _normalized_binding(arm["directory"]) == reference_binding, arm["label"]


def _normalized_binding(directory: Path) -> dict[str, Any]:
    """Compare binding proofs while hashing only their run-local diff artifacts."""
    receipt = deepcopy(_read_json(directory / "profile-input-binding.json"))

    def replace_diff_paths(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: (
                    hashlib.sha256(Path(item).read_bytes()).hexdigest()
                    if key == "unified_diff"
                    else replace_diff_paths(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [replace_diff_paths(item) for item in value]
        return value

    return replace_diff_paths(receipt)


def _global_step(arm: dict[str, Any], row: dict[str, Any]) -> int:
    step = int(row.get("step", 0))
    if row.get("kind") == "newton" or row.get("phase") == "newton":
        return arm["pncg_steps"] + step
    return step


def _split_by_stiffness(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    segments: list[list[dict[str, Any]]] = []
    for row in rows:
        if not segments or row.get("stiffness_mpa") != segments[-1][-1].get(
            "stiffness_mpa"
        ):
            segments.append([row])
        else:
            segments[-1].append(row)
    return segments


def _plot(arms: list[dict[str, Any]], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    figure.set_constrained_layout_pads(h_pad=0.16, hspace=0.10)
    colors = plt.get_cmap("tab10").colors
    trigger_nm = 1e-6 * float(arms[0]["stiffness"]["bbox_diagonal_m"]) * 1e9
    failure_label_added = False
    valid_count = sum(
        bool(
            arm["result"]["success"]
            and arm["result"].get("collision", {}).get("state_feasible")
            and arm["inversions"] == 0
            and arm["terminal_force"] is not None
            and float(arm["terminal_force"]) <= 1e-8
        )
        for arm in arms
    )
    for index, arm in enumerate(arms):
        color = colors[index % len(colors)]
        states = arm["states"]
        if states:
            steps = [_global_step(arm, row) for row in states]
            times = [float(row["elapsed_seconds"]) for row in states]
            forces = [float(row["force"]) for row in states]
            axes[0, 0].plot(steps, forces, marker=".", color=color, label=arm["label"])
            axes[0, 1].plot(times, forces, marker=".", color=color, label=arm["label"])
            for segment_number, segment in enumerate(_split_by_stiffness(states)):
                axes[1, 2].plot(
                    [float(row["elapsed_seconds"]) for row in segment],
                    [float(row["energy"]) for row in segment],
                    marker=".",
                    color=color,
                    linestyle="-",
                    label=(arm["label"] if segment_number == 0 else None),
                )
        if not arm["result"]["success"] and arm["terminal_force"] is not None:
            endpoint_step = arm["pncg_steps"] + arm["newton_steps"]
            axes[0, 0].scatter(
                [endpoint_step],
                [float(arm["terminal_force"])],
                marker="x",
                color="#b2182b",
                s=42,
                zorder=4,
                label=None if failure_label_added else "failed endpoint",
            )
            failure_label_added = True
            axes[0, 1].scatter(
                [float(arm["result"]["forward_wall_seconds"])],
                [float(arm["terminal_force"])],
                marker="x",
                color="#b2182b",
                s=42,
                zorder=4,
            )
        gaps = arm["gap_rows"]
        if gaps:
            axes[1, 0].plot(
                [_global_step(arm, row) for row in gaps],
                [float(row["minimum_distance_m"]) * 1e9 for row in gaps],
                marker=".",
                color=color,
                label=arm["label"],
            )
            axes[1, 1].step(
                [_global_step(arm, row) for row in gaps],
                [
                    float(row.get("stiffness_after", row.get("stiffness")))
                    for row in gaps
                ],
                where="post",
                color=color,
                label=arm["label"],
            )
    for axis in (axes[0, 0], axes[0, 1]):
        axis.set_yscale("log")
        axis.axhline(1e-8, color="#333", linestyle="--", linewidth=1)
    axes[0, 0].set(
        xlabel="Accepted state index",
        ylabel="Force norm",
        title="Force by accepted state",
    )
    axes[0, 1].set(
        xlabel="Elapsed forward time (s)",
        ylabel="Force norm",
        title="Force by wall time",
    )
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.96,
        f"{valid_count} / {len(arms)} valid forward solves\n\nκ0: 0.1693 MPa\n10 nm gap guard\n{trigger_nm:.1f} nm adaptive trigger\n\nStatic cold Smile forward\nEnergy lines break when κ changes.",
        va="top",
        fontsize=10,
    )
    axes[1, 0].axhline(
        10, color="#333", linestyle="--", linewidth=1, label="10 nm guard"
    )
    axes[1, 0].axhline(
        trigger_nm, color="#666", linestyle=":", linewidth=1, label="adaptive trigger"
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        xlabel="Global accepted update",
        ylabel="Minimum active gap (nm)",
        title="Contact gap",
    )
    axes[1, 1].set(
        xlabel="Global accepted update",
        ylabel="Barrier stiffness κ (MPa)",
        title="κ schedule",
    )
    axes[1, 2].set(
        xlabel="Elapsed forward time (s)",
        ylabel="Energy (MPa m^3)",
        title="Energy within fixed-κ segments",
    )
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1], axes[1, 2]):
        axis.grid(visible=True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _table(arms: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for arm in arms:
        result = arm["result"]
        collision = result.get("collision", {})
        rows.append(
            {
                "arm": arm["label"],
                "success": bool(result["success"]),
                "termination": "converged"
                if result["success"]
                else result.get("failure", "failed"),
                "wall_seconds": float(result["forward_wall_seconds"]),
                "pncg_steps": arm["pncg_steps"],
                "newton_steps": arm["newton_steps"],
                "terminal_force": arm["terminal_force"],
                "terminal_gap_nm": None
                if arm["terminal_gap_m"] is None
                else float(arm["terminal_gap_m"]) * 1e9,
                "contact_feasible": collision.get("state_feasible"),
                "inverted_tetrahedra": arm["inversions"],
                "valid_forward": bool(
                    result["success"]
                    and collision.get("state_feasible")
                    and arm["inversions"] == 0
                    and arm["terminal_force"] is not None
                    and float(arm["terminal_force"]) <= 1e-8
                ),
                "kappa_events": len(arm["events"]),
                "final_kappa_mpa": arm["stiffness"]["final_stiffness"],
            }
        )
    return rows


def _html_table(rows: list[dict[str, Any]]) -> str:
    columns = list(rows[0])
    head = "".join(f"<th>{html.escape(column)}</th>" for column in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[column]))}</td>" for column in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _write_documents(
    cfg: Config, rows: list[dict[str, Any]], evidence: dict[str, Any]
) -> None:
    commands = [
        record
        for arm in evidence["arms"]
        for record in arm["logs"]
        if record.get("command")
    ]
    markdown_rows = "\n".join(
        "| {arm} | {success} | {wall_seconds:.3f} | {pncg_steps} | {newton_steps} | {terminal_force} | {terminal_gap_nm} | {contact_feasible} | {inverted_tetrahedra} | {valid_forward} | {kappa_events} | {final_kappa_mpa} |".format(
            **row
        )
        for row in rows
    )
    final_kappas = ", ".join(
        f"{row['arm']}: {row['final_kappa_mpa']} MPa" for row in rows
    )
    command_blocks = "\n\n".join(
        "```bash\n"
        f"# {record['name'] or 'Cherries run'}\n"
        "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 "
        f"{record['command']}\n```"
        for record in commands
    )
    comet_links = "\n".join(
        f"- [{record['name'] or 'Comet run'}]({record['comet_url']})"
        for arm in evidence["arms"]
        for record in arm["logs"]
        if record.get("comet_url")
    )
    cfg.document_path.write_text(
        "# Adaptive IPC stiffness cold Smile test\n\n"
        "All arms share the saved Smile activation, jaw, neutral seed, collision coverage, IPCTK/PyTorch lock, and Newton settings. "
        "`κ0 = 0.1693 MPa = 0.1 * 1.693 MPa`, where aponeurosis is the stiffest deformable material; frozen skin spans 0.127382-0.257861 MPa. Bones and eyes are rigid obstacles. All three recorded runs retained nonempty contact sets; the fixed run preceded the later empty-contact helper correction, which was not exercised here.\n\n"
        + FORWARD_PROBLEM
        + "\n\n"
        "| Arm | Success | Wall s | PNCG | Newton | Terminal force | Terminal gap nm | Contact feasible | Inversions | Valid forward | κ events | Final κ MPa |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |\n"
        + markdown_rows
        + "\n\n"
        "A feasible contact gap alone is not a valid forward solve: force tolerance, zero inverted tetrahedra, and the solver success receipt are also required. Failures are time-to-failure observations, not speedups. Adaptive κ changes the physical barrier objective; energy is only connected within constant-κ segments. No inverse solve or adjoint was run.\n\n"
        f"Recorded terminal physical κ: {final_kappas}. These values describe each terminal barrier objective; they do not establish a valid forward result.\n\n"
        f"Outputs: `{cfg.output_dir / 'adaptive-ipc.png'}`, `{cfg.output_dir / 'adaptive-ipc.svg'}`, and `{cfg.output_dir / 'evidence.json'}`.\n\n"
        + ("## Recorded commands\n\n" + command_blocks + "\n\n" if commands else "")
        + ("## Comet\n\n" + comet_links + "\n" if comet_links else "")
    )


def main(cfg: Config) -> None:
    directories, labels = _paths(cfg.run_dirs), _values(cfg.labels)
    assert directories
    assert len(directories) == len(labels)
    assert not cfg.output_dir.exists(), cfg.output_dir
    arms = [
        _arm(directory, label)
        for directory, label in zip(directories, labels, strict=True)
    ]
    _assert_shared_inputs(arms)
    cfg.output_dir.mkdir(parents=True)
    _plot(arms, cfg.output_dir / "adaptive-ipc")
    rows = _table(arms)
    evidence = {
        "schema": "adaptive-ipc-stiffness-report-v1",
        "same_input_hashes_verified": True,
        "force_target": 1e-8,
        "minimum_gap_guard_nm": 10.0,
        "adaptive_trigger_nm": 1e-6
        * float(arms[0]["stiffness"]["bbox_diagonal_m"])
        * 1e9,
        "material_young_mpa": arms[0]["protocol"]["contact_stiffness_policy"][
            "young_moduli_mpa"
        ],
        "rows": rows,
        "arms": [
            {
                "label": arm["label"],
                "directory": str(arm["directory"]),
                "protocol": arm["protocol"],
                "result": arm["result"],
                "stiffness": arm["stiffness"],
                "logs": arm["logs"],
            }
            for arm in arms
        ],
    }
    (cfg.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    _write_documents(cfg, rows, evidence)
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>Adaptive IPC stiffness</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:13px}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto}}</style>
<main><h1>Adaptive IPC stiffness: cold Smile forward</h1><p>κ0 = 0.1693 MPa (10% of aponeurosis E = 1.693 MPa). Inputs and solver settings were verified equal across arms; all runs retained nonempty contact sets, including the earlier fixed arm before the empty-contact helper correction.</p>
<p>{html.escape(FORWARD_PROBLEM)}</p>
{_html_table(rows)}<img src=\"adaptive-ipc.svg\" alt=\"Force, contact-gap, barrier-stiffness and energy traces\">
<p>A feasible contact gap alone is insufficient: the force target, zero inversions, and solver success are all required. Failures are time-to-failure observations, not speedups. Changing κ changes the physical barrier objective; energy curves are split at κ changes.</p>
<p><a href=\"evidence.json\">Evidence JSON</a></p></main>"""
    (cfg.output_dir / "adaptive-ipc.html").write_text(page)
    for path in (
        cfg.output_dir / "adaptive-ipc.html",
        cfg.output_dir / "adaptive-ipc.png",
        cfg.output_dir / "adaptive-ipc.svg",
        cfg.output_dir / "evidence.json",
        cfg.document_path,
    ):
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
