"""Render matched fixed/adaptive IPC stiffness solves of the neutral face."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
import math
import re
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from liblaf import cherries

HERE = Path(__file__).resolve().parent
EXPERIMENT = HERE.parent
FORWARD_PROBLEM = (
    "This is one static quasistatic neutral forward problem. It starts from a repaired "
    "geometric contact initializer in the constitutive reference, with no saved equilibrium "
    "displacement. The jaw, muscle activation, and bulk additive stress are zero. The energy "
    "contains Stable Neo-Hookean fat, muscle, and aponeurosis; heterogeneous plane-stress "
    "Stable Neo-Hookean skin with its existing prescribed baseline tension; and physical IPC "
    "contact against complete source bones and fixed eyes. It contains no inertia, target fit, "
    "regularization, inverse update, or load continuation."
)
spec = importlib.util.spec_from_file_location(
    "neutral_report_benchmark", HERE / "10-benchmark.py"
)
assert spec is not None
assert spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class Config(cherries.BaseConfig):
    run_dirs: str = (
        "data/neutral-adaptive-ipc-fixed-001,"
        "data/neutral-adaptive-ipc-adaptive-001,"
        "data/neutral-adaptive-ipc-final-fixed-001"
    )
    labels: str = "fixed kappa0,adaptive kappa,fixed final kappa"
    output_dir: Path = EXPERIMENT / "data/neutral-adaptive-ipc-report-001"
    document_path: Path = EXPERIMENT / "docs/64-neutral-adaptive-ipc.md"


def _items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _nested(value: dict[str, Any], *paths: tuple[str, ...]) -> Any:
    for path in paths:
        current: Any = value
        for key in path:
            if not isinstance(current, dict) or key not in current:
                break
            current = current[key]
        else:
            return current
    return None


def _logs(directory: Path) -> list[dict[str, str]]:
    records = []
    for path in sorted(
        {
            *EXPERIMENT.glob(f"tmp/*{directory.name}*.log"),
            *EXPERIMENT.glob(f"logs/*{directory.name}*.log"),
        }
    ):
        text = path.read_text(errors="replace")
        command = re.findall(r"cherries/cmd\s+: (.+)", text)
        comet = re.findall(r"cherries/comet/url\s+: (https://\S+)", text)
        name = re.findall(r"\sname\s+: (.+)", text)
        if command or comet or name:
            records.append(
                {
                    "path": str(path),
                    "command": command[-1] if command else "",
                    "comet_url": comet[-1] if comet else "",
                    "name": name[-1] if name else "",
                }
            )
    return records


def _binding(directory: Path) -> dict[str, Any]:
    """Keep binding proofs strict except for run-local diff destinations."""
    receipt = deepcopy(_json(directory / "profile-input-binding.json"))

    def normalize(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: hashlib.sha256(Path(item).read_bytes()).hexdigest()
                if key == "unified_diff"
                else normalize(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value

    return normalize(receipt)


def _arm(directory: Path, label: str) -> dict[str, Any]:
    for name in (
        "protocol.json",
        "summary.json",
        "stiffness.json",
        "timing.json",
        "trace.jsonl",
        "profile-input-binding.json",
    ):
        assert (directory / name).is_file(), directory / name
    protocol, summary, stiffness, timing = (
        _json(directory / name)
        for name in ("protocol.json", "summary.json", "stiffness.json", "timing.json")
    )
    result = summary["result"]
    trace = _trace(directory / "trace.jsonl")
    states = [row for row in trace if row.get("kind") in {"initial", "pncg", "newton"}]
    pncg = [
        row for row in stiffness.get("observations", []) if row.get("phase") == "pncg"
    ]
    gaps = [
        row
        for row in stiffness.get("observations", [])
        if row.get("minimum_distance_m") is not None
    ]
    collision = result.get("collision", {})
    return {
        "label": label,
        "directory": directory,
        "protocol": protocol,
        "result": result,
        "stiffness": stiffness,
        "timing": timing,
        "states": states,
        "gaps": gaps,
        "pncg_steps": max((int(row["step"]) for row in pncg), default=0),
        "newton_steps": sum(row.get("kind") == "newton" for row in states),
        "force": result.get("terminal_force"),
        "gap_m": collision.get("minimum_active_distance_m"),
        "inversions": result.get("geometry", {}).get("inverted_tetrahedra"),
        "contact_feasible": collision.get("state_feasible"),
        "events": stiffness.get("events", []),
        "logs": _logs(directory),
    }


def _timing(arm: dict[str, Any]) -> dict[str, float]:
    """Extract non-overlapping phase totals and the summed CCD leaf work."""
    root = arm["timing"]["tree"]["forward"]
    children = root.get("children", {})

    def ccd_seconds(node: dict[str, Any]) -> float:
        return (
            float(node["inclusive_seconds"]) if node.get("_name") == "ipc/ccd" else 0.0
        ) + sum(
            ccd_seconds({**child, "_name": name})
            for name, child in node.get("children", {}).items()
        )

    return {
        "pncg_seconds": float(children.get("pncg", {}).get("inclusive_seconds", 0.0)),
        "newton_seconds": float(
            children.get("newton", {}).get("inclusive_seconds", 0.0)
        ),
        "ccd_seconds": ccd_seconds({**root, "_name": "forward"}),
    }


def _shared(arms: list[dict[str, Any]]) -> None:
    reference = arms[0]["protocol"]
    assert reference["schema"] == "natural-reference-adaptive-ipc-hybrid-v1"
    fixture = reference["fixture"]
    anchor = reference["contact_stiffness_policy"]
    for arm in arms[1:]:
        other = arm["protocol"]
        assert other["schema"] == reference["schema"]
        for key in ("neutral_manifest", "eyes_manifest", "seed", "seed_receipt"):
            assert other["fixture"][key]["sha256"] == fixture[key]["sha256"], (
                key,
                arm["label"],
            )
        for key in (
            "name",
            "prior_equilibrium_used",
            "fem_reference_rebased",
            "activation",
            "jaw_rotation_rad",
        ):
            assert other["fixture"][key] == fixture[key], (key, arm["label"])
        assert other["coverage"] == reference["coverage"], arm["label"]
        assert other["materials"] == reference["materials"], arm["label"]
        assert other["solver"] == reference["solver"], arm["label"]
        for key in ("boundary_projection", "prewarm_scope", "timing"):
            assert other[key] == reference[key], (key, arm["label"])
        assert (
            other["sources"]["neutral_newton"]["sources"]
            == reference["sources"]["neutral_newton"]["sources"]
        ), arm["label"]
        assert other["versions"] == reference["versions"], arm["label"]
        for key in (
            "anchor_rule",
            "anchor_stiffness_mpa",
            "epsilon_scale",
            "tolerance_anchor_force",
            "effective_force_tolerance",
            "tolerance_rule",
        ):
            assert other["contact_stiffness_policy"][key] == anchor[key], (
                key,
                arm["label"],
            )
    proof = _binding(arms[0]["directory"])
    for arm in arms[1:]:
        assert _binding(arm["directory"]) == proof, arm["label"]
    source_proof = _runtime_sources(arms[0]["directory"])
    for arm in arms[1:]:
        assert _runtime_sources(arm["directory"]) == source_proof, arm["label"]


def _runtime_sources(directory: Path) -> dict[str, str]:
    """Compare mechanics/solver bytes, deliberately excluding presentation scripts."""
    sources = _json(directory / "provenance.json")["sources"]
    solver_files = {
        "solver-performance/63-test-neutral-adaptive-ipc.py",
        "solver-performance/pncg_first.py",
        "solver-performance/hybrid_first_solver.py",
        "solver-performance/accelerated_solvers.py",
        "solver-performance/hybrid_hessian.py",
        "solver-performance/adaptive_ipc_stiffness.py",
        "solver-performance/profile_input_binding.py",
        "solver-performance/inverse_timing.py",
        "solver-performance/mesh_step_scale.py",
    }
    assert solver_files <= sources.keys(), (solver_files - sources.keys(), directory)
    selected = {
        key: value
        for key, value in sources.items()
        if key.startswith(("apple/", "joint-experiment/", "tensor-reference/"))
        or key in solver_files
    }
    assert selected
    return selected


def _force_target(protocol: dict[str, Any]) -> float:
    return float(protocol["contact_stiffness_policy"]["effective_force_tolerance"])


def _global(arm: dict[str, Any], row: dict[str, Any]) -> int:
    step = int(row.get("step", 0))
    return (
        arm["pncg_steps"] + step
        if row.get("kind") == "newton" or row.get("phase") == "newton"
        else step
    )


def _segments(states: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    output: list[list[dict[str, Any]]] = []
    for state in states:
        if not output or state.get("stiffness_mpa") != output[-1][-1].get(
            "stiffness_mpa"
        ):
            output.append([state])
        else:
            output[-1].append(state)
    return output


def _valid(arm: dict[str, Any], target: float) -> bool:
    return bool(
        arm["result"]["success"]
        and arm["contact_feasible"]
        and arm["inversions"] == 0
        and arm["force"] is not None
        and float(arm["force"]) <= target
    )


def _plot(arms: list[dict[str, Any]], target: float, output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    figure.set_constrained_layout_pads(h_pad=0.16, hspace=0.10)
    colors = plt.get_cmap("tab10").colors
    trigger = 1e-6 * float(arms[0]["stiffness"]["bbox_diagonal_m"]) * 1e9
    invalid_label = False
    for index, arm in enumerate(arms):
        color = colors[index % len(colors)]
        states = arm["states"]
        if states:
            axes[0, 0].plot(
                [_global(arm, row) for row in states],
                [float(row["force"]) for row in states],
                marker=".",
                color=color,
                label=arm["label"],
            )
            axes[0, 1].plot(
                [float(row["elapsed_seconds"]) for row in states],
                [float(row["force"]) for row in states],
                marker=".",
                color=color,
                label=arm["label"],
            )
            for segment_index, segment in enumerate(_segments(states)):
                axes[1, 2].plot(
                    [float(row["elapsed_seconds"]) for row in segment],
                    [float(row["energy"]) for row in segment],
                    marker=".",
                    color=color,
                    label=arm["label"] if segment_index == 0 else None,
                )
        if not _valid(arm, target) and arm["force"] is not None:
            axes[0, 0].scatter(
                [arm["pncg_steps"] + arm["newton_steps"]],
                [float(arm["force"])],
                marker="x",
                color="#b2182b",
                s=42,
                label=None if invalid_label else "invalid endpoint",
            )
            axes[0, 1].scatter(
                [float(arm["result"]["forward_wall_seconds"])],
                [float(arm["force"])],
                marker="x",
                color="#b2182b",
                s=42,
            )
            invalid_label = True
        if arm["gaps"]:
            axes[1, 0].plot(
                [_global(arm, row) for row in arm["gaps"]],
                [float(row["minimum_distance_m"]) * 1e9 for row in arm["gaps"]],
                marker=".",
                color=color,
                label=arm["label"],
            )
            axes[1, 1].step(
                [_global(arm, row) for row in arm["gaps"]],
                [
                    float(row.get("stiffness_after", row.get("stiffness")))
                    for row in arm["gaps"]
                ],
                where="post",
                color=color,
                label=arm["label"],
            )
    for axis in (axes[0, 0], axes[0, 1]):
        axis.set_yscale("log")
        axis.axhline(target, color="#333", linestyle="--", linewidth=1)
    axes[0, 0].set(
        xlabel="Global applied update",
        ylabel="Force norm (MPa m^2)",
        title="Force by update",
    )
    axes[0, 1].set(
        xlabel="Elapsed forward time (s)",
        ylabel="Force norm (MPa m^2)",
        title="Force by wall time",
    )
    axes[0, 2].axis("off")
    axes[0, 2].text(
        0.02,
        0.96,
        f"{sum(_valid(arm, target) for arm in arms)} / {len(arms)} valid forward solves\n\nκ0: 0.1693 MPa\nForce target: {target:.3e}\n10 nm gap guard\n{trigger:.1f} nm trigger\n\nNeutral static fixture",
        va="top",
        fontsize=10,
    )
    axes[1, 0].axhline(
        10, color="#333", linestyle="--", linewidth=1, label="10 nm guard"
    )
    axes[1, 0].axhline(
        trigger, color="#666", linestyle=":", linewidth=1, label="adaptive trigger"
    )
    axes[1, 0].set_yscale("log")
    axes[1, 0].set(
        xlabel="Global applied update",
        ylabel="Minimum active gap (nm)",
        title="Contact gap",
    )
    axes[1, 1].set(
        xlabel="Global applied update",
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


def _rows(arms: list[dict[str, Any]], target: float) -> list[dict[str, Any]]:
    output = []
    for arm in arms:
        valid = _valid(arm, target)
        output.append(
            {
                "arm": arm["label"],
                "status": "valid equilibrium"
                if valid
                else (
                    "force converged; invalid geometry"
                    if arm["result"]["success"]
                    else "failed before force convergence"
                ),
                "success": bool(arm["result"]["success"]),
                "termination": "converged"
                if arm["result"]["success"]
                else arm["result"].get("failure", "failed"),
                "wall_seconds": float(arm["result"]["forward_wall_seconds"]),
                "pncg_steps": arm["pncg_steps"],
                "newton_steps": arm["newton_steps"],
                "terminal_force": arm["force"],
                "terminal_gap_nm": None
                if arm["gap_m"] is None
                else float(arm["gap_m"]) * 1e9,
                "contact_feasible": arm["contact_feasible"],
                "inverted_tetrahedra": arm["inversions"],
                "valid_forward": valid,
                "kappa_events": len(arm["events"]),
                "final_kappa_mpa": arm["stiffness"]["final_stiffness"],
                **_timing(arm),
            }
        )
    return output


def _html_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(str(row[key]))}</td>" for key, _ in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _display_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "arm": row["arm"],
            "status": row["status"],
            "wall": f"{row['wall_seconds']:.3f}",
            "steps": f"{row['pncg_steps']} / {row['newton_steps']}",
            "force": f"{float(row['terminal_force']):.3e}",
            "gap": f"{float(row['terminal_gap_nm']):.3f}",
            "inversions": str(row["inverted_tetrahedra"]),
            "kappa": f"{float(row['final_kappa_mpa']):.6g} ({row['kappa_events']} changes)",
            "pncg": f"{row['pncg_seconds']:.3f}",
            "newton": f"{row['newton_seconds']:.3f}",
            "ccd": f"{row['ccd_seconds']:.3f}",
        }
        for row in rows
    ]


def _conclusion(rows: list[dict[str, Any]]) -> str:
    valid = [row["arm"] for row in rows if row["valid_forward"]]
    invalid = [row["arm"] for row in rows if not row["valid_forward"]]
    if valid and invalid:
        valid_stiffness = [
            float(row["final_kappa_mpa"]) for row in rows if row["valid_forward"]
        ]
        matched_fixed = any(
            not row["valid_forward"]
            and any(
                math.isclose(
                    float(row["final_kappa_mpa"]),
                    stiffness,
                    rel_tol=1e-12,
                    abs_tol=1e-12,
                )
                for stiffness in valid_stiffness
            )
            for row in rows
        )
        conclusion = f"{', '.join(valid)} reached a valid equilibrium; {', '.join(invalid)} did not."
        if matched_fixed:
            conclusion += " On this fixture, the stiffness schedule matters: a stronger fixed barrier at the same final κ did not reproduce the valid outcome."
        return conclusion
    if valid:
        return f"All recorded arms reached valid equilibria: {', '.join(valid)}."
    return "No recorded arm reached a valid equilibrium."


def _document(
    cfg: Config, rows: list[dict[str, Any]], target: float, evidence: dict[str, Any]
) -> None:
    display = _display_rows(rows)
    table = "\n".join(
        "| {arm} | {status} | {wall} | {steps} | {force} | {gap} | {inversions} | {kappa} |".format(
            **row
        )
        for row in display
    )
    timing_table = "\n".join(
        "| {arm} | {pncg} | {newton} | {ccd} |".format(**row) for row in display
    )
    commands = [
        record
        for arm in evidence["arms"]
        for record in arm["logs"]
        if record["command"]
    ]
    command_text = "\n\n".join(
        "```bash\n" + f"# {record['name'] or 'Cherries run'}\n{record['command']}\n```"
        for record in commands
    )
    links = "\n".join(
        f"- [{record['name'] or 'Comet run'}]({record['comet_url']})"
        for arm in evidence["arms"]
        for record in arm["logs"]
        if record["comet_url"]
    )
    cfg.document_path.write_text(
        "# Neutral adaptive IPC stiffness forward test\n\n" + FORWARD_PROBLEM + "\n\n"
        "All arms have matching recorded fixture/input hashes, solver/material runtime source hashes, runtime binding proof, collision coverage, and hybrid solver settings. κ0 is 0.1693 MPa (0.1 times aponeurosis E = 1.693 MPa); adaptive κ uses epsilon scale 1e-6 and a 100x cap. The force target is anchored once at κ0, before any fixed-final multiplier: `max(1e-8, 1e-3 * initial force at κ0)` = `"
        + f"{target:.17g}`.\n\n"
        + _conclusion(rows)
        + "\n\n"
        "| Arm | Result | Wall s | PNCG / Newton updates | Terminal force | Gap nm | Inversions | Final κ MPa |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |\n"
        + table
        + "\n\n"
        "| Arm | PNCG s | Newton s | CCD leaf work s |\n| --- | ---: | ---: | ---: |\n"
        + timing_table
        + "\n\n"
        "The timing tree synchronizes CUDA at scoped boundaries. PNCG and Newton are inclusive forward-solve phase times; CCD is a summed `ipc/ccd` leaf subset already included in those totals and must not be added again. The separately measured model setup/operator prewarm is excluded; the first sparse construction remains in Newton. Contact feasibility alone is insufficient: valid forward also requires the force gate, zero inverted tetrahedra, and solver success. A failed arm is only a time-to-failure observation, never a speedup. κ changes the physical barrier objective, so energy is connected only within fixed-κ segments. These hybrid timings are not comparable to the separate neutral-newton Newton-only driver.\n\n"
        f"Outputs: `{cfg.output_dir / 'neutral-adaptive-ipc.png'}`, `{cfg.output_dir / 'neutral-adaptive-ipc.svg'}`, and `{cfg.output_dir / 'evidence.json'}`.\n\n"
        + ("## Recorded commands\n\n" + command_text + "\n\n" if command_text else "")
        + ("## Comet\n\n" + links + "\n" if links else "")
    )


def main(cfg: Config) -> None:
    directories, labels = (
        [EXPERIMENT / item for item in _items(cfg.run_dirs)],
        _items(cfg.labels),
    )
    assert directories
    assert len(directories) == len(labels)
    assert not cfg.output_dir.exists(), cfg.output_dir
    arms = [
        _arm(directory, label)
        for directory, label in zip(directories, labels, strict=True)
    ]
    _shared(arms)
    target = _force_target(arms[0]["protocol"])
    assert all(_force_target(arm["protocol"]) == target for arm in arms)
    cfg.output_dir.mkdir(parents=True)
    rows = _rows(arms, target)
    _plot(arms, target, cfg.output_dir / "neutral-adaptive-ipc")
    evidence = {
        "schema": "neutral-adaptive-ipc-report-v1",
        "same_input_hashes_verified": True,
        "force_target": target,
        "minimum_gap_guard_nm": 10.0,
        "adaptive_trigger_nm": 1e-6
        * float(arms[0]["stiffness"]["bbox_diagonal_m"])
        * 1e9,
        "rows": rows,
        "arms": [
            {
                "label": arm["label"],
                "directory": str(arm["directory"]),
                "protocol": arm["protocol"],
                "result": arm["result"],
                "stiffness": arm["stiffness"],
                "timing": arm["timing"],
                "logs": arm["logs"],
            }
            for arm in arms
        ],
    }
    (cfg.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    _document(cfg, rows, target, evidence)
    display = _display_rows(rows)
    result_columns = [
        ("arm", "Arm"),
        ("status", "Result"),
        ("wall", "Wall s"),
        ("steps", "PNCG / Newton"),
        ("force", "Terminal force"),
        ("gap", "Gap nm"),
        ("inversions", "Inversions"),
        ("kappa", "Final κ MPa"),
    ]
    timing_columns = [
        ("arm", "Arm"),
        ("pncg", "PNCG s"),
        ("newton", "Newton s"),
        ("ccd", "CCD leaf work s"),
    ]
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>Neutral adaptive IPC</title><style>body{{font:16px system-ui,sans-serif;max-width:1400px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:13px;margin:1rem 0}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto}}</style><main><h1>Neutral adaptive IPC stiffness forward test</h1><p>{html.escape(FORWARD_PROBLEM)}</p><p>κ0 = 0.1693 MPa; force target = {target:.3e}. Shared fixture and actual runtime provenance hashes were verified.</p><p><strong>{html.escape(_conclusion(rows))}</strong></p><h2>Forward validity</h2>{_html_table(display, result_columns)}<h2>Recorded timing breakdown</h2>{_html_table(display, timing_columns)}<p>PNCG and Newton are inclusive forward-solve phase times. CCD is a leaf-work subset already included in those totals, so it must not be added again. Operator prewarm is excluded; first sparse construction remains in Newton.</p><img src=\"neutral-adaptive-ipc.svg\" alt=\"Neutral force, gap, stiffness, and energy traces\"><p>Contact feasibility alone is insufficient. Failed runs are time-to-failure records, not speed comparisons. κ changes the barrier objective; energy lines split at κ changes.</p><p><a href=\"evidence.json\">Evidence JSON</a></p></main>"""
    (cfg.output_dir / "neutral-adaptive-ipc.html").write_text(page)
    for path in (
        cfg.output_dir / "neutral-adaptive-ipc.html",
        cfg.output_dir / "neutral-adaptive-ipc.png",
        cfg.output_dir / "neutral-adaptive-ipc.svg",
        cfg.output_dir / "evidence.json",
        cfg.document_path,
    ):
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
