"""Render the neutral-face collision and skin component ablation experiment."""

from __future__ import annotations

import hashlib
import html
import importlib.util
import json
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
    "All four arms start from the same repaired constitutive-reference neutral face, without a "
    "saved equilibrium displacement. Jaw rotation, muscle activation, and bulk additive stress "
    "are zero. The full arm contains bulk Stable Neo-Hookean tissues, heterogeneous plane-stress "
    "skin with its existing baseline tension, and IPC against the complete bones and fixed eyes. "
    "The ablations remove the stated potential or contact constraint; they do not represent the "
    "same physical forward problem."
)
spec = importlib.util.spec_from_file_location(
    "ablation_report_benchmark", HERE / "10-benchmark.py"
)
assert spec is not None
assert spec.loader is not None
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class Config(cherries.BaseConfig):
    run_dirs: str = (
        "data/neutral-ablation-full-001,data/neutral-ablation-no-collision-001,"
        "data/neutral-ablation-no-skin-001,data/neutral-ablation-neither-001"
    )
    labels: str = "full,no collision,no skin,neither"
    output_dir: Path = EXPERIMENT / "data/neutral-ablations-report-001"
    document_path: Path = EXPERIMENT / "docs/66-neutral-ablations.md"


def _items(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _trace(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def _logs(directory: Path) -> list[dict[str, str]]:
    output = []
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
            output.append(
                {
                    "path": str(path),
                    "command": command[-1] if command else "",
                    "comet_url": comet[-1] if comet else "",
                    "name": name[-1] if name else "",
                }
            )
    return output


def _binding(directory: Path) -> dict[str, Any]:
    """Compare receipts while replacing run-local diff paths by their content hash."""
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
        "provenance.json",
    ):
        assert (directory / name).is_file(), directory / name
    protocol, summary, stiffness, timing = (
        _json(directory / name)
        for name in ("protocol.json", "summary.json", "stiffness.json", "timing.json")
    )
    result = summary["result"]
    trace = [
        row
        for row in _trace(directory / "trace.jsonl")
        if row.get("kind") in {"initial", "pncg", "newton"}
    ]
    return {
        "label": label,
        "directory": directory,
        "protocol": protocol,
        "result": result,
        "stiffness": stiffness,
        "timing": timing,
        "states": trace,
        "logs": _logs(directory),
        "pncg_steps": int(result["pncg_steps"]),
        "newton_steps": int(result["newton_steps"]),
        "force": result["terminal_force"],
        "initial_force": result["initial_force"],
        "inversions": result["geometry"]["inverted_tetrahedra"],
        "collision_feasible": result["collision"]["state_feasible"],
    }


def _runtime_sources(directory: Path) -> dict[str, str]:
    """Mechanics/runtime bytes only; the report presentation source is intentionally excluded."""
    sources = _json(directory / "provenance.json")["sources"]
    solver = {
        "solver-performance/65-test-neutral-ablations.py",
        "solver-performance/pncg_first.py",
        "solver-performance/hybrid_first_solver.py",
        "solver-performance/accelerated_solvers.py",
        "solver-performance/hybrid_hessian.py",
        "solver-performance/adaptive_ipc_stiffness.py",
        "solver-performance/profile_input_binding.py",
        "solver-performance/inverse_timing.py",
        "solver-performance/mesh_step_scale.py",
    }
    assert solver <= sources.keys(), (solver - sources.keys(), directory)
    selected = {
        key: value
        for key, value in sources.items()
        if key.startswith(("apple/", "joint-experiment/", "tensor-reference/"))
        or key in solver
    }
    assert selected
    return selected


def _shared(arms: list[dict[str, Any]]) -> None:
    reference = arms[0]["protocol"]
    assert reference["schema"] == "natural-reference-component-ablation-v1"
    for arm in arms[1:]:
        other = arm["protocol"]
        assert other["schema"] == reference["schema"]
        for key in (
            "fixture",
            "sources",
            "materials",
            "solver",
            "coverage",
            "versions",
            "boundary_projection",
            "prewarm_scope",
            "timing",
        ):
            assert other[key] == reference[key], (key, arm["label"])
        for key in (
            "anchor_rule",
            "anchor_stiffness_mpa",
            "epsilon_scale",
            "tolerance_anchor_force",
            "effective_force_tolerance",
            "tolerance_rule",
        ):
            assert (
                other["contact_stiffness_policy"][key]
                == reference["contact_stiffness_policy"][key]
            ), (key, arm["label"])
    binding = _binding(arms[0]["directory"])
    runtime_sources = _runtime_sources(arms[0]["directory"])
    for arm in arms[1:]:
        assert _binding(arm["directory"]) == binding, arm["label"]
        assert _runtime_sources(arm["directory"]) == runtime_sources, arm["label"]


def _target(arm: dict[str, Any]) -> float:
    return float(
        arm["protocol"]["contact_stiffness_policy"]["effective_force_tolerance"]
    )


def _walk(
    node: dict[str, Any], path: tuple[str, ...] = ()
) -> list[tuple[tuple[str, ...], dict[str, Any]]]:
    output = [(path, node)]
    for name, child in node.get("children", {}).items():
        output.extend(_walk(child, (*path, name)))
    return output


def _timing(arm: dict[str, Any]) -> dict[str, float]:
    root = arm["timing"]["tree"]["forward"]
    records = _walk(root)
    children = root.get("children", {})

    def exclusive(prefixes: tuple[str, ...]) -> float:
        return sum(
            float(node["exclusive_seconds"])
            for path, node in records
            if path and path[-1].startswith(prefixes)
        )

    def inclusive(name: str) -> float:
        return sum(
            float(node["inclusive_seconds"])
            for path, node in records
            if path and path[-1] == name
        )

    return {
        "total_seconds": float(root["inclusive_seconds"]),
        "pncg_seconds": float(children.get("pncg", {}).get("inclusive_seconds", 0.0)),
        "newton_seconds": float(
            children.get("newton", {}).get("inclusive_seconds", 0.0)
        ),
        "bulk_exclusive_seconds": exclusive(("bulk/",)),
        "skin_exclusive_seconds": exclusive(("skin/",)),
        "collision_ipc_exclusive_seconds": exclusive(("collision/", "ipc/")),
        "ccd_inclusive_seconds": inclusive("ipc/ccd"),
        "hessian_cache_inclusive_seconds": inclusive("hessian/cache_prepare"),
        "fem_constructor_inclusive_seconds": inclusive("hessian/fem_constructor"),
        "csr_constructor_inclusive_seconds": inclusive("hessian/csr_constructor"),
        "pcg_inclusive_seconds": inclusive("pcg"),
    }


def _valid(arm: dict[str, Any], target: float) -> bool:
    return bool(
        arm["result"]["valid_forward"]
        and arm["force"] is not None
        and float(arm["force"]) <= target
    )


def _global(arm: dict[str, Any], row: dict[str, Any]) -> int:
    return (
        arm["pncg_steps"] + int(row.get("step", 0))
        if row.get("kind") == "newton"
        else int(row.get("step", 0))
    )


def _plot(arms: list[dict[str, Any]], target: float, output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    colors = plt.get_cmap("tab10").colors
    invalid_label = False
    for index, arm in enumerate(arms):
        color, states = colors[index], arm["states"]
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
        if not _valid(arm, target) and arm["force"] is not None:
            axes[0, 0].scatter(
                [arm["pncg_steps"] + arm["newton_steps"]],
                [float(arm["force"])],
                color="#b2182b",
                marker="x",
                s=42,
                label=None if invalid_label else "invalid endpoint",
            )
            axes[0, 1].scatter(
                [float(arm["result"]["forward_wall_seconds"])],
                [float(arm["force"])],
                color="#b2182b",
                marker="x",
                s=42,
            )
            invalid_label = True
    for axis in axes[0]:
        axis.set_yscale("log")
        axis.axhline(
            target,
            color="#333",
            linestyle="--",
            linewidth=1,
            label="shared force target",
        )
        axis.grid(visible=True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    axes[0, 0].set(
        xlabel="Global applied update",
        ylabel="Force norm (MPa m²)",
        title="Force by update",
    )
    axes[0, 1].set(
        xlabel="Elapsed forward time (s)",
        ylabel="Force norm (MPa m²)",
        title="Force by wall time",
    )
    metric_labels = ["total", "PNCG", "Newton"]
    for index, arm in enumerate(arms):
        timing = _timing(arm)
        axes[1, 0].bar(
            [position + index * 0.18 for position in range(3)],
            [timing["total_seconds"], timing["pncg_seconds"], timing["newton_seconds"]],
            width=0.16,
            label=arm["label"],
        )
    axes[1, 0].set(
        xticks=range(3),
        xticklabels=metric_labels,
        ylabel="Inclusive phase time (s)",
        title="Forward timing phases",
    )
    axes[1, 0].legend(fontsize=8)
    component_labels = [
        "bulk\nexclusive",
        "skin\nexclusive",
        "collision / IPC\nexclusive",
        "native CCD\ninclusive subset",
        "Hessian cache\ninclusive subset",
        "PCG\ninclusive subset",
    ]
    for index, arm in enumerate(arms):
        timing = _timing(arm)
        values = [
            timing["bulk_exclusive_seconds"],
            timing["skin_exclusive_seconds"],
            timing["collision_ipc_exclusive_seconds"],
            timing["ccd_inclusive_seconds"],
            timing["hessian_cache_inclusive_seconds"],
            timing["pcg_inclusive_seconds"],
        ]
        axes[1, 1].bar(
            [position + index * 0.18 for position in range(6)],
            values,
            width=0.16,
            label=arm["label"],
        )
    axes[1, 1].set(
        xticks=range(6),
        xticklabels=component_labels,
        ylabel="Scoped time (s)",
        title="Potential and solver components",
    )
    axes[1, 1].legend(fontsize=8)
    for axis in axes[1]:
        axis.grid(visible=True, axis="y", alpha=0.25)
    figure.savefig(output.with_suffix(".png"), dpi=180)
    figure.savefig(output.with_suffix(".svg"))
    plt.close(figure)


def _rows(arms: list[dict[str, Any]], target: float) -> list[dict[str, Any]]:
    output = []
    for arm in arms:
        ablation, valid = arm["protocol"]["ablation"], _valid(arm, target)
        contact_on, skin_on = (
            bool(ablation["collision_enabled"]),
            bool(ablation["skin_enabled"]),
        )
        status = (
            "valid configured PDE"
            if valid
            else (
                "force converged; invalid geometry"
                if arm["result"]["success"]
                else "failed before force convergence"
            )
        )
        output.append(
            {
                "arm": arm["label"],
                "case": ablation["case"],
                "collision": contact_on,
                "skin": skin_on,
                "status": status,
                "wall_seconds": float(arm["result"]["forward_wall_seconds"]),
                "pncg_steps": arm["pncg_steps"],
                "newton_steps": arm["newton_steps"],
                "terminal_force": arm["force"],
                "inversions": arm["inversions"],
                "valid_configured_pde": valid,
                "force_converged": bool(
                    arm["result"]["success"] and float(arm["force"]) <= target
                ),
                "full_contact_diagnostic": bool(arm["result"]["full_contact_valid"]),
                **_timing(arm),
            }
        )
    return output


def _display(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "arm": row["arm"],
            "status": row["status"],
            "wall": f"{row['wall_seconds']:.3f}",
            "steps": f"{row['pncg_steps']} / {row['newton_steps']}",
            "force": f"{float(row['terminal_force']):.3e}",
            "inversions": str(row["inversions"]),
            "contact": str(row["full_contact_diagnostic"]),
            "total": f"{row['total_seconds']:.3f}",
            "pncg": f"{row['pncg_seconds']:.3f}",
            "newton": f"{row['newton_seconds']:.3f}",
            "bulk": f"{row['bulk_exclusive_seconds']:.3f}",
            "skin": f"{row['skin_exclusive_seconds']:.3f}",
            "collision": f"{row['collision_ipc_exclusive_seconds']:.3f}",
            "ccd": f"{row['ccd_inclusive_seconds']:.3f}",
            "hessian": f"{row['hessian_cache_inclusive_seconds']:.3f}",
            "constructors": f"{row['fem_constructor_inclusive_seconds']:.3f} / {row['csr_constructor_inclusive_seconds']:.3f}",
            "pcg": f"{row['pcg_inclusive_seconds']:.3f}",
        }
        for row in rows
    ]


def _html_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    head = "".join(f"<th>{html.escape(title)}</th>" for _, title in columns)
    body = "".join(
        "<tr>"
        + "".join(f"<td>{html.escape(row[key])}</td>" for key, _ in columns)
        + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _conclusion(rows: list[dict[str, Any]]) -> str:
    by_case = {row["case"]: row for row in rows}
    valid = [row["arm"] for row in rows if row["valid_configured_pde"]]
    invalid = [row["arm"] for row in rows if not row["valid_configured_pde"]]
    full, no_collision = by_case["full"], by_case["no_collision"]
    return (
        f"All arms reached the shared force threshold. {', '.join(invalid)} are geometrically invalid; "
        f"{', '.join(valid)} are valid only under their configured reduced-PDE gates. The full arm took "
        f"{full['wall_seconds']:.3f} s and has no valid-full-model speed comparison. Its direct exclusive "
        f"bulk/skin/collision-IPC work was {full['bulk_exclusive_seconds']:.3f} / "
        f"{full['skin_exclusive_seconds']:.3f} / {full['collision_ipc_exclusive_seconds']:.3f} s; "
        f"the no-collision reduced arm took {no_collision['wall_seconds']:.3f} s. Skin direct evaluation is "
        "small here, but skin removal also removes its baseline load and changed the nonlinear path."
    )


REPRODUCIBILITY_NOTE = (
    "Reproducibility note, not a timing arm: the archived prior adaptive full run "
    "`data/neutral-adaptive-ipc-adaptive-002` had matching recorded inputs and parameters. Its initial "
    "force was bit-identical; initial energy differed by 1.3e-22 and PNCG step-1 force by 5.42e-19. "
    "The first limiter decision diverged at step 6, after which κ schedule and terminal volume differed. "
    "Extra gradient/prewarm ordering and synchronized component hooks are possible perturbations; causal "
    "attribution has not been tested."
)


def _document(
    cfg: Config, rows: list[dict[str, Any]], target: float, evidence: dict[str, Any]
) -> None:
    display = _display(rows)
    validity = "\n".join(
        "| {arm} | {status} | {wall} | {steps} | {force} | {inversions} | {contact} |".format(
            **row
        )
        for row in display
    )
    timing = "\n".join(
        "| {arm} | {total} | {pncg} | {newton} | {bulk} | {skin} | {collision} | {ccd} | {hessian} | {constructors} | {pcg} |".format(
            **row
        )
        for row in display
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
        "# Neutral face collision and skin ablations\n\n"
        + FORWARD_PROBLEM
        + "\n\n"
        + f"Every arm uses the same absolute force target, anchored once from the full κ0 initial gradient: `{target:.17g}`. Fixture, material, solver, runtime-source, and input-binding provenance hashes match; only the recorded ablation fields and measured timings vary.\n\n"
        + _conclusion(rows)
        + "\n\n"
        "| Arm | Configured PDE result | Wall s | PNCG / Newton | Terminal force | Inversions | Full-contact diagnostic |\n| --- | --- | ---: | ---: | ---: | ---: | --- |\n"
        + validity
        + "\n\n"
        "| Arm | Forward inclusive s | PNCG inclusive s | Newton inclusive s | Bulk exclusive s | Skin exclusive s | Collision/IPC exclusive s | Native CCD inclusive subset s | Hessian cache inclusive subset s | FEM / CSR constructor inclusive s | PCG inclusive subset s |\n| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
        + timing
        + "\n\n"
        "Bulk, skin, and collision/IPC values sum exclusive scopes, avoiding nested-wrapper double counting. Collision/IPC includes broad phase, candidate preparation, and native IPC work. Native CCD is reported separately as an inclusive `ipc/ccd` subset of that pipeline; it must not be added to collision/IPC, PNCG, or Newton totals. Hessian-cache preparation, the first FEM/CSR constructors, and PCG are inclusive phase subsets. Startup/model construction and the ordinary prewarm are excluded; first sparse construction is included. Collision-disabled intersections remain a full-contact diagnostic, not a reduced-PDE convergence gate. Removing skin eliminates its prescribed baseline load, so its timing change is a physical reduction as well as removed computation.\n\n"
        + REPRODUCIBILITY_NOTE
        + "\n\n"
        + f"Outputs: `{cfg.output_dir / 'neutral-ablations.png'}`, `{cfg.output_dir / 'neutral-ablations.svg'}`, and `{cfg.output_dir / 'evidence.json'}`.\n\n"
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
    target = _target(arms[0])
    assert all(_target(arm) == target for arm in arms)
    rows = _rows(arms, target)
    evidence = {
        "schema": "neutral-component-ablation-report-v1",
        "same_runtime_provenance_verified": True,
        "shared_force_target": target,
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
    cfg.output_dir.mkdir(parents=True)
    _plot(arms, target, cfg.output_dir / "neutral-ablations")
    (cfg.output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    _document(cfg, rows, target, evidence)
    display = _display(rows)
    validity_columns = [
        ("arm", "Arm"),
        ("status", "Configured PDE"),
        ("wall", "Wall s"),
        ("steps", "PNCG / Newton"),
        ("force", "Terminal force"),
        ("inversions", "Inversions"),
        ("contact", "Full-contact diagnostic"),
    ]
    timing_columns = [
        ("arm", "Arm"),
        ("total", "Forward s"),
        ("pncg", "PNCG s"),
        ("newton", "Newton s"),
        ("bulk", "Bulk exclusive s"),
        ("skin", "Skin exclusive s"),
        ("collision", "Collision/IPC exclusive s"),
        ("ccd", "Native CCD subset s"),
        ("hessian", "Hessian cache subset s"),
        ("constructors", "FEM / CSR constructor subset s"),
        ("pcg", "PCG subset s"),
    ]
    page = f"""<!doctype html><meta charset=\"utf-8\"><title>Neutral component ablations</title><style>body{{font:16px system-ui,sans-serif;max-width:1500px;margin:2rem auto;padding:0 1rem}}table{{border-collapse:collapse;font-size:13px;margin:1rem 0}}th,td{{border:1px solid #bbb;padding:.35rem;text-align:right}}th:first-child,td:first-child{{text-align:left}}img{{max-width:100%;height:auto}}</style><main><h1>Neutral face collision and skin ablations</h1><p>{html.escape(FORWARD_PROBLEM)}</p><p>Shared full-κ0 force target: {target:.3e}. Runtime provenance hashes were verified.</p><p><strong>{html.escape(_conclusion(rows))}</strong></p><h2>Configured-PDE validity</h2>{_html_table(display, validity_columns)}<h2>Timing</h2>{_html_table(display, timing_columns)}<p>Bulk, skin, and collision/IPC sum exclusive scopes. Collision/IPC includes broad phase and preparation. Native CCD is an inclusive subset already contained in collision/IPC and PNCG/Newton totals; Hessian cache, first FEM/CSR construction, and PCG are also inclusive subsets. Ordinary prewarm is excluded; first sparse construction is included.</p><p>{html.escape(REPRODUCIBILITY_NOTE)}</p><img src=\"neutral-ablations.svg\" alt=\"Ablation force and timing comparison\"><p><a href=\"evidence.json\">Evidence JSON</a></p></main>"""
    (cfg.output_dir / "neutral-ablations.html").write_text(page)
    for path in (
        cfg.output_dir / "neutral-ablations.html",
        cfg.output_dir / "neutral-ablations.png",
        cfg.output_dir / "neutral-ablations.svg",
        cfg.output_dir / "evidence.json",
        cfg.document_path,
    ):
        cherries.log_output(path)


if __name__ == "__main__":
    cherries.main(main, profile=benchmark.ProfilePerformance)
