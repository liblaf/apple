"""Write a compact report from the verified four-stage MouthOpen chain."""

# ruff: noqa: C901, EM101, EM102, PLR0912, PLR0915, RUF001, S105, TRY003
from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Any

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]
ROOT = GROUP.parents[4]
sys.path.append(str(ROOT / "exp/2026/09/21/stress-activation-loss/src"))
from experiment import Profile  # noqa: E402

STAGES = ("symmetric6", "psd6", "rankone_fixed", "rankone_learned")
LABELS = {
    "symmetric6": ("Symmetric, unrestricted", 6),
    "psd6": ("PSD, contraction only", 6),
    "rankone_fixed": ("Rank one, fixed axis", 1),
    "rankone_learned": ("Rank one, learned axis", 3),
}


class Config(cherries.BaseConfig):
    output: Path = Path("87-four-stage-report")
    chain: Path = GROUP / "data/70-mouthopen-four-stage"
    analysis: Path = Path("80-four-stage-analysis/analysis.json")
    render: Path = Path("85-mouthopen-four-stage")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }


def check(item: dict[str, Any], label: str) -> Path:
    path = Path(item["path"])
    if sha256(path) != item["sha256"]:
        raise ValueError(f"{label} receipt mismatch: {path}")
    return path


def fmt(value: Any, digits: int = 4) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def sci(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3g}"


def doc_link(path: Path, report_dir: Path) -> str:
    return Path(os.path.relpath(path.resolve(), report_dir.resolve())).as_posix()


def load_log(
    stage: str,
    marker: str,
    output_name: str,
    *,
    required: bool = True,
    allow_default_output: bool = False,
) -> dict[str, Any]:
    matches = []
    for path in sorted((GROUP / "logs").glob(f"{stage}-*.log")):
        text = path.read_text(errors="replace")
        if marker not in text or "cherries/end_time" not in text:
            continue
        command_line = re.search(r"\bcherries/cmd\s*:\s*(.*)$", text, re.MULTILINE)
        if command_line is None:
            continue
        command = shlex.split(command_line.group(1).strip())
        output = None
        for index, token in enumerate(command):
            if token == "--output" and index + 1 < len(command):
                output = command[index + 1]
                break
            if token.startswith("--output="):
                output = token.partition("=")[2]
                break
        if output is None and allow_default_output:
            output = output_name
        if output is not None and Path(output).name == output_name:
            matches.append((path, text))
    if not matches:
        if required:
            raise FileNotFoundError(
                f"no completed {stage} log matched {marker!r} and output {output_name!r}"
            )
        return {"status": "not_found"}
    path, text = max(matches, key=lambda pair: pair[0].stat().st_mtime_ns)

    def field(name: str) -> str | None:
        regex = re.compile(rf"\bcherries/{re.escape(name)}\s*:\s*(.*)$")
        found = [
            m.group(1).strip()
            for line in text.splitlines()
            if (m := regex.search(line))
        ]
        return found[-1] if found else None

    url = field("comet/url")
    if url is None:
        match = re.search(r"https://www\.comet\.com/[^\s]+", text)
        url = match.group(0) if match else None
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "command": field("cmd"),
        "comet_url": url,
        "start_time": field("start_time"),
        "end_time": field("end_time"),
        "git_sha": field("git/sha"),
    }


def main(cfg: Config) -> None:
    chain_dir = cfg.chain
    analysis_path = cherries.input(cfg.analysis)
    render_dir = cherries.input(cfg.render)
    out = cherries.output(cfg.output)
    report_path = GROUP / "docs/87-mouthopen-four-stage-results.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite {report_path}")
    out.mkdir(parents=True, exist_ok=False)
    chain = json.loads((chain_dir / "chain-status.json").read_text())
    protocol = json.loads((chain_dir / "protocol.json").read_text())
    analysis = json.loads(analysis_path.read_text())
    render = json.loads((render_dir / "manifest.json").read_text())
    analysis_link = doc_link(analysis_path, report_path.parent)
    figure_link = doc_link(Path(render["figure"]["path"]), report_path.parent)
    preview_link = doc_link(Path(render["preview"]["path"]), report_path.parent)
    render_manifest_link = doc_link(render_dir / "manifest.json", report_path.parent)
    if (
        chain.get("status") != "completed_attempt_budgets"
        or tuple(chain["completed"]) != STAGES
    ):
        raise RuntimeError("four-stage chain is incomplete; report withheld")
    if protocol.get("schema") != "mouthopen-four-stage-chain-v1":
        raise ValueError("unexpected chain protocol")
    declared_config = protocol["config"]
    eta = float(declared_config["smooth_weight"])
    beta = float(declared_config["normal_weight"])
    if eta != 7.2e-6 or beta != 1.0:
        raise ValueError(f"unexpected objective weights: eta={eta}, beta={beta}")
    if analysis.get("schema") != "mouthopen-four-stage-cpu-audit-v1":
        raise ValueError("unexpected CPU analysis schema")
    if render.get("schema") != "mouthopen-four-stage-render-v1" or not render.get(
        "all_stages_terminal_200_attempts"
    ):
        raise ValueError("renderer manifest does not certify all four terminal stages")
    for key, item in protocol["inputs"].items():
        check(item, f"chain input {key}")
    for key in ("protocol", "mesh"):
        check(render[key], f"render {key}")
    for item in render["inputs"].values():
        check(item, "renderer input")
    for item in render["sources"]:
        check(item, "renderer source")
    for key in ("figure", "preview"):
        check(render[key], f"render {key}")
    if render["chain_receipts"]["status"]["sha256"] != sha256(
        chain_dir / "chain-status.json"
    ):
        raise ValueError("renderer used a different chain status receipt")

    rows = {}
    for mode in STAGES:
        summary_path = chain_dir / mode / "summary.json"
        checkpoint = chain_dir / mode / "last.npz"
        gradient_path = chain_dir / mode / "gradient-balance.json"
        summary = json.loads(summary_path.read_text())
        stage = analysis["stages"][mode]
        for input_name, item in summary["inputs"].items():
            check(item, f"{mode} declared input {input_name}")
        source_manifest_path = chain_dir / mode / "source-manifest.json"
        source_manifest = json.loads(source_manifest_path.read_text())
        for source_name, item in source_manifest.items():
            check(item, f"{mode} frozen source {source_name}")
        if (
            summary["status"] != "completed_attempt_budget"
            or int(summary["attempted_updates"]) != 200
        ):
            raise ValueError(f"{mode} did not complete the 200-attempt budget")
        if (
            float(summary["config"]["smooth_weight"]) != eta
            or float(summary["config"]["normal_weight"]) != beta
        ):
            raise ValueError(f"{mode} objective weights differ from chain protocol")
        if int(summary["optimizer_updates"]) + int(summary["skipped_updates"]) != 200:
            raise ValueError(f"{mode} attempt accounting does not balance")
        if stage["checkpoint"]["sha256"] != sha256(checkpoint):
            raise ValueError(f"CPU analysis checkpoint differs for {mode}")
        if summary["final_checkpoint"]["sha256"] != sha256(checkpoint):
            raise ValueError(f"summary checkpoint receipt differs for {mode}")
        receipts = [
            json.loads(line)
            for line in (chain_dir / mode / "solver-receipts.jsonl")
            .read_text()
            .splitlines()
            if line
        ]
        primal = [float(row["forward"]["accepted_force_norm"]) for row in receipts]
        adjoint = [
            float(row["adjoint"]["relative_residual"])
            for row in receipts
            if not row["adjoint"].get("zero_rhs", False)
        ]
        geometry = receipts[-1]["forward"].get("geometry")
        if not isinstance(geometry, dict) or "has_intersections" not in geometry:
            raise ValueError(
                f"{mode} final forward receipt lacks boundary geometry audit"
            )
        gradient = json.loads(gradient_path.read_text())
        if gradient.get("status") == "available":
            check(gradient["checkpoint"], f"{mode} endpoint gradient checkpoint")
            check(gradient["components"], f"{mode} gradient components")
            ratio = float(gradient["smoothness_to_l2_gradient_ratio"])
            displacement_drift_um = 1e6 * float(
                gradient["displacement_change_from_saved_m"]
            )
        else:
            ratio = displacement_drift_um = None
        roughness = stage["roughness"]
        amp_key = (
            "rankone_amplitude"
            if mode.startswith("rankone")
            else (
                "signed_eigenvalue_amplitude"
                if mode == "symmetric6"
                else "eigenvalue_amplitude"
            )
        )
        direction_key = (
            "rankone_direction" if mode.startswith("rankone") else "orientation"
        )
        rows[mode] = {
            "label": LABELS[mode][0],
            "dof": LABELS[mode][1],
            "budget": stage["budget"],
            "metrics": stage["metrics"],
            "roughness": roughness,
            "amplitude_roughness": roughness[amp_key],
            "directional_roughness": roughness[direction_key],
            "gradient_ratio": ratio,
            "endpoint_displacement_drift_um": displacement_drift_um,
            "endpoint_diagnostic": stage["endpoint_diagnostic"],
            "solver_converged": stage["solver_converged"],
            "orientation_valid": stage["orientation_valid"],
            "solver_residuals": {
                "primal_min": min(primal),
                "primal_max": max(primal),
                "adjoint_relative_min": min(adjoint) if adjoint else None,
                "adjoint_relative_max": max(adjoint) if adjoint else None,
                "evaluation_count": len(receipts),
            },
            "boundary_geometry": geometry,
            "summary": record(summary_path),
            "source_manifest": record(source_manifest_path),
            "declared_inputs": summary["inputs"],
            "checkpoint": record(checkpoint),
            "gradient_balance": record(gradient_path),
            "analysis_checkpoint": stage["checkpoint"],
        }

    logs = {
        "70 numerical chain": load_log(
            "70",
            "70-run-four-stage-mouthopen.py",
            chain_dir.name,
            allow_default_output=True,
        ),
        "80 CPU analysis": load_log(
            "80", "80-analyze-four-stage-mouthopen.py", analysis_path.parent.name
        ),
        "85 renderer": load_log(
            "85", "85-render-four-stage-mouthopen.py", render_dir.name
        ),
    }
    chain_inputs = dict(protocol["inputs"])
    md = [
        "# MouthOpen activation parameterization chain",
        "",
        f"The four activation parameterizations used `η = {eta:.2g}` (10× the earlier `7.2e-7` setting) and `β = {beta:g}`. They received 200 optimizer attempts each; this fixed budget does not establish optimizer convergence. The full tensor field drove each saved geometry. Glyph amplitude is the singular-value amplitude `a = σmax(B) - 1`; the figure shows its largest positive principal mode.",
        "",
        f"![Four-stage MouthOpen comparison]({preview_link})",
        "",
        f"[Full-resolution figure]({figure_link}) · [render manifest]({render_manifest_link}) · [independent CPU analysis]({analysis_link})",
        "",
        "| Parameterization | DoF/cell | Attempts | Updates | Skips | Position RMS (mm) | Normal RMS (°) | Amplitude roughness | Direction roughness | Full-S smoothness/L2 gradient ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for mode in STAGES:
        row = rows[mode]
        budget, metrics_row = row["budget"], row["metrics"]
        md.append(
            f"| {row['label']} | {row['dof']} | {budget['attempted_updates']} | {budget['optimizer_updates']} | {budget['skipped_updates']} | {fmt(metrics_row['fit_rms_mm'], 3)} | {fmt(metrics_row['normal_angle_rms_deg'], 3)} | {fmt(row['amplitude_roughness'], 6)} | {fmt(row['directional_roughness'], 6)} | {fmt(row['gradient_ratio'], 5)} |"
        )
    md += [
        "",
        "The spectral roughness terms sum to the direct weighted Frobenius roughness; the CPU audit checked that identity for each endpoint. For unrestricted symmetric activation, amplitude roughness uses signed eigenvalues. For PSD and rank-one stages it uses nonnegative eigenvalues; rank-one directional roughness uses the squared-axis alignment identity.",
        "",
        "The gradient ratio is `η ||∇S R|| / ||∇S L2||` in the symmetric full-tensor space, using dual effective-active-volume weighting; the normal-loss term is excluded. It is an endpoint diagnostic. Its independent forward solve may slightly change displacement, as reported per stage in `report.json`.",
        "",
        "## Solver and orientation diagnostics",
        "",
        "| Stage | Primal free-force residual range | Relative adjoint residual range | Inverted cells | Minimum det(F) | Inverted rest-volume fraction | Boundary self-intersection audit |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for mode in STAGES:
        row = rows[mode]
        metrics_row, residuals = row["metrics"], row["solver_residuals"]
        md.append(
            f"| {mode} | {sci(residuals['primal_min'])}–{sci(residuals['primal_max'])} | {sci(residuals['adjoint_relative_min'])}–{sci(residuals['adjoint_relative_max'])} | {metrics_row['inverted_all_cells']} | {fmt(metrics_row['detF_min'], 5)} | {fmt(metrics_row['inverted_rest_volume_fraction'], 6)} | {'Intersections detected' if row['boundary_geometry']['has_intersections'] else 'None detected'} |"
        )
    md += [
        "",
        "The saved primal and adjoint receipts passed the numerical solver gates. Each accepted endpoint also has a self-intersection audit of the complete extracted FEM boundary; that audit does not test separate bone obstacles, containment, or contact forces. Inversions and boundary intersections remain geometric diagnostics, so the contact-off chain does not validate mechanical realism.",
        "",
        "The target is a transferred MouthOpen blendshape, with jaw motion prescribed from the chin-derived rigid pose. The chain retained the `IsFixed` constraints, pruned fixture, and recorded material model. Stage 1 starts at zero activation. Stage 2 PSD-projects stage 1; stage 3 keeps the largest nonnegative eigenmode of stage 2; stage 4 carries stage 3's rank-one tensor into learned-axis controls. Each stage carries its predecessor's displacement and starts fresh Adam. In the learned-axis stage, zero-amplitude axes are released using the saved total tensor gradient.",
        "",
        "## Run records",
        "",
        f"Chain protocol: `{record(chain_dir / 'protocol.json')['sha256']}` · chain status: `{record(chain_dir / 'chain-status.json')['sha256']}` · source manifest: `{record(chain_dir / 'source-manifest.json')['sha256']}`.",
        "",
        "Declared input receipts:",
    ]
    for name, item in chain_inputs.items():
        md.append(f"- `{name}`: `{item['sha256']}`")
    md += [
        "",
        "Completed Cherries commands and URLs are included only when present in the corresponding logs:",
        "",
        "| Run | Command | Comet URL | Start–end | Log SHA-256 |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name, item in logs.items():
        url = (
            f"[Comet]({item['comet_url']})" if item.get("comet_url") else "not recorded"
        )
        md.append(
            f"| {name} | `{item.get('command') or 'not recorded'}` | {url} | `{item.get('start_time')} – {item.get('end_time')}` | `{item.get('sha256', 'n/a')}` |"
        )
    md += [
        "",
        "Stage summaries, endpoint checkpoints, source manifests, solver receipts, gradient components, and figure/source hashes are retained in the linked output directories and `report.json`.",
        "",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(md))
    result = {
        "schema": "mouthopen-four-stage-report-v1",
        "status": "complete",
        "chain": record(chain_dir / "chain-status.json"),
        "protocol": record(chain_dir / "protocol.json"),
        "analysis": record(analysis_path),
        "render_manifest": record(render_dir / "manifest.json"),
        "stages": rows,
        "logs": logs,
        "figure": render["figure"],
        "preview": render["preview"],
        "inputs": {
            str((chain_dir / name).resolve()): record(chain_dir / name)
            for name in (
                "protocol.json",
                "chain-status.json",
                "source-manifest.json",
                "mesh.npz",
            )
        },
        "markdown": record(report_path),
        "mechanical_validity_claim": False,
    }
    (out / "report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    cherries.log_metrics(
        {f"{mode}/fit_rms_mm": rows[mode]["metrics"]["fit_rms_mm"] for mode in STAGES}
    )


if __name__ == "__main__":
    cherries.main(main, profile=Profile)
