"""Render observed neutral-balance trends and the current readiness record."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pydantic_settings as ps
from joint_common import GROUP, ProfileJoint, write_json

from liblaf import cherries

LOG = logging.getLogger(__name__)


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    output_dir: Path = cherries.output("progress-report", mkdir=True)


def main(cfg: Config):
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=True)
    cases = {
        name: json.loads((GROUP / "data" / name / "summary.json").read_text())
        for name in ("neutral-prestress-001", "neutral-prestress-010")
    }
    oral_audit = json.loads(
        (GROUP / "data/prepared/neutral-pilot-oral-audit.json").read_text()
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for name, color in zip(cases, ("#086e72", "#bc622a"), strict=True):
        trace = json.loads((GROUP / "data" / name / "trace.json").read_text())
        xs = [row["update"] for row in trace]
        label = f"{trace[0]['skin_resultant_n_per_m']:.3g} N/m skin stress"
        for ax, key in (
            (axes[0, 0], "surface_motion_rms_mm"),
            (axes[0, 1], "muscle_centroid_motion_rms_mm"),
            (axes[1, 1], "detF_min"),
        ):
            ax.plot(
                xs,
                [row["metrics"][key] for row in trace],
                "-o",
                label=label,
                color=color,
                markersize=3,
            )
        axes[1, 0].plot(
            xs,
            [row["objective"] / trace[0]["objective"] for row in trace],
            "-o",
            label=label,
            color=color,
            markersize=3,
        )
    axes[0, 0].axhline(
        0.25, color="#66717b", linestyle="--", linewidth=1, label="surface budget"
    )
    titles = [
        "Neutral surface drift",
        "Neutral muscle centroid drift",
        "Objective / initial objective",
        "Smallest tetrahedral volume ratio",
    ]
    labels = [
        "Area-weighted RMS (mm)",
        "Volume-weighted RMS (mm)",
        "Relative objective",
        "min det(F)",
    ]
    for ax, title, label in zip(axes.flat, titles, labels, strict=True):
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(label)
        ax.set_xlabel("Neutral calibration update")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="center right")
    fig.suptitle("Neutral balance pilots · zero activation · full mesh", fontsize=14)
    fig.savefig(output / "neutral-trends.png", dpi=220)
    fig.savefig(output / "neutral-trends.pdf")
    plt.close(fig)
    rows = []
    for case in cases.values():
        first, last = case["first"], case["terminal"]
        metrics = last["metrics"]
        rows.append(
            f"| {first['skin_resultant_n_per_m']:.3g} N/m | {case['accepted_evaluations'] - 1} | {first['metrics']['surface_motion_rms_mm']:.3f} → {metrics['surface_motion_rms_mm']:.3f} mm | {metrics['muscle_centroid_motion_rms_mm']:.3f} mm | {metrics['detF_min']:.3f} | 0 |"
        )
    validations = []
    for name in (
        "material-validation",
        "field-validation",
        "equilibrium-validation",
        "coupled-validation",
        "face-gradient-validation",
        "equilibrium-initial-guard-validation",
        "coupled-initial-guard-validation",
    ):
        path = GROUP / "data" / name / "summary.json"
        if path.exists():
            value = json.loads(path.read_text())
            validations.append(
                {"name": name, "success": value["success"], "path": str(path)}
            )
        else:
            validations.append({"name": name, "success": None, "path": str(path)})
    manifest = json.loads((GROUP / "data/prepared/manifest.json").read_text())
    status = {
        "scope": "implementation and neutral pilots; no joint trajectory yet",
        "full_joint_run_ready": False,
        "validations": validations,
        "jaw_gate": manifest["gates"]["full_six_dof_jaw"],
        "neutral_cases": cases,
        "neutral_pilot_oral_audit": oral_audit,
        "pending_modeling_choice": "provisional inherited-contact exclusions versus lip/contact repair",
    }
    write_json(output / "status.json", status)
    oral_rows = []
    for run in oral_audit["runs"]:
        audit = run["audit"]
        upper = audit["fem_mandible_upper_oral"]
        lower = audit["fem_mandible_lower_oral"]
        oral_rows.append(
            f"| {run['run']} | {upper['new_contact_pairs']} | {lower['new_contact_pairs']} | {lower['worsened_inherited_pairs']} |"
        )
    full_face = json.loads(
        (GROUP / "data/face-gradient-validation/summary.json").read_text()
    )
    report = (
        """# Implementation and neutral-pilot results

**The full-mesh stress model and neutral pilots run. The multi-expression joint trajectory has not run.** The oral-geometry gate remains blocked by inherited intersections; a user choice about a provisional diagnostic variant is pending.

## Observed neutral trends

| Skin baseline | Updates | Surface drift | Final muscle drift | Final minimum det(F) | Inversions |
|---|---:|---|---|---:|---:|
"""
        + "\n".join(rows)
        + """

![Neutral balance trends](../data/progress-report/neutral-trends.png)

Both pilots retained zero activation and met the proposed **numerical deformation budgets**: surface RMS ≤0.25 mm and muscle-centroid RMS ≤0.5 mm. Surface fit improved while muscle-centroid drift increased slightly. All shared bulk stress tensors and skin stiffness were optimized; the prescribed skin stress stayed fixed during each continuation stage. **Neither passed the oral geometry audit.**

These skin stresses are **1% and 10% of an 80.6 N/m literature-derived proxy**, not full recovery of that proxy or a measured prestress map. The second stage starts from the first stage's best numerical-budget checkpoint, with fresh Adam moments. The saved filename `best-admissible.pt` predates the oral audit and only denotes the original numerical budget; it does not certify oral validity. The shared basis is spatially constant: its spatial smoothness is identically zero. No activation-smoothness result can be inferred from a zero-activation pilot.

The post-hoc audit checks the actual deformed FEM surface. Mandible support remains exactly at zero pose. Neither pilot adds or worsens contacts in the lip-only classifier, but both introduce and worsen mandible/oral contacts:

| Pilot | New upper-oral pairs | New lower-oral pairs | Worsened inherited lower-oral pairs |
|---|---:|---:|---:|
"""
        + "\n".join(oral_rows)
        + """

Pair counts and intersection-segment lengths are diagnostic proxies, not penetration depths or a contact law. The source/FEM oral correspondence is incomplete; these tests cannot certify every contact. The primary joint runner rejects these states.

## Implemented pieces

- Signed additive bulk stress and exact plane-stress Stable Neo-Hookean membrane with signed tangential stress.
- Twenty shared coefficients; dense six-component activation per active muscle tetrahedron; six jaw pose coordinates per expression.
- Owned equilibrium snapshots and direct plus implicit Dirichlet-pose gradients.
- Fixed input hashes, complete recovered mandibular support, four training targets and two reserved targets, and a same-muscle conductance graph.
- A gated joint/control runner with checkpointing. It remains unvalidated as a complete optimization workflow until the admission gates pass and it is executed.

## Readiness and limitation

The recovered fixture has 228,660 points, 1,146,517 tetrahedra and 288,235 active cells. There are 7,510 mapped mandibular boundary nodes. The source template contains 17 upper/lower-lip intersections after excluding shared seam vertices. Bone contacts include posterior joint-region contacts and anterior contacts that change with opening. A sampled hinge or pose box does not establish a valid deforming oral-contact model.

Material, field, implicit-boundary, coupled, and full-face derivative checks passed. The coupled test uses an independent explicit three-coordinate Newton solve for finite differences. The full-face test covers all shared baseline-stress families, skin stiffness, dense activation, and jaw rotation/translation at two step sizes (16 checks). Its maximum relative error is **FULL_FACE_ERROR_PERCENT%**, below the unchanged 2% criterion; 33 forward solves were used. An earlier attempt at absolute force tolerance 1e-13 stopped on strict Armijo near roundoff. The successful run used 1e-12 absolute and 1e-6 relative forward tolerances and 1e-7 adjoint tolerance. Its source snapshot and the failed-attempt receipt are preserved.

The final equilibrium adapter now recognizes an initially balanced state using the declared absolute free-force tolerance and clears stale solver state. Fresh synthetic equilibrium/coupled regressions pass, including exact neutral at 2.25e-24 free-force norm and zero iterations. The full-face derivative receipt predates this narrow change; a full-face revalidation will be required after the oral-model decision and final input preparation.

The current machine-readable validation and oral-audit status is in [status.json](../data/progress-report/status.json). Strong-smoothness calibration, a control trajectory, a joint trajectory, and a complete joint checkpoint-resume run remain unexecuted. The new runner checks artifact lineage and reconstructs objective components, freezes shared optimizer state in the control, and requires repeated smoothness calibration at tighter solver tolerance before fitting. The first pilot's activation neighbor-RMS budget is frozen at 0.05 normalized units (0.616 kPa); calibration must pass within it. The current neutral checkpoint also needs a new replay under the final manifest because the oral audit changed the manifest after the pilot.

The normal pilots produced local Cherries outputs; Comet warned that environment/Git-patch metadata did not finish logging. The first full-face validation also hit a Local-plugin log-copy error after writing its successful scientific receipt. The experiment profile now initializes Logging before Local so its reset does not remove Local's log handler; report generation verified that the snapshot log is written. A CLI help invocation also initialized a metadata-only Comet entry before it was interrupted; it executed no physics and is excluded from results. Local source snapshots, manifests, traces, checkpoints and scientific receipts remain available. Timing was collected while another GPU experiment was active and is not a clean joint-epoch benchmark.

## Commands and evidence

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```bash
CHERRIES_NAME="Joint inverse nonzero neutral prestress pilot" CHERRIES_TAGS="joint-inverse,neutral,prestress,pilot" uv run --frozen python src/20-neutral-pilot.py --output-dir data/neutral-prestress-001 --updates 8 --skin-prestress-fraction 0.01
CHERRIES_NAME="Joint inverse ten-percent prestress continuation" CHERRIES_TAGS="joint-inverse,neutral,prestress,continuation" uv run --frozen python src/20-neutral-pilot.py --output-dir data/neutral-prestress-010 --initial-checkpoint data/neutral-prestress-001/best-admissible.pt --updates 12 --skin-prestress-fraction 0.1 --learning-rate 0.003
```

Each run directory contains `summary.json`, `trace.json`, `protocol.json`, source hashes/snapshots and terminal/best-admissible checkpoints. Existing working-tree edits were preserved; no Git commit or push was made.
"""
    )
    report = report.replace(
        "FULL_FACE_ERROR_PERCENT",
        f"{100 * full_face['maximum_relative_error']:.4f}",
    )
    (GROUP / "docs/20-implementation-and-pilots.md").write_text(report)
    brief = """# Implementation progress

**Core implemented; full joint fitting is held at the oral-geometry gate.**

- All derivative checks passed; full-face maximum relative error: **0.035%**.
- At **8.06 N/m** skin stress, neutral surface drift fell **0.219 → 0.188 mm**, with no inverted elements. Muscle drift rose to **0.108 mm**.
- Both neutral pilots failed the oral-contact audit. Joint fitting awaits the modeling choice: a provisional diagnostic run with explicit contact exclusions, or geometry/contact repair first.

The skin stress is 10% of a literature proxy. These are neutral-balance trends; there is no joint trajectory yet.

![Neutral balance trends](/progress/neutral-trends.png)

[Detailed evidence](/progress/details/) · [Approved plan](/)
"""
    (GROUP / "docs/21-mobile-progress.md").write_text(brief)
    LOG.info("Rendered neutral trends and progress report to %s", output)
    cherries.log_output(output)


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
