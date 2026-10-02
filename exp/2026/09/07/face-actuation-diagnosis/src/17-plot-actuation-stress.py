"""Plot the frozen fixed-kinematics active-stress diagnostic."""

# ruff: noqa: EM101, EM102, RUF001, TRY003

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np

mpl.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = ROOT / "data" / "16-actuation-stress" / "summary.json"
DEFAULT_OUTPUT = ROOT / "data" / "17-actuation-stress-figure"

COLORS = {
    "fiber": "#3972a3",
    "expansion": "#d07c28",
    "compression": "#36877a",
}


def digest(path: Path, *, recorded_path: str) -> dict[str, str | int]:
    data = path.read_bytes()
    return {
        "path": recorded_path,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def classify(cases: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    if len(cases) != 3:
        raise ValueError(f"expected exactly three stress cases, got {len(cases)}")
    result: dict[str, dict[str, Any]] = {}
    offset_norms: list[float] = []
    for case in cases:
        matrix = np.asarray(case["Ainv"], dtype=float)
        if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
            raise ValueError(f"invalid inverse active map in {case.get('name')!r}")
        offset_norms.append(float(np.linalg.norm(matrix - np.eye(3))))
        det_f = float(case["detF_physical"])
        det_a = float(case["detAinv_parameter"])
        det_g = float(case["detG_elastic"])
        stress = float(case["first_piola_frobenius_MPa"])
        if case.get("F_fixed") != "identity_3x3" or det_f != 1.0:
            raise ValueError(f"case is not evaluated at fixed F=I: {case.get('name')}")
        if not math.isclose(det_g, det_f * det_a, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"determinant identity failed for {case.get('name')}")
        if not all(
            math.isfinite(value) and value > 0 for value in (det_a, det_g, stress)
        ):
            raise ValueError(
                f"plot requires finite positive values: {case.get('name')}"
            )
        kind = (
            "fiber"
            if math.isclose(det_a, 1.0, rel_tol=1e-12, abs_tol=1e-12)
            else "expansion"
            if det_a > 1.0
            else "compression"
        )
        if kind in result:
            raise ValueError(f"duplicate classified case: {kind}")
        result[kind] = case
    if set(result) != set(COLORS):
        raise ValueError(f"missing classified stress case: {set(COLORS) - set(result)}")
    if max(offset_norms) - min(offset_norms) > 1e-12:
        raise ValueError("cases do not have the same Frobenius active-map offset norm")
    return [(kind, result[kind]) for kind in ("fiber", "expansion", "compression")]


def case_labels() -> list[str]:
    return [
        "Fiber model\n50% natural contraction",
        "Inverse active-map\nexpansion",
        "Inverse active-map\ncompression",
    ]


def plot(cases: list[tuple[str, dict[str, Any]]], path: Path) -> None:
    mpl.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "font.size": 9.5,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    labels = case_labels()
    x = np.arange(len(cases), dtype=float)
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 6.6))
    figure.subplots_adjust(left=0.075, right=0.985, bottom=0.27, top=0.84, wspace=0.24)

    determinant_series = (
        ("Physical det(F)", "detF_physical", -0.17, "o", "#4f5964"),
        ("Inverse active det(A_inv)", "detAinv_parameter", 0.0, "s", "#d07c28"),
        ("Elastic det(G)", "detG_elastic", 0.17, "D", "#36877a"),
    )
    for legend, key, offset, marker, color in determinant_series:
        values = np.asarray([float(case[key]) for _, case in cases])
        axes[0].scatter(
            x + offset,
            values,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.7,
            s=75,
            label=legend,
            zorder=3,
        )
    for position, (_, case) in enumerate(cases):
        values = [
            float(case["detF_physical"]),
            float(case["detAinv_parameter"]),
            float(case["detG_elastic"]),
        ]
        if max(values) - min(values) < 1e-12:
            annotations = [(position, values[0], f"{values[0]:.5g} each")]
        else:
            annotations = [
                (position - 0.17, values[0], f"{values[0]:.5g}"),
                (position + 0.085, values[1], f"{values[1]:.5g} each"),
            ]
        for label_x, value, label in annotations:
            axes[0].annotate(
                label,
                (label_x, value),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes[0].set_yscale("log")
    axes[0].set_ylim(0.032, 8.5)
    axes[0].set_xlim(-0.42, 2.42)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Determinant (log scale)")
    axes[0].set_title("A  Physical and active-map determinants", loc="left")
    axes[0].grid(axis="y", which="both", alpha=0.22)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    stresses_kpa = np.asarray(
        [float(case["first_piola_frobenius_MPa"]) * 1000 for _, case in cases]
    )
    baseline = stresses_kpa[0]
    for position, ((kind, _), value) in enumerate(
        zip(cases, stresses_kpa, strict=True)
    ):
        axes[1].vlines(position, 3.0, value, color=COLORS[kind], linewidth=5)
        axes[1].scatter(
            position,
            value,
            color=COLORS[kind],
            edgecolor="white",
            linewidth=0.8,
            s=90,
            zorder=3,
        )
        ratio = value / baseline
        ratio_label = "reference" if position == 0 else f"{ratio:.3g}× fiber"
        axes[1].annotate(
            f"{value:.3f} kPa\n{ratio_label}",
            (position, value),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold" if position == 1 else "normal",
        )
    axes[1].set_yscale("log")
    axes[1].set_ylim(3.0, 7000)
    axes[1].set_xlim(-0.35, 2.35)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("First-Piola stress Frobenius norm (kPa, log scale)")
    axes[1].set_title("B  Constitutive stress magnitude", loc="left")
    axes[1].grid(axis="y", which="both", alpha=0.22)

    offset_norm = float(
        np.linalg.norm(np.asarray(cases[0][1]["Ainv"], dtype=float) - np.eye(3))
    )
    figure.suptitle(
        "Active-map determinant and stress at fixed physical deformation",
        fontsize=14,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.035,
        (
            f"All cases: fixed F = I and det(F) = 1; same ‖A_inv − I‖F = "
            f"{offset_norm:.6f}.  Pure-muscle constituent before mixture weighting; "
            "no equilibrium solve."
        ),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#343a40",
    )
    figure.savefig(path, dpi=300)
    plt.close(figure)


def main(input_path: Path, output: Path) -> None:
    input_path = input_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    document = json.loads(input_path.read_text())
    if document.get("schema_version") != 1:
        raise ValueError("stress summary must use schema version 1")
    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list):
        raise TypeError("stress summary cases must be a list")
    cases = classify(raw_cases)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".17-stress-", dir=output.parent) as tmp:
        stage = Path(tmp) / "artifact"
        stage.mkdir()
        png = stage / "actuation-stress.png"
        pdf = stage / "actuation-stress.pdf"
        plot(cases, png)
        plot(cases, pdf)
        source_record = digest(
            input_path,
            recorded_path=Path(input_path.relative_to(ROOT)).as_posix(),
        )
        script_path = Path(__file__).resolve()
        manifest = {
            "schema_version": 1,
            "scope": (
                "Plot of the frozen CPU constitutive diagnostic at fixed F=I; "
                "no equilibrium solve and no mixture weighting"
            ),
            "source": source_record,
            "script": digest(
                script_path,
                recorded_path=Path(script_path.relative_to(ROOT)).as_posix(),
            ),
            "same_Ainv_minus_I_frobenius": float(
                document["raw6_equal_offset_norm"]["fiber_Ainv_minus_I_frobenius"]
            ),
            "cases": [
                {
                    "kind": kind,
                    "display_label": case_labels()[index].replace("\n", " "),
                    "source_name": case["name"],
                    "detF_physical": case["detF_physical"],
                    "detAinv_parameter": case["detAinv_parameter"],
                    "detG_elastic": case["detG_elastic"],
                    "first_piola_frobenius_MPa": case["first_piola_frobenius_MPa"],
                }
                for index, (kind, case) in enumerate(cases)
            ],
            "outputs": {
                "png": digest(png, recorded_path=png.name),
                "pdf": digest(pdf, recorded_path=pdf.name),
            },
        }
        (stage / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
        )
        stage.rename(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    main(arguments.input, arguments.output)
