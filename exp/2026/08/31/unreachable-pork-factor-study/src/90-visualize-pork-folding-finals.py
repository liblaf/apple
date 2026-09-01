"""Render literal final 2-D folding states and matched factor-pair sheets."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

import matplotlib as mpl
import numpy as np
import pydantic_settings as ps
import pyvista as pv
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from PIL import Image

from liblaf import cherries

mpl.use("Agg")
import matplotlib.pyplot as plt

FACTORS = ("geometry", "muscle_extent", "activation_sharing", "poisson")
FAT, MUSCLE, TARGET, TOP = "#E8C9A1", "#B85C5C", "#C2185B", "#17212B"


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    input_roots: str = "data/60-pork-folding-2d"
    output_dir: Path = cherries.output("90-folding-final-shapes", mkdir=True)
    dpi: int = 200


def fail(message: str) -> None:
    raise ValueError(message)


def mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{context} must be an object")
    return value


def require(value: dict[str, Any], key: str, context: str) -> Any:
    if key not in value:
        fail(f"{context} missing {key!r}; present={sorted(value)}")
    return value[key]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return h.hexdigest()


def poisson(materials: dict[str, Any], path: Path) -> float:
    muscle = mapping(
        require(materials, "muscle", f"{path}.materials"),
        f"{path}.materials.muscle",
    )
    try:
        value = float(require(muscle, "nu", f"{path}.materials.muscle"))
    except (TypeError, ValueError):
        fail(f"{path}.materials.muscle.nu must be numeric")
    if not np.isfinite(value):
        fail(f"{path}.materials.muscle.nu must be finite")
    fat_value = require(materials, "fat", f"{path}.materials")
    if fat_value is not None:
        fat = mapping(fat_value, f"{path}.materials.fat")
        try:
            fat_nu = float(require(fat, "nu", f"{path}.materials.fat"))
        except (TypeError, ValueError):
            fail(f"{path}.materials.fat.nu must be numeric")
        if not np.isfinite(fat_nu) or fat_nu != value:
            fail(f"{path} has unequal/non-finite fat and muscle Poisson ratios")
    return value


def roots(value: str) -> list[Path]:
    output = [Path(item).resolve() for item in value.split(",") if item.strip()]
    if not output:
        fail("--input-roots is empty")
    if not all(item.is_dir() for item in output):
        fail(f"non-directory input root: {output}")
    return output


def nonnegative_int(value: Any, context: str) -> int:
    if isinstance(value, bool):
        fail(f"{context} must be an integer, got bool")
    try:
        result = int(value)
    except (TypeError, ValueError):
        fail(f"{context} must be an integer")
    if result < 0 or result != value:
        fail(f"{context} must be a nonnegative integer, got {value!r}")
    return result


def status_label(item: dict[str, Any]) -> str:
    return (
        f"gate {'OK' if item['practical_stationarity_gate'] else 'FAIL'}"
        f" · fwd={item['accepted_forward_failure_count']}"
        f" trial={item['trial_forward_failure_count']}"
    )


def load(path: Path) -> dict[str, Any]:
    summary = mapping(json.loads(path.read_text(encoding="utf-8")), str(path))
    case, geometry = (
        mapping(require(summary, "case", str(path)), f"{path}.case"),
        mapping(require(summary, "geometry", str(path)), f"{path}.geometry"),
    )
    materials, activation = (
        mapping(require(summary, "materials", str(path)), f"{path}.materials"),
        mapping(require(summary, "activation", str(path)), f"{path}.activation"),
    )
    # These are part of the factorial-study contract even though this renderer
    # does not plot them.  Rejecting a partial summary keeps rendered states
    # traceable to a complete inverse-physics result.
    for key in ("counts", "inverse", "metrics"):
        mapping(require(summary, key, str(path)), f"{path}.{key}")
    inverse = mapping(require(summary, "inverse", str(path)), f"{path}.inverse")
    convergence = mapping(
        require(inverse, "convergence", f"{path}.inverse"),
        f"{path}.inverse.convergence",
    )
    gate = require(
        convergence,
        "practical_stationarity_gate",
        f"{path}.inverse.convergence",
    )
    if not isinstance(gate, bool):
        fail(f"{path}.inverse.convergence.practical_stationarity_gate must be bool")
    failures = mapping(
        require(inverse, "failures", f"{path}.inverse"), f"{path}.inverse.failures"
    )
    failure_counts = {
        key: nonnegative_int(
            require(failures, key, f"{path}.inverse.failures"),
            f"{path}.inverse.failures.{key}",
        )
        for key in (
            "forward",
            "inverse",
            "adjoint",
            "nonfinite",
            "refinement_trial_forward",
        )
    }
    domain = require(geometry, "domain", f"{path}.geometry")
    if not isinstance(domain, list) or len(domain) != 2:
        fail(f"{path} is not a 2-D case")
    final = path.parent / "final.vtu"
    if not final.is_file():
        raise FileNotFoundError(final)
    grid = pv.read(final)
    if (
        not {"Displacement", "TargetDisplacement"} <= set(grid.point_data)
        or "MuscleMask" not in grid.cell_data
    ):
        fail(f"{final} missing final visualization arrays")
    cells = np.asarray(grid.cells).reshape(grid.n_cells, 4)
    if not np.all(cells[:, 0] == 3):
        fail(f"{final} is not triangles")
    points = np.asarray(grid.points[:, :2], float)
    displacement = np.asarray(grid.point_data["Displacement"][:, :2], float)
    target = np.asarray(grid.point_data["TargetDisplacement"][:, :2], float)
    if (
        not np.isfinite(points).all()
        or not np.isfinite(displacement).all()
        or not np.isfinite(target).all()
    ):
        fail(f"{final} has non-finite points/vectors")
    return {
        "name": str(require(case, "name", f"{path}.case")),
        "geometry": str(require(geometry, "geometry_id", f"{path}.geometry")),
        "muscle_extent": str(require(geometry, "muscle_extent_id", f"{path}.geometry")),
        "activation_sharing": str(
            require(activation, "sharing_id", f"{path}.activation")
        ),
        "poisson": f"{poisson(materials, path):.17g}",
        "triangles": cells[:, 1:],
        "muscle": np.asarray(grid.cell_data["MuscleMask"], bool),
        "reference": points,
        "deformed": points + displacement,
        "target": points + target,
        "practical_stationarity_gate": gate,
        "failure_counts": failure_counts,
        "failure_count": sum(failure_counts.values()),
        "accepted_forward_failure_count": failure_counts["forward"],
        "trial_forward_failure_count": failure_counts["refinement_trial_forward"],
        "final": final,
        "summary": path,
    }


def bounds(
    items: list[dict[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    points = np.concatenate(
        [np.concatenate((item["deformed"], item["target"])) for item in items]
    )
    lo, hi = points.min(axis=0), points.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    return (float(lo[0] - 0.03 * span[0]), float(hi[0] + 0.03 * span[0])), (
        float(lo[1] - 0.08 * span[1]),
        float(hi[1] + 0.08 * span[1]),
    )


def draw(
    axis: Any,
    item: dict[str, Any],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    title: str | None = None,
) -> None:
    faces = np.where(item["muscle"], MUSCLE, FAT)
    axis.add_collection(
        PolyCollection(
            item["deformed"][item["triangles"]],
            facecolors=faces,
            edgecolors="#29323A",
            linewidths=0.22,
            alpha=0.9,
            rasterized=True,
        )
    )
    top = np.isclose(item["reference"][:, 1], item["reference"][:, 1].max())
    order = np.argsort(item["reference"][top, 0])
    axis.plot(
        item["target"][top][order, 0],
        item["target"][top][order, 1],
        color=TARGET,
        linestyle=(0, (4, 2)),
        linewidth=1.3,
    )
    axis.plot(
        item["deformed"][top][order, 0],
        item["deformed"][top][order, 1],
        color=TOP,
        linewidth=1.1,
    )
    axis.set(xlim=xlim, ylim=ylim, aspect="equal")
    axis.grid(alpha=0.25)
    axis.set_axisbelow(True)
    if title:
        axis.set_title(title, fontsize=9, loc="left")


def save_item(
    item: dict[str, Any],
    out: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    dpi: int,
) -> None:
    fig, axis = plt.subplots(figsize=(6, 2.8))
    draw(axis, item, xlim, ylim, f"{item['name']}\n{status_label(item)}")
    axis.set(xlabel="x", ylabel="y")
    fig.suptitle(
        "Literal final state · shared physical axes · no vertical exaggeration",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)


def png_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        return {"width": image.width, "height": image.height, "mode": image.mode}


def pair_sheet(
    items: list[dict[str, Any]], factor: str, out: Path, dpi: int
) -> list[dict[str, Any]]:
    levels = sorted({item[factor] for item in items})
    other = [name for name in FACTORS if name != factor]
    pairs = []
    for combo in itertools.product(
        *(sorted({item[name] for item in items}) for name in other)
    ):
        match = [
            item
            for item in items
            if all(
                item[name] == value for name, value in zip(other, combo, strict=True)
            )
        ]
        if len(match) == 2 and {item[factor] for item in match} == set(levels):
            pairs.append(sorted(match, key=lambda item: item[factor]))
    if len(pairs) != 8:
        fail(f"{factor} needs 8 matched pairs, found {len(pairs)}")
    fig, axes = plt.subplots(4, 4, figsize=(15, 8), sharex=False, sharey=False)
    pair_axes = []
    for row, pair in enumerate(pairs):
        xlim, ylim = bounds(pair)
        pair_axes.append(
            {
                "case_names": [item["name"] for item in pair],
                "case_statuses": [
                    {
                        "name": item["name"],
                        "practical_stationarity_gate": item[
                            "practical_stationarity_gate"
                        ],
                        "failure_counts": item["failure_counts"],
                        "accepted_forward_failure_count": item[
                            "accepted_forward_failure_count"
                        ],
                        "trial_forward_failure_count": item[
                            "trial_forward_failure_count"
                        ],
                    }
                    for item in pair
                ],
                "shared_axes": {"x": xlim, "y": ylim},
                "equal_aspect": True,
                "no_vertical_exaggeration": True,
            }
        )
        for column, item in enumerate(pair):
            axis = axes[row // 2, (row % 2) * 2 + column]
            draw(
                axis,
                item,
                xlim,
                ylim,
                f"{factor}={item[factor]}\n{item['name']}\n{status_label(item)}",
            )
    fig.suptitle(f"Matched literal-final pairs: {factor}", fontsize=15)
    fig.legend(
        handles=[
            Patch(facecolor=FAT, label="fat"),
            Patch(facecolor=MUSCLE, label="muscle"),
            Line2D([], [], color=TOP, label="final top"),
            Line2D([], [], color=TARGET, linestyle=(0, (4, 2)), label="target"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.10, top=0.90, hspace=0.50, wspace=0.15)
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return pair_axes


def all_final_shapes_sheet(
    items: list[dict[str, Any]],
    geometries: list[str],
    out: Path,
    dpi: int,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Render all 16 cases without forcing unlike physical lengths to share axes."""
    fig = plt.figure(figsize=(16, 10), layout="constrained")
    groups = fig.subfigures(2, 1)
    axes_by_geometry: dict[str, dict[str, tuple[float, float]]] = {}
    for subfig, geometry in zip(groups, geometries, strict=True):
        group = sorted(
            (item for item in items if item["geometry"] == geometry),
            key=lambda item: item["name"],
        )
        if len(group) != 8:
            fail(f"{geometry} needs 8 cases, found {len(group)}")
        xlim, ylim = bounds(group)
        axes_by_geometry[geometry] = {"x": xlim, "y": ylim}
        axes = subfig.subplots(2, 4, sharex=True, sharey=True)
        for axis, item in zip(axes.flat, group, strict=True):
            draw(axis, item, xlim, ylim, f"{item['name']}\n{status_label(item)}")
            axis.set(xlabel="x", ylabel="y")
        subfig.suptitle(
            f"geometry={geometry} · shared physical axes within this geometry",
            fontsize=12,
        )
    fig.suptitle(
        "All 16 literal final states · equal aspect · no vertical exaggeration",
        fontsize=15,
    )
    fig.legend(
        handles=[
            Patch(facecolor=FAT, label="fat"),
            Patch(facecolor=MUSCLE, label="muscle"),
            Line2D([], [], color=TOP, label="final top"),
            Line2D([], [], color=TARGET, linestyle=(0, (4, 2)), label="target"),
        ],
        loc="lower center",
        ncol=4,
        frameon=False,
    )
    fig.savefig(out, dpi=dpi, facecolor="white")
    plt.close(fig)
    return axes_by_geometry


def main(cfg: Config) -> None:
    if cfg.output_dir.exists() and any(cfg.output_dir.iterdir()):
        raise FileExistsError(cfg.output_dir)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    final_dir, sheets, contacts = (
        cfg.output_dir / "finals",
        cfg.output_dir / "factor-pairs",
        cfg.output_dir / "contact-sheets",
    )
    final_dir.mkdir()
    sheets.mkdir()
    contacts.mkdir()
    paths = sorted(
        {
            path
            for root in roots(cfg.input_roots)
            for path in root.rglob("summary.json")
            if (path.parent / "final.vtu").is_file()
        }
    )
    items = [load(path) for path in paths]
    if len(items) != 16 or len({item["name"] for item in items}) != 16:
        fail(f"requires 16 unique 2-D cases, found {len(items)}")
    levels = {factor: sorted({item[factor] for item in items}) for factor in FACTORS}
    if any(len(value) != 2 for value in levels.values()) or {
        tuple(item[factor] for factor in FACTORS) for item in items
    } != set(itertools.product(*(levels[factor] for factor in FACTORS))):
        fail("2-D factorial matrix is not an exact 2^4")
    receipt: dict[str, Any] = {
        "case_count": len(items),
        "status_label_semantics": (
            "gate is practical stationarity; fwd is accepted forward-solve "
            "failures; trial is rejected refinement-trial forward-solve failures"
        ),
        "cases": [],
        "factor_pair_sheet_count": len(FACTORS),
        "factor_pair_sheets": [],
        "practical_stationarity_gate_pass_count": sum(
            item["practical_stationarity_gate"] for item in items
        ),
        "accepted_forward_failure_case_count": sum(
            item["accepted_forward_failure_count"] > 0 for item in items
        ),
        "refinement_trial_forward_failure_case_count": sum(
            item["trial_forward_failure_count"] > 0 for item in items
        ),
    }
    for geometry in levels["geometry"]:
        group = [item for item in items if item["geometry"] == geometry]
        xlim, ylim = bounds(group)
        for item in group:
            output = final_dir / f"{item['name']}.png"
            save_item(item, output, xlim, ylim, cfg.dpi)
            receipt["cases"].append(
                {
                    "name": item["name"],
                    "geometry": geometry,
                    "final_vtu": {
                        "path": str(item["final"]),
                        "sha256": sha(item["final"]),
                    },
                    "summary": {
                        "path": str(item["summary"]),
                        "sha256": sha(item["summary"]),
                    },
                    "png": {
                        "path": str(output),
                        "sha256": sha(output),
                        **png_metadata(output),
                    },
                    "shared_axes": {"x": xlim, "y": ylim},
                    "practical_stationarity_gate": item["practical_stationarity_gate"],
                    "failure_counts": item["failure_counts"],
                    "failure_count": item["failure_count"],
                    "accepted_forward_failure_count": item[
                        "accepted_forward_failure_count"
                    ],
                    "trial_forward_failure_count": item["trial_forward_failure_count"],
                    "status_label": status_label(item),
                    "no_vertical_exaggeration": True,
                }
            )
    for factor in FACTORS:
        output = sheets / f"{factor}-pairs.png"
        pair_axes = pair_sheet(items, factor, output, cfg.dpi)
        receipt["factor_pair_sheets"].append(
            {
                "factor": factor,
                "path": str(output),
                "sha256": sha(output),
                **png_metadata(output),
                "no_vertical_exaggeration": True,
                "axis_sharing": "physical axes are shared within each matched pair",
                "matched_pair_axes": pair_axes,
            }
        )
    contact_output = contacts / "all-16-final-shapes.png"
    contact_axes = all_final_shapes_sheet(
        items, levels["geometry"], contact_output, cfg.dpi
    )
    receipt["all_final_shapes_contact_sheet"] = {
        "path": str(contact_output),
        "sha256": sha(contact_output),
        **png_metadata(contact_output),
        "geometry_axis_groups": [
            {
                "geometry": geometry,
                "case_count": 8,
                "practical_stationarity_gate_pass_count": sum(
                    item["practical_stationarity_gate"]
                    for item in items
                    if item["geometry"] == geometry
                ),
                "failure_case_count": sum(
                    item["failure_count"] > 0
                    for item in items
                    if item["geometry"] == geometry
                ),
                "accepted_forward_failure_case_count": sum(
                    item["accepted_forward_failure_count"] > 0
                    for item in items
                    if item["geometry"] == geometry
                ),
                "refinement_trial_forward_failure_case_count": sum(
                    item["trial_forward_failure_count"] > 0
                    for item in items
                    if item["geometry"] == geometry
                ),
                "shared_axes": contact_axes[geometry],
                "equal_aspect": True,
                "no_vertical_exaggeration": True,
            }
            for geometry in levels["geometry"]
        ],
        "semantics": (
            "all 16 literal final states; each geometry has its own shared "
            "physical axes so short and long specimens remain legible"
        ),
    }
    (cfg.output_dir / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    cherries.log_metrics(
        {
            "finals/cases": len(items),
            "finals/pair_sheets": len(FACTORS),
            "finals/contact_sheets": 1,
        }
    )


if __name__ == "__main__":
    cherries.main(main)
