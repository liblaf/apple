# ruff: noqa: PLR0915, PT018
"""CPU-only endpoint audit for normalized L2, gradient, and mixed face losses."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import pyvista as pv
from liblaf.cherries import core, plugins, profiles

from liblaf import cherries

GROUP = Path(__file__).resolve().parents[1]


class ProfileVerification(profiles.Profile):
    def init(self) -> core.Run:
        run = core.run
        run.plugins.register(
            plugins.Comet(run=run, disabled=os.environ.get("DEBUG") == "1")
        )
        run.plugins.register(plugins.Git(run=run, commit=False))
        run.plugins.register(plugins.Logging(run=run))
        run.plugins.register(plugins.Local(run=run))
        return run


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    comparison_dir: Path = GROUP / "data/20-comparison"
    preparation_dir: Path = GROUP / "data/10-preparation"
    output: Path = GROUP / "data/25-verification/checks.json"


def _json(path: Path) -> dict:
    value = json.loads(path.read_text())
    assert isinstance(value, dict)
    return value


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _trace(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]
    assert rows and [row["step"] for row in rows] == list(range(len(rows)))
    return rows


def _state(path: Path) -> dict[str, np.ndarray | int | float]:
    with np.load(path, allow_pickle=False) as saved:
        required = {
            "s",
            "n",
            "q",
            "u",
            "step",
            "solver_valid",
            "physical_volume_energy",
            "smoothness_coefficient",
        }
        assert not required - set(saved.files)
        assert bool(saved["solver_valid"]) and bool(saved["physical_volume_energy"])
        return {
            "s": np.asarray(saved["s"], dtype=float),
            "n": np.asarray(saved["n"], dtype=float),
            "q": np.asarray(saved["q"], dtype=float),
            "u": np.asarray(saved["u"], dtype=float),
            "step": int(saved["step"]),
            "coefficient": float(saved["smoothness_coefficient"]),
        }


def _pack(s: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = n / np.linalg.norm(n, axis=1, keepdims=True)
    delta = s[:, None, None] * n[:, :, None] * n[:, None, :]
    return np.column_stack(
        (
            delta[:, 0, 0],
            delta[:, 1, 1],
            delta[:, 2, 2],
            delta[:, 0, 1],
            delta[:, 1, 2],
            delta[:, 0, 2],
        )
    )


def _regularizer(
    q: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    weight: np.ndarray,
    volume: np.ndarray,
    ell: float,
) -> float:
    d = q[i] - q[j]
    frobenius2 = np.sum(d[:, :3] ** 2, axis=1) + 2 * np.sum(d[:, 3:] ** 2, axis=1)
    return float(ell**2 * np.sum(weight * frobenius2) / volume.sum())


def _close(actual: float, reported: float) -> float:
    error = abs(actual - reported)
    assert error <= 1e-9 * max(1, abs(actual), abs(reported)), (actual, reported, error)
    return error


def _surface_losses(
    u: np.ndarray,
    target: np.ndarray,
    rest: np.ndarray,
    ids: np.ndarray,
    tri: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    residual = u[ids] - target
    l2 = float(1e6 / 3 * np.sum(weights[:, None] * residual**2))
    p = rest[ids][tri]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    norm2 = np.sum(normal**2, axis=1)
    assert np.all(norm2 > 0)
    grad = (
        np.stack(
            (
                np.cross(normal, p[:, 2] - p[:, 1]),
                np.cross(normal, p[:, 0] - p[:, 2]),
                np.cross(normal, p[:, 1] - p[:, 0]),
            ),
            axis=1,
        )
        / norm2[:, None, None]
    )
    field = np.einsum("tvi,tvj->tij", residual[tri], grad)
    area = 0.5 * np.sqrt(norm2)
    lg = float(np.sum(area * np.sum(field**2, axis=(1, 2))) / area.sum())
    return l2, lg


def _data_objective(branch: dict, l2: float, gradient: float, scales: dict) -> float:
    L20, Lg0, K = (float(scales[key]) for key in ("L20", "Lg0", "K"))
    if branch["kind"] == "l2":
        return K * l2 / L20
    if branch["kind"] == "gradient":
        return K * gradient / Lg0
    beta = float(branch["beta"])
    return K * (l2 / L20 + beta * gradient / Lg0) / (1 + beta)


def main(cfg: Config) -> None:
    comparison = cherries.input(cfg.comparison_dir.resolve())
    preparation = cherries.input(cfg.preparation_dir.resolve())
    output = cherries.output(cfg.output.resolve(), mkdir=True)
    protocol, prepared = (
        _json(comparison / "protocol.json"),
        _json(preparation / "protocol.json"),
    )
    scales = protocol["loss_normalization"]
    loss_selection_path = GROUP / "data/09-loss-selection/selection.json"
    smooth_selection_path = GROUP / "data/15-selection/selection.json"
    loss_selection = _json(loss_selection_path)
    smooth_selection = _json(smooth_selection_path)
    assert loss_selection["status"] == smooth_selection["status"] == "selected"
    assert loss_selection["selected_beta"] == protocol["selected_beta"]
    assert prepared["selected_beta"] == protocol["selected_beta"]
    assert (
        smooth_selection["selected_coefficient"]
        == protocol["branches"]["mixed-on"]["coefficient"]
    )
    assert protocol["branches"]["mixed-off"]["coefficient"] == 0
    assert set(scales) >= {"L20", "Lg0", "K", "formula"}
    assert float(scales["K"]) == float(scales["L20"])
    assert protocol["fixture"] == prepared["fixture"]
    assert protocol["loss_normalization"] == prepared["loss_normalization"]
    assert protocol["initial_axes"] == prepared["initial_axes"]
    assert _json(preparation / "gradient-validation.json")["status"] == "passed"
    assert (
        _json(GROUP / "data/06-validation/gradient-validation.json")["status"]
        == "passed"
    )
    cpu_gate = _json(GROUP / "data/05-loss/checks.json")
    assert cpu_gate["passed"]
    for filename, digest in cpu_gate["source_sha256"].items():
        assert _digest(GROUP / "src" / filename) == digest
    for key, source in prepared["sources"].items():
        assert protocol["sources"][key]["sha256"] == source["sha256"]
    for source in protocol["sources"].values():
        assert _digest(Path(source["path"])) == source["sha256"]
        assert _digest(Path(source["snapshot"])) == source["sha256"]
    for fixture_record in protocol["fixture"].values():
        assert _digest(Path(fixture_record["path"])) == fixture_record["sha256"]
    inherited = protocol["initial_axes"]
    assert _digest(Path(inherited["source"])) == inherited["sha256"]
    assert _digest(comparison / "initialization.npz") == inherited["sha256"]
    initial_axes = np.load(comparison / "initialization.npz")["axes"]
    with np.load(comparison / "mesh.npz", allow_pickle=False) as mesh:
        rest_saved, initial_u = (
            np.asarray(mesh["rest_points"]),
            np.asarray(mesh["initial_u"]),
        )
        i, j, w = (
            np.asarray(mesh[key], dtype=np.int64 if key != "edge_weight" else float)
            for key in ("edge_i", "edge_j", "edge_weight")
        )
        volume, active = (
            np.asarray(mesh["active_volumes"]),
            np.asarray(mesh["active_ids"], dtype=np.int64),
        )
        skin_ids = np.asarray(mesh["skin_ids"])
        triangles = np.asarray(mesh["triangles"])
        target = np.asarray(mesh["target_displacement_skin"])
        skin_weights = np.asarray(mesh["skin_vertex_weights"])
    neutral_l2, neutral_lg = _surface_losses(
        initial_u, target, rest_saved, skin_ids, triangles, skin_weights
    )
    normalization_errors = {
        "L20": _close(neutral_l2, scales["L20"]),
        "Lg0": _close(neutral_lg, scales["Lg0"]),
    }
    fixture = pv.read(protocol["fixture"]["volume.vtu"]["path"])
    rest = np.asarray(fixture.points)
    assert np.array_equal(rest, rest_saved)
    tets = np.asarray(fixture.cells).reshape(-1, 5)[:, 1:]
    V0 = (
        np.linalg.det(np.transpose(rest[tets[:, 1:]] - rest[tets[:, :1]], (0, 2, 1)))
        / 6
    )
    assert np.all(V0 > 0)
    branches, starts, traces = {}, {}, {}
    assert protocol["selected_beta"] == protocol["branches"]["mixed-off"]["beta"]
    assert protocol["selected_beta"] == protocol["branches"]["mixed-on"]["beta"]
    for name in ("mixed-off", "mixed-on"):
        folder = comparison / name
        summary, trace = _json(folder / "summary.json"), _trace(folder / "trace.csv")
        start, end = _state(folder / "step-0000.npz"), _state(folder / "last.npz")
        assert np.array_equal(start["n"], initial_axes)
        assert (
            int(start["step"]) == 0
            and np.max(np.abs(start["q"])) == 0
            and np.max(np.abs(start["u"] - initial_u)) < 1e-14
        )
        assert int(end["step"]) == int(summary["last_step"]) == int(trace[-1]["step"])
        assert summary["status"] in {
            "completed_budget_not_convergence_certified",
            "stationarity_and_plateau_thresholds_met",
        }
        s, n, q, u = (np.asarray(end[key]) for key in ("s", "n", "q", "u"))
        assert s.min() >= 0 and np.max(np.abs(np.linalg.norm(n, axis=1) - 1)) < 1e-12
        q_error = float(np.max(np.abs(q - _pack(s, n))))
        assert q_error < 1e-12
        C = np.zeros((len(q), 3, 3))
        C[:, 0, 0], C[:, 1, 1], C[:, 2, 2] = q[:, :3].T
        C[:, 0, 1] = C[:, 1, 0] = q[:, 3]
        C[:, 1, 2] = C[:, 2, 1] = q[:, 4]
        C[:, 0, 2] = C[:, 2, 0] = q[:, 5]
        ev = np.linalg.eigvalsh(C)
        assert ev.min() >= -1e-12 and np.max(np.abs(ev[:, :2])) < 1e-10
        R = _regularizer(
            q, i, j, w, volume, float(protocol["config"]["smooth_length_m"])
        )
        branch = protocol["branches"][name]
        row, coefficient = summary["last_metrics"], float(branch["coefficient"])
        assert coefficient == float(end["coefficient"]) == float(summary["coefficient"])
        l2, lg = _surface_losses(u, target, rest, skin_ids, triangles, skin_weights)
        data = _data_objective(
            branch,
            l2,
            lg,
            scales,
        )
        objective = data + coefficient * R
        x = rest + u
        J = (
            np.linalg.det(np.transpose(x[tets[:, 1:]] - x[tets[:, :1]], (0, 2, 1)))
            / 6
            / V0
        )
        errors = {
            "position_loss": _close(l2, float(row["position_loss_component_mm2"])),
            "gradient_loss": _close(lg, float(row["surface_gradient_loss"])),
            "data": _close(data, float(row["data_objective"])),
            "R": _close(R, float(row["activation_smoothness"])),
            "objective": _close(objective, float(row["objective"])),
            "trace": _close(objective, trace[-1]["objective"]),
            "detF_min": _close(float(J.min()), float(row["detF_min"])),
            "detF_max": _close(float(J.max()), float(row["detF_max"])),
        }
        assert int(np.count_nonzero(J <= 0)) == int(row["inverted_all_cells"])
        assert int(np.count_nonzero(J[active] <= 0)) == int(
            row["inverted_active_cells"]
        )
        receipts = [
            json.loads(line)
            for line in (folder / "solver-receipts.jsonl").read_text().splitlines()
        ]
        assert len(receipts) == int(end["step"]) + 1 and all(
            r["forward"]["success"] and r["adjoint"]["success"] for r in receipts
        )
        branches[name] = {
            "step": int(end["step"]),
            "q_reconstruction_inf": q_error,
            "rank_one_error": float(np.max(np.abs(ev[:, :2]))),
            "regularizer": R,
            "position_loss": l2,
            "gradient_loss": lg,
            "detF_min": float(J.min()),
            "detF_max": float(J.max()),
            "inverted_all": int(np.count_nonzero(J <= 0)),
            "inverted_active": int(np.count_nonzero(J[active] <= 0)),
            "errors": errors,
            "receipts": len(receipts),
        }
        starts[name], traces[name] = start, trace
    assert all(
        np.array_equal(starts["mixed-off"][key], starts["mixed-on"][key])
        for key in ("q", "u", "n")
    )
    matched = min(map(len, traces.values()))
    assert (
        max(
            abs(
                traces["mixed-off"][k]["learning_rate"]
                - traces[name][k]["learning_rate"]
            )
            for name in traces
            for k in range(matched)
        )
        == 0
    )
    output.write_text(
        json.dumps(
            {
                "scales": scales,
                "neutral_loss_errors": normalization_errors,
                "source_count": len(protocol["sources"]),
                "verifier_sha256": _digest(Path(__file__)),
                "selection_receipts": {
                    "loss": {
                        "path": str(loss_selection_path),
                        "sha256": _digest(loss_selection_path),
                    },
                    "smoothness": {
                        "path": str(smooth_selection_path),
                        "sha256": _digest(smooth_selection_path),
                    },
                },
                "branches": branches,
                "matched_lr_steps": matched,
                "passed": True,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileVerification)
