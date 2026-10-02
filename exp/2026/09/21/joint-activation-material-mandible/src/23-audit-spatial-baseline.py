"""Compare constant and component-aware baseline force spaces at exact F=I."""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pydantic_settings as ps
import scipy.linalg
import scipy.sparse as sp
import torch
from joint_common import GROUP, ProfileJoint, archive_sources, sha256, write_json
from joint_data import PreparedInputs
from joint_equilibrium import configure_cuda
from joint_fields import BULK_TISSUES, research_informed_material_config
from joint_physics import JointPhysics
from scipy.sparse.csgraph import connected_components

from liblaf import cherries

LOG = logging.getLogger(__name__)
COMPLETED = False


class Config(cherries.BaseConfig):
    model_config = ps.SettingsConfigDict(cli_parse_args=True, cli_kebab_case=True)
    prepared_dir: Path = GROUP / "data/prepared"
    contact_spec: Path = GROUP / "data/contact/config.json"
    output_dir: Path
    rho_m: float = 0.005
    smooth_length_m: float = 0.005
    skin_target_n_per_m: float = 80.6
    smooth_weights: tuple[float, ...] = (0.0, 1.0, 100.0, 10000.0)


def matrices(value: np.ndarray) -> np.ndarray:
    value = value.reshape(-1, 6)
    result = np.zeros((len(value), 3, 3))
    result[:, 0, 0], result[:, 1, 1], result[:, 2, 2] = value[:, :3].T
    for coordinate, i, j in ((3, 0, 1), (4, 1, 2), (5, 0, 2)):
        result[:, i, j] = result[:, j, i] = value[:, coordinate] / math.sqrt(2)
    return result


def project(value: np.ndarray) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eigh(matrices(value))
    tensors = (vectors * np.clip(eigenvalues, -0.9, 10)[:, None, :]) @ vectors.swapaxes(
        -1, -2
    )
    return np.column_stack(
        (
            tensors[:, 0, 0],
            tensors[:, 1, 1],
            tensors[:, 2, 2],
            math.sqrt(2) * tensors[:, 0, 1],
            math.sqrt(2) * tensors[:, 1, 2],
            math.sqrt(2) * tensors[:, 0, 2],
        )
    ).ravel()


def make_basis(  # noqa: PLR0915
    points: np.ndarray,
    tets: np.ndarray,
    volumes: np.ndarray,
    fraction: np.ndarray,
    *,
    secondary: bool,
    rho: float,
    ell: float,
):
    ids = np.flatnonzero(fraction > 1.0e-6)
    centers = points[tets[ids]].mean(axis=1)
    mass = volumes[ids] * fraction[ids]
    pattern = np.asarray(((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)))
    faces = np.sort(tets[ids][:, pattern].reshape(-1, 3), axis=1)
    owner = np.repeat(np.arange(len(ids), dtype=np.int32), 4)
    order = np.lexsort(faces.T[::-1])
    faces, owner = faces[order], owner[order]
    paired = np.flatnonzero(np.all(faces[1:] == faces[:-1], axis=1))
    assert not np.any(np.diff(paired) == 1)
    i, j = owner[paired], owner[paired + 1]
    xyz = points[faces[paired]]
    area = (
        np.linalg.norm(np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0]), axis=1)
        / 2
    )
    distance = np.linalg.norm(centers[i] - centers[j], axis=1)
    f = fraction[ids]
    conductance = area / distance * 2 * f[i] * f[j] / (f[i] + f[j])
    assert np.all(np.isfinite(conductance))
    assert np.all(conductance > 0)
    adjacency = sp.coo_matrix(
        (np.ones(2 * len(i)), (np.r_[i, j], np.r_[j, i])), shape=(len(ids), len(ids))
    ).tocsr()
    count, labels = connected_components(adjacency, directed=False)
    component_mass = np.bincount(labels, weights=mass, minlength=count)
    components = np.argsort(-component_mass, kind="stable")
    main_ids = np.flatnonzero(labels == components[0])
    x = centers[main_ids]
    w = mass[main_ids] / mass[main_ids].sum()
    center = np.sum(w[:, None] * x, axis=0)
    covariance = (x - center).T @ (w[:, None] * (x - center))
    eigenvalues, vectors = np.linalg.eigh(covariance)
    assert eigenvalues.min() > 0
    y = (x - center) @ (vectors / np.sqrt(eigenvalues))
    chosen = [int(np.argmin(np.sum(y * y, axis=1)))]
    chosen.append(int(np.argmax(np.sum((y - y[chosen[0]]) ** 2, axis=1))))
    line = y[chosen[1]] - y[chosen[0]]
    offset = y - y[chosen[0]]
    chosen.append(int(np.argmax(np.linalg.norm(np.cross(offset, line), axis=1))))
    normal = np.cross(line, y[chosen[2]] - y[chosen[0]])
    chosen.append(int(np.argmax(np.abs(offset @ normal))))
    assert len(set(chosen)) == 4
    assert abs(np.linalg.det(y[chosen[1:]] - y[chosen[0]])) > 1.0e-8
    local_anchors = main_ids[chosen].tolist()
    anchor_components = [int(components[0])] * 4
    if secondary:
        members = np.flatnonzero(labels == components[1])
        centroid = np.average(centers[members], axis=0, weights=mass[members])
        local_anchors.append(
            int(members[np.argmin(np.sum((centers[members] - centroid) ** 2, axis=1))])
        )
        anchor_components.append(int(components[1]))
    positions = centers[local_anchors]
    phi = np.zeros((len(ids), len(local_anchors)))
    for component in range(count):
        members = np.flatnonzero(labels == component)
        anchors = np.flatnonzero(np.asarray(anchor_components) == component)
        if len(anchors):
            d2 = np.sum(
                (centers[members, None] - positions[anchors][None]) ** 2, axis=2
            )
            weights = 1 / (d2 + rho**2)
            weights /= weights.sum(axis=1, keepdims=True)
            phi[np.ix_(members, anchors)] = weights
        else:
            centroid = np.average(centers[members], axis=0, weights=mass[members])
            nearest = int(np.argmin(np.sum((positions - centroid) ** 2, axis=1)))
            phi[members, nearest] = 1
    assert np.isfinite(phi).all()
    assert phi.min() >= 0
    assert np.max(np.abs(phi.sum(axis=1) - 1)) < 5.0e-16
    delta = phi[i] - phi[j]
    smooth = ell**2 / mass.sum() * delta.T @ (conductance[:, None] * delta)
    magnitude = phi.T @ ((mass / mass.sum())[:, None] * phi)
    assert np.max(np.abs(smooth @ np.ones(phi.shape[1]))) < 1.0e-12
    probe = np.random.default_rng(20260921).normal(size=(phi.shape[1], 6))
    field = phi @ probe
    direct_smooth = (
        ell**2 / mass.sum() * np.sum(conductance[:, None] * (field[i] - field[j]) ** 2)
    )
    direct_magnitude = np.sum((mass / mass.sum())[:, None] * field**2)
    assert np.isclose(direct_smooth, np.sum(probe * (smooth @ probe)), rtol=1.0e-12)
    assert np.isclose(
        direct_magnitude, np.sum(probe * (magnitude @ probe)), rtol=1.0e-12
    )
    receipt = {
        "positive_cells": len(ids),
        "edges": len(i),
        "components": int(count),
        "anchor_cell_ids": ids[local_anchors].tolist(),
        "anchor_components": anchor_components,
        "largest_component_volume_shares": (
            component_mass[components[:2]] / mass.sum()
        ).tolist(),
        "unanchored_component_policy": "entire component inherits nearest anchor to its volume centroid",
        "partition_sum_maximum_error": float(np.max(np.abs(phi.sum(axis=1) - 1))),
        "graph_energy_relative_error": float(
            abs(direct_smooth - np.sum(probe * (smooth @ probe))) / direct_smooth
        ),
        "magnitude_relative_error": float(
            abs(direct_magnitude - np.sum(probe * (magnitude @ probe)))
            / direct_magnitude
        ),
    }
    return ids, phi, smooth, magnitude, receipt


def fit(matrix: np.ndarray, target: np.ndarray):
    scale = np.linalg.norm(matrix, axis=0)
    value, _, rank, singular = np.linalg.lstsq(matrix / scale, target, rcond=None)
    return value / scale, {
        "rank": int(rank),
        "scaled_condition": float(singular[0] / singular[-1]),
    }


def bounded_quadratic(
    matrix: np.ndarray,
    target: np.ndarray,
    smooth: np.ndarray,
    magnitude: np.ndarray,
    weight: float,
):
    norm = np.linalg.norm(target)
    a, b = matrix / norm, target / norm
    gram, rhs = a.T @ a + weight * smooth, a.T @ b
    unconstrained = np.linalg.solve(gram, rhs)

    def slack(x: np.ndarray):
        values = np.linalg.eigvalsh(matrices(x))
        return np.r_[(values + 0.9).ravel(), (10 - values).ravel()]

    def objective(x: np.ndarray):
        return float(0.5 * x @ gram @ x - rhs @ x + 0.5)

    if slack(unconstrained).min() >= 0:
        value, iterations = unconstrained, 0
    else:
        lipschitz = float(np.linalg.eigvalsh(gram).max())
        value = project(unconstrained)
        extrapolated = value.copy()
        momentum = 1.0
        for iteration in range(1, 200001):
            iterations = iteration
            candidate = project(extrapolated - (gram @ extrapolated - rhs) / lipschitz)
            if objective(candidate) > objective(value) + 1.0e-14:
                extrapolated = value.copy()
                momentum = 1.0
                candidate = project(value - (gram @ value - rhs) / lipschitz)
            residual = candidate - project(candidate - (gram @ candidate - rhs))
            if np.max(np.abs(residual)) <= 1.0e-8:
                value = candidate
                break
            next_momentum = (1 + math.sqrt(1 + 4 * momentum**2)) / 2
            extrapolated = candidate + (momentum - 1) / next_momentum * (
                candidate - value
            )
            value, momentum = candidate, next_momentum
        else:
            message = "Spectral quadratic did not meet its KKT tolerance"
            raise RuntimeError(message)
    gradient = gram @ value - rhs
    stationarity = float(np.max(np.abs(value - project(value - gradient))))
    assert slack(value).min() >= -1.0e-8
    assert stationarity < 1.1e-8, stationarity
    return value, {
        "smoothness_weight": weight,
        "weight_status": "fixed force-space sensitivity, not final inverse calibration",
        "objective": objective(value),
        "optimizer_iterations": iterations,
        "projected_gradient_inf": stationarity,
        "minimum_spectral_slack": float(slack(value).min()),
        "smoothness": float(value @ smooth @ value),
        "magnitude": float(value @ magnitude @ value),
    }


def main(cfg: Config):  # noqa: C901, PLR0915 - fixed scientific audit sequence.
    global COMPLETED  # noqa: PLW0603
    started = time.perf_counter()
    output = cfg.output_dir
    output.mkdir(parents=True, exist_ok=False)
    archive_sources(output)
    prepared = PreparedInputs.load(
        cfg.prepared_dir / "inputs.npz", cfg.prepared_dir / "manifest.json"
    )
    spec = research_informed_material_config()
    assert cfg.skin_target_n_per_m == 80.6
    configure_cuda()
    physics = JointPhysics(
        prepared.volume_path,
        prepared.skin_path,
        prepared.arrays,
        bulk_young_mpa={t: spec["materials"][t]["young_mpa"] for t in BULK_TISSUES},
        bulk_nu={t: spec["materials"][t]["poisson"] for t in BULK_TISSUES},
        skin_young_mpa=spec["materials"]["skin"]["reference_map"]["young_mpa"],
        skin_nu=spec["materials"]["skin"]["poisson"],
        thickness_m=spec["materials"]["skin"]["thickness_m"],
        contact_config=json.loads(cfg.contact_spec.read_text()),
    )
    bases, arrays, basis_receipts = {}, {}, {}
    for tissue in BULK_TISSUES:
        LOG.info("Building component-aware %s basis", tissue)
        basis = make_basis(
            physics.points,
            physics.tets,
            physics.volumes,
            np.asarray(physics.mesh.cell_data[tissue.title() + "Fraction"]),
            secondary=tissue == "muscle",
            rho=cfg.rho_m,
            ell=cfg.smooth_length_m,
        )
        bases[tissue] = basis
        ids, phi, smooth, magnitude, basis_receipts[tissue] = basis
        arrays.update(
            {
                f"{tissue}_cell_ids": ids,
                f"{tissue}_phi": phi,
                f"{tissue}_G": smooth,
                f"{tissue}_M": magnitude,
            }
        )
    model, state = physics.runtime.forward.model, physics.runtime.forward.state
    zero_bulk = torch.zeros((3, 3, 3))
    zero_skin = torch.zeros((2, 2))
    base = physics.materials(zero_bulk, zero_skin, torch.ones(()))
    model.dof_map.fixed_values = physics.boundary(torch.zeros(6))
    state.u = torch.zeros_like(physics.points_t)
    state.collision = model.collision.state_at(state.u)

    def gradient(
        tissue: str | None = None,
        stress: np.ndarray | None = None,
        skin: float = 0.0,
    ):
        values = {name: dict(fields) for name, fields in base.items()}
        if tissue is not None:
            values[tissue]["active_stress"] = torch.as_tensor(
                stress, dtype=torch.float64
            )
        values["skin"]["baseline_stress"] = (
            (torch.eye(2) * skin * 1.0e-6)
            .expand(physics.skin.n_cells, 2, 2)
            .contiguous()
        )
        model.set_materials(values)
        value = physics.runtime.forward.problem.grad(state)
        assert torch.isfinite(value).all()
        return value.detach().cpu().numpy()

    g0 = gradient()
    target = -(gradient(skin=cfg.skin_target_n_per_m) - g0)
    columns, embedding_errors = [], []
    coordinate_tensors = matrices(np.eye(6))
    for tissue in BULK_TISSUES:
        ids, phi, _, _, _ = bases[tissue]
        mu = spec["materials"][tissue]["mu_mpa"]
        tissue_columns = []
        for anchor in range(phi.shape[1]):
            for coordinate in range(6):
                stress = np.zeros((physics.mesh.n_cells, 3, 3))
                stress[ids] = (
                    mu * phi[:, anchor, None, None] * coordinate_tensors[coordinate]
                )
                tissue_columns.append(gradient(tissue, stress) - g0)
            LOG.info("Assembled %s anchor %d/%d", tissue, anchor + 1, phi.shape[1])
        tissue_matrix = np.column_stack(tissue_columns)
        for coordinate in range(6):
            stress = np.broadcast_to(
                mu * coordinate_tensors[coordinate], (physics.mesh.n_cells, 3, 3)
            ).copy()
            constant = gradient(tissue, stress) - g0
            embedded = tissue_matrix[:, coordinate::6].sum(axis=1)
            error = np.linalg.norm(embedded - constant) / np.linalg.norm(constant)
            assert error < 1.0e-10, (tissue, coordinate, error)
            embedding_errors.append(float(error))
        columns.append(tissue_matrix)
    matrix = np.column_stack(columns)
    nodal_volume = np.zeros(len(physics.points))
    for local in range(4):
        np.add.at(nodal_volume, physics.tets[:, local], physics.volumes / 4)
    free = model.dof_map.free_indices.detach().cpu().numpy()
    weights = 1 / np.sqrt(nodal_volume[free // 3])
    weights /= np.median(weights)
    weighted_matrix, weighted_target = weights[:, None] * matrix, weights * target
    smooth = scipy.linalg.block_diag(
        *[np.kron(bases[t][2], np.eye(6)) / 3 for t in BULK_TISSUES]
    )
    magnitude = scipy.linalg.block_diag(
        *[np.kron(bases[t][3], np.eye(6)) / 3 for t in BULK_TISSUES]
    )
    constant_matrix = np.column_stack(
        [
            block[:, coordinate::6].sum(axis=1)
            for block in columns
            for coordinate in range(6)
        ]
    )

    def metrics(value: np.ndarray, mat: np.ndarray = matrix):
        residual = mat @ value - target
        return {
            "raw_relative_residual": float(
                np.linalg.norm(residual) / np.linalg.norm(target)
            ),
            "weighted_relative_residual": float(
                np.linalg.norm(weights * residual) / np.linalg.norm(weighted_target)
            ),
        }

    results = {}
    for name, mat, response_mat in (
        ("constant18", constant_matrix, constant_matrix),
        ("spatial78", matrix, matrix),
    ):
        for kind, row_weights in (
            ("raw", np.ones(len(weights))),
            ("dual_volume", weights),
        ):
            value, diagnostic = fit(row_weights[:, None] * mat, row_weights * target)
            results[f"{name}_{kind}_unconstrained"] = {
                **diagnostic,
                **metrics(value, response_mat),
                "coefficients": value.tolist(),
            }
    for weight in cfg.smooth_weights:
        value, diagnostic = bounded_quadratic(
            weighted_matrix, weighted_target, smooth, magnitude, weight
        )
        results[f"spatial78_bounded_smooth_{weight:g}"] = {
            **diagnostic,
            **metrics(value),
            "coefficients": value.tolist(),
            "anchor_eigenvalues_dimensionless": np.linalg.eigvalsh(
                matrices(value)
            ).tolist(),
        }
    arrays.update(
        {
            "gram_raw": matrix.T @ matrix,
            "rhs_raw": matrix.T @ target,
            "gram_weighted": weighted_matrix.T @ weighted_matrix,
            "rhs_weighted": weighted_matrix.T @ weighted_target,
            "G": smooth,
            "M": magnitude,
        }
    )
    np.savez_compressed(output / "basis-and-normal-equations.npz", **arrays)
    contact = model.collision.diagnostics(state.collision, state.u)
    assert contact["contact_numerically_valid"]
    receipt = {
        "schema": "joint-neutral-spatial-baseline-audit-v2",
        "success": True,
        "scope": "exact F=I force representability only; no equilibrium solve or neutral shape feasibility claim",
        "active_model_changed": False,
        "skin_target_n_per_m": cfg.skin_target_n_per_m,
        "basis": basis_receipts,
        "bulk_columns": 78,
        "shared_coefficients_if_activated": 80,
        "column_units": "MPa*m^2 nodal force per dimensionless anchor coordinate; tensor scale is tissue mu",
        "objective_contract": {
            "force_term": "0.5 * ||W(Aq-b)||^2 / ||Wb||^2",
            "smoothness_term": "0.5 * beta * sum_t tr(C_t^T G_t C_t) / 3",
            "tissue_reduction": "unweighted mean of three individually volume-normalized tissue fields",
            "magnitude": "sum_t tr(C_t^T M_t C_t) / 3; diagnostic only, no magnitude penalty",
            "target": "negative skin-only force increment at F=I, excluding fixed passive/contact force",
        },
        "bounds": {"kind": "per-anchor tensor eigenvalues", "lower": -0.9, "upper": 10},
        "target_force_norm_n": float(np.linalg.norm(target) * 1.0e6),
        "constant_embedding_maximum_relative_error": max(embedding_errors),
        "results": results,
        "contact": contact,
        "hashes": {
            "input_arrays": sha256(cfg.prepared_dir / "inputs.npz"),
            "input_manifest": sha256(cfg.prepared_dir / "manifest.json"),
            "contact_spec": sha256(cfg.contact_spec),
            "basis_arrays": sha256(output / "basis-and-normal-equations.npz"),
        },
        "elapsed_seconds": time.perf_counter() - started,
        "supersedes": "003 used unscaled columns, coordinate box bounds, and a different basis; it is not valid evidence for this contract",
    }
    write_json(output / "summary.json", receipt)
    (output / "report.md").write_text(
        "# Exact-rest regional baseline audit\n\nNo equilibrium or inverse optimization ran. These residuals do not decide nonlinear neutral-motion feasibility.\n\n"
        + json.dumps(receipt, indent=2)
        + "\n"
    )
    cherries.log_output(output)
    COMPLETED = True


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
    if not COMPLETED:
        raise SystemExit(1)
