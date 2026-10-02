"""Audit labeled lip vertices against the saved rigid-inverse fixed map."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree

mpl.use("Agg")
import matplotlib.pyplot as plt

from liblaf import cherries

GROUP = Path(__file__).resolve().parent.parent
ROOT = GROUP.parents[4]
JOINT = ROOT / "exp/2026/09/21/joint-activation-material-mandible"
sys.path.insert(0, str(JOINT / "src"))
from joint_common import ProfileJoint, write_json  # noqa:E402


class Config(cherries.BaseConfig):
    output: Path = GROUP / "data/lip-boundary-audit-003-isfixed-policy"
    volume: Path = (
        GROUP / "data/forward-repaired-reference-005/rebased-reference-volume.vtu"
    )
    endpoint: Path = GROUP / "data/inverse-mouthopen-rigid-trial-001/endpoint.npz"
    state: Path = JOINT / "data/frozen-neutral-004/state.npz"
    neutral_protocol: Path = GROUP / "data/forward-repaired-reference-005/protocol.json"


def rec(p: Path) -> dict[str, str]:
    b = p.read_bytes()
    return {"path": str(p.resolve()), "sha256": hashlib.sha256(b).hexdigest()}


def main(c: Config) -> None:
    assert not c.output.exists()
    c.output.mkdir(parents=True)
    m = pv.read(c.volume)
    x = np.asarray(m.points)
    with np.load(c.endpoint) as a:
        u = a["displacement_m"][: len(x)]
    with np.load(c.state) as a:
        h, cra, man = (
            a[k]
            for k in (
                "historical_fixed_node_ids",
                "cranium_node_ids",
                "mandible_node_ids",
            )
        )
    lip = np.asarray(m.point_data["IsLip"]).astype(bool)
    protocol = json.loads(c.neutral_protocol.read_text())
    geometry = protocol["reference_configuration"]["rigid_geometry"]["full_skull"]
    geometry_path = Path(geometry["geometry_path"])
    assert rec(geometry_path)["sha256"] == geometry["geometry_sha256"]
    with np.load(geometry_path, allow_pickle=False) as a:
        soft_ids = np.asarray(a["soft_global_ids"])
        soft_faces = np.asarray(a["soft_faces"])
    assert soft_ids.max() < len(x)
    lower = lip & (x[:, 1] <= np.median(x[lip, 1]))
    ids = np.arange(len(x))
    saved_mask = np.asarray(m.point_data["FixedMask"], dtype=bool).any(axis=1)
    intended_jaw = np.intersect1d(h, man, assume_unique=True)
    old_extra_cranium = np.setdiff1d(cra, h, assume_unique=True)
    old_extra_mandible = np.setdiff1d(man, h, assume_unique=True)
    assert np.array_equal(
        np.flatnonzero(saved_mask), np.union1d(h, np.union1d(cra, man))
    )
    # Group labels do not authorize a clamp. The two saved-old-policy groups
    # are rendered only to expose why this archived trial is invalidated.
    groups = {
        "intended_isfixed_zero": lip & np.isin(ids, np.setdiff1d(h, intended_jaw)),
        "intended_isfixed_jaw": lip & np.isin(ids, intended_jaw),
        "intended_free": lip & ~np.isin(ids, h),
        "saved_old_policy_extra_cranium_should_be_free": lip
        & np.isin(ids, old_extra_cranium),
        "saved_old_policy_extra_mandible_should_be_free": lip
        & np.isin(ids, old_extra_mandible),
    }
    posed = x + u
    on_soft = np.isin(ids, soft_ids)
    soft_tree = cKDTree(posed[soft_ids])

    def surface_stats(mask: np.ndarray) -> dict[str, object]:
        surface_mask = mask & on_soft
        off_surface_mask = mask & ~on_soft
        incident = np.any(np.isin(soft_ids[soft_faces], ids[surface_mask]), axis=1)
        nearest = soft_tree.query(posed[off_surface_mask])[0]
        return {
            "on_rendered_soft_surface_count": int(surface_mask.sum()),
            "off_rendered_soft_surface_count": int(off_surface_mask.sum()),
            "rendered_soft_faces_incident_to_group": int(incident.sum()),
            "off_surface_nearest_rendered_vertex_distance_m": {
                "min": float(nearest.min()) if len(nearest) else None,
                "median": float(np.median(nearest)) if len(nearest) else None,
                "max": float(nearest.max()) if len(nearest) else None,
            },
        }

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.triplot(
        posed[soft_ids, 0],
        posed[soft_ids, 1],
        soft_faces,
        color="#aaaaaa",
        alpha=0.18,
        linewidth=0.08,
        zorder=0,
    )
    for name, color in [
        ("intended_free", "#777777"),
        ("saved_old_policy_extra_cranium_should_be_free", "#d62728"),
        ("saved_old_policy_extra_mandible_should_be_free", "#1f77b4"),
        ("intended_isfixed_zero", "#9467bd"),
        ("intended_isfixed_jaw", "#2ca02c"),
    ]:
        z = groups[name]
        ax.scatter(
            posed[z, 0],
            posed[z, 1],
            s=7,
            c=color,
            label=f"{name} ({z.sum()}; soft {(z & on_soft).sum()})",
            zorder=2,
        )
    ax.set(
        xlabel="world x (m)",
        ylabel="world y (m)",
        title="IsLip vertices: intended IsFixed policy and archived old-mask extras",
    )
    ax.legend()
    ax.set_aspect("equal")
    fig.savefig(c.output / "mouth-lip-fixed-overlay.png", dpi=220)
    plt.close(fig)
    out = {
        "schema": "lip-boundary-audit-v2-isfixed-policy",
        "inputs": {
            "volume": rec(c.volume),
            "endpoint": rec(c.endpoint),
            "frozen_state": rec(c.state),
            "neutral_protocol": rec(c.neutral_protocol),
            "rendered_soft_geometry": rec(geometry_path),
            "boundary_source": rec(JOINT / "src/joint_full_skull_contact.py"),
        },
        "definition": "IsLip point label; lower half is a geometric display heuristic. IsFixed is the intended sole FEM clamp; GroupId labels route jaw motion only after intersection with IsFixed. Saved old-policy extras are reported separately and do not authorize a clamp.",
        "boundary_policy": {
            "intended_fixed_nodes": len(h),
            "intended_jaw_nodes": len(intended_jaw),
            "saved_archived_fixedmask_nodes": int(saved_mask.sum()),
            "saved_old_policy_extra_cranium_nodes": len(old_extra_cranium),
            "saved_old_policy_extra_mandible_nodes": len(old_extra_mandible),
        },
        "lip_count": int(lip.sum()),
        "lower_count": int(lower.sum()),
        "lip_y_median_m": float(np.median(x[lip, 1])),
        "groups": {
            k: {
                "count": int(v.sum()),
                "lower_count": int((v & lower).sum()),
                "median_displacement_m": float(np.median(np.linalg.norm(u[v], axis=1)))
                if v.any()
                else 0,
                "max_displacement_m": float(np.max(np.linalg.norm(u[v], axis=1)))
                if v.any()
                else 0,
                "rendered_surface": surface_stats(v),
            }
            for k, v in groups.items()
        },
        "rendered_soft_surface": {
            "vertex_count": len(soft_ids),
            "triangle_count": len(soft_faces),
            "lip_on_rendered_soft_surface_count": int((lip & on_soft).sum()),
            "lip_off_rendered_soft_surface_count": int((lip & ~on_soft).sum()),
        },
        "overlay": rec(c.output / "mouth-lip-fixed-overlay.png"),
        "saved_trial_policy_status": "invalidated_old_policy; saved FixedMask is reported for comparison only.",
        "boundary_evidence": "joint_full_skull_contact.py FullSkullGeometry.full_boundary_displacement initializes zero then assigns supplied original mandible IDs rigid_displacement; corrected callers must supply IsFixed intersect Mandible.",
    }
    write_json(c.output / "receipt.json", out)
    cherries.log_output(c.output)
    cherries.log_metrics(
        {
            "lip/labeled": int(lip.sum()),
            "lip/saved_old_policy_extra": int(
                (
                    groups["saved_old_policy_extra_cranium_should_be_free"]
                    | groups["saved_old_policy_extra_mandible_should_be_free"]
                ).sum()
            ),
        }
    )


if __name__ == "__main__":
    cherries.main(main, profile=ProfileJoint)
