"""Use a saved loaded equilibrium as the origin of expression displacements.

The adopted face is not a stress-free FEM reference. Constitutive and contact
reference geometry remain unchanged; total displacement is u0 + increment.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, override

import numpy as np
import torch
from joint_common import GROUP, sha256
from joint_data import PreparedInputs, array_sha256

from liblaf.apple.forward._problem import ForwardProblem


def load_script(filename: str) -> Any:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class FrozenNeutral:
    """Hash-bound neutral state, transferred targets and regularization geometry."""

    arrays: dict[str, np.ndarray]
    manifest: dict[str, Any]
    directory: Path

    @classmethod
    def load(cls, directory: Path | None = None) -> FrozenNeutral:
        if directory is None:
            selection = json.loads((GROUP / "data/current-neutral.json").read_text())
            directory = Path(selection["directory"])
            assert sha256(directory / "manifest.json") == selection["manifest_sha256"]
        manifest = json.loads((directory / "manifest.json").read_text())
        assert manifest["schema"] == "joint-frozen-neutral-v1"
        assert manifest["success"] is True
        for record in [
            *manifest["sources"].values(),
            *manifest["artifacts"].values(),
            *manifest["runtime_sources"].values(),
        ]:
            assert sha256(Path(record["path"])) == record["sha256"]
        with np.load(directory / "state.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        assert set(arrays) == set(manifest["arrays"])
        for key, value in arrays.items():
            assert array_sha256(value) == manifest["arrays"][key]["sha256"]
        return cls(arrays, manifest, directory)

    def build_physics(self) -> tuple[Any, dict[str, dict[str, torch.Tensor]]]:
        """Reconstruct the original constitutive model and its fixed material state."""
        for record in self.manifest["runtime_sources"].values():
            assert sha256(Path(record["path"])) == record["sha256"], record["path"]
        runner = load_script("68-run-simple-skin-forward.py")
        protocol = json.loads(
            Path(self.manifest["sources"]["protocol"]["path"]).read_text()
        )
        inputs = protocol["inputs"]
        prepared = PreparedInputs.load(
            Path(inputs["prepared_npz"]), Path(inputs["prepared_manifest"])
        )
        geometry_receipt = inputs["geometry"]["geometry"]
        geometry = runner.load_full_skull_geometry(
            Path(geometry_receipt["geometry_path"]),
            Path(geometry_receipt["audit_path"]),
        )
        canonical = runner.research_informed_material_config()["materials"]
        mechanics = protocol["mechanics"]
        runtime_arrays = dict(self.arrays)
        runtime_arrays["target_displacement_m"] = self.arrays[
            "target_total_displacement_m"
        ]
        physics = runner.FullSkullJointPhysics(
            prepared.volume_path,
            prepared.skin_path,
            runtime_arrays,
            bulk_young_mpa={
                name: canonical[name]["young_mpa"] for name in runner.BULK_TISSUES
            },
            bulk_nu={
                name: mechanics["poisson_ratios"][name] for name in runner.BULK_TISSUES
            },
            skin_young_mpa=canonical["skin"]["reference_map"]["young_mpa"],
            skin_nu=0.49,
            thickness_m=canonical["skin"]["thickness_m"],
            full_skull_geometry=geometry,
            full_skull_admission=json.loads(Path(inputs["admission_path"]).read_text()),
            full_skull_contact_config=mechanics["contact"],
            rtol=0.0,
            atol=self.manifest["force_threshold_code"],
            max_steps=protocol["solver"]["max_steps"],
            forward_method="pncg",
            adjoint_rtol=1e-7,
        )
        skin, _ = runner.load_skin_field(
            Path(inputs["skin_field_path"]),
            Path(inputs["skin_field_manifest_path"]),
            prepared=prepared,
            expected_triangles=physics.skin_tri,
        )
        baseline = runner.heterogeneous_materials(physics, skin)
        physics.runtime.forward.model.set_materials(baseline)
        return physics, baseline

    def expression_materials(
        self,
        baseline: dict[str, dict[str, torch.Tensor]],
        active_ids: torch.Tensor,
        *,
        skin_multiplier: torch.Tensor,
        active_stress: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Expose stiffness and expression activation, with no baseline parameters.

        Activation tensors use the original constitutive reference frame. The
        caller applies the declared symmetric/PSD parameterization and priors.
        A changed stiffness requires a new equilibrium and neutral-drift check.
        """
        assert active_stress.shape == (len(active_ids), 3, 3)
        assert bool(torch.isfinite(active_stress).all())
        assert bool(torch.allclose(active_stress, active_stress.transpose(-1, -2)))
        assert (
            skin_multiplier.ndim == 0
            or skin_multiplier.shape == baseline["skin"]["mu"].shape
        )
        assert bool(torch.isfinite(skin_multiplier).all() & (skin_multiplier > 0).all())
        values = {name: dict(fields) for name, fields in baseline.items()}
        values["muscle"]["active_stress"] = baseline["muscle"][
            "active_stress"
        ].index_add(0, active_ids, active_stress)
        for key in ("mu", "lmbda"):
            values["skin"][key] = baseline["skin"][key] * skin_multiplier
        return values

    def total_displacement(self, increment: torch.Tensor) -> torch.Tensor:
        origin = torch.as_tensor(
            self.arrays["neutral_displacement_m"],
            device=increment.device,
            dtype=increment.dtype,
        )
        assert increment.shape == origin.shape
        return origin + increment

    def expression_residual(
        self, total: torch.Tensor, target_index: int
    ) -> torch.Tensor:
        ids = torch.as_tensor(self.arrays["observation_node_ids"], device=total.device)
        target = torch.as_tensor(
            self.arrays["target_total_displacement_m"][target_index],
            device=total.device,
            dtype=total.dtype,
        )
        return total[ids] - target


class NeutralIncrementProblem(ForwardProblem):
    """Translate free coordinates without changing forces, tangent or CCD paths."""

    def __init__(self, model: Any, neutral_full_displacement: torch.Tensor) -> None:
        super().__init__(model=model)
        self.origin = model.dof_map.to_free(neutral_full_displacement).detach().clone()

    @override
    def update(self, state: Any, increment: torch.Tensor) -> None:
        super().update(state, self.origin + increment)
