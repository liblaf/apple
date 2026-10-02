"""Profile-only current-runtime binding for immutable frozen-neutral inputs."""

from __future__ import annotations

import ast
import contextlib
import copy
import difflib
import hashlib
import json
import subprocess
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
from joint_common import sha256, write_json
from joint_data import array_sha256
from joint_frozen_neutral import FrozenNeutral

IMPORT_MIGRATIONS = {
    "apple/forward/_forward.py",
    "apple/forward/_problem.py",
    "apple/inverse/_diff_forward.py",
    "experiment/joint_equilibrium.py",
    "experiment/joint_newton.py",
    "tensor-reference/face_physics.py",
}
_VERSION_METADATA = "apple/_version.py"
_LOCKFILE = "uv.lock"
# The inverse experiment uses only ``_AdjointProblem`` from this module through
# joint_equilibrium._Implicit.  Updating these reviewed bytes requires a new
# explicit binding rather than silently inheriting the forward-only exception.
CURRENT_OWNED_ADJOINT_SHA256 = (
    "0334053c9c21b7b5e7a8d3e084091c68946f1c9eb76dc41c0089530ce5d24ba4"
)

# This explicit profile/diagnostic opt-in permits the two directional-
# quadratic-form clamps while retaining the frozen input manifest and proving
# every other AST node is identical.
ISFIXED_BOUNDARY_CHANGES = {
    "experiment/joint_physics.py": {
        "functions": frozenset({"JointPhysics.__init__"}),
        "current_only_functions": frozenset(),
        "class_docstrings": frozenset(),
    },
    "experiment/joint_full_skull_contact.py": {
        "functions": frozenset(
            {
                "FullSkullJointPhysics.__init__",
                "FullSkullJointPhysics.full_skull_receipt",
            }
        ),
        "current_only_functions": frozenset(),
        "class_docstrings": frozenset(),
    },
}

PNCG_CURVATURE_CLAMP_CHANGES = {
    "experiment/joint_materials.py": {
        "functions": frozenset({"_membrane_hess_quad_kernel"}),
        "current_only_functions": frozenset(),
        "class_docstrings": frozenset(),
    },
    "apple/collision/_collision.py": {
        "functions": frozenset({"Collision.hess_quad"}),
        # This narrow read-only helper exposes each native PyPI term so the
        # validation can prove clamping happens before summation.
        "current_only_functions": frozenset({"Collision.raw_hess_quad_terms"}),
        "class_docstrings": frozenset(),
    },
}


def _normalized_imports(source: str) -> list[str]:
    return sorted(
        ast.dump(node, include_attributes=False).replace(
            "liblaf.peach", "liblaf.apple.solvers"
        )
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )


def _code_without_imports(source: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            node.module = node.module.replace("liblaf.peach", "liblaf.apple.solvers")
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    return ast.dump(tree, include_attributes=False)


def _named_code_without_imports(source: str, name: str) -> str:
    """Normalize one top-level runtime class while ignoring import relocation."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            node.module = node.module.replace("liblaf.peach", "liblaf.apple.solvers")
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    assert len(tree.body) == 1, name
    return ast.dump(tree, include_attributes=False)


class _RemoveNamedNodes(ast.NodeTransformer):
    """Remove only explicitly named function bodies and class docstrings."""

    def __init__(
        self, *, functions: frozenset[str], class_docstrings: frozenset[str]
    ) -> None:
        self.functions = functions
        self.class_docstrings = class_docstrings
        self.scope: list[str] = []
        self.removed_functions: set[str] = set()
        self.removed_class_docstrings: set[str] = set()

    def _qualified(self, name: str) -> str:
        return ".".join([*self.scope, name])

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST | None:
        qualified = self._qualified(node.name)
        if qualified in self.functions:
            self.removed_functions.add(qualified)
            return None
        self.scope.append(node.name)
        try:
            return self.generic_visit(node)
        finally:
            self.scope.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST | None:
        qualified = self._qualified(node.name)
        if qualified in self.functions:
            self.removed_functions.add(qualified)
            return None
        self.scope.append(node.name)
        try:
            return self.generic_visit(node)
        finally:
            self.scope.pop()

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        qualified = self._qualified(node.name)
        self.scope.append(node.name)
        try:
            node = self.generic_visit(node)
        finally:
            self.scope.pop()
        if qualified in self.class_docstrings and ast.get_docstring(node) is not None:
            assert isinstance(node.body[0], ast.Expr)
            node.body.pop(0)
            self.removed_class_docstrings.add(qualified)
        return node


def _code_without_permitted_nodes(
    source: str, *, functions: frozenset[str], class_docstrings: frozenset[str]
) -> tuple[str, dict[str, list[str]]]:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            node.module = node.module.replace("liblaf.peach", "liblaf.apple.solvers")
    tree.body = [
        node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    visitor = _RemoveNamedNodes(functions=functions, class_docstrings=class_docstrings)
    tree = visitor.visit(tree)
    assert tree is not None
    return ast.dump(tree, include_attributes=False), {
        "functions": sorted(visitor.removed_functions),
        "class_docstrings": sorted(visitor.removed_class_docstrings),
    }


def _without_version_metadata(source: str) -> list[str]:
    return [
        line
        for line in source.splitlines()
        if not line.startswith(("__version__ =", "__version_tuple__ ="))
    ]


def _unified_diff(old: str, new: str, *, name: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"historical/{name}",
            tofile=f"current/{name}",
        )
    )


def _runtime_archive(neutral_dir: Path, name: str, expected_sha256: str) -> Path:
    archived = neutral_dir / "sources" / name
    assert archived.is_file(), archived
    assert sha256(archived) == expected_sha256, archived
    return archived


def _historical_lock(current_path: Path, expected_sha256: str) -> tuple[bytes, str]:
    repository = current_path.parent
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository, text=True
    ).strip()
    historical = subprocess.check_output(
        ["git", "show", "HEAD:uv.lock"], cwd=repository
    )
    assert hashlib.sha256(historical).hexdigest() == expected_sha256
    return historical, head


def _packages(lock_bytes: bytes) -> dict[str, dict[str, Any]]:
    document = tomllib.loads(lock_bytes.decode())
    result: dict[str, dict[str, Any]] = {}
    for row in document.get("package", []):
        name = row["name"]
        assert name not in result, name
        result[name] = row
    return result


def _package_changes(old: bytes, new: bytes) -> dict[str, Any]:
    before, after = _packages(old), _packages(new)
    return {
        "added": {name: after[name] for name in sorted(after.keys() - before.keys())},
        "removed": {
            name: before[name] for name in sorted(before.keys() - after.keys())
        },
        "changed": {
            name: {"historical": before[name], "current": after[name]}
            for name in sorted(before.keys() & after.keys())
            if before[name] != after[name]
        },
    }


def load_profile_binding(  # noqa: C901, PLR0912, PLR0915
    neutral_dir: Path,
    output_dir: Path,
    *,
    allow_pncg_curvature_clamps: bool = False,
    allow_isfixed_boundary: bool = False,
    unused_inverse_sha256: str | None = None,
    current_owned_adjoint_sha256: str | None = None,
) -> FrozenNeutral:
    """Verify historical neutral inputs and explicitly rebind runtime-only bytes.

    The original manifest and source directory remain immutable.  The returned
    object owns a copied manifest whose runtime entries name the current files.
    """
    neutral_dir = neutral_dir.resolve()
    output_dir = output_dir.resolve()
    assert not (
        unused_inverse_sha256 is not None and current_owned_adjoint_sha256 is not None
    )
    manifest_path = neutral_dir / "manifest.json"
    original = json.loads(manifest_path.read_text())
    assert original["schema"] == "joint-frozen-neutral-v1"
    assert original["success"] is True

    for item in [*original["sources"].values(), *original["artifacts"].values()]:
        assert sha256(Path(item["path"])) == item["sha256"], item["path"]
    with np.load(neutral_dir / "state.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    assert set(arrays) == set(original["arrays"])
    for name, value in arrays.items():
        assert array_sha256(value) == original["arrays"][name]["sha256"], name

    rebound = copy.deepcopy(original)
    changes: dict[str, Any] = {}
    diff_directory = output_dir / "runtime-diffs"
    for name, item in original["runtime_sources"].items():
        current_path = Path(item["path"])
        assert current_path.is_file(), current_path
        current_hash = sha256(current_path)
        if name == _LOCKFILE:
            historical_bytes, head = _historical_lock(current_path, item["sha256"])
            if current_hash == item["sha256"]:
                continue
            assert name == _LOCKFILE
            diff = _unified_diff(
                historical_bytes.decode(), current_path.read_text(), name=name
            )
            diff_directory.mkdir(parents=True, exist_ok=True)
            diff_path = diff_directory / "uv.lock.diff"
            diff_path.write_text(diff)
            new_item = {
                **item,
                "sha256": current_hash,
                "bytes": current_path.stat().st_size,
            }
            rebound["runtime_sources"][name] = new_item
            changes[name] = {
                "historical": item,
                "current": new_item,
                "verified_historical_source": f"git HEAD:{head}:uv.lock",
                "reason": "current solver/PyPI environment migration; lockfile differences recorded without package-set assumptions",
                "unified_diff": str(diff_path.resolve()),
                "package_changes": _package_changes(
                    historical_bytes, current_path.read_bytes()
                ),
            }
            continue

        archived = _runtime_archive(neutral_dir, name, item["sha256"])
        if current_hash == item["sha256"]:
            continue
        old_text, current_text = archived.read_text(), current_path.read_text()
        diff_directory.mkdir(parents=True, exist_ok=True)
        diff_path = diff_directory / (name.replace("/", "__") + ".diff")
        diff_path.write_text(_unified_diff(old_text, current_text, name=name))
        permitted_changes: dict[str, dict[str, frozenset[str]]] = {}
        if allow_pncg_curvature_clamps:
            permitted_changes.update(PNCG_CURVATURE_CLAMP_CHANGES)
        if allow_isfixed_boundary:
            permitted_changes.update(ISFIXED_BOUNDARY_CHANGES)
        allowed = permitted_changes.get(name)
        if allowed is not None:
            imports_equal = _normalized_imports(old_text) == _normalized_imports(
                current_text
            )
            old_body, old_removed = _code_without_permitted_nodes(
                old_text,
                functions=allowed["functions"],
                class_docstrings=allowed["class_docstrings"],
            )
            current_body, current_removed = _code_without_permitted_nodes(
                current_text,
                functions=allowed["functions"] | allowed["current_only_functions"],
                class_docstrings=allowed["class_docstrings"],
            )
            assert imports_equal, name
            assert old_body == current_body, name
            assert old_removed["functions"] == sorted(allowed["functions"]), name
            assert current_removed["functions"] == sorted(
                allowed["functions"] | allowed["current_only_functions"]
            ), name
            assert (
                old_removed["class_docstrings"] == current_removed["class_docstrings"]
            ), name
            proof = {
                "kind": (
                    "only_named_isfixed_boundary_bodies_changed"
                    if name in ISFIXED_BOUNDARY_CHANGES
                    else "only_named_pncg_quadratic_form_bodies_changed"
                ),
                "normalized_imports_equal": True,
                "non_import_ast_equal_excluding": {
                    "functions": sorted(allowed["functions"]),
                    "current_only_functions": sorted(allowed["current_only_functions"]),
                    "class_docstrings": sorted(allowed["class_docstrings"]),
                },
                "removed_from_historical_ast": old_removed,
                "removed_from_current_ast": current_removed,
                "exact_diff_recorded": True,
            }
        elif (
            name == "apple/inverse/_diff_forward.py"
            and unused_inverse_sha256 is not None
        ):
            # Forward-only callers must guard the differentiable entry points
            # and pin the exact reviewed bytes. This is not an inverse replay.
            assert current_hash == unused_inverse_sha256, name
            proof = {
                "kind": "unused_inverse_wrapper_in_guarded_forward_only_run",
                "reviewed_sha256": unused_inverse_sha256,
                "exact_diff_recorded": True,
                "non_import_ast_equal": False,
            }
        elif (
            name == "apple/inverse/_diff_forward.py"
            and current_owned_adjoint_sha256 is not None
        ):
            # The inverse route does not call DifferentiableForward.  Its
            # custom joint_equilibrium._Implicit calls only _AdjointProblem;
            # prove that owned class remained identical after import migration.
            assert current_hash == current_owned_adjoint_sha256, name
            assert _named_code_without_imports(
                old_text, "_AdjointProblem"
            ) == _named_code_without_imports(current_text, "_AdjointProblem"), name
            proof = {
                "kind": "current_owned_adjoint_problem_in_joint_implicit_inverse",
                "reviewed_sha256": current_owned_adjoint_sha256,
                "adjoint_problem_non_import_ast_equal": True,
                "public_differentiable_forward_used": False,
                "custom_joint_equilibrium_implicit_used": True,
                "exact_diff_recorded": True,
            }
        elif name == _VERSION_METADATA:
            assert _without_version_metadata(old_text) == _without_version_metadata(
                current_text
            )
            proof = {
                "kind": "generated_version_metadata_only",
                "non_version_lines_equal": True,
            }
        else:
            assert name in IMPORT_MIGRATIONS, name
            imports_equal = _normalized_imports(old_text) == _normalized_imports(
                current_text
            )
            body_equal = _code_without_imports(old_text) == _code_without_imports(
                current_text
            )
            if body_equal:
                assert imports_equal, name
                proof = {
                    "kind": "imports_only",
                    "normalized_imports_equal": True,
                    "non_import_ast_equal": True,
                }
            else:
                # Current forward backends intentionally add sparse-Hessian
                # support.  Other runtime records must still be import-only.
                assert name in {
                    "apple/forward/_forward.py",
                    "apple/forward/_problem.py",
                }, name
                proof = {
                    "kind": "additional_runtime_backend_code",
                    "normalized_imports_equal": imports_equal,
                    "non_import_ast_equal": False,
                    "exact_diff_recorded": True,
                }
        new_item = {
            **item,
            "sha256": current_hash,
            "bytes": current_path.stat().st_size,
        }
        rebound["runtime_sources"][name] = new_item
        changes[name] = {
            "historical": item,
            "current": new_item,
            "verified_historical_source": str(archived.resolve()),
            "proof": proof,
            "unified_diff": str(diff_path.resolve()),
        }

    physical_runtime_names = {
        "apple/collision/_collision.py",
        "experiment/joint_materials.py",
        "experiment/joint_physics.py",
        "experiment/joint_contact.py",
        "experiment/joint_full_skull_contact.py",
    }
    changed_physical = sorted(physical_runtime_names & changes.keys())
    allowed_physical = sorted(
        set(PNCG_CURVATURE_CLAMP_CHANGES if allow_pncg_curvature_clamps else ())
        | set(ISFIXED_BOUNDARY_CHANGES if allow_isfixed_boundary else ())
    )
    assert changed_physical == allowed_physical, changed_physical
    receipt = {
        "schema": "profile-current-runtime-binding-v1",
        "historical_manifest": {
            "path": str(manifest_path),
            "sha256": sha256(manifest_path),
        },
        "historical_data_and_arrays_verified": True,
        "historical_manifest_modified": False,
        "physical_material_reference_and_contact_source_hashes_unchanged": not bool(
            changed_physical
        ),
        "allowed_physical_runtime_changes": changed_physical,
        "allowed_physical_change_scope": (
            "Only the named PNCG hess_quad bodies and the narrow read-only "
            "raw_hess_quad_terms helper are excluded from AST equality. Physical "
            "energy, force, and Hessian-product code remain in the compared AST."
            if allow_pncg_curvature_clamps
            else None
        ),
        "isfixed_boundary_correction": (
            {
                "enabled": True,
                "reason": "Correct historical boundary-policy error.",
                "scope": "IsFixed is the sole FEM clamp; jaw = IsFixed intersect Mandible; runtime geometry fixed IDs are rebound; historical neutral equilibrium is invalidated.",
                "excluded_ast_functions": {
                    name: sorted(rule["functions"])
                    for name, rule in ISFIXED_BOUNDARY_CHANGES.items()
                },
                "source_diffs_recorded": True,
                "physics_unchanged": False,
            }
            if allow_isfixed_boundary
            else None
        ),
        "runtime_changes": changes,
    }
    write_json(output_dir / "profile-input-binding.json", receipt)
    return FrozenNeutral(arrays=arrays, manifest=rebound, directory=neutral_dir)


@contextlib.contextmanager
def bind_frozen_neutral_load(
    neutral_dir: Path,
    output_dir: Path,
    *,
    allow_pncg_curvature_clamps: bool = False,
    allow_isfixed_boundary: bool = False,
    unused_inverse_sha256: str | None = None,
    current_owned_adjoint_sha256: str | None = None,
) -> Iterator[FrozenNeutral]:
    """Temporarily bind only one historical neutral directory to current runtime.

    This is deliberately narrow: other ``FrozenNeutral.load`` calls retain the
    original strict hash behavior.  Restore the class descriptor on every exit.
    """
    target = neutral_dir.resolve()
    bound = load_profile_binding(
        target,
        output_dir,
        allow_pncg_curvature_clamps=allow_pncg_curvature_clamps,
        allow_isfixed_boundary=allow_isfixed_boundary,
        unused_inverse_sha256=unused_inverse_sha256,
        current_owned_adjoint_sha256=current_owned_adjoint_sha256,
    )
    original_descriptor = FrozenNeutral.__dict__["load"]
    original_load = FrozenNeutral.load

    def load_bound(
        _cls: type[FrozenNeutral], directory: Path | None = None
    ) -> FrozenNeutral:
        if directory is not None and Path(directory).resolve() == target:
            return bound
        return original_load(directory)

    FrozenNeutral.load = classmethod(load_bound)
    try:
        yield bound
    finally:
        FrozenNeutral.load = original_descriptor
