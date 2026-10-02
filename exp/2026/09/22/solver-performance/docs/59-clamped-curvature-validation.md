# Saved Smile per-contribution PNCG curvature clamp validation

This diagnostic evaluates the previously saved Smile PNCG handoff state and direction after adding the requested clamps. It does not run a forward iteration, Newton step, inverse update, or performance benchmark.

The new PNCG curvature is positive:

| Contribution | Curvature |
| --- | ---: |
| Fat | +1.889165151850197e-08 |
| Aponeurosis | +5.146641954413307e-09 |
| Muscle | +2.865877620523420e-08 |
| Skin membrane, per-triangle clamped | +3.933488622284179e-10 |
| All tissue | +5.309041854037789e-08 |
| IPC contact, sum of per-contact clamps | +7.801364531947458e-10 |
| **PNCG directional curvature** | **+5.387055499357264e-08** |

The prior raw IPC aggregate at this identical state was -1.089637922091028e-07. The 4,670 active contact terms contain 1,249 negative and 3,421 positive values. Each term is now clamped before it is summed, producing +7.801364531947458e-10. This proves the result is not `max(raw_contact_sum, 0)`, which would be zero here.

The generic tetrahedral kernel already clamped every cell/quadrature term. The custom membrane kernel now computes a triangle-local scalar `h_quad` and atomically adds `max(h_quad, 0)`. Contact now enumerates one public PyPI IPCTK Gauss--Newton quadratic-form value per active collision and sums `max(term, 0)`.

## Scope and provenance

The command was run from `exp/2026/09/22/solver-performance`:

```bash
CHERRIES_NAME='Saved Smile per-contribution PNCG curvature clamp validation' \
CHERRIES_TAGS='pncg,curvature,clamp,smile,active-stress,pypi-ipctk' \
.venv/bin/python \
src/59-validate-clamped-curvature.py \
--output-dir data/pncg-clamped-curvature-004 \
> tmp/pncg-clamped-curvature-004.console.log 2>&1
```

An earlier `pncg-clamped-curvature-001` attempt stopped before model setup because the historical profile protocol has no `jaw_sha256` field. It performed no curvature evaluation. Subsequent runs verify the saved jaw tensor directly against the original Smile checkpoint. Runs `-002` and `-003` checked the initial implementation; the table and linked final evidence use run `-004` after the array-layout correction.

The saved state has the same activation and jaw tensors as the original Smile checkpoint. It reconstructs the same active-stress, prestressed tissue, mandible pose, bone, and eyeball collision model. The frozen-neutral manifest, every referenced input artifact, and every stored array hash were verified without modification.

The runtime binding permits only these explicit source differences from the historical input runtime:

- `joint_materials._membrane_hess_quad_kernel`;
- `Collision.hess_quad`; and
- the new `Collision.raw_hess_quad_terms` helper, used by the clamped PNCG wrapper and by this validation to expose the unclamped terms.

It records the complete textual diffs and proves AST equality after removing only those named bodies. Therefore physical energy, force, and Hessian-product implementations remain equal to the historical runtime. The opt-in is used by the current hybrid PNCG profile and this saved-state validation. It changes only the PNCG directional-curvature surrogate; it does not PSD-project the assembled Newton operator.

## Timing observation

The final run made one synchronized diagnostic evaluation of the raw contact-term enumeration in 0.0217 s and of the clamped wrapper in 0.0208 s. These calls include CPU transfers and Python iteration across 4,670 contacts and were measured once after model setup. The helper converts full vertex and direction arrays to Fortran order once before the contact loop; it avoids the earlier repeated conversion inside every `collision.dof` call.

The initial unoptimized validation run (`-002`) measured 0.630 s and 0.740 s respectively. It is preserved as development evidence but is not comparable performance data because it repeatedly converted C-order arrays. Neither measurement is a full-forward benchmark or an estimate of full-solver speedup.

## Evidence

- [Machine-readable result](../data/pncg-clamped-curvature-004/clamped-curvature.json)
- [Binding and AST proof receipt](../data/pncg-clamped-curvature-004/profile-input-binding.json)
- [Current source snapshot](../data/pncg-clamped-curvature-004/provenance.json)
- [Exact source diffs](../data/pncg-clamped-curvature-004/runtime-diffs/)
- [Cherries console log](../tmp/pncg-clamped-curvature-004.console.log)
- [Comet run](https://www.comet.com/liblaf/apple/3b2aa741c59146f6b66292a2e5c33c55)

Cherries completed with exit code 0. A pre-existing loader requested an unrelated nonexistent `data/simple-skin-forward` asset during shutdown; the result JSON, binding receipt, source snapshot, and Comet metrics were all written and inspected.
