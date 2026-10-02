# PNCG negative curvature audit

The negative PNCG curvature in `hybrid-first-profile-002` came from IPC contact. Replaying only its first 11 accepted PNCG updates reproduced the handoff, with no Newton iterations or inverse updates. This is a diagnostic, not another performance benchmark.

At the next proposed direction, the measured contributions were:

| Contribution | Directional curvature |
| --- | ---: |
| Fat | +1.889165151850200e-08 |
| Aponeurosis | +5.146641954413293e-09 |
| Muscle | +2.865877620523413e-08 |
| Skin membrane | +3.933486677947452e-10 |
| All tissue | +5.309041834594900e-08 |
| IPC Gauss–Newton contact | -1.089637922091028e-07 |
| Total | -5.587337386316005e-08 |

The original profile recorded -5.587337388521334e-08. The relative replay difference is about 4e-10. Initial free-force norm also matched: 1.267878263414983e-05.

## Where clamping applies

The generic tetrahedral `WarpPotentialFem.make_hess_quad_kernel` clamps each cell/quadrature contribution to zero before accumulation (`src/liblaf/apple/warp/fem/_base.py:334-339`). `StableNeoHookeanStress` uses that default. The custom membrane kernel in `joint_materials.py:704` directly accumulates its signed triangle curvature and does not clamp. Its net contribution happened to be positive in this replay.

`Collision.hess_quad` directly returns the public PyPI `ipctk` Gauss–Newton quadratic form. Neither our wrapper nor that native path clamps or projects it. In PyPI 1.6.0, with distance vector t and its local differential a along p, the contact expression is:

```text
w * [4 * f''(||t||²) * (a·t)² + 2 * f'(||t||²) * ||a||²]
```

The active log barrier has negative f', so the second term can be negative even for a positive contact weight. “Gauss–Newton” here does not imply a positive-semidefinite operator. Improved-max contacts can additionally carry signed weights. The decomposition above identifies contact as the cause, but does not separate the contributions of signed weights and tangential curvature inside contact.

Verified against the official v1.6.0 sources: [quadratic-form implementation](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/potentials/normal_potential.cpp#L272-L307), [barrier derivatives](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/barrier/barrier.cpp#L24-L43), and [signed contact construction](https://github.com/ipc-sim/ipc-toolkit/blob/v1.6.0/src/ipc/collisions/normal/normal_collisions_builder.cpp#L340-L380). The quadratic-form and barrier sources bundled with the public PyPI cp314 wheel were byte-identical to the official v1.6.0 tag.

Therefore the current PNCG denominator is a sum of clamped bulk contributions, raw membrane contributions, and raw IPC GN contact curvature. It does not implement a clamp on every contribution. A consistently nonnegative PNCG surrogate would require addressing both membrane and contact. Clamping a scalar directional quadratic form also does not PSD-project the assembled Newton Hessian; the exact Newton/adjoint operators can still be indefinite.

No solver or physical-model behavior was changed by this audit.

## Reproduction and evidence

Working directory: `exp/2026/09/22/solver-performance`.

```bash
CHERRIES_NAME='PNCG negative curvature component audit' \
CHERRIES_TAGS='pncg,curvature,diagnostic,pypi-ipctk' \
.venv/bin/python \
src/58-audit-pncg-curvature.py \
> tmp/pncg-curvature-audit-001.console.log 2>&1
```

The script refuses to overwrite an existing output directory; choose a new `--output-dir` for a repeat. It verifies checkpoint hashes, `uv.lock`, and relevant archived material, collision, PNCG, and runtime sources against the preceding profile. It uses the same explicit frozen-input runtime binding, active stress, prestress, jaw pose, and bone/eyeball contact. The repository remains a dirty experimental checkout; exact executed source snapshots are retained. Post-run edits to the audit script only address lint conventions.

- [Machine-readable decomposition and complete PNCG trace](../data/pncg-curvature-audit-001/curvature-audit.json)
- [Saved handoff state and direction](../data/pncg-curvature-audit-001/negative-curvature-state.pt)
- [Source provenance](../data/pncg-curvature-audit-001/provenance.json)
- [Console log and complete Comet summary](../tmp/pncg-curvature-audit-001.console.log)
- [Comet run](https://www.comet.com/liblaf/apple/eb7999892f174cb99de534b698a32867)

The process and Cherries shutdown completed with exit code 0. The Comet summary records the five component values in the table, the run name above, and entrypoint `src/58-audit-pncg-curvature.py`. A loader queued a nonexistent unrelated `data/simple-skin-forward` asset and emitted a warning at shutdown; the curvature JSON, handoff checkpoint, and source snapshot were saved and inspected successfully.
