# Jaw numerical preflight

`src/28-run-jaw-preflight.py` is a no-optimization full-face check between a
neutral checkpoint and the jaw proposal domain. It uses the checkpoint's exact
shared field, Newton-CG, soft-tissue/bone IPC, topology-aware FEM checks, and the
separately validated pure-FEM rigid-bone linear CCD guard.

`diagnostic_only` may inspect a frozen intermediate neutral checkpoint, but its
receipt always records `final_launch_ready: false`. `final_launch_ready` also
requires the converged full-target neutral contract. Every run uses a fresh
output directory.

## Version 2 correction

Diagnostic 002 is preserved at
`data/jaw-preflight-spatial25-diagnostic-002/summary.json` (SHA-256
`120e2c1287376e08fcc6a3a305bddd737fe523bb8dbe5bba7659a6d73282d481`).
Its zero pose passed equilibrium, contact, shape, and FEM checks, but the solved
state's maximum absolute coordinate difference from the frozen primal was
`1.9021703167654294e-7 m`. Version 1
incorrectly compared displacement with the `1e-12` absolute **force** tolerance
and called the result an identity test. The legacy one-degree candidate passed
the rigid guard and was rejected by the owned soft-tissue/bone boundary CCD at
fraction `0.0361328125`; all 12 broad endpoints were retained as rigid-CCD
rejections.

Version 2 declares a separate maximum Euclidean nodal-displacement difference
budget of `1e-6 m`. This model-QA choice is one micrometre, or `1/250` of the
`0.25 mm` neutral surface-motion RMS budget. It also reports the RMS Euclidean
difference over the unique skin/FEM surface nodes and over every volume node.
Because the solved and frozen displacements share the same reference points,
their world-coordinate difference equals their displacement difference. This
is a reproducibility check, not bitwise identity, and it does not change the
force, contact, shape, or FEM gates.

Version 2 retains the failed manifest pose as `legacy_one_degree_center`, role
`diagnostic_legacy_one_degree_center`. It adds the source-declared
`world_x_positive_0p01_degree`, exactly `+0.01 degree` in the world-frame
rotation-vector x component about `mandible_pivot_m`, with zero other
coordinates, as role `required_nonzero_world_x_0p01_degree`. This candidate is
fixed in source before the new run. It is not selected adaptively after a
trial, and the failed one-degree evidence remains in every v2 receipt.

## Declared probes

Every probe starts independently from an exact clone of the saved neutral
primal and records the common seed hash. The 15 rows are:

1. `zero`, role `required_zero_geometry_reproducibility`;
2. twelve signed coordinate endpoints at `+/-10 degrees` for each rotation
   component and `+/-5 mm` for each translation component, role
   `diagnostic_broad_axis_endpoint`;
3. `legacy_one_degree_center`, role `diagnostic_legacy_one_degree_center`; and
4. `world_x_positive_0p01_degree`, role
   `required_nonzero_world_x_0p01_degree`.

The rigid-bone guard runs before equilibrium. A failed linear boundary path is
recorded as `rigid_bone_boundary_ccd_rejection`. Soft-tissue/bone prescribed
boundary CCD failure, nonlinear Newton failure, and post-solve numerical
rejection remain distinct. There is no clipping, fallback solver, smaller
substitute, or adaptive retry. Extreme endpoint failures are domain evidence;
they neither fail the receipt nor establish whole-box feasibility.

Every accepted probe writes `state-<label>.npz` with float64 pose and complete
solved displacement. Its row binds path, SHA-256, shape, dtype, checkpoint,
input hashes, and neutral-seed hash for rendering without another solve.

Preflight `success` requires both:

1. zero pose passes equilibrium/contact/shape/FEM checks and its maximum
   Euclidean nodal displacement difference from the frozen neutral primal is at
   most `1e-6 m`; and
2. the predeclared `+0.01 degree` world-x candidate passes all numerical gates.

## Scope and admission

The rigid-bone CCD guard checks linear interpolation of pure-FEM boundary
vertex positions, matching the solver's prescribed-boundary motion. It adds no
energy, force, or derivative and does not certify a rigid rotation arc,
registered source anatomy, joint constraints, or the whole box. Soft-tissue/
bone IPC remains the mechanical contact term. Receipts keep
`anatomical_validation: false`, `whole_pose_box_validated: false`, and
`bone_bone_energy_added: false`.

The summary schema is `joint-jaw-preflight-v2`. Downstream final admission must
require exact neutral/input/contact/rigid/source hashes, the Newton contract and
production tolerances, `success: true`, `admission_mode: final_launch_ready`,
`final_launch_ready: true`, `probe_count == 15`,
`broad_axis_endpoint_count == 12`,
`geometry_reproducibility.budget_m == 1e-6`,
`zero_geometry_reproducibility_met: true`, and
`required_nonzero_world_x_0p01_degree_met: true`. It must validate the exact row
labels, roles, common seed, accepted-state hashes, and successful forward
receipts. The legacy candidate and endpoint acceptance counts remain
diagnostic.

Run from the experiment directory after a full-target neutral checkpoint is
available:

```bash
CHERRIES_NAME="Full-target jaw numerical preflight" \
CHERRIES_TAGS="joint-inverse,jaw,contact,preflight" \
uv run --frozen python src/28-run-jaw-preflight.py \
  --admission-mode final_launch_ready \
  --neutral-checkpoint <full-target-neutral-terminal.pt> \
  --contact-validation <matching-contact-validation-summary.json> \
  --output-dir <fresh-jaw-preflight-directory>
```

For intermediate evidence, use `--admission-mode diagnostic_only` and a fresh
diagnostic directory. It enforces the same numerical tests and lineage but
cannot authorize calibration or final optimization.

## Completed diagnostic 003

The normal Cherries run completed with exit 0 in
`data/jaw-preflight-spatial25-diagnostic-003/`. Its summary SHA is
`bf9a490ec699b9889f03d464c4d67fd58ac6f68a938d6c9f3e3bdc3b79bc6c0e`;
[Comet](https://www.comet.com/liblaf/apple/f475ad6f2fa2447092225806a7d08435)
records the run. Status is `passed_diagnostic_only`, with
`final_launch_ready: false`.

| Check | Observed result |
| --- | ---: |
| Zero maximum Euclidean nodal difference | 0.224882 µm |
| Zero skin-node RMS difference | 0.000042011 µm |
| Zero volume-node RMS difference | 0.000652176 µm |
| Zero Newton steps / final force norm | 3 / 9.3991e-14 |
| +0.01° Newton steps / final force norm | 5 / 7.2865e-13 |
| Minimum active gap, zero / +0.01° | 49.689 / 49.856 µm |
| Accepted states | Zero and +0.01° |
| Rejected diagnostics | All 12 extreme endpoints and legacy +1° |

Force norms above use the solver's MPa·m² units. They are not quoted in newtons.

Every probe used the same frozen 25% intermediate seed. Both accepted states
passed contact, FEM and shape checks, with no inverted tetrahedra. The legacy
pose was rejected by soft-tissue/bone boundary CCD at fraction `0.0361328125`.
This proves one small local feasible jaw direction, not a feasible broad pose
box. The final optimizer retains contact-aware trial rejection and backtracking.
The [eight rendered figures](49-jaw-domain-visuals.md) include the isolated
candidate-minus-zero motion.

Reproduce the diagnostic from this experiment directory with a fresh output:

```bash
CHERRIES_NAME='Spatial25 contact jaw preflight diagnostic 003' \
CHERRIES_TAGS=joint-inverse,jaw,contact,preflight \
uv run --frozen python src/28-run-jaw-preflight.py \
  --admission-mode diagnostic_only \
  --neutral-checkpoint data/review-snapshots/neutral-convergence-025-contact-spatial80-metric-bfgs-001-u0072-e45c7a58/terminal.pt \
  --contact-validation data/contact-validation/summary.json \
  --output-dir data/jaw-preflight-spatial25-diagnostic-reproduction
```
