# Prescribed-skin forward solve with Poisson ratio 0.49

The current forward continuation uses **ν = 0.49 for fat, aponeurosis, muscle,
and skin**, as requested. It uses PNCG throughout, complete source cranium and
mandible collision, zero activation and zero bulk baseline stress. Skin stiffness
and baseline membrane resultant (N0) remain the frozen heterogeneous literature-derived fields.
Inverted volume cells are diagnostics and do not stop the solve. The independent,
unprojected free-force convergence target remains **0.000151920034752 N**.

**Converged and independently audited.** The terminal segment is
`data/simple-skin-forward-011`. The iOS review is available at
<PRIVATE_PREVIEW_URL>.

| Final quantity | Result |
| --- | ---: |
| Exact unprojected free-force norm | **0.000141868289 N** |
| Original fixed force threshold | **0.000151920035 N** |
| Inverted tetrahedra | **0 / 1,146,517** |
| Physical det(F) range | 0.374200–1.600054 |
| Surface displacement RMS | 1.431518 mm |
| Minimum active IPC-pair distance | 17.0666 µm |
| Independent soft–bone intersection check | **No intersections** |
| Negative contact weights | **0** |

The final active-pair distance is about 1,707 times the 10 nm numerical
clearance, so that constraint is inactive at this force-converged endpoint.
The tissue and contact gradient norms are each approximately 0.249168 N and
balance to the much smaller total residual above. The recorded wall time along
the retained checkpoint path is 36.81 minutes for 18,213 accepted iterations,
including setup and checkpoint overhead. This excludes discarded branches and
diagnostics and is not an isolated performance benchmark. The last resumed
segment needed two further steps; this is not a two-step solve from reference.

![Convergence and physical volume diagnostics](../data/simple-skin-forward-visuals-011/01-solver-and-volume-trends.png)

![Reference and force-converged skin at matched true scale](../data/simple-skin-forward-visuals-011/03-overlay-front.png)

## Physical model and parameter provenance

The bulk is passive polynomial Stable Neo-Hookean. Young's moduli remain
0.0112 MPa for fat, 1.693 MPa for aponeurosis and 0.012 MPa for muscle. Skin uses
the exact plane-stress version, 1 mm thickness, per-face Young's modulus
127.382–257.861 kPa and isotropic baseline resultant 30.400–80.830 N/m. These
skin fields are smooth transfers of sparse regional inverse fits, with manual
anatomical anchors and bilateral assumptions; they are not measurements on this
subject. See [the material provenance](66-skin-forward-literature.md) and the
hash-bound `simple-skin-forward-nu049-inputs-001/skin-field-manifest.json`.

Only the Poisson-ratio array changed in the new skin artifact. Its other arrays
(E, thickness, baseline resultant and triangle order) exactly match the original
artifact. ν = 0.49 is the user's common near-incompressibility assumption. The
implementation uses `mu=E/(2*(1+nu))` and
`lambda_code=lambda_classical+mu` for the polynomial SNH convention.

The repaired interior soft geometry is rebased as the FEM reference. The source
bones, outer skin and fixed nodes were not moved by that rebase. Both source
bones remain fixed at neutral jaw pose. Collision retains all 35,162 cranium and
18,948 mandible triangles and tests them against the 65,580 selected pure-soft
boundary triangles. Bonded mixed transition faces, soft-soft contact and
bone-bone contact are outside this forward model. This standalone reference has
not been admitted to the joint inverse experiment.

## Contact and solver corrections

The earlier `IMPROVED_MAX_APPROX` area-weighted contact discretization produced
negative aggregate contact energy at deformed states. Run 003 was stopped for
that contact defect, not for its 152 inverted cells. It is excluded from the
retained solution path. [Repeated reconstruction and derivative checks](75-contact-rebuild-diagnostic.md)
and [a synthetic derivative test](76-positive-ipc-contact-validation.md) support
using standard area-weighted `IPC`, which has nonnegative collision weights.
This changes the contact discretization; the corrected model was restarted from
the reference in run 004. Barrier stiffness remains 0.01 MPa, activation distance
0.1 mm, and friction is absent.

Run 004 approached the singular contact layer and stalled inside native CCD;
its process was interrupted, and its accepted checkpoint 200 was retained. PNCG
with numerical damping and short steps recovered all 98 inverted cells at that
checkpoint. Subsequent step-cap and restart changes are solver continuations,
not material or load changes. The retained ancestry is:

`004:200 → 005:2739 → 006:3343 → 008:1466 → 009:8000 → 010:2463 → 011:2`.

Run 007 is an excluded branch: the 0.5 mm cap repeatedly created femtometre gaps
and expensive CCD calls. Its energy decreased but its force did not converge.
[The solver diagnostic](79-pncg-progress-diagnostic.md) records both the overly
restrictive 10 µm cap and the aggressive-contact failure.

Run 009 adds a validated **10 nm CCD-only minimum clearance**. Barrier `dmin`
remains zero, so the contact energy, gradient and Hessian are unchanged. This
buffer does restrict the numerical feasible set for trial paths. A successful
endpoint must have a gap above the buffer and meet the same unprojected force
tolerance; projected stationarity at the buffer is not accepted. The initial
minimum active IPC-pair distance for run 009 was 17.246 µm. CCD uses Tight Inclusion with an iteration budget
of 1,000, and all accepted steps still require Armijo decrease.

Run 009 reduced force to approximately 0.00083 N, then progressed very slowly.
It ended at local step 8,784; the retained branch into run 010 begins from its
checkpoint 8,000. Run 010 strengthens Armijo's decrease coefficient from 10⁻⁴
to 0.25. [A frozen-state derivative diagnostic](80-late-pncg-cycle.md) confirms
the energy gradient and exact HVP, while the approximate directional curvature
used by PNCG underestimates the tested curvature by 60.16%. This explains the
overlong nearly alternating steps in the late trace. A separate exact-HVP
curvature probe improved quadratic-model agreement but did not establish faster
force convergence; the retained solve continues with approximate curvature and
the stronger line search. No convergence tolerance was relaxed.

Run 010 then exposed a stopping-state mismatch: upstream PNCG's convergence
state held the gradient from before the accepted update. At local step 2,463
it reported primary success, but the independent terminal check correctly
rejected its final force of 0.000191106 N. The local monitored optimizer now
recomputes the accepted-state force before every stopping decision. A
[CPU regression](../data/accepted-force-stop-validation-001/summary.json) checks
both tolerance-crossing directions and verifies that the iteration count is
unchanged. Run 011 resumed the saved accepted state and passed the original
threshold after two more steps, followed by a second terminal recomputation.

The independent [endpoint audit](../data/simple-forward-terminal-audit-011/summary.json)
rebuilds full-source contact on CPU, checks intersections and contact weights,
recomputes all tetrahedral Jacobians, verifies the ν = 0.49 skin artifact, and
checks fixed boundary motion against coordinate-scale floating-point roundoff.
It passed with no intersections, no negative weights and no inverted cells.

## Reproduction and evidence

Run from `exp/2026/09/21/joint-activation-material-mandible`:

```bash
CHERRIES_NAME='PNCG nu0.49 accepted-state equilibrium convergence' \
CHERRIES_TAGS=joint-inverse,forward,pncg,nu049,positive-ipc,continuation \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
uv run --frozen python src/68-run-simple-skin-forward.py \
  --prepared-dir data/simple-skin-forward-inputs-001/prepared \
  --geometry data/simple-skin-forward-inputs-001/geometry.npz \
  --geometry-audit data/simple-skin-forward-inputs-001/geometry-audit.json \
  --admission data/simple-skin-forward-inputs-001/admission.json \
  --skin-field data/simple-skin-forward-nu049-inputs-001/skin-field.npz \
  --skin-field-manifest data/simple-skin-forward-nu049-inputs-001/skin-field-manifest.json \
  --output-dir data/simple-skin-forward-011 \
  --common-poisson 0.49 --allow-inverted-cells true \
  --contact-collision-set-type IPC --ccd-max-iterations 1000 \
  --ccd-min-distance-m 1e-8 --pncg-restart-interval-steps 200 \
  --hessian-damping-initial 0.001 \
  --restart-checkpoint data/simple-skin-forward-010/checkpoint-terminal-step-02463.npz \
  --force-threshold-override 1.5192003475221146e-10 \
  --rtol 1e-5 --atol 1e-11 --max-steps 100000 \
  --line-search-max-steps 60 --line-search-armijo 0.25 --max-step-norm-m 0.0005 \
  --wall-cap-seconds 21600 --watchdog-seconds 60 --heartbeat-seconds 15 \
  --telemetry-interval-steps 25 --checkpoint-interval-steps 500
```

Use a new output directory for reproduction. Each run archives the dirty source,
exact command, runtime and input hashes. No commit or push was made. The force
threshold is inherited from the original corrected-model reference force of
15.1920034752 N, rather than recomputed from each resumed state's smaller force.
Code force units are MPa·m²; multiply by 10⁶ for newtons. Total energy may be
negative because of the skin-stress reference; contact energy must remain
nonnegative.

The final [PNCG run](https://www.comet.com/liblaf/apple/c2c0f967eb4b4ecaa515a1190d85746b),
[independent geometry audit](https://www.comet.com/liblaf/apple/66acde2fc48745edaea04f545d2bac81)
and [stopping regression](https://www.comet.com/liblaf/apple/faf4c4edc7d849e9bed3a55a82a43e52)
all completed with exit code zero. Their receipts and exact commands are stored
beside their source archives. The final checkpoint SHA-256 is
`fc88096e657c895b2a401e6e8601ddf774f8ef92d656ca06ef3524e25ffaf1ed`.

The [final visualization run](https://www.comet.com/liblaf/apple/592d3a0774a945c28da28b6e3dadd908)
produced eight inspected figures: stiffness, baseline resultant, cumulative
convergence/volume diagnostics, front and side displacement, matched reference
comparisons, and full-skull context. The tailnet page and served figure hashes
were verified against these local artifacts.

Force convergence, cell orientation, contact validity and anatomical plausibility
are reported separately. Even a force-converged forward state is not a proof of
mechanical stability, accurate attachments, or readiness of the large inverse fit.
