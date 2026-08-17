# Smile Skin Material Heuristic Sweep: Design And Smoke

## Purpose

This experiment tests a low-dimensional skin-material heuristic on one fixed
human-face `Smile` target. It does not optimize one material variable per
triangle. Every material candidate gets a fresh, independent inverse solve of
the per-muscle-tet unconstrained 6-DoF `ActivationInv`; activation and forward
state are never transferred between candidates.

The prepared input is reused from:

```text
exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu
```

It contains 228,660 points, 1,146,517 tetrahedra, 288,235 active muscle
tetrahedra, and therefore 1,729,410 inverse activation DoFs.

## Material Grid

The grid is `2 x 3`:

- minimum Young's-modulus scale in expanding regions: `1.0`, `0.25`;
- prestrain gain in contracting regions: `0.0`, `0.5`, `1.0`.

The baseline skin Young's modulus is `E0 = 0.20 MPa`; the strong local
softening candidate has a theoretical floor of `0.05 MPa`. After the calibrated
spatial diffusion, its observed minimum is `0.0748078 MPa`. The six candidate
labels are `e100-p000`, `e100-p050`, `e100-p100`, `e025-p000`, `e025-p050`,
and `e025-p100`.

For a target/rest triangle area ratio `r`, the candidate preparation first:

1. forms the signed field `s = log(r)` on every finite all-vertices-`IsFace`
   triangle;
2. applies a symmetric soft deadband
   `sign(s) max(|s| - log(1.01), 0)`;
3. caps the positive and negative severities separately at their rest-area
   weighted 99th percentiles;
4. diffuses this one signed field over the complete eligible patch by solving
   `(M + t K) u = M s`, where `M` is rest triangle area,
   `t = (5 mm)^2 / 2`, and the face boundary has zero Dirichlet data;
5. decodes positive `u` as Young's-modulus softening and negative `u` as
   isotropic in-plane prestrain.

The finite-volume conductance uses triangle centers implicitly: an interior
shared edge of length `l` has weight `3 l^2 / (2 (Ai + Aj))`; a boundary edge
has weight `3 l^2 / (2 Ai)`. This physical 5 mm scale replaces the former
same-sign mask smoothing and avoids hard per-triangle material jumps.

The material maps are:

```text
E = E0 * exp(log(EminScale) * max(u, 0) / positive_cap)
Ainv_diag = exp(0.5 * prestrain_gain * max(-u, 0)) - 1
```

Only the in-plane skin `ActivationInv` diagonal entries are set. Outside the
finite all-vertices-`IsFace` masks, Young's modulus remains `0.20 MPa` and skin
prestrain remains zero.

To keep this experiment a controlled comparison against the existing Smile
baseline, `E` and `nu = 0.49` are converted with the repository's existing 3D
isotropic Lamé convention,
`lambda = E nu / ((1 + nu) (1 - 2 nu))` and
`mu = E / (2 (1 + nu))`. In a 2D membrane this is a plane-strain-like
coefficient choice, not a claim that a thin skin shell is under plane stress.
A plane-stress conversion would change the baseline membrane physics and is
therefore reserved for a separate material-axis experiment.

## Candidate Preparation Smoke

Command:

```bash
cd /home/liblaf/Projects/liblaf/apple/exp/2026/08/17/human-face-smile-material-heuristic-sweep
DEBUG=1 \
CHERRIES_NAME="Smile heat-smoothed skin material candidate preparation" \
CHERRIES_TAGS="human-face,smile,skin,young-modulus,prestrain,heuristic,2x3,heat-diffusion,debug" \
uv run --frozen python src/10-prepare-material-candidates.py
```

The final strict DEBUG run exited successfully in 15.7 seconds. The input mesh identity was
stable (`SHA-256 8131d694...563`), and all six VTP files passed finite-value,
formula, region, range, shape, file-identity, topology-content,
material-content, solver-content, and readback validation for all
solver-consumed and heuristic fields. Expression point arrays retain expected
NaNs where the target is undefined outside its ROI; those arrays are not
consumed by Koiter.

The formal producer fixes the primary design at 1%/99%/5 mm and fixes all
calibrated gate values. `Fraction`, `Lambda`, `Mu`, `ActivationInv`, geometry,
connectivity, and `GlobalPointId` are included in the live solver-content hash;
`Fraction` must be finite and exactly one. The manifest writer rejects
NaN/Infinity rather than emitting permissive JSON.

The eligible patch contains 29,899 triangles (`0.04287998 m^2`), one connected
component, 44,495 interior edges, and 707 Dirichlet boundary edges. There are
43 nonmanifold edges elsewhere on the extracted surface; none touches the
eligible patch. For the primary 5 mm field:

- the weighted expansion/contraction caps are `0.394987` and `0.503041`;
- relative infinity residual is `1.48e-14`, with zero maximum-principle
  violation;
- interior normalized jump q99/max is `0.06020/0.16391`;
- boundary normalized jump q99/max is `0.06037/0.14069`;
- rest-area RMS attenuation is `0.78994`, and correlation with the capped input
  is `0.94154`.

The strong combined `e025-p100` candidate has area-weighted mean
`E = 0.190110 MPa`, `Ainv_diag max = 0.232116`, and minimum stress-free area
ratio `0.658714`. Its expansion/contraction sign coverage is 55.12%/44.88% of
eligible rest area.

A summary-only sensitivity sweep evaluated all 27 combinations of diffusion
scale `2.5/5/10 mm`, deadband `0.5/1/2%`, and weighted cap
`97.5/99/99.5%`. All rows were finite. All nine 5 mm rows and all nine 10 mm
rows passed the calibrated jump gates; all nine 2.5 mm rows failed at least the
interior maximum-jump gate. This supports 5 mm as the smallest tested primary
scale that clears the spatial-frequency screen. No extra sensitivity VTPs were
written.

Outputs:

- `data/10-material-candidates-manifest.json`
- `data/10-material-candidates-table.md`
- `data/10-material-candidates/skin-*.vtp`
- `logs/10-prepare-material-candidates.log`

## Inverse Interface Smoke

Command:

```bash
DEBUG=1 \
CHERRIES_NAME="Smile material factorial corners plus no-skin Agg strict one-evaluation smoke" \
CHERRIES_TAGS="human-face,smile,inverse,skin,no-skin,young-modulus,prestrain,heat-field,fresh-zero,per-tet,6dof,strict-provenance,agg,smoke,debug" \
uv run --frozen python src/20-inverse-material-sweep.py \
  --candidate-set corners-with-no-skin \
  --stage smoke \
  --inverse-max-steps 0 \
  --mandatory-baseline-steps 0 \
  --output-summary tmp/strict-smoke-v2/20-material-smoke-summary.json \
  --output-table tmp/strict-smoke-v2/20-material-smoke-table.md \
  --live-plot-dir tmp/strict-smoke-v2/live-material-smoke
```

The first multi-case attempt exposed a real environment defect: Matplotlib had
selected `TkAgg`, and Tcl object destruction from a PyTorch worker thread
terminated the process after two cases. The entrypoint now selects and verifies
the non-interactive `Agg` backend before importing the reference inverse loop.
The clean rerun completed all five cases in about 128 seconds.

The smoke covers the uniform baseline, prestrain-only corner, softening-only
corner, combined corner, and the explicit `no-skin` control. It verified for
every case:

- fresh activation RMS and maximum are exactly zero;
- no initial displacement was reused;
- 288,235 independent active tetrahedra produce 1,729,410 activation DoFs;
- forward result is `primary_success` and adjoint result is `success`;
- result VTU and one-frame temporal VTKHDF read back successfully;
- zero inverted tetrahedra, zero surface normal reversals, and no collapsed or
  excessively stretched skin triangles;
- `I + muscle ActivationInv` is SPD at the zero-activation checkpoint.

The combined `e025-p100` prestrained rest equilibrium has
`error/target = 0.90398`, `det(F)` q0.1% `= 0.8791`, and skin area-ratio
q0.1%/q99.9% `= 0.6930/1.0077`. The `no-skin` control has
`error/target = 1.0` at zero activation. These are interface-smoke diagnostics,
not optimized comparisons.

The later formal temporal scan supersedes this smoke as a physical-admissibility
test. With the same topology/material/solver-content hashes, formal
`e025-p100@0` had two inverted tetrahedra and two folded skin triangles. The
smoke and formal displacement fields differ by only `2.11e-6 m` RMS and
`2.44e-4 m` maximum, so this corner is numerically close to a local fold rather
than robustly admissible. Static material screening therefore needs repeated
independent forward solves, not one smoke result.

The smoke intentionally stopped at the step limit and is not an optimization
result. The aggregate and table are explicitly logged at their redirected tmp
paths; Cherries additionally emitted two harmless warnings while checking the
unused default screen paths.

## Formal Run Plan And Cost

Historical skin-enabled 200-step runs on this mesh took 7,106 to 7,605 seconds
per candidate. A direct six-candidate 200-step sweep would therefore cost about
12 hours and generate roughly 56 GB of temporal history, before adding the
`no-skin` control.

The staged plan is:

1. Screen `no-skin` plus all six material candidates from fresh zero for exactly
   40 optimizer steps (41 evaluations), about 3 hours and 13-14 GB.
2. Scan temporal checkpoints and construct the target-fidelity/bumpiness Pareto
   set using only physically valid states. Keep `no-skin` as a control rather
   than treating it as a material candidate.
3. Rerun from fresh zero the `no-skin` control, uniform-skin baseline, and the
   physically valid nondominated material candidates selected by Stage A for
   exactly 200 steps. Do not warm-start from screening.
4. Compare fixed-budget best states and matched-fidelity temporal checkpoints.

## Scientific Gates

- Material fields must be finite, positive where required, capped, confined to
  the intended expansion/contraction regions, and unchanged outside `IsFace`.
- Every candidate must start from exactly zero activation and displacement,
  build its own forward model, parameter tensor, and Adam optimizer, and use
  per-muscle-tet unconstrained 6-DoF activation.
- Best-state forward and adjoint solves, final forward and adjoint solves, and
  stored residual/gradient metrics must be successful and finite.
- Stage A must contain the complete ordered steps `0..40`, exactly 41
  evaluations and 40 optimizer updates, no learning-rate deviation, and no
  forward or adjoint failure. A truncated or backtracked fixed-budget path is
  invalid rather than silently ranked.
- A preliminary best state is ineligible for the Pareto set if it has any
  inverted tetrahedra, `det(F)` q0.1% below `0.2`, any surface normal reversal,
  skin area-ratio q0.1% below `0.1` or q99.9% above `10`, or a non-SPD
  `I + muscle ActivationInv` (minimum eigenvalue below `1e-6`).
- A material candidate must also keep target error within 5% of the uniform
  skin baseline at the compared checkpoint. Relative inversion counts are
  recorded as a secondary baseline diagnostic.
- The 40-step screen is only a fixed-budget ranking, not convergence evidence.
  A material change is useful only if the longer independent rerun preserves
  target fidelity while reducing bumpiness; different activations are expected
  and are not compared by direct transfer.

## Current Status

Strict candidate preparation, the five-case factorial/control inverse smoke,
the formal seven-case 40-step screen, the full 287-frame physical-prefix
analysis, the formal two-process zero-activation A1 sweep, and the `E=0.75`
A2 boundary refinement are complete.
The A1 sweep executed all 18 forwards successfully, but `e025-p100` and
`e050-p075` failed the replicated inversion/fold gates, so `safe_low=None` and
no candidate was approved for a dynamic Stage B rerun. In A2, `e075-p050` and
`e075-p075` passed, while `e075-p100` was branch-unstable with three inversions
and three folds in one replicate; `safe_low` therefore remains unset. The next
decision is either a singleton-process branch diagnostic or a newly
pre-registered rectangle with maximum prestrain gain reduced to `0.75`. See
[20-material-screen-findings.md](20-material-screen-findings.md) for the formal
results and decision. The repository `uv.lock` hash stayed unchanged during
all experiment runs.
