# Neutral-start activation, smoothness and loss comparison

The user requested four activation parameterizations × two activation smoothness
settings × two data losses, all optimized from scratch. They selected target
height 0.20 only, so this experiment contains exactly 16 runs.

Each run independently starts with control q=0, displacement u=0, activation
B=I, and Adam moments m=v=0. `initial-state.npz` records these values, and the
runner checks them exactly. No fitted state from the earlier continuation study
is used. The only reused pieces are validated numerical implementations.

## Protocol

The [frozen protocol](00-protocol.md) specifies the 100 by 10 mesh, fixed
sides/bottom, corrected physical-J energy, fixed materials and 400 active muscle
triangles. Parameterizations are unrestricted symmetric (3 DoF), PSD
contraction-only (3), learned-axis rank-one contraction (2), and fixed-x
contraction (1), independently on each muscle triangle. Learned axes initialize
at theta=0 with strength 0, so their initial angle gradient is zero.

All runs use 1,200 Adam updates, initial learning rate .03 with .99 decay per
update, betas .9/.999, epsilon 1e-8, feasibility projection, forward tolerance 1e-10,
and maximum 250 Newton iterations. The objective is

```text
J = L2 + beta * (L2_neutral / N_neutral) * N + weight * h^2 * R.
```

L2 is mean squared vector displacement error on free top vertices. N matches
oriented target normals (equivalently unit tangents) on corresponding deformed
top segments, with fixed reference edge-length weights. R is mean squared
Frobenius difference of neighboring activation tensors B. The normal branch
uses beta=.05; L2-only uses beta=0. Smoothness-on uses the historical weight 1;
off uses 0. There is no coefficient sweep or selection in this factorial study.
At h=.20, L2_neutral=0.021548821333333332 and N_neutral=0.08415960104896235.
The smoothness-on coefficient is .04, and the normal coefficient is fixed at
.05 times the ratio of those neutral losses.

Primary within-row comparisons use the latest saved update common to all four
variants of that activation parameterization. Full endpoints and failures are
reported separately. A failed solve is retained as a failed proposal; other runs
continue from their own independent neutral starts. No fallback solver or
checkpoint continuation is used.

## Commands and provenance

Working directory:
`exp/2026/09/21/normal-matching-scratch`.

```bash
env DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CHERRIES_NAME='2D normal matching scratch: preflight validation' CHERRIES_TAGS='2d,normal-matching,scratch,validation' .venv/bin/python src/05-verify.py
CHERRIES_NAME='2D scratch factorial smoke' CHERRIES_TAGS='2d,from-scratch,normal-matching,smoke' DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/10-run.py --output 09-smoke --max-updates 5
CHERRIES_NAME='2D from-scratch activation smoothness loss factorial' CHERRIES_TAGS='2d,from-scratch,normal-matching,smoothness,4x2x2' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 uv run python src/10-run.py
```

Main Comet record:
[a22cd86609ec4ad8872a74f25cfdd065](https://www.comet.com/liblaf/apple/a22cd86609ec4ad8872a74f25cfdd065).
The run snapshots the resolved numerical source files, source hashes, gate
receipt, protocol and configuration. Source paths and versions are in
`data/10-comparison/protocol.json`. Git HEAD is
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the workspace contains existing
uncommitted research and physics work, so the source snapshots define the
numerical provenance rather than HEAD alone. Fits are deterministic neutral
initializations; random numbers are used only for derivative checks.

## Verification

The fresh preflight gate passed direct and mapped normal derivatives, full
implicit derivatives for all 16 model/regularizer/loss combinations, activation
smoothness pullbacks, and neutral-state checks. Maximum combined implicit
relative derivative error was 1.99935e-5 (threshold 3e-5). Direct/mapped normal
relative errors were 3.42e-10 / 7.68e-11. With beta 0, objectives match the
historical L2-plus-smoothness implementation exactly at the tested controls, and
gradients differ by at most 1.70e-21. The five-update smoke run completed all 16
cases and checked exact q/u/B/m/v initial values.

The independent final audit also passed. All 16 neutral receipts, common-step
states and endpoint/checkpoint pairs were checked. Recomputed position, normal,
slope, second-derivative, activation roughness, objective, area, determinant and
force metrics agree with the traces; constraints and fixed displacements pass.
Maximum force residual across common steps and endpoints is 9.015e-11.
All accepted trace states are inversion-free. All 16 common-step physical
Hessians have positive smallest eigenvalues (9.698e-06 to 4.664e-05);
this local discrete forward check is separate from inverse convergence.

Seven of the eight fresh L2 baselines reproduce the archived controls and
displacements bitwise. Learned-axis with smoothness on differs only by maximum
control error 4.747e-11 and displacement error 5.089e-13, within the explicit
roundoff comparison tolerances. The difference follows the changed floating-point
addition order for the smoothness pullback. These archive reads happen only in
post-run verification, never to initialize an optimization.

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CHERRIES_NAME='2D normal matching scratch: independent factorial verification' CHERRIES_TAGS='2d,normal-matching,scratch,verification,factorial' .venv/bin/python src/20-verify.py --input-dir 10-comparison --output 20-verification
```

Verification Comet record:
[29e794cd643c49ce82926ca363be848a](https://www.comet.com/liblaf/apple/29e794cd643c49ce82926ca363be848a).

## Results

**Activation smoothing reliably reduces activation-tensor variation. Adding
normal matching improves target-normal agreement, but does not consistently
reduce the second-derivative shape residual.** All 16 combinations were optimized
from neutral. Fifteen reached 1,200 updates; unrestricted L2 without smoothing
failed its forward solve at proposal 263, retaining accepted state 262. Its four
conditions are compared at common saved update 260; other rows use update 1,200.
The numerical cases took 451.38 seconds in total.

The full factorial is displayed as four activation rows by four condition
columns (smoothness off/L2, off/L2+normal, on/L2, on/L2+normal). Additional 4 by 2
figures overlay the two losses in each smoothness column. All comparison states
come from the independent verification receipt; no best-weight selection occurs.

### Effect of adding normal matching

Changes are relative to L2-only at the same activation model, smoothness setting
and saved update. D2 is RMS second reference-coordinate derivative of vector
displacement error, not geometric curvature. Projection is dot(u,u_target) /
||u_target||^2 and measures motion along the target pattern, not peak height.

| Activation | Smoothness | Step | Position RMS change | Normal-angle RMS change | D2 residual change | Target-projection change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Unrestricted symmetric | off | 260 | +3.537% | -34.776% | -37.276% | -30.487% |
| Unrestricted symmetric | on | 260 | +0.467% | -15.507% | +82.523% | -14.446% |
| PSD contraction-only | off | 1200 | +0.212% | -17.065% | +5.145% | +8.194% |
| PSD contraction-only | on | 1200 | +0.039% | -0.877% | +4.492% | -0.299% |
| Learned axis | off | 1200 | -0.020% | -3.792% | -23.824% | +2.791% |
| Learned axis | on | 1200 | -0.020% | -0.724% | +11.206% | +0.035% |
| Fixed x | off | 1200 | -0.004% | -1.944% | +15.026% | -1.470% |
| Fixed x | on | 1200 | +0.004% | -0.870% | +13.909% | -0.678% |

Normal-angle RMS decreases in all eight pairs. D2 decreases in only two:
unrestricted and learned-axis, both with smoothing off. It increases in all
four smoothing-on pairs. Therefore normal matching is not an interchangeable
replacement for either activation regularization or a high-frequency residual
penalty.

The learned-axis smoothing-off comparison is favorable: position RMS changes
from 0.139843 to 0.139815 (-0.020%), normal-angle RMS 24.821 to 23.880 degrees
(-3.792%), D2 18.195 to 13.861 (-23.824%), and target projection rises 2.791%.
With smoothing already on, adding normal matching gives only 0.724% lower normal
angle and increases D2 by 11.206%, while positional RMS is effectively unchanged.

The unrestricted smoothing-off apparent shape gain has a substantial amplitude
cost: normal matching reduces target projection 30.49% and raises position RMS
3.54% at the matched update. The L2 control also has four backtracking top edges
at that step and later fails. This is not a clean fit-preserving improvement.
The normal branch's completion alone does not negate those comparison limits.

### Effect of activation smoothing

Percent changes below compare smoothing-on to smoothing-off at fixed data loss
and the same row's common update. R is squared neighboring B-tensor variation.

| Activation | Data loss | R change | D2 residual change | Position RMS change | Target-projection change |
| --- | --- | ---: | ---: | ---: | ---: |
| Unrestricted symmetric | L2 | -99.575% | -77.822% | +3.044% | -24.514% |
| Unrestricted symmetric | L2 + normal | -99.459% | -35.463% | -0.011% | -7.094% |
| PSD contraction-only | L2 | -99.886% | -65.005% | +0.142% | -3.698% |
| PSD contraction-only | L2 + normal | -99.897% | -65.222% | -0.030% | -11.257% |
| Learned axis | L2 | -99.027% | -35.264% | +0.106% | -0.548% |
| Learned axis | L2 + normal | -97.310% | -5.495% | +0.107% | -3.214% |
| Fixed x | L2 | -96.848% | -2.329% | +0.040% | -4.110% |
| Fixed x | L2 + normal | -99.013% | -3.277% | +0.048% | -3.339% |

Smoothing reduces activation R by 96.85–99.90% across all eight comparisons and
reduces D2 in each. The size of the surface-residual improvement differs strongly
by parameterization. These tensor and surface metrics must remain separate.

### All 16 conditions at common steps

Length units are uncalibrated model units, not millimeters. Normal angles are
degrees. These are forward-equilibrated states, not demonstrated inverse optima.

| Activation | Condition | Step | Position RMS | Normal-angle RMS | D2 residual | Activation R | Target projection | Min J |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unrestricted symmetric | smooth-off-l2 | 260 | 0.133069 | 35.995 | 70.489 | 1.010222 | 0.25958 | 0.012230 |
| Unrestricted symmetric | smooth-off-normal | 260 | 0.137775 | 23.477 | 44.214 | 0.515449 | 0.18044 | 0.321936 |
| Unrestricted symmetric | smooth-on-l2 | 260 | 0.137120 | 27.799 | 15.633 | 0.004295 | 0.19594 | 0.632610 |
| Unrestricted symmetric | smooth-on-normal | 260 | 0.137760 | 23.489 | 28.534 | 0.002786 | 0.16764 | 0.439101 |
| PSD contraction-only | smooth-off-l2 | 1200 | 0.139696 | 28.153 | 33.669 | 1.757549 | 0.11201 | 0.245868 |
| PSD contraction-only | smooth-off-normal | 1200 | 0.139992 | 23.349 | 35.401 | 2.171621 | 0.12119 | 0.217059 |
| PSD contraction-only | smooth-on-l2 | 1200 | 0.139895 | 24.037 | 11.782 | 0.002000 | 0.10787 | 0.832477 |
| PSD contraction-only | smooth-on-normal | 1200 | 0.139950 | 23.827 | 12.312 | 0.002244 | 0.10754 | 0.823360 |
| Learned axis | smooth-off-l2 | 1200 | 0.139843 | 24.821 | 18.195 | 0.133724 | 0.10309 | 0.783753 |
| Learned axis | smooth-off-normal | 1200 | 0.139815 | 23.880 | 13.861 | 0.043290 | 0.10597 | 0.793846 |
| Learned axis | smooth-on-l2 | 1200 | 0.139992 | 24.012 | 11.779 | 0.001301 | 0.10253 | 0.874275 |
| Learned axis | smooth-on-normal | 1200 | 0.139964 | 23.839 | 13.099 | 0.001165 | 0.10256 | 0.871370 |
| Fixed x | smooth-off-l2 | 1200 | 0.140107 | 24.318 | 11.581 | 0.038042 | 0.11225 | 0.857661 |
| Fixed x | smooth-off-normal | 1200 | 0.140102 | 23.846 | 13.321 | 0.103928 | 0.11060 | 0.871500 |
| Fixed x | smooth-on-l2 | 1200 | 0.140164 | 24.019 | 11.311 | 0.001199 | 0.10764 | 0.892051 |
| Fixed x | smooth-on-normal | 1200 | 0.140169 | 23.810 | 12.884 | 0.001026 | 0.10691 | 0.876632 |

### Full endpoints and failure

Successful 1,200-step endpoints below are intentionally kept separate from the
260-step primary comparison for unrestricted activation.

| Activation | Condition | Accepted endpoint | Position RMS | Normal-angle RMS | D2 residual | Minimum J | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Unrestricted symmetric | smooth-off-l2 | 262 | 0.133008 | 35.957 | 70.513 | 0.001764 | Failed proposal 263; residual 3.191e-6 |
| Unrestricted symmetric | smooth-off-normal | 1200 | 0.137673 | 23.366 | 46.678 | 0.309409 | 1,200-update budget |
| Unrestricted symmetric | smooth-on-l2 | 1200 | 0.136948 | 28.091 | 15.011 | 0.620950 | 1,200-update budget |
| Unrestricted symmetric | smooth-on-normal | 1200 | 0.137625 | 23.358 | 30.267 | 0.426163 | 1,200-update budget |
| PSD contraction-only | smooth-off-l2 | 1200 | 0.139696 | 28.153 | 33.669 | 0.245868 | 1,200-update budget |
| PSD contraction-only | smooth-off-normal | 1200 | 0.139992 | 23.349 | 35.401 | 0.217059 | 1,200-update budget |
| PSD contraction-only | smooth-on-l2 | 1200 | 0.139895 | 24.037 | 11.782 | 0.832477 | 1,200-update budget |
| PSD contraction-only | smooth-on-normal | 1200 | 0.139950 | 23.827 | 12.312 | 0.823360 | 1,200-update budget |
| Learned axis | smooth-off-l2 | 1200 | 0.139843 | 24.821 | 18.195 | 0.783753 | 1,200-update budget |
| Learned axis | smooth-off-normal | 1200 | 0.139815 | 23.880 | 13.861 | 0.793846 | 1,200-update budget |
| Learned axis | smooth-on-l2 | 1200 | 0.139992 | 24.012 | 11.779 | 0.874275 | 1,200-update budget |
| Learned axis | smooth-on-normal | 1200 | 0.139964 | 23.839 | 13.099 | 0.871370 | 1,200-update budget |
| Fixed x | smooth-off-l2 | 1200 | 0.140107 | 24.318 | 11.581 | 0.857661 | 1,200-update budget |
| Fixed x | smooth-off-normal | 1200 | 0.140102 | 23.846 | 13.321 | 0.871500 | 1,200-update budget |
| Fixed x | smooth-on-l2 | 1200 | 0.140164 | 24.019 | 11.311 | 0.892051 | 1,200-update budget |
| Fixed x | smooth-on-normal | 1200 | 0.140169 | 23.810 | 12.884 | 0.876632 | 1,200-update budget |

### Limits of this benchmark

The global target remains strongly underfit in every condition. With the fixed
sides and bottom, the sampled h=.20 target top would bound an area 2.3332
times the reference area; the matched simulations have ratios 0.9833–0.9966.
This measured area discrepancy highlights the material/boundary/actuation
challenge in this demo. It does not prove that a better optimization cannot fit
the target, nor separate material stiffness from activation capacity.

The optimizer uses the historical rapidly decaying learning rate and a finite
budget. Flat curves at late updates are not a convergence certificate. The
physical Hessian concerns forward displacement equilibrium at fixed activation;
it is not the inverse-objective Hessian. Normal loss, activation R, positional
fit, motion and D2 measure different properties. This experiment supports
activation smoothing for controlling tensor variation, with an optional normal
term for directional agreement; it does not establish a universally better
mesh-shape loss or an intrinsic ranking of activation models.

## Artifacts and run completion

- `data/10-comparison/<activation>/<condition>/initial-state.npz`: exact neutral
  controls, displacement, B, and zero Adam moments for each run.
- `history.npz`, `checkpoint.npz`, `trace.csv`, `summary.json`: every condition's
  saved states, last accepted optimizer state, per-update metrics and stop reason.
- `data/10-comparison/unconstrained/smooth-off-l2/failure.json` and
  `failed-proposal.npz`: preserved unsuccessful proposal.
- `data/20-verification/comparison.json` and `checks.json`: independent neutral,
  source, historical-replay, metric, constraint, residual and Hessian audit.
- `data/30-figures/all-factorial-meshes.png`: explicit 4 by 4 conditions.
- `data/30-figures`: paired profile, mesh and objective/projected-gradient figures,
  plus endpoint receipts and renderer source.
- `logs/10-run.log`, `logs/20-verify.log`: full numerical/verification logs and
  Comet summaries.

Main and verification runs exited 0 after Cherries/Comet shutdown. A known
imported mesh-helper default registers unused `data/10-pork-2d`, producing a
missing-asset warning; actual outputs were independently verified. It is not a
numerical error or a reason to hide the failed unrestricted proposal.

Main Comet summary excerpt (full block in `logs/10-run.log`):

```text
name: 2D from-scratch activation smoothness loss factorial
url: https://www.comet.com/liblaf/apple/a22cd86609ec4ad8872a74f25cfdd065
cherries/cmd: .venv/bin/python src/10-run.py
cherries/start_time: 2026-09-21 02:15:23.612144+08:00
cherries/end_time: 2026-09-21 02:22:55.321578+08:00
cherries/git/sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
```

The four final figure types were visually inspected: the explicit 4 by 4 mesh
plate, paired profiles, paired internal meshes, and objective/projected-gradient
histories. The mesh plots share physical axis scales and show only the prescribed
target boundary, without inventing target interior displacements. The history
plot shows full trajectories and marks the common comparison step. Its normalized
objectives contain different terms and must not be ranked as positional fit.
Ruff checks pass for the experiment source directory.

Final rendering command, from the same experiment directory:

```bash
CHERRIES_NAME='Neutral-start normal matching: factorial figures' CHERRIES_TAGS='target-normal,normal-matching,2d,neutral-start,factorial,figures' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/30-render.py
```

Final rendering Comet record:
[c35cc52b8c4d4aaaa438c1ec9789142c](https://www.comet.com/liblaf/apple/c35cc52b8c4d4aaaa438c1ec9789142c).
The renderer exited 0 after full Cherries/Comet shutdown. The final figures are
in `data/30-figures`; earlier renders remain in `data/30-figures-initial` and
`data/30-figures-before-margin-fix`. The first revision corrected the Hessian
receipt key; the last enlarged the history plot's right margin to avoid a clipped
axis label. Neither revision changed numerical results. The final history image
was visually checked for complete labels, legend and footer.
