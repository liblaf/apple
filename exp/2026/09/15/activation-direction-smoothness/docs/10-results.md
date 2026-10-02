# 2D activation direction and smoothness comparison

## Main result

All four models were compared with the same L2 data loss, corrected Stable Neo-Hookean energy, fixed material parameters, and the original muscle band. The weight sweep contains **32 fits**: four activation models × four weights × two targets. The main off/on comparison uses **weight 1**. It reduces mean squared neighboring activation-tensor differences by **96.8–99.9%** at the saved endpoints; the worst raw fitting-loss increase is **6.01%**. At the shared update 250, reductions are **95.8–99.9%**, with at most **5.95%** higher fitting loss.

The learned single-axis model fits nearly as well as the free-directions contraction model in these finite Adam runs. Free activation fits best, but it uses active extension and its unregularized matrices are often invalid as pure active stretches. The fixed x-direction model has the largest fitting error.

[Open the figure gallery](../data/70-readable-glyphs/index.html) · [Full numeric table (CSV)](../data/40-analysis/comparisons.csv) · [Weight selection](../data/40-analysis/selection.json) · [Pre-run protocol](00-protocol.md)

[Aligned final shape and activation for h=0.20](80-aligned-h200.md)

[Why the free/off run stopped, and a tested continuation](100-continuation.md)

[Continuation through failed forward solves to step 1200, with updated figure](120-inexact-continuation.md): the activation update budget completed, but the retained shape remained unconverged.

[Resetting forward displacement after failure](140-reset-continuation.md): seven resets restored forward convergence; the final endpoint has lower L2 but is mechanically unstable.

## Models and fixed physics

Here B=A⁻¹ is the inverse activation tensor. Controls are independent in every one of the **400 muscle triangles**. The learned model optimizes its direction during the inverse fit; it does not freeze a direction obtained from another fit.

| Model | Parameterization | DoF/triangle | Total DoF |
| --- | --- | ---: | ---: |
| Free activation | B=I+S; S symmetric, unrestricted | 3 | 1,200 |
| Contraction-only, free directions | B=I+S; S positive semidefinite | 3 | 1,200 |
| Contraction-only, learned direction | B=I+s n(θ)n(θ)ᵀ; s≥0; n=(cos θ,sin θ) | 2 | 800 |
| Contraction-only, fixed x-direction | B=diag(1+s,1); s≥0 | 1 | 400 |

The free-directions contraction model can shorten along both principal axes. The learned-direction model shortens along one learned axis and has transverse active stretch exactly one. After Adam, PSD controls are projected by eigenvalue clipping; scalar strengths are clamped to zero from below. Angles are unconstrained.

The rectangle is 1 × 0.1 with a 100 × 10 grid (2,000 triangles), muscle at y∈[0.04,0.06], passive fat elsewhere, and fixed bottom/side nodes. Young’s moduli are 0.03 (muscle) and 0.003 (fat), with ν=0.49 throughout. These material parameters are fixed across all cases. Targets retain top displacement (0, 4 h x(1−x)) for h=0.05 and 0.20. Length units are uncalibrated model units, not millimeters.

Every solve uses the existing experiment-local corrected energy:

```text
W(F,B) = μ/2 (||F B||²_F − 2) − μ(J−1) + λ/2 (J−1)²
J = det(F)
```

The physical determinant is independent of activation. Contraction-only restricts A; surrounding elastic tissue can still expand in the solved F.

## Effective smoothness weight

```text
L_data  = mean free-top nodes ||u − u_target||²₂
R       = mean shared-edge muscle neighbors ||B_i − B_j||²_F
L_total = L_data + α h² R
```

This is squared L2 vector error, without an extra 1/2 or coordinate average. Smoothness acts on B throughout the muscle band, with **498 within-muscle neighbor edges** and no muscle/fat boundary penalty. The Frobenius norm counts the off-diagonal entry twice. Smoothing B treats all four parameterizations consistently and avoids axis-sign or angle-wrap ambiguity. Dividing L_total by h² gives L_data/h²+αR. The chosen weight is specific to this fixed mesh and normalization.

Every candidate received the full historical **1,200-update Adam budget**, initial learning rate 0.03, per-update decay 0.99, betas (0.9,0.999), epsilon 1e−8. Each run starts independently from B=I; learned θ starts at zero. There is no activation magnitude cap, extra regularizer, or inverse line search.

The selection criterion was at least 75% less R for every model and target. The smallest tested shared weight meeting this criterion at both last-valid endpoints and shared update 250, with all eight regularized fits finishing and all endpoint Hessians positive, is α=1. This is an effective tested setting, not a globally optimal weight.

| α | Minimum endpoint reduction in R | Minimum reduction at update 250 | Largest endpoint L2 increase | Full-budget fits |
| ---: | ---: | ---: | ---: | ---: |
| 0.01 | 63.67% | 60.19% | 0.60% | 8/8 |
| 0.1 | 95.16% | 93.92% | 2.95% | 7/8 |
| 1 | 96.85% | 95.76% | 6.01% | 8/8 |

Weight 0.01 reduces variation but misses the declared effectiveness threshold for some cases and has one unstable endpoint. Weight 0.1 is effective as a smoother, but free activation at h=0.20 fails on update 304 (last solved update 303). Weight 1 completes all eight fits. The unregularized free h=0.20 run fails on update 263 (last solved update 262). The smaller weight 0.01 reaches the full budget with L2/h²=0.439634, but its endpoint has a negative smallest Hessian eigenvalue (−0.00329381): it is a mechanically unstable stationary state, not a valid locally stable equilibrium. That point is retained only as diagnostic evidence. Completion alone is insufficient to judge forward validity.

![Weight sweep](../data/70-readable-glyphs/tuning-tradeoff.png)

## Fitting error: smoothness off versus on

Tables report **pure L_data/h²**, excluding the regularizer; lower is better. The starred endpoint is an early stopped run.

| Model | h=0.05, off | h=0.05, on | h=0.20, off | h=0.20, on |
| --- | ---: | ---: | ---: | ---: |
| Free activation | 0.441198 | 0.450303 | 0.442281* | 0.468871 |
| Contraction-only, free directions | 0.478325 | 0.486909 | 0.487878 | 0.489267 |
| Contraction-only, learned direction | 0.480874 | 0.489901 | 0.488904 | 0.489945 |
| Contraction-only, fixed x-direction | 0.507097 | 0.510570 | 0.490751 | 0.491145 |

All unstarred endpoints are at update 1,200. The larger-target free/off endpoint is at 262 and cannot be interpreted as an equal-budget optimum. For an equal-update comparison, the same ordering is visible at **update 250**, before either forward failure:

| Model | h=0.05, off | h=0.05, on | h=0.20, off | h=0.20, on |
| --- | ---: | ---: | ---: | ---: |
| Free activation | 0.444478 | 0.452428 | 0.443748 | 0.470156 |
| Contraction-only, free directions | 0.482028 | 0.488705 | 0.489373 | 0.490763 |
| Contraction-only, learned direction | 0.483502 | 0.491542 | 0.490792 | 0.491594 |
| Contraction-only, fixed x-direction | 0.508030 | 0.510738 | 0.492105 | 0.492481 |

![Fit and activation variation through optimization](../data/70-readable-glyphs/loss-and-roughness.png)

## Variation and physical deformation

|                              h | Model                               | R reduction with α=1 | Relative R reduction |                                                                                                                                                                                                                                                                                                                                              min physical J, off → on |
| -----------------------------: | ----------------------------------- | -------------------: | -------------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                           0.05 | Free activation                     |               99.40% |               99.26% |                                                                                                                                                                                                                                                                                                                                                       0.4660 → 0.8220 |
|                           0.05 | Contraction-only, free directions   |               99.78% |               99.65% |                                                                                                                                                                                                                                                                                                                                                       0.5826 → 0.8751 |
|                           0.05 | Contraction-only, learned direction |               99.82% |               99.78% |                                                                                                                                                                                                                                                                                                                                                       0.7669 → 0.9011 |
|                           0.05 | Contraction-only, fixed x-direction |               99.86% |               99.79% |                                                                                                                                                                                                                                                                                                                                                       0.7979 → 0.9346 |
|                           0.20 | Free activation                     |               99.56% |               99.27% |                                                                                                                                                                                                                                                                                                                                                       0.0018 → 0.6210 |
|                           0.20 | Contraction-only, free directions   |               99.89% |               99.76% |                                                                                                                                                                                                                                                                                                                                                       0.2459 → 0.8325 |
|                           0.20 | Contraction-only, learned direction |               99.03% |               98.89% |                                                                                                                                                                                                                                                                                                                                                       0.7838 → 0.8743 |
|                           0.20 | Contraction-only, fixed x-direction |               96.85% |               96.64% |                                                                                                                                                                                                                                                                                                                                                       0.8577 → 0.8921 |
| “Relative R” divides R by mean |                                     |                  B−I |                      | ²_F, so its reduction also controls for a change in overall activation magnitude. Its reduction ranges from 96.6% to 99.8%; the improvement is not solely lower activation amplitude. R is a **squared** variation measure: the corresponding neighboring-tensor RMS decreases by approximately 82.2–96.6%. Neither metric is a direct measure of surface smoothness. |

All main regularized endpoints have positive physical J; the smallest is about 0.621. Free activation still permits active extension in every muscle triangle at α=1. Without smoothing, det(B) is nonpositive in 39.5% / 60.75% of the free-model triangles for the small/large targets, respectively. At α=1 these fractions are zero, but smoothness is not an invertibility constraint. The three contraction-only models enforce B⪰I by construction.

![Smaller-target deformed meshes](../data/70-readable-glyphs/deformation-h050.png)

![Larger-target deformed meshes](../data/70-readable-glyphs/deformation-h200.png)

### Activation on the deformed shape

Updated on 2026-09-16 from the saved solutions. Each glyph sits at the mean of its triangle's deformed vertices, X+u. Its reference activation eigenvector n is transported to F n / ||F n||, where F is reconstructed from that triangle. Each glyph is now a thin, equal-length segment: 0.45-point width and full length 1.4 times the median reference muscle edge (0.0168995 model units). Length and width communicate orientation without shrinking weak activations to dots. Color encodes the signed eigenvalue of B−I, with one linear scale from −4.43036 to +4.43036 shared across all models and both targets. Both eigenmodes are retained; zero-strength modes are hidden.

These are transported material activation axes. They are not spatial stress eigenvectors; repeated activation eigenvalues have nonunique axes. In the fixed-x model, x is fixed in the reference configuration, so its displayed axis can tilt with the deformation. Every panel within a target uses the same physical bounds and equal x/y scale. The gallery includes both whole-shape views and central muscle close-ups.

![Smaller-target activation on the deformed shape](../data/70-readable-glyphs/activation-h050.png)

![Larger-target activation on the deformed shape](../data/70-readable-glyphs/activation-h200.png)

[Smaller-target muscle close-up](../data/70-readable-glyphs/activation-zoom-h050.png) · [Larger-target muscle close-up](../data/70-readable-glyphs/activation-zoom-h200.png)

## Interpretation and limits

- **A learned single axis is close to the full contraction-only tensor here.** Relative to free-directions contraction, learned-axis endpoint L2 is 0.53% / 0.21% higher without smoothing and 0.61% / 0.14% higher with smoothing for the small/large targets. This is evidence about these finite optimization trajectories. Most unsmoothed PSD activations are effectively rank one: only 2.0% / 5.5% have a second eigenvalue of B−I above 1e−6; those fractions become 11.0% / 17.75% with smoothing.
- **The axes actually learn.** For active learned-axis cells, mean absolute angle from x is 14.46° / 6.55° without smoothing and 13.19° / 6.67° with smoothing. Angles are interpreted modulo π. At zero strength, the angle gradient vanishes; x initialization therefore biases which inactive cells can begin moving. These are effective learned axes, not measured anatomical fibers.
- **No inverse convergence claim.** Adam’s final learning rate is approximately 1.75e−7; small late changes do not establish stationarity. Nonzero coordinate-dependent gradient diagnostics remain. Different model coordinates and numbers of controls affect Adam, so equal hyperparameters do not establish globally optimal model capacity.
- **Contraction amplitudes remain unbounded by the model.** At α=1 the largest B singular values among contraction-only cases are about 1.82–3.63, allowing active principal stretches as low as about 0.275. These are synthetic model choices, not physiological measurements.
- **Smoothing B does not prove smooth or realistic tissue deformation.** The near-incompressible, fixed-boundary parabola target retains a substantial fitting mismatch. We do not label it mathematically unreachable without proof. Physical det(F), activation det(B), and visible shape quality remain distinct checks.

## Validation

Independent checks cover all 32 case endpoints and 3,690 saved displacement/control states. Results are in [baseline checks](../data/20-baseline-checks/checks.json) and [regularized checks](../data/21-regularized-checks/checks.json), with [saved-state and artifact checks](../data/50-artifact-checks/checks.json).

- All six pre-existing unsmoothed baseline endpoints reproduce the earlier controls and displacement arrays **exactly** (maximum absolute difference zero).
- Full implicit fitting-plus-smoothness directional finite differences pass for every model; maximum relative error is **8.62e−7**. Standalone control-pullback and smoothness-gradient maximum absolute errors are **2.57e−10** and **5.73e−10**.
- All requested endpoint residuals are at most **9.76e−11**; fixed boundary displacement errors are zero. The retained saved states have positive physical J.
- All saved contraction-only activation eigenvalues satisfy B⪰I within roundoff. Worst minimum eigenvalue is 1−6.7e−16; learned/fixed secondary components are at most 6.54e−16.
- Every selected α=1 endpoint passes the positive-Hessian test for local forward stability; the smallest algebraic eigenvalue among these eight cases is **8.81e−6**. This is a forward-stability check, not inverse convergence.
- The full sweep identified one unstable endpoint: free activation, h=0.20, α=0.01. It is excluded from stable-solution interpretation.
- Sparse ARPACK failed to converge on an ill-conditioned sweep Hessian. The regularized endpoint diagnostic was explicitly changed to dense symmetric `scipy.linalg.eigh(..., subset_by_index=[0,0], driver="evr")`; the maximum eigenpair residual is 2.24e−15. No forward simulation or activation history was changed.
- The deformed activation render passes a rotation/shear/translation check and an independent audit of all 16 displayed cases: center and signed-strength errors are zero; the maximum transported-direction error is 2.32e−15. See the [glyph audit and per-case geometry](../data/70-readable-glyphs/glyph-audit.json).
- Ruff, formatting, source compilation, report links, and final figure layouts were checked. Figures show identical physical axes within each target, and label stopped or unstable results.

## Reproducibility and run records

Working directory: `exp/2026/09/15/activation-direction-smoothness`. The original working tree already contained unrelated changes; this work adds only this experiment group and does not commit or push. Git HEAD was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`. Every sweep directory includes the executed source snapshot and hashes in `protocol.json`; the six historical baseline endpoints reproduce the earlier saved controls and displacements exactly.

Use a fresh output name when rerunning; the runner refuses to overwrite existing results. For each `(OUT, WEIGHT)` pair `(tune-w0,0)`, `(tune-w001,0.01)`, `(tune-w01,0.1)`, `(tune-w1,1)`, the computation command was:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
CHERRIES_NAME="2D activation smoothness weight $WEIGHT" \
CHERRIES_TAGS="2d,activation,learned-direction,l2,physical-volume,smoothness-sweep" \
uv run python src/10-run.py --output "$OUT" --weights "$WEIGHT"
```

The baseline used the descriptive name “2D activation comparison without smoothness”. Cherries Git commits were explicitly disabled. The full default snapshot interval is 10 updates; each final or failed-run last-valid state is also retained. `trace.csv` contains every successful update, while `failed-proposal.npz` separates the failing proposal from the retained solution.

```bash
DEBUG=1 uv run python src/40-analyze.py
DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python src/20-verify.py --input-dirs tune-w0 --output 20-baseline-checks
DEBUG=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run python src/20-verify.py --input-dirs tune-w001,tune-w01,tune-w1 --output 21-regularized-checks
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  uv run python src/30-render.py --selected-weight 1
```

Comet git-patch collection encountered the dirty repository’s Git LFS filter and delayed shutdown of the numerical jobs. Local numerical artifacts are the evidence source. Renderer git-patch autologging was disabled through the documented installed SDK configuration; explicit source snapshots/hashes remain. Final Comet status and summary receipts are recorded below.

### Cherries / Comet receipts

All four numerical processes exited with code **0**, after their Cherries shutdown hooks. Each produced a Comet summary, but all four reported incomplete environment-detail collection and a final **“Failed to log run in comet.com”** warning. Complete remote synchronization is therefore not claimed. The local snapshots, traces, checkpoints, terminal logs, and checks above are retained.

[Full Comet summary blocks](../data/40-analysis/comet-summaries.txt)

| Weight | Comet run (remote synchronization incomplete) | Terminal receipt |
| ---: | --- | --- |
| 0 | [Run](https://www.comet.com/liblaf/apple/b12620632d5946d69f9d43ceeeda14d9) | [Log](../logs/tune-w0-terminal.log) |
| 0.01 | [Run](https://www.comet.com/liblaf/apple/48751e3f5724468dbf6a694378f00353) | [Log](../logs/tune-w001-terminal.log) |
| 0.1 | [Run](https://www.comet.com/liblaf/apple/4755c9ccb8404f4ca5b7568c0670f607) | [Log](../logs/tune-w01-terminal.log) |
| 1 | [Run](https://www.comet.com/liblaf/apple/89f75b934b0146d8b17fca279b75bcaf) | [Log](../logs/tune-w1-terminal.log) |

The legacy mesh import registers an unused `10-pork-2d` output path and emits a missing-artifact warning at shutdown. It is unrelated to this study’s requested outputs; all 32 requested case directories are complete.

The original figure render also exited 0: [Comet run](https://www.comet.com/liblaf/apple/6bcc5825dcc448d797dbdccdcbdd5018), [terminal receipt](../logs/render-delivery-terminal.log). The scatter coordinates and connecting curves were audited against the same fit/roughness data, and final label spacing was inspected.

The **2026-09-16 deformed-activation render** exited 0: [Comet run](https://www.comet.com/liblaf/apple/9aa4fc3ef9264c86b68b44245df93ae9), [terminal receipt](../logs/render-deformed-terminal.log). Its [gallery](../data/60-deformed-figures/index.html) contains eight PNG/PDF figure pairs and 16 glyph-geometry archives. The numerical fits are the saved solutions above. Whole-shape and close-up activation layouts were visually inspected for both targets. The displayed target parabola now uses the full reference domain x∈[0,1], matching the fitting target.

The **readable-glyph revision** uses constant-length, thin segments and a shared signed-strength colorbar: [updated gallery](../data/70-readable-glyphs/index.html), [Comet run](https://www.comet.com/liblaf/apple/8d04cb6861d44cd687e163e184fc6636), [terminal receipt](../logs/render-readable-glyphs-terminal.log), and [delivery checks](../data/70-readable-glyphs/delivery-checks.json). All 16 saved glyph archives match the previous render array-for-array. The normal Cherries run completed with exit code 0 and generated all eight PNG/PDF pairs. The Local plugin reported log-asset snapshot errors; the direct terminal log, executed source snapshot, figures, and numerical glyph archives are present and verified.

```bash
COMET_AUTO_LOG_GIT_PATCH=false OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
CHERRIES_NAME='Readable activation axes on deformed 2D muscle' \
CHERRIES_TAGS='2d,activation,deformed-glyphs,smoothness,figures' \
uv run python src/30-render.py --output 70-readable-glyphs \
  > logs/render-readable-glyphs-terminal.log 2>&1
```
