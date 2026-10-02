# Constraining muscle activation to reduce surface bumps

Controlled nonlinear tetrahedral experiments · 6–7 September 2026 · Apple

**A fixed fiber direction plus spatially smooth scalar contraction is the strongest candidate in this study.** Bounding a general activation tensor is insufficient to prevent fitting surface noise. Restricting each tetrahedron to one contraction scalar helps, but that scalar can still oscillate between neighbors. Penalizing those differences removes most of the excess surface roughness while preserving the intended large-scale motion. A magnitude penalty provides an additional preference for weak activation; its benefit is smaller and depends on the target.

Across three 5% noise samples, F-S and F-MS reduce excess high-pass roughness by about **88%** relative to Raw6, while retaining about **99.7–100%** of the intended low-frequency amplitude. They accept roughly **4.6% D** error against the noisy observations and recover the clean shape more accurately. These are measured endpoint comparisons; convergence qualifications are reported below.

[Methods](#what-was-tested) · [Noise ablations](#noise-fitting-versus-useful-deformation) · [Held-out checks](#held-out-noise-and-initialization) · [Mesh check](#check-against-a-finer-physical-mesh) · [Downloads](#records-and-downloads)

This conclusion comes from a three-layer block with known fibers and synthetic targets. It establishes a controlled mechanism and a candidate activation prior. It does not establish physiological activation or demonstrate that the previous facial folds are fixed.

## What was tested

The block has footprint 1 × 1 and thickness 0.1: bottom fat 0.04, muscle 0.02, top fat 0.04. Only the bottom is fixed; the sides and top are free. The fat and muscle moduli are 0.003 and 0.03 MPa, with Poisson ratio 0.49. The inverse mesh contains 34,560 tetrahedra, including 6,912 active muscle tetrahedra, and 6,875 vertices. All methods observe the same top surface and use the repository's `StableNeoHookeanActive` energy and implicit adjoint. There is no skin, surface-smoothing objective, added contact energy, or post-processing of the fitted activation.

Let `P = f fᵀ` for a unit reference fiber and `Q = I − P`. The scalar model uses

```text
A_inv(a) = exp(a) P + exp(−γ a) Q
A(a)     = exp(−a) P + exp( γ a) Q
0 ≤ a ≤ −log(0.65),     γ = 1/2 in the main comparisons.
```

The active natural fiber stretch is `exp(−a)`, so positive `a` means contraction even though the fiber eigenvalue of `A_inv` increases. This constrains the active natural strain; the actual solved fiber stretch also depends on loads and surrounding tissue. With γ = 1/2, the transverse expansion is determined by volume preservation; it is not an independently fitted transverse activation. The maximum permitted natural shortening is 35%, an experimental cap rather than a measured physiological limit. The contraction-only model follows the general idea of a prescribed reference fiber and scalar activation used in muscle simulation; the constitutive law here remains this repository's law. [Teran et al., 2005](https://graphics.cs.wisc.edu/Papers/2005/TSBNLF05/muscles_tvcg_2005.pdf).

| Method | Per-tet control | Purpose |
| --- | --- | --- |
| Raw6 | Six unrestricted entries of symmetric `A_inv − I` | Historical control freedom |
| G6 | Six entries of symmetric `H`, `A_inv = exp(H)`, bounded Frobenius norm | Positive definite, bounded general activation |
| G | Five trace-free entries of `H`, same norm bound | General isochoric activation |
| F | One bounded contraction scalar, known fiber `f = (1,0,0)` | Fiber restriction |
| Shared | One contraction scalar for the entire muscle | Deliberately restrictive reference |

Raw6 and G6 have 41,472 scalar controls, G has 34,560, F has 6,912, and Shared has one. G and F each have four variants: no penalty, magnitude only (M), smoothness only (S), and both (MS). The general norm bound is `||H||F ≤ √(3/2) a_max`; the fiber model is therefore a subset of G with the same cap. Raw6 versus G6 changes positivity, parameterization, and bounds together. G6 versus G isolates active-volume freedom within the bounded parameterization. G versus F isolates the fiber-aligned, contraction-only subset under the shared isochoric assumption.

The minimized objective is

```text
J = area_mean(||u − u_target||²) / D² + λm Rm + λs Rs
Rm = Σ Ve ae² / (Vmuscle a_ref²)
Rs = ℓ² Σ face_edges (Sij / dij) (ai − aj)² / (Vmuscle a_ref²)
a_ref = −log(0.8),     ℓ = 0.1.
```

The face graph includes each pair of face-sharing active tetrahedra once. This penalty encourages small jumps; it does not impose exact continuity of a per-tetrahedron field. Even F still has 6,912 control scalars for only 625 observed surface points, so the fiber restriction alone leaves considerable freedom. For G, replace `a²` and `(ai−aj)²` by `||H||F²/(3/2)` and `||Hi−Hj||F²/(3/2)`. These penalties agree exactly on the constant-fiber subspace. Enabled weights are 0.01 in the 33-fit screen. The assembled MS strength comparison uses equal weights of 0.001, 0.01, 0.1, and 1; it is a diagonal sweep, not a complete two-parameter frontier. Zero penalty is supplied by the G and F screen rows.

Three targets separate the questions:

1. **Clean:** a forward equilibrium generated with `a_true = 0.10 + 0.10 exp(−r²/(2 × 0.2²))`, centered on the block. Its top displacement RMS is `D = 0.007783240962` in the geometry's length units.
2. **Noisy:** the clean target plus reference-normal cosine-product noise at wave numbers 2–3, with area RMS `0.02 D`. Only the noisy top is fitted; the clean target is retained for evaluation. Three additional seeds at `0.05 D` test robustness at frozen weights.
3. **Smooth mismatch:** the clean target plus an upward `16x(1−x)z(1−z)` field with RMS `0.5 D`. This tests a request for extra motion. It is not certified unreachable.

The original design draft is preserved. The [execution protocol](../docs/12-execution-protocol.md) records the initial scope and the change from noise wave numbers 4–6 to 2–3, giving at least eight lateral cells per wavelength. The broader draft's anatomical face study, five-seed full factorial, ring toy, spatially varying fiber errors, material-mismatch sweep, and G6-MS frontier were not part of this completed block study.

## How bumpiness is measured

The main metric is the area-weighted RMS of `HP[(u − u_clean) · n0] / D`, where the high-pass operator subtracts a reflected Gaussian low-pass filter of fixed physical length 0.06. Lengths 0.03 and 0.12 and the central surface crop are also reported. The target error and raw displacement high-pass are separate measurements: genuine target curvature must not automatically be counted as an artifact.

For the smooth mismatch target, excess high-pass relative to the original clean shape includes some deliberately requested motion. Its value is therefore a deformation diagnostic rather than a stand-alone artifact score. The report also supplies high-pass residuals relative to the actual target and target-reference values.

Retained motion is the projection of the low-pass fitted displacement onto the low-pass clean displacement, with both total and spatial-mean-removed versions. The projection residual measures pattern error. For the mismatch case the reference is its prescribed noiseless target. A large amplitude alone is insufficient: Shared retains much of the amplitude but misses the spatial pattern.

## Forward mechanism check

On a 48 × 10 × 48 mesh, sinusoidal activation patterns at frequencies 1 and 4 have equal volume-weighted mean and RMS. Their surface responses are measured relative to a uniform activation equilibrium. At amplitude 0.03, the higher-frequency activation produces 72.5% less total surface response but 70.1% more high-pass response at length 0.06 and about 166% more surface Laplacian RMS. Halving the activation amplitude approximately halves the response.

![Activation frequency versus surface response](../data/10-frequency/frequency-response.png)

The high-pass ratio depends on the physical cutoff: high/low-frequency ratios are 2.42 at length 0.03, 1.70 at 0.06, and 0.655 at 0.12. Thus finer activation can increase a specified measure of bumps despite a smaller overall response, but this is not a cutoff-independent statement that all high frequencies are amplified. All nine forward solves succeeded, with minimum deformation determinant 0.9596. [Full forward measurements](../docs/15-frequency-findings.md).

## Noise fitting versus useful deformation

The 2% noise screen gives the following endpoints. Errors and excess high-pass RMS are percentages of the clean displacement RMS D. Retained amplitude is the spatial-mean-removed low-frequency projection. “Stationary” means the declared projected KKT threshold was reached; budget endpoints remain optimization-limited.

| Method | Noisy target error (% D) | Clean error (% D) | Excess HP (% D) | Retained amplitude | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw6 | 0.1424 | 2.0038 | 1.1572 | 99.999% | 160-step budget |
| G6 | 0.1627 | 2.0031 | 1.1560 | 99.995% | 160-step budget |
| G | 0.0752 | 1.9977 | 1.1552 | 100.000% | stationary |
| G-M | 0.6534 | 1.9286 | 1.0561 | 99.842% | stationary |
| G-S | 0.8226 | 1.4134 | 0.7617 | 99.999% | 160-step budget |
| G-MS | 1.0518 | 1.4838 | 0.7326 | 99.831% | stationary |
| F | 0.4466 | 1.8704 | 1.0620 | 100.000% | 160-step budget |
| F-M | 1.4483 | 0.9799 | 0.4337 | 99.702% | stationary |
| F-S | 1.8439 | 0.5318 | 0.1475 | 99.993% | stationary |
| F-MS | 1.8776 | 0.6083 | 0.1417 | 99.691% | stationary |
| Shared | 17.5222 | 17.4075 | 0.5272 | 96.862% | stationary |

**The decisive matched comparison is F-MS versus G-MS:** both reach projected stationarity. F-MS has 80.7% less excess high-pass error and 59.0% less clean-target error, while its error against the noisy observation rises from 1.052% D to 1.878% D. Both retain over 99.6% of the intended low-frequency amplitude. Thus the fiber-aligned, contraction-only restriction adds value even after a general isochoric tensor has magnitude and smoothness penalties.

Against Raw6, F-MS reduces excess high-pass RMS by 87.8% and the common log-tensor neighbor jump by 95.7%. Its clean-target error falls from 2.004% D to 0.608% D. Raw6 exhausts its iteration budget, so this is a comparison of measured endpoints, not a certificate comparing global optima. Raw6 already fits almost all the observation noise; the smaller training residual is accompanied by worse recovery of the clean shape.

**Smoothness does most of the additional work within the fiber family.** F alone still produces strongly varying neighboring controls. F-S reduces its excess high-pass RMS from 1.062% D to 0.1475% D. Adding magnitude (F-MS) lowers this only another 3.95%, while raising clean error from 0.5318% D to 0.6083% D and slightly attenuating the intended motion. F alone is budget-limited; F-S and F-MS are stationary.

Magnitude alone does help this noisy case: F-M reduces F’s high-pass error by 59.2%, whereas G-M reduces G’s by only 8.6%. It is therefore inaccurate to dismiss magnitude regularization entirely. Its effect is target- and control-space-dependent, and it does not substitute for spatial smoothness.

![Excess normal displacement height fields](../data/50-analysis/noisy-excess-height-fields.png)

These height fields show error relative to the clean shape, using the same vertical and color scale. They are error visualizations, not rendered physical geometry.

![Noisy-target surface high-pass residual maps](../data/50-analysis/noisy-residual-highpass-maps.png)

The maps above use residuals relative to the noisy target, so a denoised solution retains visible residual structure. The principal artifact score in the table instead uses the known clean target. The line profiles below show the clean signal, noisy observations, and fitted motion directly.

![Normal displacement and high-pass line profiles](../data/50-analysis/noisy-z-half-line-profiles.png)

![Activation field diagnostics](../data/50-analysis/noisy-muscle-activation-maps.png)

The activation panels identify their quantity individually: offset norm, log-strain magnitude, or scalar contraction. Their shared color scale is not a claim that these quantities are identical. Quantitative cross-model comparisons use the common symmetric log tensor when the activation map is positive definite.

## Clean target and identifiability

The aligned fiber model retains the generating motion. F-S has clean RMS error 0.0589% D and excess high-pass error 0.00681% D; F-MS has clean RMS error 0.3243% D and excess high-pass error 0.0193% D. F-S introduces less bias here. Magnitude alone increases clean-target excess high-pass error in both the G and F families at weight 0.01: shrinking the controls can itself change the desired shape.

Surface fit does not identify the internal activation field. In the clean test, Raw6 fits the surface to 0.1322% D but 45.0% of its volume-weighted squared log-tensor magnitude lies outside the known generating fiber line. This diagnostic drops to 22.4% for G6, 17.6% for G, and 0.955% for G-S; it is zero by construction for F. These quantities measure agreement with the synthetic generating direction, not physiological plausibility on unknown anatomy. Raw6’s normalized log magnitude is not necessarily larger than that of the successful fiber model: excess freedom and spatial variation matter in addition to overall activation size.

![Common log-tensor and fiber-direction diagnostics](../data/50-analysis/fiber-basis-log-diagnostics.png)

Shared is too restrictive: its clean error is 17.41% D even though its low-frequency amplitude is about 97%. Its low-pass pattern residual exposes the missing localized contraction. Retaining amplitude alone would wrongly rate this case as adequate.

## Held-out noise and initialization

At the same fixed weights, three new noise seeds have normal-displacement noise RMS 5% D. F-S uses byte-identical copies of the existing held-out fixtures. The table reports the arithmetic mean over these three paired seeds; n = 3 is a limited robustness check, not a population estimate.

| Method | Noisy target error (% D) | Clean error (% D) | Excess HP (% D) | Retained amplitude | Stationary endpoints |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw6 | 0.1200 | 4.9991 | 2.8745 | 100.000% | 0/3 |
| G-MS | 2.1777 | 3.5195 | 1.8219 | 99.832% | 3/3 |
| F-S | 4.6015 | 1.3169 | 0.3512 | 99.996% | 2/3 |
| F-MS | 4.6293 | 1.3037 | 0.3323 | 99.694% | 2/3 |

Relative to each paired Raw6 endpoint, F-S reduces excess high-pass RMS by 86.4–89.3% (mean 87.8%) and clean-target error by 72.7–74.7% (mean 73.7%). F-MS gives 87.1–89.9% less high-pass RMS (mean 88.4%) and 73.1–74.9% less clean-target error (mean 73.9%). These are paired percentage reductions, not reductions inferred from unpaired averages.

F-MS has slightly lower mean clean error and excess high-pass at this noise level; F-S fits the noisy target slightly better and preserves amplitude more accurately. F-S also had lower clean error in the clean and 2% cases. The evidence supports a smooth fiber-contraction field as the primary prior, with magnitude as a tunable secondary preference. It does not establish a universally best magnitude weight.

![Three paired held-out noise samples](../data/50-analysis/smoothing-heldout-comparison.png)

Raw6 exhausts all three 240-step budgets; G-MS is stationary in all three. F-S and F-MS each have two stationary endpoints and one stalled endpoint in the original runs. Selected tighter-tolerance checks are described below and kept separate from this table.

Changing the first held-out seed’s initialization from zero to 0.08 barely changes F-MS: clean error changes by −0.0149%, excess high-pass by −0.0217%, and noisy-target error by +0.000821%, all relative to their respective original metric. Raw6’s noisy-target error improves by 25.2%, but its clean error and high-pass RMS change by only −0.0137% and +0.00234%. The observed denoising contrast survives this one additional initialization; this is not a multi-start global optimization study.

## Check against a finer physical mesh

A separate clean forward target was generated on a 48 × 20 × 48 mesh (276,480 tetrahedra). The fine target was restricted to exactly coincident nodes on the original mesh, and the inverse fit used only the restricted 625-node top surface. After fitting, each coarse piecewise-constant activation tensor was replayed unchanged in its fine child tetrahedra. All 55,296 fine active tetrahedra lie wholly inside their selected coarse parent; every coarse active tetrahedron has eight children. The replay adds no control parameters.

| Method | Coarse target-fit RMS (% coarse D) | Fine replay error (% fine D) | Fine excess HP (% fine D) | Fine min det(F) | Coarse inverse status |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw6 | 0.3276 | 12.9704 | 1.6882 | 0.7511 | step_budget |
| F-MS | 1.9230 | 3.4777 | 0.3972 | 0.9785 | line_search_stalled |

![Coarse fit and refined replay comparison](../data/50-analysis/refinement-comparison.png)

Raw6 achieves the better coarse fit, but its saved control transfers poorly to the refined forward solve. F-MS has 73.2% lower fine-mesh shape error and 76.5% lower fine-mesh high-pass error. The fine replay uses the same physical filter length 0.06 and the refined clean target for evaluation. Coarse errors use the restricted target’s displacement RMS; fine errors use the fine target’s RMS, so percentages across those columns have slightly different denominators.

Fresh rest-start and target-initialized fine solves differ by only 2.68 × 10⁻⁵ D for Raw6 and 9.86 × 10⁻⁷ D for F-MS. That branch discrepancy is much smaller than the replay errors. Neither coarse inverse reached the KKT threshold: Raw6 ended at 0.01058 and F-MS at 0.00190; no fine-mesh reoptimization was performed. The comparison supports the combined F-MS prior for these saved controls and does not isolate its separate components. This shows substantially greater sensitivity to coarse discretization for the saved Raw6 endpoint in this example. It remains a single refined target with the same material law and known fiber construction, not an anatomical validation or a convergence-rate study. [Full refinement records](../docs/40-refinement-findings.md).

## Regularization strength and the price of smoothness

The diagonal MS sweep changes magnitude and smoothness weights together. All values below come from the original runs; tighter-tolerance repeats are reported separately. Errors are percentages of D.

| Method | λm = λs | Target error (% D) | Clean error (% D) | Excess HP (% D) | Retained amplitude | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| G-MS | 0.001 | 0.412 | 1.845 | 1.0367 | 99.988% | stationary |
| G-MS | 0.01 | 1.052 | 1.484 | 0.7326 | 99.831% | stationary |
| G-MS | 0.1 | 3.256 | 2.897 | 0.2917 | 97.612% | stalled |
| G-MS | 1 | 21.431 | 21.345 | 0.2986 | 78.974% | stalled |
| F-MS | 0.001 | 1.693 | 0.814 | 0.3004 | 99.969% | stationary |
| F-MS | 0.01 | 1.878 | 0.608 | 0.1417 | 99.691% | stationary |
| F-MS | 0.1 | 3.662 | 3.122 | 0.0928 | 96.981% | stalled |
| F-MS | 1 | 24.707 | 24.628 | 0.2529 | 75.592% | stalled |

For F-MS, weight 0.1 is smoother than 0.01 but introduces more signal bias; weight 1 reduces retained amplitude below 80% and raises the clean error to about 25% D. G-MS also drops below 80% amplitude at weight 1. Stronger penalties do not monotonically improve the excess-error metric because attenuating the intended shape creates error too.

The 0.01 setting was fixed before the new-seed tests. It is a useful starting point under this normalization, not a transferable constant for every mesh, muscle, or observation mask. The plot shows actual sampled endpoints and flags nonstationary cases. There is no interpolated claim of superiority at exactly matched target error.

![Fit versus clean-referenced surface high-pass error](../data/50-analysis/fit-versus-excess-highpass.png)

For mismatch targets, the clean-referenced vertical axis contains deliberately requested motion. The companion plot uses the actual target residual, and the target-reference plot shows how much high-pass content was prescribed.

[Target-residual comparison](../data/50-analysis/fit-versus-target-residual-highpass.png) · [Target high-pass references](../data/50-analysis/target-excess-highpass-references.png)

## Smooth mismatch: constraints can reject requested motion

Adding a smooth upward displacement of RMS 0.5 D exposes a large fit cost. The following values are measured endpoints, not a proof of the best attainable fit. “Total amplitude” includes the spatial mean; the pattern residual is the low-pass vector projection residual, normalized by the original clean D.

| Method | Target error (% D) | Total amplitude | Demeaned amplitude | Pattern residual (% D) | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| Raw6 | 0.187 | 100.00% | 100.00% | 0.031 | 160-step budget |
| G6 | 0.232 | 100.00% | 100.00% | 0.033 | 160-step budget |
| G | 25.737 | 91.33% | 100.55% | 22.737 | 160-step budget |
| G-MS | 27.201 | 91.06% | 100.48% | 23.740 | stalled |
| F | 48.062 | 84.46% | 103.39% | 42.841 | 160-step budget |
| F-S | 48.822 | 83.93% | 102.95% | 43.375 | stalled |
| F-MS | 48.828 | 83.67% | 102.63% | 43.247 | stalled |
| Shared | 53.672 | 80.91% | 97.49% | 46.199 | stationary |

The volume-changing Raw6 and G6 endpoints fit this target much more closely than the isochoric G/F endpoints. The restricted models retain much of the original contraction pattern while missing the extra upward motion. For example, F-MS has a demeaned amplitude above 100% but a 43.25% D low-pass pattern residual. This again shows why amplitude alone is an inadequate quality criterion.

F-MS has smaller high-pass displacement relative to the original clean shape here, but it also rejects deliberately prescribed motion. That smaller number is not evidence that artifacts were removed from the mismatch target. None of these results certify unreachability or a capacity limit: all methods except Shared remain nonstationary, and F remains strongly nonstationary.

![Smooth-mismatch displacement maps](../data/50-analysis/mismatch-normal-maps.png)

## Sensitivity to fibers, shortening bounds, and transverse stretch

The remaining tests fit the aligned clean target while changing one prior assumption at a time: coherent fiber rotations of 10° and 25°, shortening caps of 10% and 20%, and γ = 0 (fixed transverse natural stretch). The 20% cap reproduces the 35%-cap result because the fitted maximum shortening is about 18%. A 10% cap attenuates the required motion. Rotating fibers or changing the active-volume convention also produces substantial residual error at the saved endpoints.

These are deliberately mismatched priors against a target generated with aligned, isochoric fibers. They measure sensitivity and optimization behavior; fitting this target cannot choose the physiologically correct fiber field or active-strain law. Several original endpoints stalled, including the 25° and γ = 0 cases at large KKT residuals. The tighter-tolerance results below distinguish any numerical improvement from the original constraint change.

| Prior setting | Original clean error (% D) | Tightened clean error (% D) | Tightened KKT | Tightened status |
| --- | ---: | ---: | ---: | --- |
| Aligned, 35% cap, γ = 1/2 | 0.3243 | — | — | Original stationary |
| 20% cap | 0.3243 | — | — | Original stationary |
| 10% cap | 23.8383 | 23.8383 | 0.000894136 | projected_stationary |
| 10° fiber rotation | 24.1622 | 24.1623 | 0.000874885 | projected_stationary |
| 25° fiber rotation | 54.0888 | 52.0934 | 0.0991352 | step_budget |
| γ = 0 | 38.0434 | 38.0349 | 0.0833886 | line_search_stalled |

The 10° and 10%-cap endpoints reach the computed projected-stationarity threshold with almost unchanged shape error. The 25° endpoint improves from 54.09% D to 52.09% D error but remains strongly nonstationary. Its surface changes by 8.57% D during the extra 100 steps, so the original result was materially affected by optimization. The γ = 0 endpoint remains stalled with KKT 0.0834. Neither difficult case establishes a fit floor or proves that a particular prior is incapable of fitting the target.

## What the tighter numerical checks changed

| Endpoint | Original KKT | Tightened KKT | Surface change (% D) | Tightened status |
| --- | ---: | ---: | ---: | --- |
| cap-0.1 / F-MS | 0.00460952 | 0.000894136 | 0.007073 | projected_stationary |
| fiber-10deg / F-MS | 0.00362663 | 0.000874885 | 0.009722 | projected_stationary |
| fiber-25deg / F-MS | 0.52839 | 0.0991352 | 8.574642 | step_budget |
| noise005-seed20260917 / F-MS | 0.00110405 | 0.00067205 | 0.000413 | projected_stationary |
| strength-0.1 / F-MS | 0.00728366 | 0.00109828 | 0.006934 | line_search_stalled |
| strength-0.1 / G-MS | 0.0041524 | 0.000822949 | 0.006373 | projected_stationary |
| strength-1 / F-MS | 0.031648 | 0.00678433 | 0.008738 | line_search_stalled |
| strength-1 / G-MS | 0.0260514 | 0.00171814 | 0.005445 | line_search_stalled |
| transverse-natural-stretch-fixed / F-MS | 0.0579602 | 0.0833886 | 0.088615 | line_search_stalled |

Four of nine tightened endpoints reach the computed threshold; four still stall and the 25° case exhausts its extra budget. The four strong-penalty endpoints and the selected 5% noisy endpoint change by at most 0.00874% D. Their original tradeoff is stable despite incomplete stationarity in several cases. The larger changes under wrong fibers are reported separately above.

Very small directional derivatives require care. The original tightened checks use central differences at Euclidean coordinate steps 0.001 and 0.0003. The cap, 25°, and γ = 0 audits use feasible interior-coordinate directions and therefore are not full-gradient audits. The 10° endpoint has large small-step discrepancies even though its computed KKT passes. A separate fixed-control audit expands the difference steps without changing any fitted controls:

| Endpoint | FD relative error at 0.01 | At 0.003 | At 0.001 | At 0.0003 |
| --- | ---: | ---: | ---: | ---: |
| strength-0.1/F-MS | 0.063% | 0.461% | 2.554% | 18.291% |
| strength-1/F-MS | 0.281% | 0.983% | 4.559% | 12.677% |
| strength-1/G-MS | 0.237% | 0.655% | 0.109% | 4.820% |
| fiber-10deg/F-MS | 2.896% | 8.422% | 27.347% | 94.717% |

At step 0.01 the three strong-penalty checks agree within 0.3%; the 10° check agrees within 2.9%. Smaller steps can be much worse, and repeated same-step estimates vary. This is consistent with a finite numerical-difference/inner-solve floor, but it does not identify the cause or establish a derivative pass criterion. All 32 signed perturbation solves succeeded. The large qualitative fit and roughness differences are much larger than the observed endpoint changes in the main noisy comparisons; fine distinctions in stationarity should not be overinterpreted.

The polish batch saved all nine summaries, final fields, finite-difference receipts, and end-of-run logging, but its process returned exit code 143. The cause and final remote-upload completion remain unverified. Its local endpoint files were independently checked for geometry consistency and intersections and are retained with an [execution-status receipt](../data/35-polish/execution-status.json).

## Choosing fibers for the face

The block deliberately uses a known fiber field. The prepared face mesh does not currently have verified, registered fibers and attachment or centerline annotations sufficient for this experiment. Assigning a world axis would make the control space simpler without making it anatomically meaningful.

For muscles with identifiable attachments, use registered anatomical fibers or a field derived from annotated origin and insertion patches. An attachment-based Laplace construction can provide a smooth candidate field; the attachment labels, boundary flux balance, and unresolved gradients still need inspection. Geometry does not uniquely identify anatomy, and a Laplace field can miss twisting architecture. [Choi and Blemker, 2013](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0077576).

For Orbicularis oris, begin with tangents to a registered reference centerline or an anatomical atlas. Fiber direction and surface-displacement direction are different: circumferential contraction can produce inward radial motion. Freeze the field during the primary inverse comparison, then test small coherent directional perturbations. Freely optimizing a direction in every tetrahedron would restore much of the freedom the prior removes.

On curved fibers, penalize differences of the scalar activation within each muscle. A constant activation legitimately produces changing global tensor entries as the fibers turn. Report fiber-orientation roughness separately and do not smooth across muscle compartments. If fixed fibers remain too restrictive, a few shared directional corrections or a small bounded residual tensor are later experiments; they are not validated by this study.

## Numerical checks and reproducibility

The initial screen allows 160 accepted projected L-BFGS steps per case; follow-ups allow 240. The projected KKT stopping threshold is 0.001. The gradient step is scaled by the number of active tetrahedra (Shared uses 1), then the projected residual is RMS-normalized over scalar control entries and divided by `a_ref`. Every trial begins from the last accepted equilibrium, each case starts from rest, and final activations are solved again from rest to audit branch agreement. Budget exhaustion and stalled line searches are retained in the records. A small shape error does not certify inverse stationarity.

The adapter checks zero activation, contraction sign, determinant, packing, bounds, and derivatives for all five coordinate families. Ten end-to-end finite-difference checks across those families have maximum relative gradient error 3.40 × 10⁻⁵. An initial pilot exposed an autograd state-alias issue; the experiment wrapper now allocates a fresh detached state buffer before each solve. The failed pilot is preserved, and pilot outputs are excluded from reported comparisons. Production library files were not modified.

The main forward tolerances are relative 10⁻⁶ and absolute 10⁻¹¹, and the adjoint relative tolerance is 10⁻⁷. Selected endpoint checks use forward tolerances 10⁻⁸ and 10⁻¹³ and adjoint tolerance 10⁻⁹. They retain the same objective and control constraints and save the original and tightened results separately.

Positive definite activation is not a guarantee of valid deformation. The main and follow-up optimizers record determinants of total deformation, activation, and elastic deformation over accepted trajectories. The minimum values across the 33-case screen are 0.5271, 0.4956, and 0.5718, respectively, with no nonpositive determinants. Its maximum fresh-rest branch discrepancy is 4.46 × 10⁻⁵ D. An independent IPC Toolkit audit checks the full exterior boundary: 231 main-run files, 166 follow-up files, four coarse/refined finals, and nine polished finals have no detected intersections. Endpoint and checkpoint files are counted separately, including duplicate endpoint states. This does not claim continuous collision detection between optimizer steps.

The γ = 0 sensitivity fixes the transverse natural stretch and permits active volume change. The repository evaluates `W(F A_inv)` without a separate `det(A)` multiplier. Active-strain formulations can include that multiplier, so γ = 0 and volume-changing G6 results must be read as sensitivities of this existing constitutive model, not as a validated alternative muscle law. The main G/F comparison is isochoric. [Giantesio and Musesti, 2017](https://dmf.unicatt.it/~musesti/pubblicazioni/Darmstadt.pdf).

The source revision is `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`. The recorded main environment uses Python 3.14.6, PyTorch 2.12.0+cu130, NumPy 2.4.6, SciPy 1.17.1, Warp 1.14.0, and an NVIDIA RTX 4090. Per-run source copies, hashes, seeds, configuration, solver receipts, metrics, checkpoints, and final fields are saved locally. Cherries records local artifacts and Comet metrics with automatic Git commits disabled. The installed Cherries Comet asset hook does not upload mesh or figure assets; the downloadable local records are the artifact evidence. The first forward Comet run was marked incomplete during environment-collection shutdown even though its local numerical work completed. Later runs disable automatic environment and Git-patch collection.

## Practical starting point

For the next controlled face comparison, start with a fixed, inspected fiber field, one nonnegative contraction scalar per muscle tetrahedron, a declared shortening cap, and a scalar neighbor penalty within each muscle. Compare F-S with F-MS at a small magnitude weight; retain G-MS and Raw6 as matched references. The 0.01 weight is only a starting value with the objective normalization used here. Choose it against retained motion and acceptable target error, rather than maximizing smoothness alone.

The graph penalty encourages regularity but does not enforce exact continuity. A continuous scalar basis could be a later ablation if exact continuity is required; its reduced control capacity must be compared separately. The block experiments do not reproduce the severe historical facial folding. Facial fiber registration, attachments, contact, material mismatch, and equilibrium selection still need a matched face experiment before this prior can be called a fix for that case.

## Records and downloads

The numerical study contains **60 inverse fits**: 33 initial-screen fits, 25 strength/noise/initialization/prior-sensitivity fits, and two fits to the refined target. Nine selected endpoints receive additional tighter-tolerance optimization. The fixed-control finite-difference scale audit is separate from those inverse fits.

- [Complete endpoint metrics](../data/50-analysis/endpoint-metrics.csv), [initial three-target table](../data/50-analysis/three-target-metrics.csv), and [paired held-out metrics](../data/50-analysis/smoothing-heldout-comparison.csv).
- [Refinement comparison](../data/50-analysis/refinement-comparison.csv), [polish comparison](../data/50-analysis/polish-comparison.csv), and [finite-difference scale audit](../data/37-gradient-scale/summary.json), and [10° gradient audit](../data/37-gradient-scale-fiber10/summary.json).
- [Main surface audit](../data/45-audit/summary.json), [refinement surface audit](../data/45-audit-refinement/summary.json), and [polished surface audit](../data/45-audit-polish/summary.json). The reproduction bundle also includes the per-group follow-up audits.
- [PNG/PDF figure bundle](../site/records/figures.zip) for independent reuse, and [compact reproduction bundle](../site/records/reproducibility.zip) containing current and frozen source files, configuration, hashes, CSVs, and JSON receipts.
- [Reproduction and service instructions](../README.md), [initial execution protocol](../docs/12-execution-protocol.md), and [design draft](../docs/10-experiment-plan.md).

Full NPZ fields and VTU meshes remain in this experiment's local `data/` directories. Reproduction requires the recorded repository revision and environment; the compact bundle excludes those large fields and meshes. The Tailnet HTTP service is bound to `PRIVATE_HOST:8766` and runs for the current boot; the README gives its restart command.

Comet metric records: [forward probe](https://www.comet.com/liblaf/apple/810f208f5e434783aa79c766b65364c4), [33-fit screen](https://www.comet.com/liblaf/apple/a4a78f9ebf6848f187d1ac94f1d90bf2), [22 follow-ups](https://www.comet.com/liblaf/apple/c367b46af5e9442eae37571f8b179d5d), [F-S held-outs](https://www.comet.com/liblaf/apple/938505dc36e540a89ebd878326b5dc47), [refinement](https://www.comet.com/liblaf/apple/12da56ed730e4ed6999c9f685e23d914), [numerical polish](https://www.comet.com/liblaf/apple/629ac31394bc4933ba19732d4688e6d4), and [gradient scale audit](https://www.comet.com/liblaf/apple/384b5967f16a44ce92db283c741240ab).
