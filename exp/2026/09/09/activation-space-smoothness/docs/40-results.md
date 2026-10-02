# Activation-space C-smoothness results

Axis-on completed 128 updates; Axis-off failed at attempted update 29 when its forward solve reached the fixed 5,000-iteration limit. The accepted off trace ends at update 28. At that common update, Axis-on has lower surface and mechanical diagnostics, but the states are not matched in fit or motion. The frozen actual-state match, update 11 versus update 11, reduces the primary surface residual by only 0.40%. Both Raw6 arms completed 200 updates; their matched endpoints show a 0.20% reduction. Neither comparison meets the declared 10% threshold.

This uses one learned-axis initialization (seed 20260909), one target, one mesh, and fixed budgets. It does not establish physiological validity, inverse stationarity, anatomical fiber identification, or seed robustness.

## Smaller-rate follow-up

The original results above are retained. A separate rate-only test at 1.964496, one quarter of the original learned-axis rate, completed 64 updates per arm. The off arm reached 2.588 mm fit error. The on arm reached a best fit of 3.378 mm at update 45, then degraded to 5.152 mm at update 64 with 2,189 inverted tetrahedra.

Both arms first inverted at update 25 with about 1.14 mm motion, close to the original pair's first-inversion motion of about 1.17 mm at update 8. At saved states matched in fit and motion, the quarter rate produced smaller physical updates but slightly higher surface residuals. The quarter-rate off41/on45 match has a 2.25% increase in surface residual on, and also misses the 10% improvement criterion. See the [quarter-rate results](49-quarter-rate-results.md), [execution record](48-conservative-rate-execution.md), and [independent verification](../data/53-conservative-rate-verification/summary.json).

The targeted smoothed-arm test at rate 0.3 completed 256 updates with fit 2.876797 mm, motion 4.047514 mm, and 73 inverted tetrahedra. Its first-inversion motion bracket overlaps those of the higher rates. It produces smaller physical updates and avoids the quarter-rate trajectory's sharp late deterioration through the tested budgets, but its selected global-fit/motion match against the original rate has 9.35% worse surface residual. The corresponding match against the quarter rate improves residual by only 0.66%, with a changed low-frequency motion pattern. There is no rate-0.3 off/on pair. See the [full rate-0.3 results](56-rate-03-results.md), [execution record](55-rate-03-execution.md), and [final verification](../data/57-rate-03-verification/summary.json).

## Activation glyphs

The [deformed activation glyph gallery](76-deformed-activation-glyphs.md) adds overview and mouth-corner views for six saved learned-axis states. The full field contains one centered line per active muscle tetrahedron, totaling 288,235 lines. Each center is the saved deformed tetrahedron center, mean(X + u), and each reference learned direction n_rest is transformed into the spatial direction normalize(F n_rest). Line length remains 4.5 mm × commanded shortening fraction, using one common scale across all cells and states. Greater contraction therefore produces a longer line; color also uses a common linear 0–100% scale. The faint skin context uses the same saved deformation through GlobalPointId. Camera-dependent muscle labels are recomputed for each deformed state and view; they hide different muscles behind the front muscle while retaining internal tetrahedra of the visible muscle. Muscle shapes emerge from the glyph distribution and direction; no muscle surfaces or explicit outlines are drawn. The complete VTP fields retain every cell without spatial sampling. These saved states have different fits and budgets and are not an additional matched smoothness test.

## Implementation differences

The off/on comparisons change smoothness within a model. Comparisons across learned axis, corrected Raw6, and historical PSD also change the control space, initialization, optimizer scaling, and sometimes budget. Raw learning rates, penalty coefficients, and endpoint errors therefore do not provide a controlled ranking across models.

Here B is the inverse active-strain matrix, C = B − I, and Z = BBᵀ − I is the common dimensionless effective field. Historical PSD instead represents additive stress Q = μZ directly through a normalized matrix M. S denotes the spatial variation penalty, evaluated on the specified tensor field.

| Implementation | Learned axis | Corrected Raw6 | Historical PSD |
| --- | --- | --- | --- |
| Per-cell controls | 3 values v; 864,705 scalars in total | 6 unscaled symmetric entries q; 1,729,410 scalars | 6 Frobenius-orthonormal symmetric coordinates q; 1,729,410 scalars |
| Map to effective field | C = vvᵀ; B = I + C; Z = (2 + ‖v‖²)vvᵀ | C = sym_unscaled(q); B = I + C; Z = 2C + C² | M = sym_orthonormal(q); Q = Q_ref M; Q_ref = 3μ, so Z = 3M |
| Admissible tensors | C and Z are PSD of rank at most 1; B is positive definite; no magnitude cap | C and B are unconstrained symmetric; Z ≽ −I; no magnitude cap; different signs of B can produce the same Z | Q is PSD of rank 0–3 after spectral projection; eigenvalues limited to 0–0.302013 MPa |
| Muscle implementation | Physical-volume active strain: shear term uses FB, volume terms use det(F) | Same physical-volume active-strain law as learned axis | Passive stable law plus additive ½ Q:(FᵀF − I) |
| Smooth arm | 0.0004450069704 × S(C) | 0.003214147722 × S(C) | 5.905171468 × S(M), equivalent to 0.6561301632 × S(Z) because Z = 3M |
| Initialization | Seed 20260909; ‖v‖² = 0.001; one randomly sampled axis per each of 103 labels, copied to its cells; each cell then optimized independently; zero displacement seed | Exact canonical step-200 controls and saved no-skin displacement; optimizer moments reset for each off/on re-fit | Off/on-64 start at q = 0 and rest displacement; off-1024 continues saved controls, displacement, moments, and counter |
| Adam settings in the original study | Fixed rate 7.857985795; ε = 0.01; β = (0.9, 0.999) | Fixed rate 0.3; ε = 0.01; β = (0.9, 0.999) | Off/on-64: rate 0.3; off-1024: 0.3 through global update 512, then 0.6; ε = 0.01; β = (0.9, 0.999) |
| Update constraints | No control projection, outer backtracking, physical-step cap, or inversion rejection | Same absence of outer constraints as learned axis | Spectral projection after Adam, outside the differentiation graph; upper cap never bound in the recorded runs, but the lower PSD constraint did |

For the same mapped Z, these muscle laws produce the same equilibrium force and deformation Hessian: setting Q = μ(BBᵀ − I) makes their energy difference independent of F. This correspondence does not make their inverse optimization equivalent. The parameterization, allowable Z, initialization, smoothing field, and Adam/projection history still differ. In particular, “learned axis” does not impose one fixed anatomical fiber direction per muscle.

### Run protocol and comparison role

| Run | Start and optimizer | Smoothness term | Executed result | Valid comparison role |
| --- | --- | --- | --- | --- |
| Axis-off | Shared seed-20260909 controls; zero displacement seed; fresh Adam at 7.858 | None | Accepted/evaluated states through 28; fit 6.04 mm; attempted 29 fails. Best fit 3.539 mm at 15 | Common prefix with Axis-on; selected actual-state match is 11/11 |
| Axis-on | Same controls and displacement seed; fresh Adam at 7.858 | 4.45007e−4 S(C) | Completed 128 updates; fit 0.681 mm | Update 128 is unpaired because Axis-off has no corresponding state |
| Raw6-off | Canonical step-200 controls plus saved no-skin displacement; fresh Adam at 0.3 | None | Completed 200 re-fit updates; fit 1.3978 mm | Equal-start, equal-budget off/on pair |
| Raw6-on | Exact same controls and displacement; fresh Adam at 0.3 | 0.00321415 S(C) | Completed 200 re-fit updates; fit 1.4047 mm | Selected actual-state match is endpoint 200/200 |
| PSD-off-64 | Zero controls, rest displacement, fresh Adam at 0.3 | None | Completed 64 updates; fit 3.82 mm | Historical equal-start, equal-budget pair with PSD-on-64 |
| PSD-on-64 | Same zero/rest/fresh start at 0.3 | 5.90517 S(M) | Completed 64 updates; fit 4.03 mm | Historical controlled smoothness comparison |
| PSD-off-1024 | Continues saved controls, displacement, moments, and counter; rate 0.3 then 0.6 after 512 | None | Completed global update 1024; fit 1.54 mm | Longer-run context; no matched smooth continuation |

The learned-axis initial q/C/Z arrays are bitwise identical across off/on, and equilibrated initial displacement differs by 1.38e−14 mm RMS. The first-update q difference is 1.97e−6, exceeding the predeclared 1e−6 gate. The report retains that failed verification gate rather than treating the pair as numerically identical throughout.

See the [full implementation comparison](46-implementation-comparison.md) for shared material/solver settings, the smoothing formula, and source pointers. Plot colors are consistent across model comparisons: learned-axis off/on use vermillion/blue; Raw6 off/on use magenta/green. Solid/dashed lines and distinct markers provide a second distinction. Exact-section targets are charcoal and dotted.

## Learned-axis result

The common-update comparison ends at update 28. The Axis-on endpoint is separate; no Axis-off state exists at update 128.

| State | Fit RMS (mm) | Motion RMS (mm) | Union residual HP (mm) | Union displacement HP (mm) | S(C) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Axis-off, update 28 | 6.04 | 8.30 | 0.616 | 0.766 | 6.74e6 |
| Axis-on, update 28 | 2.04 | 4.65 | 0.233 | 0.489 | 1.36e3 |
| Axis-on, update 128 | 0.681 | 5.18 | 0.167 | 0.465 | 286 |

At update 28, the union residual high-pass RMS is lower by 0.383 mm (62.1%), displacement high-pass RMS by 0.277 mm, and S(C) by 99.98%. Fit RMS differs by 4.00 mm and motion RMS by 3.65 mm, so these are descriptive equal-update changes rather than a matched surface result. The low-frequency projection ratio is 0.620, below the declared 0.90 retention threshold.

For learned-axis parameter v, C = vvᵀ, B = I + C, and Z = (2 + ‖v‖²)vvᵀ.

| State | P_U | S(Z) | Inversions | p99 shortening | min det(F) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Axis-off, update 28 | 1.12 | 7.99e13 | 12,657 | 98.9% | -12.9 |
| Axis-on, update 28 | 0.694 | 4.74e6 | 335 | 81.6% | -2.01 |
| Axis-on, update 128 | 0.907 | 1.27e5 | 94 | 86.6% | -1.53 |

The Axis-on endpoint is its best-fit state under the 128-update budget. It retains 94 inverted tetrahedra. Its maximum Z eigenvalue is 4.03e3; with the recorded muscle shear modulus, this is a 40.6 MPa maximum effective additive-stress scale. Solver completion does not supply physiological validation.

![Unpaired target and Axis-on update-128 endpoint](../data/42-axis-on-endpoint-v2/geometry/side-context/target-axis-on-0128.png)

![Unpaired Axis-on update-128 right mouth-corner close-up](../data/42-axis-on-endpoint-v2/geometry/region1-mouth-corner/target-axis-on-0128.png)

![Exact sections through the unpaired Axis-on update-128 endpoint](../data/44-report-figures-v2/sections/axis-on-endpoint-0128.png)

![Fresh-pair fit, motion, primary surface score, and low-frequency projection trajectories](../data/44-report-figures-v2/new-pairs-trajectories.png)

### Matched saved states

The state search selects update 11 in both arms. Fit RMS is 3.77 mm off and 3.76 mm on; motion RMS is 3.07 mm in both. The differences, 0.0101 mm and 0.00216 mm, are within the 0.05 mm tolerances.

| State | Union residual HP (mm) | Union displacement HP (mm) | S(C) | P_U | Inversions | p99 shortening |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Axis-off, update 11 | 0.247 | 0.311 | 325 | 0.366 | 19 | 34.7% |
| Axis-on, update 11 | 0.246 | 0.311 | 296 | 0.369 | 18 | 34.7% |

The matched residual reduction is 0.000985 mm, or 0.40%; displacement changes by -0.000077 mm, and S(C) is 9.17% lower. The low-frequency projection ratio is 1.01. The tested 10% threshold is not met. Axis-off supplies no later matched trajectory, so this result does not rule out other weights or later matched fits.

![Learned-axis matched saved states at the right mouth corner](../data/40-comparison/geometry/learned-axis-matched/region1-mouth-corner-target-off-on.png)

![Exact sections through the learned-axis matched states](../data/44-report-figures-v2/sections/learned-axis-matched.png)

## Corrected Raw6 result

Both Raw6 arms completed 200 updates, and their endpoints also form the selected actual-state match.

| State | Fit RMS (mm) | Motion RMS (mm) | Union residual HP (mm) | Union displacement HP (mm) | S(C) | P_U |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw6-off, update 200 | 1.40 | 4.82 | 0.180 | 0.467 | 9.40 | 0.920 |
| Raw6-on, update 200 | 1.40 | 4.81 | 0.180 | 0.466 | 8.37 | 0.919 |

Fit and motion differ by 0.00690 mm and 0.0110 mm, within both 0.05 mm tolerances. The primary residual is lower by 0.000362 mm, or 0.20%; S(C) is 11.0% lower, S(Z) is 14.0% lower, and the low-frequency projection ratio is 0.998. Both states have four inverted tetrahedra. The residual criterion fails, so the tested matched-state threshold is not met. Raw6-on uses learning rate 0.3 and C-smoothness coefficient 0.003214147722027223. Raw6 and learned-axis coefficients are coordinate-specific and are not matched physical prior strengths.

![Raw6 matched endpoints at the right mouth corner](../data/40-comparison/geometry/raw6-matched/region1-mouth-corner-target-off-on.png)

![Exact sections through the Raw6 matched endpoints](../data/44-report-figures-v2/sections/raw6-matched.png)

### Regional matched-state residual HP reduction (negative means increase)

| Frozen support ([detailed metrics](../data/43-regional-matched-metrics/regional-matched-metrics.csv)) | Learned-axis residual HP reduction | Raw6 residual HP reduction |
| --- | ---: | ---: |
| Right mouth corner | 0.480% | 0.311% |
| Right lateral cheek | -0.108% | 0.0523% |
| Right lower cheek/jaw | -0.0604% | 0.192% |

| Protected nose-to-mouth check | Off fit RMS (mm) | On fit RMS (mm) | Off low-frequency projection | On low-frequency projection |
| --- | ---: | ---: | ---: | ---: |
| Learned axis, update 11 | 6.05 | 6.03 | 0.605 | 0.596 |
| Raw6, update 200 | 1.93 | 1.95 | 0.823 | 0.820 |

## Historical PSD context

These reused PSD states use earlier protocols and penalty settings. They provide context rather than an equal-start or consistently equal-budget ranking across models.

| Context state | Fit RMS (mm) | Motion RMS (mm) | Union residual HP (mm) | Union displacement HP (mm) | P_U | S(Z) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PSD-off, update 64 | 3.82 | 1.81 | 0.284 | 0.178 | 0.264 | 7.88 |
| PSD-on, update 64 | 4.03 | 1.56 | 0.306 | 0.152 | 0.227 | 0.357 |
| PSD-off, update 1024 | 1.54 | 4.43 | 0.146 | 0.415 | 0.822 | 196 |

![Attained fit and surface error with historical PSD context](../data/44-report-figures-v2/historical-context-tradeoff.png)

## Learning rate and distorted intermediate states

The learned-axis rate **7.86 is a strong suspect in the later Axis-off instability**, but learning rate alone has not been established as the cause of all distortion. The selected rate was the fastest in a 16-update calibration whose `stable` flag checked solver success and short-term loss behavior. It did **not** require inversion-free geometry.

| Fit-only pilot rate | Fit RMS at update 16 (mm) | Motion RMS (mm) | Inverted tetrahedra | First inverted update |
| ---: | ---: | ---: | ---: | ---: |
| 0.982 | 5.302 | 0.015 | 0 | None through 16 |
| 1.964 | 5.233 | 0.155 | 0 | None through 16 |
| 3.929 | 4.376 | 1.760 | 1 | 14 |
| **7.858, selected** | **3.028** | **4.210** | **225** | **8** |

![Four learning-rate calibration pilots](../data/44-report-figures-v2/learning-rate-calibration-trajectories.png)

The half-rate pilot first inverts at 1.122 mm motion RMS, versus 1.167 mm at the selected rate. It delays the first inversion from update 8 to update 14, while reaching a similar deformation at onset. The lower-rate pilots have not been continued to comparable final fit or motion, so their smaller early inversion counts do not demonstrate a complete remedy.

The primary Axis-off fit worsens from its best 3.539 mm at update 15 to 6.038 mm at update 28; inverted tetrahedra increase from 541 to 12,657 before update 29 fails. These are optimization iterates, not a physical motion sequence. They are exact saved states, rather than interpolated animation frames.

Adam uses fixed η with β = (0.9, 0.999) and ε = 0.01. This runner has no outer loss-decrease test, backtracking, physical-step cap, or inversion-based rejection. Its inner equilibrium line search does not choose the activation update. In addition, Z grows nonlinearly with v, and S(C) penalizes spatial differences without limiting activation magnitude or det(F). The [learning-rate diagnosis](45-learning-rate-diagnosis.md) gives the evidence and the limits of this interpretation; no new fitting run was performed.

### Calibration and execution sensitivity

The original learned-axis rate grid failed. The [closed-grid revision](15-calibration-revision.md) selected the upper tested rate, 7.857985794554741, and C-smoothness coefficient 0.0004450069704277614. Its discarded 16-update weight pilot retained 100.17% of fit progress and 98.43% of incremental target projection while reducing S(C) by 40.38%. This justifies the frozen setting within the grid; it does not establish an optimum.

The [divergence figure](../data/44-report-figures-v2/calibration-vs-primary-first16.png) compares the selected pilot and primary Axis-off through update 16.

![Selected calibration pilot and primary Axis-off through update 16](../data/44-report-figures-v2/calibration-vs-primary-first16.png)

The [read-only audit](../data/18-calibration-main-divergence-audit/summary.json) finds no archived implementation or configuration mismatch. The evidence supports sensitivity along an increasingly inverted, nonconvex trajectory, but it does not identify the initiating perturbation or prove a roundoff-only cause; no deterministic GPU replay was run.

The primary off/on pair has bitwise-identical step-0 q, C, and Z arrays; equilibrated u differs by 1.38e-14 mm RMS. The first-update q maximum absolute difference is 1.97e-6 against the 1e-6 gate, so first-step numerical equivalence remains failed. The [partial verification](../data/24-learned-axis-partial-verification/summary.json) confirms accepted steps 0–28 and keeps attempted update 29 solver-invalid. The [definitive verification](../data/50-verification-v2/summary.json) passes retained-evidence integrity; `study_checks_passed` is false because the Axis-off budget and first-update q gate failed.

## Measurements and evidence

Fit and motion are uniform vertex-vector RMS over finite fitted face vertices. The primary surface score is the area-weighted 5 mm high-pass RMS of rest-normal target residual over 1,592 unique skin vertices in the frozen right mouth-corner, right lateral-cheek, and right lower-cheek/jaw supports; it is not whole-face roughness. P_U is the mass-weighted projection of achieved low-pass normal displacement onto the target low-pass field over that union.

The matched search uses saved post-update states only, requires fit and motion differences no greater than 0.05 mm, and does not interpolate. A useful matched effect requires at least 10% primary residual reduction, lower S(C), and P_U,on/P_U,off at least 0.90 when P_U,off is positive. See the [protocol](12-learned-axis-plan.md), [comparison summary](../data/40-comparison/summary.json), [render receipt](../data/42-axis-on-endpoint-v2/summary.json), [selected states](../data/40-comparison/selected-states.csv), and calibration receipts for [learned axis](../data/18-learned-axis-calibration-refined/summary.json) and [Raw6](../data/19-raw6-smooth-calibration/summary.json).
