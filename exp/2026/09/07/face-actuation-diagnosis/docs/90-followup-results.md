# Face activation: smoothing and independent force tests

## What this follow-up tests

The preceding face diagnosis (private preview omitted) found that a weak activation regularizer reduced field roughness without visibly removing the facial bumps. This follow-up first tests the physical effect of smoothing one fixed activation field. It then checks the skin's mechanical contribution and separates manual active force from passive muscle stiffness.

All displayed faces use actual saved equilibrium positions at their original scale. No output vertices are smoothed. Deformation determinants and activation spectra are recorded as diagnostics; they do not reject a state. The main inverse representation still has 288,235 active tetrahedra with six independent entries each, or 1,729,410 scalar controls.

Stronger diffusion reduces surface variation substantially when taken well beyond the initial 50% penalty reduction, but the expression and target fit also deteriorate. The skin membrane softens fine mouth corrugations more selectively at the initial field settings, with a similar fit cost. Explicit fiber tension increases manual expression motion at unchanged passive stiffness through gain 3; gain 10 fails the fixed forward-solve budget and has no accepted equilibrium image.

[Interactive 3D comparison](viewer.html) · [Separate PNG/PDF figures](records/figures.zip) · [Scripts and evidence](records/reproducibility.zip)

## 1. Diffusing the same bumpy activation field

The source is the best valid Raw6 continuation checkpoint, local step 50, from the preceding study. Its forward replay reproduces the original face: the area-weighted displacement difference is only 0.00000031 mm. It is a saved early-run solution, not a convergence certificate.

The graph joins tetrahedra that share a face and have the same muscle label. It has 501,409 edges and 854 connected components. Each component conserves the average of all six entries of `H = Ainv − I`, weighted by rest tetrahedral volume times muscle fraction. The diffusion equation is `(M + τ L) H = M H₀`. Its strength is selected from the activation field alone. The limiting field sets every tetrahedron to its own component's weighted mean. This is a forward diagnostic field; the inverse representation still retains all per-tetrahedron controls.

The initial label “strong diffusion” meant `R/R₀ = 0.5`. Because `R` sums squared weighted differences between neighboring activations, halving it reduces their RMS variation by only 29.3%. That setting does not imply a nearly uniform activation field or a smooth visible surface. The extended test therefore includes approximately 90% and 99% penalty reduction and the exact constant-within-component limit.

Every solve starts from the exact same saved displacement seed. Historical Young's moduli remain 0.03 MPa for muscle, 0.003 MPa for fat, and 0.1 MPa for aponeurosis; Poisson ratios remain 0.49, 0.49, and 0.35, using the historical classical Lamé convention. Fixation is unchanged, with no contact or skin energy. PNCG tolerances remain relative `5e-4`, absolute `1e-10`, with a 5,000-step limit and the existing line search. All six forward solves meet these tolerances.

| Activation field | Achieved R/R₀ | Target RMS error, mm | Motion RMS, mm | Target projection |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 1.000000 | 0.827918 | 4.864734 | 0.942467 |
| About 10% penalty reduction | 0.899977 | 0.905952 | 4.771861 | 0.922629 |
| About 50% penalty reduction | 0.499966 | 1.336869 | 4.149701 | 0.797147 |
| About 90% penalty reduction | 0.100086 | 2.568070 | 2.847175 | 0.529101 |
| About 99% penalty reduction | 0.010066 | 3.343077 | 2.037496 | 0.364743 |
| Constant within each component | 0 | 3.851625 | 1.732387 | 0.272148 |

Surface errors, motion, and projection use common rest-surface area weights within each comparison. They differ from the uniform Cartesian MSE used by the earlier inverse optimizer. The manual actuation branch in Section 3 uses its own previously established fixture and material contract; its gain comparisons are internal to that branch.

Stronger diffusion does reduce surface variation much more than the initial 50% setting. At 99% penalty reduction, mouth high-pass displacement falls by approximately 50–52% over the 2, 5, and 10 mm spatial scales. In the constant-component limit, it falls by 59–64%. These are substantial absolute changes. The associated expression-motion losses are 58.1% and 64.4%, and target RMS errors rise to 3.343 and 3.852 mm.

| Activation field | Full-face high-pass RMS at 5 mm, mm | Mouth high-pass RMS at 5 mm, mm | Mouth high-pass / total motion |
| --- | ---: | ---: | ---: |
| Unchanged | 0.291630 | 0.663597 | 13.64% |
| About 10% penalty reduction | 0.289554 | 0.658454 | 13.80% |
| About 50% penalty reduction | 0.259922 | 0.588112 | 14.17% |
| About 90% penalty reduction | 0.197374 | 0.460435 | 16.17% |
| About 99% penalty reduction | 0.136615 | 0.327515 | 16.07% |
| Constant within each component | 0.099351 | 0.239964 | 13.85% |

Here, “mouth” means vertices within 10 mm intrinsic distance of the rest lip vertices. High-pass values describe spatial variation; the target itself can contain real fine detail. A lower absolute high-pass value alone therefore does not establish target recovery.

The normalizations answer different questions. Relative to total expression motion, the mouth high-pass fraction at 5 mm increases slightly even at the constant-component limit. Relative to the mouth's own normal-displacement RMS, it decreases by 45.1% at that limit. The corresponding regional-normal fractions decrease by 37.7–45.9% over the three scales. Thus the effect is not simply uniform displacement scaling: the response changes shape and direction as well. Meanwhile, mouth target-residual high-pass increases by 35–96% in the limiting state, consistent with its worse target fit.

The matched renders show that the initial 50% setting leaves most cheek and mouth corrugations visible. At 90% reduction they soften but remain prominent. The 99% state is substantially smoother, and the component-constant state removes most of the fine corrugations, although small mouth-edge irregularities remain. At the same time the broad smile weakens and the mouth opening narrows. The extended test therefore confirms that the initial diffusion setting left substantial removable surface variation.

![Six matched mouth views across the full diffusion range](../data/95-extended-diffusion-viewer/extended-diffusion-mouth-comparison.png)

Separate assets: [mouth PDF](../data/95-extended-diffusion-viewer/extended-diffusion-mouth-comparison.pdf), [full-head PNG](../data/95-extended-diffusion-viewer/extended-diffusion-full-head-comparison.png), [full-head PDF](../data/95-extended-diffusion-viewer/extended-diffusion-full-head-comparison.pdf), [figure provenance](../data/95-extended-diffusion-viewer/extended-diffusion-comparison-receipt.json), [original three-setting figures](../data/90-field-diffusion-figure/manifest.json).

There is a limit to what increasing this diffusion strength can remove. Its constant field has exactly zero differences on the 501,409 same-muscle graph edges, but different muscle components can retain different means. An independent audit adds the 22,231 active-to-active shared faces across muscle labels. On this enlarged graph, roughness falls from 117.291516 to 0.590534 in the limiting state, rather than to zero. Active-to-inactive interfaces are outside this audit. These residual transitions are recorded; the test does not establish that they cause the remaining surface irregularities.

Conserving mean activation does not conserve active stress. The volume-weighted RMS activation-induced first Piola stress at the common displacement seed falls from 1.01735 MPa in the unchanged field to 0.34486 MPa in the component-constant field. After re-equilibration those values are 1.01735 and 0.06332 MPa. This is the pure-muscle difference `P(F,Ainv) − P(F,I)`, weighted by rest volume times muscle fraction; it is not a measured anatomical stress. The experiment therefore varies both field variation and the nonlinear mechanical response.

**Decision:** the initial 50% setting was insufficient to test the near-uniform limit. Stronger diffusion substantially reduces absolute surface variation, but the substantially smoothed fixed-field replays do not preserve the original expression and fit. The conditional 64-step inverse sweep was not launched. These forward results do not rule out recovering expression by refitting under a stronger regularizer; they show the tradeoff that such an inverse comparison would need to resolve.

Evidence: [initial forward results](../data/80-forward-field-diffusion/summary.json), [extended forward results](../data/94-extended-field-diffusion/summary.json), [extended field selection](../data/94-extended-field-diffusion/field-preparation.json), [six-case surface measures](../data/95-extended-diffusion-surface/summary.json), [initial saved-state validation](../data/84-field-diffusion-validation.json), [extended validation and boundary audit](../data/97-extended-diffusion-validation.json), [3D render validation](../data/95-extended-diffusion-viewer/qa-receipt.json).

## 2. Skin transmission at fixed activation

The two activation fields—unchanged and strongly diffused—are each replayed with skin factors 0 and 0.12. The no-skin states are reused from the preceding test. The two new skin-on solves start from the same original displacement seed; neither field is refitted.

The only added energy is the existing skin term with `E = 0.024 MPa`, `ν = 0.46`, and thickness `1 mm`. Volume materials, fixation, forward tolerances, and activation arrays remain identical. The local physics copy differs from the historical source only by removal of the guard that forbids nonzero skin energy. The current [Koiter implementation](../../../../../../src/liblaf/apple/warp/fem/_koiter.py) uses an in-plane metric energy and has no curvature/bending term. This experiment therefore tests that membrane implementation, not a full model of skin bending.

| Activation field | Skin factor | Target RMS error, mm | Motion RMS, mm | Mouth high-pass RMS at 5 mm, mm |
| --- | ---: | ---: | ---: | ---: |
| Unchanged | 0 | 0.827918 | 4.864734 | 0.663597 |
| Unchanged | 0.12 | 1.403239 | 4.156302 | 0.505262 |
| Strong diffusion | 0 | 1.336869 | 4.149701 | 0.588112 |
| Strong diffusion | 0.12 | 1.841614 | 3.597674 | 0.451582 |

At fixed unchanged activation, skin reduces mouth high-pass displacement by 26.0%, 23.9%, and 19.5% at 2, 5, and 10 mm. Total expression motion falls 14.6%, so the mouth's high-pass-to-motion ratio improves by 13.4%, 10.9%, and 5.7%. The strongly diffused field shows a similar mouth effect. This is more selective than activation diffusion alone in the mouth region.

The benefit has a substantial shape cost: target error rises from 0.828 to 1.403 mm for the unchanged field. The mouth's target-residual high-pass is almost unchanged at 2 mm and increases by 10.5% and 18.1% at 5 and 10 mm. Adding skin also changes displacement directions: regional normal motion can increase even while total motion decreases. Neither a lower absolute high-pass value nor a single normalized ratio establishes target recovery.

The matched renders show softer fine corrugations around the cheeks, mouth, and chin with the skin term. The mouth rim remains uneven, and the combined skin-plus-diffusion state has the weakest broad expression.

![Two activation fields with skin off and on](../data/89-skin-transmission-viewer/skin-transmission-mouth-2x2.png)

Separate assets: [comparison PDF](../data/89-skin-transmission-viewer/skin-transmission-mouth-2x2.pdf), [figure provenance](../data/89-skin-transmission-viewer/skin-transmission-mouth-2x2-receipt.json), [viewer validation](../data/89-skin-transmission-viewer/qa-receipt.json).

All four cells use the same activation and seed contracts. Both new forward solves pass. Their inverted-tetrahedron counts, 156 and 150, are retained in the records and did not gate execution. The skin term changes the surface response, but these fixed-field tests do not establish that it removes the bumps or that a skin-on inverse solve can preserve the original fit.

Evidence: [execution and command](85-skin-transmission-forward.md), [forward results](../data/85-forward-field-skin-transmission/summary.json), [surface measures](../data/89-skin-transmission-surface/summary.json), [independent saved-state validation](../data/86-skin-transmission-validation.json), [service completion](../data/85-forward-field-skin-transmission/service-exit-receipt.json). The four-case skin viewer (private preview omitted) shows each actual state.

## 3. Active force at unchanged passive stiffness

Increasing both muscle Lamé parameters in the previous study strengthened the prescribed smile, but changed passive resistance as well as activation-induced stress. The new diagnostic adds an explicit fiber-tension energy to the unchanged passive energy:

`W(F) = W_passive(F) + (T/2)(‖F f‖² − 1)`

Here `f` is a unit reference fiber and `T ≥ 0` sets the active second Piola tension. Its first Piola contribution is `T (F f) ⊗ f`. Setting `T = 0` restores the original passive energy, stress, and tangent. The active term is bounded below and adds a positive-semidefinite material tangent. This defines a different actuation model from a prescribed natural contraction; it does not hold the total active tangent or Cauchy stress constant as the tissue deforms.

The reference tension is `T₀ = 3 μ = 0.0246575 MPa`. It matches only the axial stress of the earlier c50 active-strain prescription at `F = I`; it does not make the two constitutive laws equivalent. Passive muscle parameters remain `E = 0.024 MPa`, `ν = 0.46`, with the current corrected Stable Neo-Hookean Lamé convention used by the manual-expression study.

The single-tetrahedron checks cover gains from 0 to 10 under no external load and an opposing nominal axial load of 5 kPa. Finite differences verify the energy gradient, tangent action, tangent diagonal, and energy curvature; the largest reported derivative error is approximately `1.1e-11 MPa`. Superposed-rotation checks also pass. Each tested equilibrium has a small force residual and positive tangent on the explicitly constrained two-degree-of-freedom axial/transverse family. This is a local check on that restricted family, not a claim of stability for every free deformation of a tetrahedron.

| Gain | Unloaded fiber stretch | Fiber stretch against 5 kPa |
| --- | ---: | ---: |
| 0 | 1.0000 | 1.2543 |
| 1 | 0.6195 | 0.6769 |
| 3 | 0.4484 | 0.4710 |
| 10 | 0.2968 | 0.3043 |

The loaded contraction emerges from force balance. It is not set equal to the gain or to a prescribed natural shortening.

![Single-tetrahedron force and volume responses](../data/81-single-tet-active-tension-extended/gain-response.png)

Separate assets and evidence: [response PDF](../data/81-single-tet-active-tension-extended/gain-response.pdf), [actual tetrahedron states PNG](../data/81-single-tet-active-tension-extended/tetrahedra.png), [tetrahedron states PDF](../data/81-single-tet-active-tension-extended/tetrahedra.pdf), [extended numerical checks](../data/81-single-tet-active-tension-extended/summary.json), [constitutive report and commands](81-single-tet-active-tension.md).

The face diagnostic then applies this law to the same 9,732 smile-elevator tetrahedra and unit reference fibers used by the preceding manual test. All other cells receive zero added tension. Every gain starts from zero displacement, with identity activation inverse, no skin, unchanged passive parameters, and no target in the equilibrium solve. The target is used only afterward to describe the motion. Forward tolerances are the manual branch's original relative `1e-5` and absolute `1e-12`, with a 10,000-step limit.

| Active-tension gain | T, MPa | Surface motion RMS, mm | Smile projection | Median loaded fiber stretch | Minimum det F | Forward outcome |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0 | 0 | 0 | 1.0000 | 1.0000 | Passed, 1 step |
| 1 | 0.024658 | 0.414898 | 0.045063 | 0.9506 | 0.6735 | Passed, 1,177 steps |
| 3 | 0.073973 | 1.033988 | 0.115819 | 0.8758 | 0.1594 | Passed, 3,189 steps |
| 10 | 0.246575 | — | — | — | — | Failed, 10,000-step limit |

The three saved equilibria have no inverted tetrahedra. At gain 3, however, the minimum determinant is already 0.159, showing substantial local volume compression. These diagnostics did not gate the runs. The median muscle shortening is only 12.4%, compared with 55.2% in the unloaded restricted tetrahedron at the same gain: surrounding tissue and fixation materially change the loaded response.

The matched face views show progressively wider lips and raised mouth corners, with the mouth remaining largely closed. The stronger response follows from increased active tension while the passive material parameters stay fixed. This supports an independent force control in this implementation; it does not calibrate physiological tension, validate the estimated fibers, or recover the open-mouth target.

Gain 10 ended with `max_steps_reached` and gradient norm `1.2197e-9` in the solver's recorded units. Its line search accepted the last step, but the forward convergence test did not pass. No equilibrium state was exported or rendered for that gain. The Cherries wrapper returned process exit status 0 despite the raised `ForwardConvergenceError`; the [terminal outcome receipt](../data/93-active-tension-face-outcome.json) therefore classifies the experiment as `partial_forward_failure` from its actual solver records. No tolerance was relaxed and the solve was not extended.

![Manual face response at unchanged passive stiffness](../data/91-active-tension-viewer/active-tension-face-comparison.png)

Separate assets: [comparison PDF](../data/91-active-tension-viewer/active-tension-face-comparison.pdf), [viewer validation](../data/91-active-tension-viewer/qa-receipt.json).

Evidence: [face setup, execution, and interpretation](88-active-tension-face.md), [per-gain trace](../data/88-active-tension-face/trace.csv), [independent saved-state validation](../data/92-active-tension-face-validation.json), [gain-10 failure](../data/88-active-tension-face/failure.json), [Warp derivative checks](../data/88-active-tension-warp-audit/warp-audit.json). The active-tension viewer (private preview omitted) contains the three accepted states.

## Reproduction and verification

The new computations extend `exp/2026/09/07/face-actuation-diagnosis/`. Numerical and constitutive experiments use Cherries and Comet with the explicit `ProfileCometNoCommit` profile. Exact source copies, configurations, input hashes, solver receipts, and output hashes accompany the runs. The downloads contain code and compact evidence; the large original fixtures and NPZ/VTK equilibrium arrays remain in the workspace and are identified by their hashes.

The diffusion implementation passed an independent unequal-mass, disconnected-component comparison against a dense linear solve. Saved finite-diffusion fields have post-correction linear residuals below `9.9e-12`. All component-mean errors, including the analytical constant-field limit, are below `1.7e-15`. Each saved NPZ and VTK pair agrees exactly on activation, topology, rest positions, displacement, and deformed positions.

The initial diffusion run completed its numerical work at 16:19 CST on September 7, 2026, with [Comet record 465c5d9b](https://www.comet.com/liblaf/apple/465c5d9b7d0d45ab8ad0a1e92ea6c417). The three extended forward cases completed at 17:11 CST with [Comet record e67b702e](https://www.comet.com/liblaf/apple/e67b702e47a343c29e9d6a0b5627d030). Both runs reported a missing log asset in the Cherries snapshot during shutdown. Complete service stdout, local numerical outputs, and Comet scalar summaries were retained; successful upload of every remote artifact is not claimed.

An initial validation-helper invocation used Cherries' default commit profile and failed while serializing a NumPy boolean. The accidental local commit was undone with a mixed reset, preserving every experiment file. An incidental lockfile mirror-URL rewrite was also restored; all 272 package name/version pairs were unchanged. The corrected helper then passed with the explicit no-commit profile. The [correction receipt](../data/87-runner-side-effect-correction/receipt.json) records the restoration to the original repository HEAD. Nothing was pushed.

To replay the field and skin forward comparisons into new directories, run the following from this experiment directory. The recorded fixture and original checkpoint must be present. Each numerical runner selects `ProfileCometNoCommit` explicitly.

```bash
export COMET_AUTO_LOG_GIT_METADATA=false
export COMET_AUTO_LOG_GIT_PATCH=false
export COMET_AUTO_LOG_ENV_DETAILS=false
export CUDA_VISIBLE_DEVICES=0
export CHERRIES_NAME='Face actuation follow-up replay'
export CHERRIES_TAGS='face,forward,followup,no-commit'

.venv/bin/python \
  src/80-forward-field-diffusion.py \
  --output-dir data/80-forward-field-diffusion-replay

.venv/bin/python \
  src/85-forward-field-skin-transmission.py \
  --field-run data/80-forward-field-diffusion-replay \
  --output-dir data/85-forward-field-skin-transmission-replay

.venv/bin/python \
  src/94-extended-field-diffusion.py \
  --output-dir data/94-extended-field-diffusion-replay
```

The extended runner verifies its original checkpoint, seed, graph, and material contract against the immutable accepted `data/80-forward-field-diffusion/summary.json` before running the three additional fields.

The [single-tetrahedron report](81-single-tet-active-tension.md) and [manual face report](88-active-tension-face.md) give their exact commands, parameters, and Comet records. The render manifests specify every source state and camera input. The site can be rebuilt from the accepted local artifacts without running a solver:

```bash
.venv/bin/python \
  src/96-publish-followup.py docs/96-followup-publication.json \
  --output data/96-followup-site-replay
```

The publisher checks local HTML and Markdown links, exact browser geometry, render receipts, JavaScript imports, download archives, and file hashes before producing the site. These checks and the inspected static renders support the delivered comparison; they do not constitute a live browser WebGL execution test.
