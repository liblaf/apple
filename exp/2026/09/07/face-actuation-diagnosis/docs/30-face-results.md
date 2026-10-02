# Face actuation diagnosis: manual expressions and smooth per-tetrahedron control

The manual tests show that the current smile response is sensitive to the balance between muscle actuation and surrounding tissue stiffness. Prescribing 50% natural fiber contraction produces only 6.3% average loaded shortening in the selected smile muscles. Keeping the same activation and multiplying those muscles' Lamé parameters by ten increases surface motion from 0.490 to 2.102 mm and produces a recognizable closed-mouth smile. This surface result retains three inverted tetrahedra and uses uncalibrated material values. The model also produces visible pursing/closure with manual orbicularis-oris activation at the original materials. These results narrow the diagnosis: the current smile response is weak, but the face model can produce recognizable expression-related motion.

The saved historical no-skin reference fits the supplied target to 0.654 mm RMS and has visible surface bumps. Its free activation space contains 1,729,410 coordinates. The new matched historical-configuration comparison uses the same control domain for both methods, retaining an independent tensor in every active tetrahedron and penalizing neighboring differences within each named muscle. Current-fixture and regional runs are separate diagnostic comparisons.

[Open the interactive comparison](viewer.html) · [Download separate figures](records/figures.zip) · [Download scripts and compact evidence](records/reproducibility.zip)

The completed early-trend comparison reaches area-weighted surface errors of 0.828 mm for Raw6 and 0.804 mm for Raw6-S. Both recover large target-like motion. The smoothness penalty reduces activation-field variation by 9.2% at the selected best states, but the decrease in surface high-frequency displacement is only about 0.3–1.1%. The rendered cheeks and lip boundaries remain visibly bumpy. Keeping every tetrahedron's controls and adding a smoothness penalty is workable; this tested weight has only a modest effect on the surface artifacts.

![Raw6 without skin energy: rest at left, selected best continuation state at right.](../data/60-final-diagnosis-viewer/historical-adam-raw6/mouth-closeup.png)

![Raw6-S with all per-tetrahedron controls retained: rest at left, selected best continuation state at right.](../data/60-final-diagnosis-viewer/historical-adam-raw6-smooth/mouth-closeup.png)

## Three different deformation maps

The physical tetrahedron deformation is `F = dx/dX`; its signed determinant describes the final local volume ratio and orientation. The constitutive kernel evaluates `G = F A_inv`. The prescribed inverse active map `A_inv` is a control, and its determinant is not the final volume ratio. Where the active map is invertible, its inverse describes the natural active deformation. Raw6 also permits indefinite or singular maps, so not every optimized raw tensor has a physiological natural-state interpretation.

For the fiber model,

```text
A_inv = exp(a) f fᵀ + exp(-a/2) (I - f fᵀ)
natural axial stretch = exp(-a)
det(A_inv) = 1
det(G) = det(F) det(A_inv)
```

Thus volume-preserving activation does not force the loaded tissue to preserve volume or achieve the prescribed axial shortening. The same activation can produce different `F` under different surrounding materials and boundary conditions. All new experiments record tetrahedron inversion, distortion and active-map spectra without using them to reject an otherwise successful numerical solve. Failed equilibrium or derivative solves remain explicit numerical failures.

## Manual activation separates actuation from inverse optimization

Three target-independent patterns were solved: bilateral smile elevators, bilateral risorius, and orbicularis oris. Each uses 10%, 30% and 50% prescribed natural contraction, with and without skin energy. The same control arrays are used for each matched skin pair. Every pattern starts at rest and follows the same 0 → 10 → 30 → 50% quasistatic continuation; these are numerical load increments, not physical time. All 18 strict equilibrium solves succeeded.

Fiber directions come from rest geometry: a muscle-fraction-weighted principal axis for each non-ring muscle, and cellwise tangents to a fitted ellipse for orbicularis oris. They were not fitted to the Smile target and are not measured anatomical fibers. [Direction construction](../../face-activation-materials/src/face_fixture.py).

| Manual pattern, 50% natural contraction | Surface motion without skin | Surface motion with skin | Interpretation |
| --- | ---: | ---: | --- |
| Smile elevators | 0.490 mm | 0.406 mm | Weak smile-related motion; no-skin target projection 0.0529 |
| Risorius | 0.161 mm | 0.128 mm | Weak horizontal pull |
| Orbicularis oris | 1.067 mm | 0.981 mm | Clear inward lip motion; no-skin lip RMS 4.260 mm |

Surface values use the same rest-surface area weights. The Smile target has 5.096 mm surface displacement RMS. Projection is the least-squares scalar coefficient of that target field; it is not a percentage of activated muscle. Orbicularis oris is a pursing test rather than a Smile fit. Its no-skin mean inward radial lip motion is 2.736 mm. The displayed shape is a qualitative expression witness, not anatomical validation.

The no-skin smile case has fraction-volume-weighted mean physical fiber stretch 0.93697, despite a prescribed natural stretch of 0.5. Its selected cells are mixed: their integrated fractions are approximately 32.9% muscle, 48.0% fat and 19.2% aponeurosis. Adjacent cells outside that selected set are mostly fat. Of 3,660 selected-muscle vertices, 492 are fixed; none overlaps the newly added artificial-cut fixation. This excludes that cut as a direct fixed-vertex explanation for the selected manual smile, while retaining the original attachments and mixed-tissue resistance as relevant mechanisms. [Manual configuration and results](../data/10-manual-activation/summary.json).

![The same prescribed smile activation with selected-muscle Lamé parameters multiplied by ten: rest at left, solved endpoint at right, at true displacement scale.](../data/60-final-diagnosis-viewer/manual-smile-elevators-c50-selected-muscle-lame-x10/mouth-closeup.png)

The lifted mouth corners and changed cheek contour are visible in the [full-face comparison](../data/60-final-diagnosis-viewer/manual-smile-elevators-c50-selected-muscle-lame-x10/full-head.png). This resembles a closed-mouth smile, while the [supplied target](../data/60-final-diagnosis-viewer/manual-smile-elevators-c50-selected-muscle-lame-x10/target-skin-mouth-closeup.png) opens the mouth substantially. A qualitative resemblance does not demonstrate a calibrated or complete Smile reconstruction.

The target displacement alone does not establish mandible motion or the correct jaw boundary condition. The manual smile-elevator prescription has no jaw-opening actuator, and the fixture omits contact. Its low projection onto the complete Smile target therefore does not isolate smile-force capacity.

![Manual orbicularis-oris activation at 30% prescribed natural contraction, without skin energy: rest at left and saved equilibrium at right.](../data/60-final-diagnosis-viewer/manual-orbicularis-oris-30pct-skin-000/mouth-closeup.png)

This intermediate orbicularis-oris case shows visible lip pursing and closure without changing the material parameters. The viewer also contains the 10% and 50% cases, both skin settings, and every manual smile/risorius case.

## Material counterfactuals show sensitivity to load resistance

The following solves preserve the exact no-skin c50 smile controls and start from its saved equilibrium. Each changes one material array and independently re-equilibrates. Both Lamé parameters are scaled together, preserving that material's Poisson ratio.

| Change | Surface motion RMS | Smile projection | Mean physical fiber stretch | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: |
| Original materials | 0.490 mm | 0.0529 | 0.9370 | 0 |
| Selected-muscle Lamé parameters × 10 | 2.102 mm | 0.2471 | 0.7737 | 3 |
| Fat Lamé parameters × 0.1 | 0.758 mm | 0.0757 | 0.9121 | 0 |
| Aponeurosis Lamé parameters × 0.1 | 0.815 mm | 0.0884 | 0.8962 | 1 |

All three counterfactuals met the declared strict equilibrium tolerance. Increasing selected-muscle stiffness produces 4.29 times the surface motion; softening either surrounding material also increases the response. These controlled sensitivities support an actuation-to-resistance limitation in this fixture. The tenfold change increases passive muscle stiffness and active-strain stress together. It is a sensitivity test, not a calibrated tissue value or a pure active-force multiplier. These warm-start equilibria have not been independently reproduced from rest. [Exact-control hashes, material changes and solver receipts](../data/11-manual-sensitivities-v2/summary.json).

The constitutive law shows that Raw6 can produce much greater initial actuation stress in some volumetric directions than bounded volume-preserving fiber contraction. For `B = A_inv`, the implemented first Piola stress at a fixed physical `F = I` is

```text
P(I, B) = mu B Bᵀ + det(B) [-mu + lambda (det(B) - 1)] I.
```

For a determinant-one active map at this fixed rest geometry, the expression reduces to `P = mu (B Bᵀ - I)`: changing the volumetric coefficient `lambda` does not amplify the initial activation stress. Under load, that coefficient still affects the equilibrium through the physical deformation `F`.

At the current muscle material, c50 fiber activation gives principal stresses `(24.66, -4.11, -4.11)` kPa. An isotropic Raw6 map with exactly the same Frobenius distance from identity gives `(1436.81, 1436.81, 1436.81)` kPa in the expansive inverse-active direction: about 98 times the stress norm. The opposite isotropic offset gives only 0.30 times the fiber stress norm, so this response is strongly asymmetric. All these comparisons hold the physical geometry at `F = I`, with `det(F) = 1`. The difference comes from how the active map enters the volumetric elastic energy. Equal numerical control size therefore does not mean equal actuation strength across these parameterizations. This calculation is for the pure-muscle constituent before mixture weighting, and does not claim that the inverse solution uses either exact isotropic pattern. [Derivation and finite-difference verification](16-actuation-stress-mechanism.md).

![The same physical deformation and active-map offset norm can produce very different elastic determinants and actuation stresses.](../data/17-actuation-stress-figure/actuation-stress.png)

[Separate vector figure](../data/17-actuation-stress-figure/actuation-stress.pdf) · [Frozen values and figure hashes](../data/17-actuation-stress-figure/manifest.json).

## The no-skin reference and the matched smooth-field experiment

The archived June endpoint has the expected combination of close target fit and rough geometry: area-weighted error 0.654 mm, motion 4.974 mm, and target projection 0.968. It includes 142 inverted tetrahedra and 159 active tensors with a nonpositive eigenvalue. The best saved state is step 194 of a 200-step budget; the original run reported six failed forward evaluations and did not establish inverse convergence. Exporting that state performed no new equilibrium solve. [Historical evidence and configuration differences](11-historical-baseline.md).

The September fixture kept 120,020 active cells in 35 expression-muscle labels, compared with 288,235 cells in 103 labels in June. It also added 6,600 fixed cut vertices and changed materials, objective weights, optimizer and tolerances. The historical endpoint moved those newly fixed vertices by 1.231 mm RMS. The excluded controls contained 32.2% of its volume-weighted squared activation offset. These diagnostics establish that the configurations differ materially; they do not assign a causal contribution to individual excluded muscles.

For each active tetrahedron `i`, Raw6 keeps six independent coordinates of `H_i = A_inv,i - I`. Raw6-S adds only the following within-muscle term:

```text
R = ell² / (Vmuscle a_ref²) * sum_(i,j) w_ij ||H_i - H_j||_F² / 1.5
w_ij = shared_face_area / center_distance * harmonic(MuscleFraction_i, MuscleFraction_j)
ell = 5 mm; a_ref = -log(0.8)
objective = fitting_error + lambda_s R
```

The graph includes face-sharing pairs with the same muscle label. No coordinate is removed. Finite regularization penalizes rapid variation and permits gradual variation within a muscle; it does not force one constant activation value per muscle or impose exact mathematical continuity on a piecewise-constant field. In graph modes, the quadratic term penalizes high-frequency modes more heavily while retaining the low modes and all higher modes in the optimization space.

On the 120,020-cell fixture, the graph has 210,187 edges and 132 connected components. The 67 isolated cells occupy only 0.0142% of muscle-fraction-weighted active volume. Exact constant-per-muscle fields have zero penalty. The independent analytical/autograd gradient check agrees within 1.12e-16, including the symmetric tensor off-diagonal multiplicity. [Graph and derivative audit](../data/45-diagnosis-comparison/regularizer-audit.json).

The endpoint comparison uses the same rest skin, area weights and vertex mapping for every method. It measures both the displacement and the target residual along the rest-surface normals. At each 2, 5 and 10 mm scale, the high-pass field is the scalar field minus its diffused low-pass, obtained from `(M + t K)y = Mx`, with `t = scale²/4` and natural no-flux boundaries. Results cover the full face and a mouth neighborhood within 10 mm intrinsic edge distance of the lip vertices. These descriptive metrics retain target detail as well as artifacts; they require joint interpretation with fit, motion and the actual geometry. They do not modify the displayed vertices.

All endpoint tables below use area-weighted surface errors. The matched Adam traces separately report the historical uniform target-vertex RMS and optimize one third of its square, in mm², plus the field penalty. Keeping these definitions separate avoids interpreting a weighting change as improved fitting.

### The matched early trend

The historical-configuration pair retains all 288,235 active tetrahedra, 1,729,410 scalar controls and 501,409 within-muscle graph edges. Both methods use the original material arrays, fixation and no-skin energy, Adam learning rate 0.3 and epsilon 0.01. Raw6-S uses `lambda_s = 0.0005`; Raw6 uses zero. The weight was selected from the initial penalty-to-fit scale, not tuned on the final surface result. The [matched contract](12-matched-historical-adam.md) records the selection and equations.

The initial runs were interrupted after steps 54 and 60. The reported continuation starts each method from its own saved step-50 controls and explicitly resets Adam moments in both. It was intentionally stopped around 64 new local steps to assess the trend: Raw6 finished step 65 and Raw6-S step 64, with exit status 0 and complete checkpoints. The archived 150-step ceilings were not reached. Local indices therefore follow an earlier 50-step prefix; they are not total uninterrupted step counts. [Execution and continuation record](18-interruption-and-continuation.md).

At the common local step 64, uniform target-vertex fit RMS is 0.842 mm for Raw6 and 0.779 mm for Raw6-S. Graph variation falls from 117.41 to 104.07, an 11.4% reduction. Raw6 has one unsuccessful finite forward evaluation at local step 51; the later evaluations pass. Raw6-S has successful forward and adjoint receipts throughout. The best-state export excludes unsuccessful evaluations.

![The common continuation trace through step 64. The red cross records Raw6's unsuccessful forward evaluation.](../data/48-matched-historical-adam-continuation/fit-rms-vs-step.png)

[Vector fit curve](../data/48-matched-historical-adam-continuation/fit-rms-vs-step.pdf) · [Field-variation curve](../data/48-matched-historical-adam-continuation/graph-variation-vs-step.png) · [Vector field-variation curve](../data/48-matched-historical-adam-continuation/graph-variation-vs-step.pdf) · [Exact paired trace and provenance audit](../data/48-matched-historical-adam-continuation/summary.json).

The selected best states are Raw6 local step 50 and Raw6-S local step 64. Raw6's extra in-flight step 65 did not improve its best state. The following values come from the actual saved meshes and common rest-surface area weights:

| Method | Selected local step | Surface fit RMS | Surface motion RMS | Historical-graph variation | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw6 | 50 | 0.828 mm | 4.865 mm | 114.64 | 158 |
| Raw6-S | 64 | 0.804 mm | 4.877 mm | 104.07 | 138 |

The selected Raw6-S state has 2.9% lower area-weighted fit error and 9.2% lower field variation, with almost unchanged motion amplitude. Across the 2, 5 and 10 mm scales and both spatial supports, normal-displacement high-pass RMS decreases by 0.26–1.10%; normal-residual high-pass RMS decreases by 1.34–3.70%. These changes are small relative to the persistent visible roughness. The [Raw6 full-face view](../data/60-final-diagnosis-viewer/historical-adam-raw6/full-head.png) and [Raw6-S full-face view](../data/60-final-diagnosis-viewer/historical-adam-raw6-smooth/full-head.png) retain the actual deformed vertices.

An equal-step geometry check uses the actual local-step-60 meshes from both completed exports. Area-weighted fit improves from 0.866 to 0.818 mm with Raw6-S, while motion changes from 4.834 to 4.859 mm. The surface differences remain small:

| Filter scale | Displacement high-pass change, full face / mouth | Residual high-pass change, full face / mouth |
| --- | ---: | ---: |
| 2 mm | −0.53% / +0.17% | −2.95% / −2.74% |
| 5 mm | −0.76% / −0.54% | −2.31% / −2.56% |
| 10 mm | −0.35% / −0.55% | −2.22% / −2.85% |

Percentages are Raw6-S relative to Raw6. The 2 mm mouth displacement measure increases slightly, so even this matched comparison does not show a uniform surface-smoothing effect. [Equal-step-60 surface audit](../data/69-completed-step60-surface/summary.json) · [Exact saved-state manifest](69-completed-step60-surface-manifest.json).

These early results already distinguish a smoother activation field from a smooth output surface. With this graph and weight, the inverse retains large expression amplitude and close target fit, while surface bumps remain. The result supports testing field regularization without reducing the control space; it does not establish that the chosen penalty strength is sufficient. [Complete common-surface metrics](../data/44-final-surface-comparison/summary.json) · [Comparison table](../data/44-final-surface-comparison/comparison-table.csv).

### Current-fixture diagnostics

The separate current-fixture pair retains 720,120 coordinates and uses the September materials, fixation, area-weighted normalized objective and strict L-BFGS solver. Both terminate through numerical line-search stalls, rather than a tetrahedron-orientation gate.

| Saved endpoint | Accepted iterations | Fit RMS | Motion RMS | Current-mask field variation | Inverted tetrahedra |
| --- | ---: | ---: | ---: | ---: | ---: |
| Raw6 | 48 | 2.023 mm | 4.381 mm | 2216.95 | 3915 |
| Raw6-S | 9 | 3.174 mm | 2.933 mm | 250.18 | 2 |

These endpoints have different optimization histories and cannot establish a final smoothness tradeoff. At the common recorded step 9, Raw6-S reduces field variation from 405.23 to 250.18, approximately 38.3%, while fit changes from 3.203 to 3.174 mm. No immutable Raw6 mesh was saved at step 9, so this trace comparison does not imply an equal-step surface-roughness result.

The last rejected inner line searches have gradient norms around `2e-12`–`2e-11` and relative energy changes around `1e-15`–`1e-14`. These receipts are consistent with precision and stopping-rule sensitivity, rather than an orientation-based rejection. They do not make the failed solves pass their declared tolerance. [Raw6 trials](../data/20-raw6-no-skin/trials.json) · [Raw6-S trials](../data/21-raw6-smooth-no-skin/trials.json).

Raw6's independent rest solve succeeds under the same relative-tolerance rule, but differs from the saved endpoint by approximately 0.963 mm surface RMS. The seed-dependent stopping threshold differs between that solve and the warm trajectory, leaving tolerance sensitivity and branch sensitivity unresolved. Raw6-S's independent rest solve reaches 10,000 iterations without success. The accepted inverse endpoints are preserved, and neither run certifies inverse stationarity or a uniquely reproduced surface. [Raw6 receipts](../data/20-raw6-no-skin/summary.json) · [Raw6-S receipts](../data/21-raw6-smooth-no-skin/summary.json).

## Release of the previous geometric floor

The constrained Region5Modes continuation uses the previous 700-coordinate controls, materials, skin factor 0.12, magnitude penalty and saved initial state. It removes the earlier geometric acceptance floor and starts a new L-BFGS history. Over 40 additional iterations, fit improves from 4.631 to 4.292 mm and surface motion increases from approximately 0.685 to 1.066 mm. An accepted intermediate state reaches `min det(F) = 0.000343`; this state is retained. The final minimum is 0.0422 with no inverted cells.

An independent equilibrium solve from rest succeeds and differs from the saved final surface by only 0.000135 mm RMS. This confirms the final forward equilibrium at the declared tolerance. The inverse run reaches its iteration budget with a nonzero projected-gradient diagnostic, so it does not establish a capacity limit. The improvement also includes 40 more optimizer steps and a restarted history; it cannot be attributed quantitatively to floor removal alone. Removing the floor permits useful further exploration but does not, in this run, recover the large target expression. [Continuation and reset receipts](../data/22-region5-no-floor/summary.json).

## What this establishes

The manual tests support investigating the strength of actuation relative to surrounding tissue resistance. Removing skin alone changes manual motion moderately; it does not restore a large smile. The constitutive stress calculation identifies an additional confound when comparing bounded fiber activation with unrestricted Raw6. Fiber directions remain geometry estimates, and attachments, jaw pose and missing contact remain unisolated modeling uncertainties.

A weak, nonstationary bounded inverse run cannot establish insufficient model capacity. A visually smooth face with little motion is also insufficient evidence that a regularizer solved the artifact problem. The relevant comparison retains per-tetrahedron freedom and reports target fit, expression amplitude, activation variation and surface roughness together. The regional low-dimensional controls remain useful as a diagnostic comparison.

## Reproduction and evidence

The [reproduction guide](../README.md) gives the environment, exact source inputs, numerical settings and analysis commands. New face trend comparisons can request 64 local steps directly. The archived continuations retain their original 150-step ceilings and the separate records of their intentional early stops.

| Experiment | Recorded outcome |
| --- | --- |
| Manual activation | All 18 strict equilibria succeeded |
| Material counterfactuals | All three exact-control equilibria succeeded |
| June no-skin reference | Exported saved best step 194; no new solve |
| Current-fixture Raw6 / Raw6-S | Numerical line-search stalls after 48 / 9 accepted iterations |
| Region5Modes without geometric floor | Completed 40 additional iterations; independent final rest solve succeeded |
| Initial historical Adam pair | Interrupted at steps 54 / 60; exit cause not captured |
| Historical Adam continuation pair | Intentionally stopped at local 65 / 64; process exits 0; best valid local states 50 / 64 |

The viewer contains 27 cases: 18 manual activation states, three material sensitivities, the saved June baseline, three current-fixture inverse endpoints and the two matched continuation endpoints. It includes actual saved optimization checkpoints and observed-target skin overlays. The [Raw6 export](../data/46-historical-adam-raw6-continuation-history/manifest.json) and [Raw6-S export](../data/47-historical-adam-raw6-s-continuation-history/manifest.json) verify their selected best states exactly against the saved arrays and successful solver receipts. [Process completion and checkpoint hashes](../data/68-trend-completion.json).

Separate figure assets include the [ten-case numerical comparison PNG](../data/70-final-comparison-figure/final-comparison.png), its [vector PDF](../data/70-final-comparison-figure/final-comparison.pdf), and the linked fit, field-variation and actuation-stress figures. The ten-case figure uses common surface metrics but includes different model configurations; only the declared matched pairs isolate the smoothness treatment.

The figure ZIP contains separate PNG/PDF files. The compact reproduction ZIP contains scripts, source snapshots, configurations, traces and solver/audit records. Full volume meshes, activation arrays, optimizer checkpoints and historical VTKHDF files remain local, identified by hashes. The final browser bundle is a surface and cutaway inspection artifact rather than a replacement for those complete numerical states.

[Geometry and provenance validation](../data/61-final-viewer-validation.json) accompanies the bundle. The private published report is available at PRIVATE_HOST:8768 (private preview omitted), with the 3D viewer (private preview omitted) at the same address.

All scientific geometry is displayed at its saved positions and 1× displacement. Rendering may average lighting normals but does not smooth vertices. The target overlay contains only the supplied observed skin displacement. Optimization histories contain actual saved equilibria and are not physical animations. Source snapshots, configuration files, input hashes and numerical receipts accompany each run; compact downloadable records and full local NPZ/VTU states have separate inventories.
