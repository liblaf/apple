# Joint activation, material, and mandible inverse experiment

> **Superseding operational note — adopted neutral.** The converged prescribed-
> skin/full-skull state is now the selected neutral. Preserve the original
> FEM/material reference and frozen equilibrium displacement; do not rebase the
> mesh stress-free. Baseline stress fields are frozen and no longer optimized.
> Expression targets preserve their transferred displacements from the adopted
> neutral. The active plan contains per-tet activation, jaw pose, and one global
> skin-stiffness multiplier; legacy Shared20/Spatial80 text below is design
> history only. See [the current contract](31-final-optimization-contract.md).

Design and readiness audit, 2026-09-21, revised after user approval. Target date: Tuesday, 2026-09-22, Asia/Shanghai; end of day is the scheduling assumption because no cutoff time was specified. **Tuesday's deliverable is inspection of the final large simultaneous joint optimization. Preparations must converge and include rich visualizations; only the final joint stage is exempt from inverse convergence by Tuesday. Strong smoothness regularization applies to every trainable continuous spatial field.** This document records the accepted design and its revisions, source inspection, and existing evidence; it is not a run report.

**Accepted direction.** Represent the effects previously assigned to passive pre-strain using shared baseline stress fields in all four tissues, and represent activation using an additional expression-specific muscle stress field. This uses one additive-stress mechanism and removes independent `Fp` optimization. Keep six expression-specific stress coordinates per active muscle tetrahedron. First establish a prestressed neutral equilibrium, then fit activation/jaw motion, and finally release the shared baseline stresses and skin stiffness in the same multi-expression objective. Prioritize a valid finite-budget joint trajectory and interpretable trends for Tuesday. Forward equilibrium, adjoint accuracy, and geometric validity remain acceptance gates; inverse stationarity is not required.

**Scope and variables.** The user explicitly selected design/readiness first and clarified that six activation degrees of freedom means **per muscle tetrahedron, per expression**. Region-level activation is therefore not a substitute for the final experiment.

The subsequent approval removes final joint inverse convergence from Tuesday's completion criteria and makes strong smoothness a primary modeling choice for all optimized spatial fields. The objective and comparisons below incorporate those instructions. Separate unordered expression targets do not imply a physical time sequence.

The latest modeling instruction unifies pre-strain effects and activation in additive stress. **Baseline stress is shared and can be signed; muscle activation is an expression-specific increment.** “Zero activation” means zero increment, not zero baseline stress. This replaces the earlier multiplicative-prestrain proposal throughout the operational plan. Its relation to that earlier model is stated below rather than assuming universal finite-strain equivalence.

| Quantity | Shared across expressions? | Proposed representation |
| --- | --- | --- |
| Fat baseline stress `S0,fat` | Yes | Signed symmetric reference stress, strongly smoothed on the fat domain |
| Generated aponeurosis baseline stress `S0,apo` | Yes | Signed symmetric reference stress, strongly smoothed on the constructed layer; reference distribution remains an evidence gap |
| Muscle baseline stress `S0,muscle` | Yes | Signed symmetric reference stress, strongly smoothed within continuous muscle domains |
| Skin baseline stress `N0,skin` | Yes | Symmetric tangential membrane stress resultant, with strong covariant surface smoothness |
| Skin stiffness | Yes | One positive global multiplier of a frozen reference distribution; any later trainable spatial variation receives strong log-stiffness smoothness; thickness fixed |
| Muscle activation increment `A_e` | No | Six symmetric tensor coordinates for every active tetrahedron; strong within-muscle tensor smoothness, PSD constraint, and fixed stress normalization; `A_0=0` |
| Mandible pose | No | Rigid SE(3) transform, three rotational and three translational coordinates; neutral pose fixed |

Fat, muscle, and aponeurosis stiffnesses and all Poisson ratios remain fixed in each run, matching the requested list of variables. Their alternative research-informed initializations are separate sensitivity cases. Shared baseline-stress fields initially have few spatial basis coefficients; dense per-tetrahedron activation does not require dense independent baseline-stress variables. All four shared baseline-stress families remain trainable in the final joint stage.

For the first neutral-feasibility test, use one constant symmetric baseline-stress tensor per bulk tissue (18 coefficients total) and one isotropic tangential skin-stress coefficient, plus the skin stiffness multiplier. This 20-parameter shared initialization tests whether a simple balance is possible; it does not assert a spatial literature distribution exists. The final shared basis may need regional refinement. Before any full run, freeze its functions, support, frames, coefficient counts, bounds, and basis-array hashes. Failure of this small basis does not prove that every possible baseline-stress field fails. Dense per-tet activation remains unchanged throughout.

## What the model actually contains

The source chain is: anatomical GLB surfaces → independently registered skin and bone templates → heuristic SMAS envelope → background tetrahedral mesh with sampled tissue fractions → transferred Faceform expression fields → Apple face fixture.

| Concern | Verified implementation | Implication for this experiment |
| --- | --- | --- |
| Anatomical source | The local `00-complete_human_head_anatomy.glb` has no recorded acquisition URL or author. Its geometry strongly matches [3D4SCI's Complete Human Head Anatomy](https://sketchfab.com/3d-models/complete-human-head-anatomy-c240eee6c2824f8cbb105129392711b2), but that is a forensic identification. | Do not describe this as a validated same-subject anatomical dataset. |
| Skin, bones, and oral cavity | The skin uses an XYZ ReadyToSculpt template registered to selected GLB skin/eyes/gingiva. Cranium and mandible are separately registered Sculptor meshes. The oral cavity inherits template geometry. | Registration uncertainty affects both passive geometry and jaw observations. |
| Muscle–skin and muscle–bone coupling | Muscle fractions are assigned after meshing. Skin and volume share displacement nodes. There are no explicit muscle origins, dermal insertions, or sliding interfaces. | Coupling exists through the continuum, but there is no independently calibrated attachment law. Freeze this topology in the primary inverse. |
| Aponeurosis | [SMAS construction](${MELON_ROOT}/exp/2026/05/27/head/src/32-construct-smas-muscle-span.py:25) uses normal rays with an 18 mm cutoff, heat extension, offsets, and Boolean subtraction. [Tissue fractions](${MELON_ROOT}/src/liblaf/melon/recipe/fractions.py:66) label generated SMAS outside muscles as aponeurosis. | Its inferred baseline stress describes this constructed layer. It cannot validate the layer's anatomical location. |
| Lips and gums | [Masks](${MELON_ROOT}/exp/2026/05/27/head/src/42-gen-masks.py:85) use overlapping 2 mm proximity bands. Source surfaces also have open/non-manifold features. | These are QA flags, not by themselves proof of erroneous fusion. Localize them and define actual moving/contact surfaces before jaw fitting. |
| Expression targets | [Gallery extraction](${MELON_ROOT}/exp/2026/05/27/head/src/60-blendshapes.py:8), [delta transfer](${MELON_ROOT}/exp/2026/05/27/head/src/61-delta-transfer.py:37), and [boundary interpolation](${MELON_ROOT}/exp/2026/05/27/head/src/62-disp-transfer.py:15) transfer unrelated expression displacements. Boundary interpolation uses a distance cutoff and zero fill. | Distinguish valid correspondence, actual zero motion, and unmapped data. Never fit zero-filled missing observations as measured stationary tissue. |
| Nasolabial folds | No controlled result establishes that aponeurosis selection causes their absence. | Inspect the target's crease first. Measure crease depth/position and landmark motion separately; retain attachment, layer, constitutive, mesh, and transfer explanations. |

Faceform describes its source blendshapes as captured shapes transferred to a generic face in its [official change log](https://faceform.com/docs/Wrap/WrapChangeLog/WrapChangeLog.html). The resulting targets are useful kinematic supervision, not measurements of this anatomy donor. More provenance detail is in the [existing pipeline audit](../../../../../../docs/research/2026-09-12-head-anatomy-pipeline-audit.md:15).

There are two important Apple fixtures: the September screened fixture has 35 activation regions and 120,020 active cells; the historical full-active fixture used by the tensor experiments has **103 regions and 288,235 active cells**. Both use 228,660 points and 1,146,517 tetrahedra. This plan uses the latter active domain, subject to a frozen input manifest. Its [fixture receipt](../../../07/face-actuation-diagnosis/data/12-historical-fixture/summary.json) records the distinction.

## Constitutive contract

**One stress representation, two roles.** Let `F` map the observed neutral geometry to the current configuration and `C=F^T F`. For each volume tissue `t`, define

$$
Q_{t,e}=S_{0,t}+\mathbf1_{t=\mathrm{muscle}}A_e,
\qquad A_0=0,
$$
$$
W_{t,e}(F)=W_{\mathrm{SNH},t}(F)+\tfrac12 Q_{t,e}:(C-I),
\qquad P_{t,e}=P_{\mathrm{SNH},t}(F)+FQ_{t,e}.
$$

Both stresses are symmetric second-Piola tensors in the neutral reference, expressed per neutral tissue volume. Multiply the complete tissue contribution by its neutral volume fraction once. Use the same constitutive formula for shared baseline stress and expression-specific activation; their supports, priors, and constraints differ. The inferred baseline is an effective residual stress, not a measured natural length or a separately optimized stress-free geometry.

Use the actual polynomial Stable Neo-Hookean variant already implemented:

$$
W_{\rm SNH}(F)=\frac{\mu}{2}(\|F\|_F^2-3)-\mu(J-1)
+\frac{\lambda_c}{2}(J-1)^2,\qquad J=\det F.
$$

Its base infinitesimal constants require `mu=E/[2(1+nu)]` and `lambda_c=E nu/[(1+nu)(1-2nu)]+mu`. These describe the unstressed base material; the added stress contributes geometric tangent terms at the prestressed state. Historical runs deliberately used a different lambda convention, so preserve their records and state the new convention explicitly. The [original paper supplement, equation 1](https://research.pixar.com/docs/2018.SiggraphPapers.SGK.b_suppl.pdf) identifies this as its initial polynomial energy. Apple's version omits the additional logarithmic term in the final paper formulation.

**Constraints differ between baseline and activation.** `S0,t` must permit both tensile and compressive eigenvalues, including pressure-like components. Keep the existing contractile activation constraint `0 <= A_e <= A_max I` in the spectral sense. Do not project the baseline, or the sum `S0+A_e`, onto the activation-only PSD cone. For bulk baseline stress, enforce

$$
\mu_t I+S_{0,t}\succeq\epsilon\mu_t I,\quad \epsilon>0,
$$

and a declared upper spectral bound. This supplies a positive directional quadratic coefficient for the current polynomial, avoids unbounded negative quadratic directions, and preserves rank-one coercivity when `lambda_c>=0`. It does not certify a positive full Hessian or stable assembled equilibrium. Baseline compression and upper bounds remain computational/prior choices to test during neutral calibration. Muscle increments retain all six orthonormal tensor coordinates per active tetrahedron and their separate spectral projection; normalization scales are fixed during optimization.

The experiment-local [tensor material](../../../07/tensor-active-stress/src/tensor_active.py:34) already has the required energy, stress, and exact additive tangent `dP[dF]=dF Q`. Reuse that mathematics for the total stress. Its existing caller contract assumes PSD `Q`, so integration must explicitly support signed baselines and test their admissibility; simply passing a signed field through the existing activation projector would erase compression.

**Relation to pre-strain.** The current [physical-volume preferred-metric energy](../../../../../../src/liblaf/apple/warp/fem/_stable_neo_hookean_active.py:20), `mu/2(||F B||²-3)+g(det F)`, has exactly the same forces and tangents as the stress form under

$$
S_0=\mu(BB^T-I).
$$

The energies differ by a parameter-dependent constant `tr(S0)/2`; this does not change equilibrium or its implicit derivative, but must not be mistaken for equality of absolute material energies. Thus the requested representation is an exact mechanical reparameterization of that project-local preferred-metric law.

It is not a universal equivalence to a multiplicative stress-free-volume law `W(F Fp)/Jp`, with `Jp=det(Fp)`. That law expands to `mu/(2 Jp) C:(Fp Fp^T) + lambda_c Jp J²/2 - (mu+lambda_c)J + constants`. At `Jp=1` the same stress mapping is exact; for general `Jp`, the changed `J²` coefficient cannot be represented by a fixed stress tensor with unchanged bulk coefficients. The operational choice here is the unified additive-stress model. No `Fp`, `log(Fp)`, or `1/det(Fp)` variables or derivative paths are needed in the new runner.

**Skin uses the corresponding surface stress.** Store a symmetric tangential membrane resultant `N0,skin` in N/m, in a frozen orthonormal neutral tangent frame. If `Fs` is the surface deformation gradient and `Cs=Fs^T Fs`, use the per-neutral-area energy

$$
\psi_s(F_s)=\psi_{s,\rm passive}(F_s)
+\tfrac12 N_{0,s}:(C_s-I_2),\qquad
P_{s,\rm added}=F_s N_{0,s}.
$$

`N0=h_ref S0,s` converts a volumetric stress to a membrane resultant only under the declared fixed reference-thickness convention. Never insert a Pa-valued 3D stress directly into surface metric coefficients. Skin may have a tensile prior, but its admissibility policy must be stated separately from muscle activation. For an `h_ref`-scaled plane-stress SNH membrane, a sufficient positive-quadratic bound uses `h_ref mu_s I + N0,s`; verify it for the final reduced energy, including the changing skin stiffness. Preserve a separate upper bound and magnitude prior.

The plan retains its SNH-skin assumption. Current [Koiter skin](../../../../../../src/liblaf/apple/warp/fem/_koiter.py:59) is a StVK-like membrane, so SNH skin still requires a surface implementation. Set `h_ref=1 mm` as a fixed modeling assumption, and use `psi_passive=h_ref W_plane_stress(Fs)` on the neutral area. For this polynomial, the eliminated normal stretch is `z=JA*(lambda_c+mu)/(lambda_c*JA²+mu)`. The tangential added-stress term does not depend on `z`, so it can be added to this reduced passive energy. This removes the prior pre-strain/thickness Jacobian bookkeeping. Consistent surface derivatives still need validation, and fine wrinkle prediction remains limited without bending.

## Neutral equilibrium and identifiability

The required neutral state is **prestressed and equilibrated**. Set `A0=0` and the neutral jaw pose, retain all shared baseline stresses, and solve the free-node equilibrium. At `F=I`, the base volume stress vanishes and each tissue contributes `S0,t`; these fields, skin membrane stress, and any declared loads/contact must balance on free nodes. Bone-support reactions need not vanish. Equilibrium does not require every tissue to have nonzero or tensile baseline stress.

Continue skin baseline stress from zero toward the selected prior while adjusting the other shared baseline stresses. At every continuation stage, solve the neutral problem. If the residual or neutral-shape budget cannot be met within the declared priors/admissibility bounds, report that inconsistency. Do not cancel it using hidden nodal forces, nonzero neutral activation increments, or an unreported reference-mesh change. Zero-centered priors are computational defaults where evidence is absent, not a requirement that all tissues remain unstressed.

Neutral muscle-shape preservation is an internal regularizer around the supplied asset, not validation against measured internal anatomy. A useful initial engineering budget is area-weighted surface RMS drift ≤0.25 mm and muscle-fraction/volume-weighted cell-centroid RMS drift ≤0.5 mm, accompanied by regional and 95th-percentile drift. Also measure muscle deformation-gradient distortion: centroid preservation alone can hide internal shear. If source muscle surfaces are embedded for evaluation, save their barycentric correspondence. These proposed tolerances must be frozen before the long run and must not be called measured uncertainty.

The expression-specific muscle energy depends on the sum `S0,muscle+A_e`. Fixing `A0=0` anchors the neutral expression, but neutral geometry observes equilibrium rather than the internal stress field: a shared self-equilibrated stress can still trade against the common part of expression increments, subject to the bounds. At mixed-tissue cells, the added force depends only on the fraction-weighted sum of baseline tensors. Independent tissue stresses that cancel in that sum are unobservable mechanically. Shared basis restrictions, strong smoothness, and magnitude/reference priors select a conditional decomposition; they do not make it uniquely biological.

Likewise, scaling all passive and active terms together leaves quasistatic equilibrium unchanged when there are no unscaled external loads, contact/barrier terms, or mechanical penalties. Unmeasured Dirichlet reactions do not identify that scale. Fixing the other tissue stiffnesses anchors the new inverse to assumed scales. Fixed skin thickness removes the direct `E*h` ambiguity, but absolute skin stiffness still inherits those assumptions.

## Research priors and initialization

No reviewed primary source supplies a facial-aponeurosis baseline-stress distribution. The checked skin research gives regional inverse prestress fits, not a registered full-face SNH parameter map. Stress parameterization permits using reported prestress directly as a source-qualified prior after its stress measure, frame, spatial mapping, and reference state are reconciled. It avoids inventing a conversion for an undefined pre-strain measure.

| Tissue | Source-qualified value | Use in this design |
| --- | --- | --- |
| Skin | Flynn et al. fit in-vivo facial skin using Ogden–QLV. A derived zero-stress tangent is about 204 kPa; central-cheek fitted prestresses are 89.4/71.8 kPa. The convention of “Equivalent Pre-strain” 0.33/0.24 is unresolved. [DOI](https://doi.org/10.1016/j.jmbbm.2013.03.004) | Supports an approximate 0.2 MPa base stiffness scale and model-dependent directional stress priors. If those prestresses are interpreted in the chosen neutral frame with `h_ref=1 mm`, their membrane-resultant scales are 89.4/71.8 N/m. This is a stated model translation, not a measurement on our asset. |
| Deep medial cheek fat | 11.2 ± 6.9 kPa from shear-wave elastography, 89 women. [Paluch full text](https://www.termedia.pl/doi_ft/10.5114/ada.2018.79778) | Apparent wave-based scale; 0.0112 MPa is a candidate research-informed seed, not measured static SNH Young's modulus. |
| Zygomaticus major | 12.0 ± 4.3 or 18.3 ± 3.7 kPa depending on probe, 15 volunteers. [Primary article](https://www.scirp.org/pdf/jbise_2019111915335847.pdf) | Two alternative apparent-modulus seeds, not interchangeable with active stress or a passive pre-strain measurement. |
| SMAS/platysma proxy | Cervical tensile modulus 1.693 ± 0.543 MPa, n=7. [Tereshenko figure 5](https://academic.oup.com/view-large/figure/541759926/ojaf126f5.jpg) | A distant regional/protocol stiffness proxy, not a baseline stress. Test its consequence on the generated layer before adopting it. |

The reported dispersions are not Bayesian uncertainty distributions for this model. Poisson ratios, thickness, registered spatial priors, and missing baseline stresses remain modeling choices. Record them separately from measured/fitted values. Passive modulus and baseline stress have the same units in the bulk but are distinct quantities; do not substitute one for the other.

A proposed research-informed stiffness sensitivity case uses the stated scales after constitutive matching; the historical `fat=0.003`, `muscle=0.03`, `aponeurosis=0.1` MPa case is retained as an evidence-continuity control. Protocol and anatomical differences prevent treating this sensitivity case as a calibrated central estimate. Do not silently replace a stiff research-proxy layer with the historical value to make a solve pass. This model change could materially alter conditioning.

For currently unsupported baseline stresses, initialize at zero and use tissue-specific stress scales, signed admissibility bounds, magnitude priors, and strong spatial smoothness. Set the numerical ranges during the neutral pilot; they are not biological confidence intervals. Prefer reported prestress, with explicit conversion, over converting an ambiguous pre-strain percentage. Remove the earlier proposed log-strain sensitivity brackets and all independent pre-strain variables from the operational configuration.

## Multi-expression objective and jaw mechanics

Let `theta` contain all shared baseline stresses and skin stiffness, and `(A_e, xi_e)` the expression-specific muscle increment and jaw pose. Solve each equilibrium

$$
r_e(q_e;\theta,A_e,\xi_e)=0,
\qquad x_e=c(\xi_e)+Bq_e.
$$

`c` contains rigidly transformed mandible nodes and fixed cranium coordinates; `B` injects free soft-tissue coordinates. The optimization minimizes

$$
L=\frac1N\sum_e D_e(x_e,y_e)
+\lambda_0 D_0(x_0,X)
+R_{\rm shared}(\theta)
+\frac1N\sum_e\big[R_A(A_e)+R_{\rm jaw}(\xi_e)\big].
$$

Use area-weighted positional residuals as the primary data term. Add a fixed, normalized target-gradient or normal residual only as a secondary term; derivative-only fitting can lose global position and jaw motion. Normalize each expression so a large-amplitude target does not automatically dominate. Confidence masks and any robust residual scale are fixed from registration QA before fitting. Report raw unweighted and area-weighted errors too.

`Rshared` includes the declared material/prior mismatch and strong spatial smoothness of every shared baseline-stress field. `RA` includes strong within-muscle smoothness of each activation increment and a nonzero magnitude penalty; preserve all six coordinates rather than imposing rank one. Regularize shared baselines and increments separately: smoothing only their sum would allow rough components to cancel. Freeze definitions and scales before comparisons. Apply regularization to reconstructed physical fields, not merely to arbitrary basis coefficients.

### Strong smoothness contract

Use first-order spatial smoothness of dimensionless field representations on the frozen neutral geometry. For a volumetric field `z`, define

$$
\mathcal S_t(z)=\frac{\ell_t^2}{V_t}
\sum_{(i,j)\in\mathcal E_t}w^{(t)}_{ij}
\lVert z_i-z_j\rVert_F^2,
\qquad
V_t=\sum_i V_i\phi_{t,i}.
$$

The graph uses shared tetrahedral faces; conductance is shared-face area divided by centroid distance, weighted by the harmonic mean of the tissue fractions. All weights, volumes, and graph edges are fixed in the neutral reference. Positive weights define a nonnegative quadratic regularizer. Keep graphs within the intended continuous tissue domains and, for muscle fields, within muscle labels. Proximity alone must not introduce smoothing across opposed lips, a sliding/contact interface, or unrelated muscles. Missing exterior neighbors impose no ghost value or artificial decay to zero. Archive graph hashes, total weights, disconnected components, singleton mass, and interface exclusions.

| Field | Quantity to smooth | Domain / metric |
| --- | --- | --- |
| Muscle activation increment `A_e` | Full symmetric `A_e / A_ref` tensor | Same-muscle tetrahedron graph, separately for each expression; smooth both magnitude and orientation through the tensor |
| Fat, aponeurosis, and muscle baseline stress | `S0,t / S_ref,t` | Corresponding tissue graph; compare tensors in a common neutral coordinate frame |
| Skin baseline stress | Tangential `N0,s / N_ref,s` | Surface graph with transport between tangent frames before tensor subtraction |
| Spatial skin stiffness | `eta=log(E_skin/E_unit)` | Surface scalar gradient or positive-weight surface graph; smooth relative stiffness variation |

`A_ref`, each bulk stress scale `S_ref,t`, each membrane-resultant scale `N_ref,s`, `E_unit`, and each spatial length `ell_t` are fixed normalization choices, not trainable variables. In particular, changing skin stiffness must not weaken regularization by changing its normalization. The existing activation graph's 5 mm length is an initial computational convention; freeze a length for each field before selecting weights. Smooth tensor differences rather than eigenvector angles, which are ambiguous near sign changes and repeated eigenvalues.

Where pressure-like and deviatoric baseline stresses have different scales, use `||dev(S0)||²/s_dev² + tr(S0)²/(3 s_hyd²)`, with both scales in stress units, and apply the same split to edge differences and prior mismatch. Fix these scales before calibration. Transform locally stored bulk tensors into the common neutral frame before comparison.

For a triangle-based surface graph, use shared-edge length divided by centroid distance as conductance and normalize by surface area rather than volume. Skin-stress differences take the form `N0,i - U_ij N0,j U_ij^T`, where `U_ij` transports the neighboring tangent frame into the current one. Define and test that transport, with consistent normals and reciprocal edges, so surface curvature or a change of local basis does not create artificial roughness. Scalar log-stiffness needs no frame transport.

Apply these penalties to the total represented field, alongside a separate reference/prior penalty; a smooth correction alone must not conceal a discontinuous base field. The initial constant bulk tensors and scalar skin correction are already smooth. If a family has no free spatial variation, its spatial penalty is identically zero or constant and is not counted as an effective optimization force. In particular, the current global skin multiplier cannot change `grad(log E_ref)`: record the fixed map's roughness, constrain the multiplier with its prior, and activate a genuine spatial stiffness penalty if regional coefficients are later introduced. This does not add new spatial degrees of freedom by itself.

**Weight selection.** Strong regularization is the default for the primary joint run. Use a short discarded calibration probe at feasible, nonuniform field states generated within the actual trainable basis. Archive that state and the normalized neighbor-RMS budget for each family before the primary run. For each spatially variable family, compare the smoothness-gradient RMS with the data/neutral-objective gradient RMS in its normalized field coordinates; use the mapped coefficients only when that coordinate metric is explicitly recorded. Choose a reference coefficient that gives comparable gradient magnitudes, and use a nominal factor of three above that coefficient as the initial strong candidate and lower bound for the primary weight. This is an optimizer calibration convention, not an anatomical measurement. Use the same expression averaging and volume/area normalization as the final objective. If either calibration gradient is zero or dominated by solver noise, do not divide by an arbitrary epsilon: use another admissible nonuniform probe, or record that the parameterization has no spatial mode to penalize.

Inspect the actual combined optimizer update after projection, rather than assuming gradient ratios determine Adam's update. Choose a stable step size for the strong objective and verify the declared roughness budgets during the pilot; if necessary, increase the weight or report a conflict between the strong prior and the attainable fit. Freeze each field's separate weight before the reported trajectory and retain it through the main run. Do not silently weaken smoothness in response to worse fit. Save achieved-strength diagnostics: neighbor RMS, field/prior amplitude, weighted smoothness contribution, data/regularizer gradient RMS, and proposed-versus-projected update roughness. A short stronger-weight sensitivity may be added if time permits; it starts from the same saved state and optimizer initialization. Broad regularization sweeps are not a Tuesday prerequisite.

Smoothness controls spatial variation but has constant-field null modes. Magnitude priors, signed baseline-stress admissibility bounds, PSD activation-increment bounds, and neutral-shape constraints remain necessary. Track each baseline, each increment, their total stress, target-fit error, and expression motion separately. An unfavorable fit/smoothness tradeoff is a valid trend to report.

Mandible pose is one rigid transform per expression, not a spatial field. Keep its pose prior and bounds. Do not smooth jaw poses or activation across arbitrary expression ordering, and do not penalize changes between optimizer iterations as if those iterations were physical time. The equilibrium displacement is governed by the mechanics; target-relative shape derivatives remain data terms rather than substitutes for material/activation smoothness.

For mandible motion, define the rigid node set, pivot, coordinate frame, neutral transform, and disjoint cranium support set. Apply rotation and translation to the bone-connected boundary, not just the displayed bone. Include the direct dependence of observed positions on pose and the implicit dependence through equilibrium. With adjoint `H^T a = L_q`, the derivative is `L_xi - a^T r_xi`; merely mutating fixed values does not supply it.

The current historical fixture has teeth and gingiva masks but no complete `IsMandible` field. Older [preparation code](../../../../05/27/inverse-face/src/10-prepare-inverse-face.py:30) did use separate cranium/mandible masks. Recover the correspondence to that source, verify the entire boundary selection, and record the transfer. Teeth/gingiva masks cannot stand in for all mandibular support nodes. A pose derivative check must include a loss on free tissue only, so an omitted implicit boundary contribution cannot pass through direct jaw-node motion alone.

Oral masks are not ready-made collision surfaces. Localize lip–lip, lip–teeth, and cavity intersections, define exclusions, and validate prescribed jaw motion before inverse fitting. Retain jaw variables in the full objective. If oral contact cannot be made reliable, restrict a pilot to a verified non-contact motion envelope and label its restricted scope; a fixed-jaw substitute does not complete the requested experiment. Outer facial motion alone weakly constrains jaw pose, so source-derived jaw/lower-tooth observations or an explicitly assumed pose prior are needed.

**Expression cohort.** Start with four training targets chosen after QA to span distinct motion patterns, including at least one jaw-sensitive target if the jaw gate passes. Reserve two other target fields before tuning shared priors. Candidate categories are smile, lip pucker, brow motion, and modest jaw opening; use the actual available names and inspect transfer quality before final selection. The fixture contains 36 named displacement arrays, including variants; do not equate that count with 36 independent measurements.

For reserved expressions, freeze shared parameters and infer only their activation and jaw pose from a preselected subset of landmarks/vertices. Score disjoint withheld observations. Fitting all reserved-expression vertices and then reporting their training error is only a material-transfer adaptation test, not a prediction test. Even spatially withheld data share the same transfer process, so this does not establish independent subject-level validation.

## Implementation readiness and computational plan

| Component | Status from inspection | Required before the large joint run |
| --- | --- | --- |
| Per-cell tensor active stress | Experiment-local path exists, with recorded constitutive/adjoint checks and full-face fits | Reuse its additive-stress mathematics for total stress; retain packed per-tet muscle increments |
| Shared baseline stress in all tissues | Tensor law is reusable; shared signed fields and priors are not integrated | Add separate baseline/increment assembly, signed-stress bounds, mixture weights, and shared-parameter derivatives |
| SNH skin | Existing skin is Koiter | Add and validate plane-stress SNH membrane if SNH applies to skin |
| Jaw optimization | Fixed-node map has no integrated pose input to inverse | Add differentiable rigid boundary and direct/implicit pose terms; verify oral geometry/contact |
| Multi-expression state | Core backward reads mutable solver state | Use owned per-call snapshots, or strictly finish each forward/backward before reusing a solver; verify expression-order invariance |
| Long-run evidence | Existing results are fixed-material and primarily single-expression | Benchmark the actual new joint epoch, including neutral and all adjoints |

Use the current Apple checkout as the implementation base after snapshotting its uncommitted dependencies. `apple-next` is an ancestor of the current branch and supplies no alternative shared-stress, jaw, or multi-expression architecture. The stress-only revision removes the need for multiplicative-prestrain kernels and Jacobian derivatives; signed-stress integration, skin, jaw, neutral balance, and state ownership remain real implementation tasks.

The [current autograd context](../../../../../../src/liblaf/apple/inverse/_diff_forward.py:124) saves material inputs but later differentiates using `ctx.forward.state`. Queuing multiple expression forwards through the same instance before backward can use the wrong state. A minimal sequential implementation accumulates gradients with parameters frozen for the whole epoch: rebuild every non-leaf material transform from shared parameter leaves, solve, call that expression's scaled loss backward immediately, store only detached metrics, and save owned detached primal/adjoint warm starts. Restore the corresponding warm starts for each expression; the adjoint solver also carries mutable state. Evaluate neutral with its own immediate backward and shared priors once, then update all blocks. Never accumulate live expression losses for one later backward through the reused solver. Verify gradients and reversed expression order against an isolated two-expression reference within declared solver tolerances; moving to batching later requires proper state ownership. Warm starts are solver hints, never alternate reference configurations.

On the historical active domain, one expression has **1,729,410** activation scalars. Four have **6,917,640**; six have **10,376,460**. Parameter, gradient, and two Adam-moment arrays alone use about **211 MiB** for four expressions in float64. This excludes dense tensor expansions, FEM state, adjoint buffers, and solver workspace. The available GPU at audit time was an RTX 4090 with 24 GiB; that snapshot is not a resource reservation. Process expression solves sequentially and release tapes promptly.

Existing tensor stress reached [1024 updates](../../../07/tensor-active-stress/docs/108-learning-rate-report.md:1) at 1.610246 mm fit RMS with zero inversions but minimum `det(F)=0.135720`; neither settling rule passed. This supports feasibility of dense activation, while showing that no-inversion alone is insufficient and convergence is not already solved. Recent [mixed-loss runs](../../../19/mixed-loss-learned-axis-face/docs/10-results.md:1) also failed their inverse convergence criteria. Their no-skin, fixed-material timing is not a benchmark for the new model.

The tensor continuation's recorded selected checkpoint chunks sum to about 2.35 hours; later 64-update chunks took about 5.7–6.8 minutes. Historical processes shared the GPU, and the initial 64-update chunk took about 40.8 minutes. This is selected-path execution time, excluding discarded alternatives, full study overhead, and the new model's costs, not a clean throughput benchmark. It is not a two-hour forecast for a joint inverse. Four expressions, neutral balancing, skin, contact, and shared baseline-stress/material changes all add work.

Measure warmed timings of forward, adjoint, regularization/projection, neutral, and checkpoint work at both neutral and a deformed state. Then estimate `T_epoch = T_neutral + sum(T_forward,e + T_adjoint,e) + T_other`. Allocate the final budget using the slower pilot case and time for validation and failure investigation. Do not select an iteration count solely by borrowing an old learning rate or wall time.

## Required soft tissue–bone contact

The user explicitly requires contact between soft tissue and bone using the **complete registered source cranium and moving mandible**, not partial bone faces extracted from the tetrahedral mesh. Retain all 35,162 cranium and 18,948 mandible source triangles and their registered coordinates. The source meshes are collision obstacles; FEM attachment nodes retain their separate mechanical role. Implement the contact potential and collision-safe line search in equilibrium; post-hoc overlap counting alone is insufficient. Keep exact global FEM node correspondence and preserve attachment constraints. Validate contact energy, forces, exact Hessian products, boundary/jaw derivatives and expression-state isolation with active contact. Each adjoint needs an owned contact state rebuilt at its saved equilibrium. Collision-safe checks must cover movement of the complete source mandible as well as its prescribed FEM support nodes.

The complete sources intersect the original soft-tissue boundary, so replacement requires a collision-free, non-inverted initial displacement before equilibrium. This must not silently change the FEM reference shape, targets, source bone coordinates, or exclusions. The partial-FEM neutral run is stopped and preserved as legacy evidence; its derivative and convergence receipts do not admit the replacement model. Report matched collision-on/off forward and adjoint times separately from contact-only operation timings. The source cranium and mandible also have 128 raw intersection pairs at their supplied neutral pose; the earlier partial-FEM bone-bone CCD guard does not establish validity for these complete meshes.

The first declared numerical barrier is the area-weighted physical IPC clamped-log potential with 0.1 mm activation distance and 0.01 MPa stiffness. These are computational contact parameters, not measured biological interface properties. Report minimum active gap, active pairs, barrier energy, CCD fractions, contact forces and attachment error, with contact-location overlays on both bones. A no-contact neutral solve is optimizer development evidence only and cannot complete final preparation.

The earlier FEM oral audit counted adjacent triangles sharing exact vertices/edges. Those bonded-junction pairs are not evidence of free-surface penetration. The corrected audit retains raw counts while excluding only contact confined to shared topology; nonadjacent intersections remain explicit geometric failures. The source template's separate 17 lip intersections and its incomplete correspondence to FEM surfaces remain model limitations, not evidence that every FEM state intersects.

## Gates and schedule for Tuesday's trend inspection

The target is a finite-budget final joint trajectory with all requested parameter families active and enough saved states to inspect trends. The user clarified that preparation stages must converge and have rich visualizations. Neutral balancing and fixed-shared activation/jaw initialization therefore require declared projected-gradient and objective-stabilization criteria, in addition to converged forward/adjoint solves and geometric diagnostics. A fixed update budget, an improving curve, or a neutral pilot does not complete preparation. Only the final simultaneous joint stage may remain inverse-unconverged by Tuesday. Monday's readiness check covers derivatives, converged neutral balance and initialization, jaw inputs, strongly regularized stability, and measured epoch time. Unresolved constitutive or jaw-gradient errors still prevent scientifically usable trends; failed gates must be reported explicitly. Set the run's time/update budget from the pilot and reserve time on Tuesday to inspect the saved trajectory.

| Stage | Earliest conditional slot | Exit evidence |
| --- | --- | --- |
| Freeze data and mechanical contract | Monday morning | Input hashes; exact fixture/active mask; tissue/target QA; skin law; shared signed stress / PSD increment convention; jaw node/contact definitions; source-qualified priors |
| Verify new derivatives and neutral balance | Monday daytime | Constitutive and mixed-derivative checks; jaw finite differences; two-expression state-isolation check; converged nonzero neutral stress balance, stationary constrained gradient and stable objective, neutral surface and muscle budgets met with zero activation |
| Full-mesh pilots and baseline | Monday afternoon/evening | Successful forward/adjoint at all selected targets; frozen strong smoothness weights and update scale; measured epoch time/memory; converged fixed-shared-material activation/jaw initialization and rich visual QA |
| Joint optimization trajectory | Monday night through Tuesday | All four shared baseline-stress families, skin stiffness, and per-expression activation/jaw blocks active under one frozen strongly regularized objective; periodic full checkpoints; no inverse-convergence requirement |
| Trend inspection and report | Tuesday, with a reserved final work block | Loss/fit/roughness/parameter/pose/neutral-validity curves; same-view checkpoint comparisons; baseline comparison; actual update budget and stop reason |

Start with neutral calibration, then activation/jaw fitting at fixed shared parameters, then release the shared parameters with scaled block steps. Finish with a simultaneous joint stage on the full active domain. Earlier staged passes are preparation and must converge; they are not the final claimed result. Diagnose a stationary but shape-invalid neutral state as a material-basis/model feasibility issue rather than calling it successful preparation. Do not silently settle for a 1% or 10% research-proxy continuation stage as the final material target.

Minimum validation before admitting an optimizer gradient:

- Constitutive energy/force/Hessian consistency and rigid-rotation objectivity. Setting both baseline and activation stress to zero recovers the unstressed base law; setting only `A0=0` retains the prestressed neutral model.
- Check the composed base-plus-total-stress material, including signed admissible baselines and PSD increments. Verify separate shared and per-expression gradients, the corrected unstressed small-strain tangent, lower/upper baseline bounds, and invariance to different baseline/increment decompositions with identical total stress when priors are excluded.
- Assert a one-to-one packed-coordinate map for all 288,235 active cells, expected control-ID coverage, zeros on inactive cells, and exactly one application of muscle fraction. Store the active-ID hash; no regionwise actuator can silently replace the required six coordinates per active tetrahedron.
- Directional finite differences for each shared baseline-stress family, skin stiffness, per-expression activation increment, and jaw rotation/translation. Check multiple step sizes with tighter forward tolerances. Existing 2% directional-gradient gates are a starting engineering tolerance, not evidence that new paths pass.
- A small synthetic problem with known parameters: demonstrate response recovery and explicitly show any parameter ambiguity. Failure to recover a non-identifiable parameter is not necessarily a derivative bug.
- Successful forward and adjoint residuals, finite values, no inverted elements, valid oral geometry/contact, signed baseline-stress admissibility, and PSD activation-increment bounds. Predeclare a compression/distortion gate using the pilot; store determinant tails and volume changes, not only inversion counts.
- Perturb-and-resolve checks around the neutral equilibrium and at least one fitted state. A small force residual alone does not certify mechanical stability; log any negative-curvature evidence.

For the reported trajectory, record projected gradients for every constrained parameter block, objective change, neutral drift, fit/motion/crease diagnostics, active-stress cap occupancy, material-prior departures, and validation errors. For the final joint stage, projected gradients and objective spans are diagnostics; its inverse-stationarity threshold need not be reached by Tuesday. Preparation stages must satisfy their predeclared stationarity and stabilization thresholds. Each accepted forward/adjoint still satisfies its declared solve tolerance. A budget-limited trajectory with substantial inverse gradients can meet the trend-inspection objective.

The Tuesday report should include:

- Total objective and its separate data, neutral, prior, magnitude, and smoothness terms; per-expression surface fit and motion. Plot against joint epoch and elapsed solve time, mark learning-rate changes and the point when each shared family was released.
- Roughness and amplitude for every spatial field, including per-tissue signed baseline-stress spectra, pressure-like/deviatoric components, skin membrane resultants, skin stiffness, and activation-increment spectra/cap occupancy. Also report total muscle stress; record lower-bound margins for `mu I+S0`. Do not present these as recovered pre-strain or a stress-free geometry.
- Mandible rotation/translation trajectories in the fixed reference frame; zero-activation surface and muscle drift; forward/adjoint residuals; physical `det(F)` tails, inversions, and oral/contact diagnostics.
- Initial, early, intermediate, and latest admissible checkpoints rendered at the same physical scale, camera, and color limits. State the actual number of accepted joint updates. Material maps and muscle cross-sections should accompany skin geometry.
- A finite-budget fixed-shared-material control under the same strong activation regularization. Compare fit, movement, roughness, and parameter changes at matched update/solve budgets, with wall time reported separately. Reserve held-out targets in advance; a full held-out adaptation study and an extra initialization are secondary to obtaining and inspecting a valid joint trajectory.

Stable, improving, stalled, or conflicting trends are all reportable. The deliverable is evidence about the evolving joint fit, not a requirement that every metric improves or that inferred parameters are uniquely biological.

The primary comparison is **fixed shared materials + activation/jaw** versus **joint shared materials + activation/jaw**, with identical targets, stress space, data losses, active-stress regularization, admissibility rules, and declared tuning opportunities. Shared-field priors/penalties are constant in the fixed-material arm; compare data metrics and common terms explicitly rather than ranking unlike total objectives. Report both solve counts and wall time. Include a short second shared-parameter initialization when runtime permits; if fitted materials vary greatly while shape fits remain similar, report the ambiguity rather than selecting one as ground truth.

Every checkpoint must contain shared and expression parameters, all optimizer moments and counters, the next learning-rate state, detached primal/adjoint warm starts, the frozen data/config/source manifest, solver receipts, and stop status. Preserve best admissible and terminal states separately. A Cherries/process exit code is not proof of equilibrium or inverse success; inspect scientific status receipts. Run new experiments through Cherries with explicit names/tags and a noncommitting profile, then package high-cardinality outputs per run before considering Git tracking.

**Tuesday completion rule.** Complete converged preparation with visual evidence, then deliver a reproducible, finite-budget final simultaneous joint trajectory with all requested parameter families included, strong smoothness on every trainable spatial field, and the trend inspection above. Inverse convergence is explicitly not required; an unconverged but valid joint trajectory can satisfy this objective. If only a fixed-jaw, activation-only, or numerically invalid run is available, report the missing scope or failed gate. Anatomical repairs and SMAS/attachment ablations remain separate follow-up experiments so the main fit has a fixed model to interpret.

## Audit record

The paragraphs below describe the original design audit. Current implementation and preparation evidence is maintained separately in `20-implementation-and-pilots.md`, `21-neutral-convergence.md`, and the preparation visualization report. The latest user correction explicitly requires converged preparation and rich visual QA; the 8/12-update neutral pilots remain preliminary only.

The audit read live code, fixture/protocol receipts, existing result reports, Melon source/history, and primary literature. It also checked the polynomial small-strain tangent and plane-stress scalar minimizer symbolically. No simulation, pilot, test suite, code refactor, commit, or deployment was performed. Existing uncommitted changes were preserved. Apple base commit was `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; the relevant recent experiments and some constitutive edits are working-tree content, so this SHA alone does not reproduce them. A future run must hash the executed source and input files.

The approval revision changed the Tuesday acceptance criteria, added the all-field strong-smoothness contract, and specified trend plots/checkpoints. The subsequent stress-only revision replaces independent pre-strain variables with shared baseline stresses, retains expression-specific activation increments, and updates priors, regularization, and derivative gates. These revisions change this design document only; the joint runner is not yet implemented.

The stress-only revision also checked the analytical bulk identities in CPU float64 on 32 random tensor states, seed 210921, using `mu=0.01` and `lambda_c=0.5`. Maximum energy-offset, force, and Hessian-vector discrepancies for `S0=mu(BB^T-I)` were `7.4e-18`, `2.8e-17`, and `4.4e-16`. A separate non-unit `Jp=1.09725` check confirmed the additional determinant-dependent term rather than fixed-stress equivalence. These are algebra checks of the formulas, not a Warp kernel test, forward simulation, or evidence of full-model readiness.
