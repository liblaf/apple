# Facial activation and material priors for the inverse FEM experiment

## Decision

The next face experiment should not optimize an independent six-component activation tensor in every muscle tetrahedron. Start with one bounded scalar contraction per `MuscleId`, applied through a fixed reference-space direction field and weighted by the existing muscle fraction. If that is too restrictive, add only two to four smooth modes inside each muscle. Keep a smooth, trace-free symmetric log-activation tensor as a diagnostic kinematic model, not as a physiological muscle model.

For passive material sensitivity, use hypodermis/fat as the reference shear scale. A defensible first face-specific baseline is

| tissue | baseline shear-like scale | ratio to hypodermis | sensitivity |
| --- | ---: | ---: | ---: |
| hypodermis/fat | 0.8 kPa | 1 | fixed in the ratio sweep |
| skin/dermis | 6.4 kPa | 8 | ratios 4, 8, 16 |
| passive muscle | 6.4 kPa | 8 | ratios 1, 8, 32 |

The first two absolute values are the small-strain isochoric scales derived from Picard et al.'s Yeoh coefficients, `mu = 2 C10`: `C10 = 400 Pa` for hypodermis and `3,180 Pa` for skin. The passive-muscle baseline is an experiment choice inside the very broad facial-model range below, not a measured facial-muscle value. If the current stable Neo-Hookean parameters have the usual infinitesimal Lamé semantics, use these numbers as `mu` and set `nu = 0.46`, hence `lambda / mu = 2 nu / (1 - 2 nu) = 11.5`. Verify the implementation's convention before translating them. Do not assign a 3-D shear modulus directly to a membrane or shell coefficient; include the physical skin thickness in that conversion.

Change activation space first while holding materials fixed. Then run the two one-factor ratio sweeps. A joint activation/material fit would make it difficult to tell whether smoothness came from the activation prior or from an artificially stiff outer layer.

## What the primary literature supports

### Named-muscle controls and fiber fields

Sifakis, Neverov, and Fedkiw built a tetrahedral face with a B-spline fiber field for each muscle. Muscle tetrahedra had transversely isotropic passive and active responses, scaled by muscle volume fraction. Their inverse variables were 32 muscle-level activations `a_i` constrained to `[0, 1]`, plus rigid head and jaw controls; `[0, 1]` is a normalized force control, not an active stretch interval. This supports regional sharing and fraction weighting. It does not support independent per-tetrahedron tensor controls. The model's anatomical fiber fields came from a transferred detailed anatomy, so its physiological interpretation depends on information absent from the current mesh.

Picard et al. provide a simpler construction that is closer to what can be reproduced here. For each named muscle, they manually placed a curve through the hypodermal tetrahedral mesh using literature and CT anatomy. Intersected elements became muscle elements, and the direction between successive curve points supplied a single fiber direction for a Hill-type active law. This supports a smoothed centerline tangent for long muscles and explicitly shows which part still needs anatomical judgment. Their model used 1 mm skin shell elements and isotropic second-order Yeoh laws for the passive skin and hypodermis.

Cong et al. describe the higher-fidelity alternative: deform an anatomical template containing the cranium, jaw, 41 muscles, and tetrahedral flesh into a target surface, using landmarks, feature curves, and a volumetric Poisson morph. This transfers the internal muscle shapes and their simulation data together. It is the credible route if actual anatomical fibers later become necessary. Running PCA independently on each `MuscleId` is only a geometry-derived estimate and is not equivalent to this anatomical transfer.

### Material values and their semantics

Picard et al. used a second-order Yeoh model with `C10 = 400 Pa`, `C20 = 1,400 Pa` for hypodermis and `C10 = 3,180 Pa`, `C20 = 14,500 Pa` for skin, with `nu = 0.46`. At infinitesimal isochoric strain, `mu = 2 C10`, giving 0.80 kPa and 6.36 kPa and a skin-to-hypodermis ratio of 7.95. The `C20` ratio is 10.36, so the contrast also persists into the nonlinear term. These values belong to a Yeoh face model and provide a scale and ratio, not a universal conversion to every constitutive law.

Barbarino et al. used a Rubin-Bodner facial tissue model and explicitly tested three passive muscle parameter sets because published muscle stiffness was uncertain. In their small-deformation interpretation, the shear-like scale is `m2 * mu0`. The table gives approximately 1.36 kPa for skin/mucosa, 2.96 kPa for SMAS, 0.259 kPa for deep fat, and 0.333, 8.14, or 33.3 kPa for the three muscle sets. Relative to deep fat, these are about 5.3 for skin and 1.3, 31, or 129 for muscle. The useful result here is the broad uncertainty and the approximate skin/fat contrast. Copying the highest muscle value as a default would overstate the evidence.

Weickenmeier, Jabareen, and Mazza performed in vivo suction experiments at the jaw, parotid, and forehead and fit an elastic-viscoplastic model. Their fitted initial shear moduli were 2.32 kPa for skin and 0.05 kPa for SMAS. Location changed maximum response by as much as 18%, while repeat measurements at one location varied by less than 12%. They set the fitted fiber contribution to zero and treated the layers as isotropic for that identification. SMAS is not interchangeable with fat or skeletal facial muscle; the 46-fold skin/SMAS ratio is therefore evidence for layer and model sensitivity, not the ratio to use in this mesh.

Ni Annaidh et al. tested 56 excised human back-skin samples in uniaxial tension at `0.012 s^-1`. They reported a mean initial slope of `1.18 +/- 0.88 MPa`, a later linear-region elastic modulus of `83.3 +/- 34.9 MPa`, failure strain `54 +/- 17%`, and a statistically significant relation between Langer-line orientation and collagen orientation. These MPa values describe excised back skin in a tensile curve and must not be substituted for the low-strain kPa-scale face parameters from suction or Yeoh identification. The study does support treating skin anisotropy and prestress as real omissions rather than tuning noise.

The local MyoFLAME manuscript was inspected only as context. It is an anonymous submission with placeholder venue and DOI fields. Its Type-A columns are precomputed linear-elastic responses and its Type-B sphincters are analytic ring-contraction response columns. Those columns are useful low-dimensional shape priors, but they do not provide an active-strain fiber field. In particular, an inward ring displacement basis must not be described as a circumferential muscle fiber direction.

## Recommended activation spaces

### A. First choice: scalar, volume-preserving fiber contraction

For tetrahedron `e`, let `f_e` be a unit, unoriented reference-space axis and let `s_e >= 0` be contraction magnitude. Define

```text
B_e = s_e (I / 3 - f_e f_e^T)
A_e = exp(B_e)
```

Then `B_e` is symmetric and trace-free, `A_e` is symmetric positive definite, and `det(A_e) = 1`. The active stretch is `exp(-2 s_e / 3)` along `f_e` and `exp(s_e / 3)` in each transverse direction. If the constitutive code consumes `A_inv`, pass `exp(-B_e)` rather than changing the meaning of the fitted variable.

Use a single `s_m` for every cell with `MuscleId = m` in the first experiment. Use the already defined `MuscleFraction` exactly once, in the constitutive mixing or energy weighting. Tapering both activation magnitude and energy by the same fraction would square the intended partial-volume effect.

As an explicitly kinematic bound, begin with active axial stretch in `[0.8, 1]`, equivalent to `s in [0, 0.335]`. Widen to `[0.7, 1]`, or `s <= 0.535`, only as a sensitivity test. These limits are cautious deformation bounds; none of the six peer-reviewed sources above measured maximum active shortening of the modeled facial muscles.

### B. Geometry-estimated direction field

The direction construction has three cases:

1. For a long, compact muscle region, take the dominant PCA axis of muscle-cell centroids as an initialization. Prefer a manually reviewed origin-to-insertion centerline when the ends can be identified, and use its local tangent instead of one constant axis. Because the energy uses `f f^T`, the sign of `f` is irrelevant.
2. For orbicularis oris or orbicularis oculi, fit a local plane and center or, preferably, a smooth closed centerline. Set the fiber direction to the circumferential tangent. In the planar approximation, `f_e` is proportional to `n cross (x_e - c)`. Circumferential shortening can generate inward radial displacement; the displacement direction itself is not the fiber direction.
3. For fan-shaped or branching regions, global PCA is not credible. Supply at least one reviewed origin and an insertion curve, then interpolate a direction field. If those landmarks are unavailable, classify the result as a kinematic regularizer rather than anatomical muscle activation.

Smooth and renormalize this field inside each `MuscleId`, with explicit handling at branch cuts and ring closure. Inspect it visually before solving. The mesh labels constrain where a muscle can act, but they do not determine its fiber directions.

### C. Second choice: a few spatial modes per muscle

If one scalar cannot match a clearly smooth target, use

```text
s_e = bounded(s_m0 + sum_k c_mk phi_mk(e)),  k = 1, ..., K
```

where `phi_mk` are the first `K = 2-4` nonconstant modes of the cell-adjacency Laplacian within that muscle. Parameterize `bounded(.)` smoothly so the axial-stretch limit is exact. Weight mode construction and reported norms by reference tetrahedron volume and `MuscleFraction`. This permits broad activation gradients without restoring a control at every cell.

Use temporal sharing as well when fitting a sequence: the same spatial coefficients define a muscle and only their amplitudes vary smoothly over frames. Bilateral equality can be an initialization or soft prior, but not a hard physiological law.

### D. Diagnostic only: smooth trace-free log tensor

When testing whether an assumed fiber direction prevents target recovery, let each muscle have a shared symmetric trace-free tensor plus at most two to four smooth spatial modes:

```text
B_e = B_m0 + sum_k c_mk Phi_mk(e)
A_e = exp(B_e)
```

Bound the eigenvalues of `B_e`, penalize adjacency differences in `B`, and retain `det(A_e) = 1`. This is a useful SPD, nonsingular kinematic active-strain space. A general trace-free tensor also permits local shear and unequal transverse stretches, however, so a successful fit does not identify a muscle's activation or fibers. Compare it with the scalar-fiber model as an upper-bound diagnostic and label it accordingly.

## Minimal experiment sequence

1. Freeze geometry, boundary/contact treatment, skin formulation, target, observation mask, and current passive materials. Compare per-cell six-component activation with region-shared scalar activation. Report target error, surface Laplacian/curvature residual, high-frequency activation energy, minimum `det(F)`, and parameter count.
2. Compare PCA/centerline directions with the smooth two-to-four-mode scalar field. For ring muscles, compare the circumferential field separately. Do not pool ring and long-muscle conclusions.
3. Run the general smooth log-tensor model only if the scalar model leaves a structured target residual. If it reduces the target error but creates non-axisymmetric eigenmodes, interpret that as model discrepancy or missing anatomy, not recovered physiology.
4. With the activation representation fixed, sweep skin/fat ratios `4, 8, 16`. Then sweep passive-muscle/fat ratios `1, 8, 32`. Keep `nu = 0.46` and the fat scale fixed. Repeat the chosen ratio once at half and twice the global shear scale to expose dependence on absolute stiffness, gravity, contact, and prescribed loads.
5. Accept a setting only if its smoothness improvement survives the material sweep and does not come from suppressing the intended facial motion. Compare the same anatomical ROI and the whole visible face; a crop is a metric, not a new mechanical boundary.

Before attributing bumps to activation freedom, also check tetrahedron quality/orientation, abrupt material-fraction transitions, muscle/skin coupling, contact, and fixed-boundary placement. A stiff skin layer or a graph penalty can visually hide defects in any of these components.

## Assumptions to state with every result

**Physiologically motivated:** activation is nonnegative; one named muscle shares a control; contraction is along a reference-space axis; the local active map is volume preserving; muscle fraction scales partial-volume contribution; orbicularis fibers are circumferential rather than radially directed.

**Geometry-informed but uncertain:** `MuscleId` boundaries are anatomically accurate; PCA identifies origin-to-insertion direction; fitted ring planes and centers represent the actual sphincter; a smoothed centerline adequately captures fan-out and branching.

**Kinematic or numerical priors:** the 20% initial and 30% expanded contraction caps; trace-free SPD log activation; graph smoothness and the number of modes; bilateral coupling; the selected passive-muscle baseline; transfer of Yeoh low-strain scales into the stable Neo-Hookean implementation.

**Not supported by the present evidence:** interpreting a general per-cell tensor as measured muscle physiology; interpreting MyoFLAME's analytic response columns as active-strain fibers; using excised back-skin `83 MPa` as the facial skin Young's modulus; claiming a unique material set from image matching alone.

## Primary sources

1. Sifakis, E., Neverov, I., and Fedkiw, R. (2005). “Automatic Determination of Facial Muscle Activations from Sparse Motion Capture Marker Data.” *ACM Transactions on Graphics* 24(3), 417–425. [doi:10.1145/1186822.1073208](https://doi.org/10.1145/1186822.1073208); [author PDF](https://pages.cs.wisc.edu/~sifakis/papers/activations_siggraph_2005.pdf).
2. Picard, M.-C., Nazari, M. A., Perrier, P., Bettega, G., Lartizien, R., Rochette, M., and Payan, Y. (2026). “A Clinically Compatible Method for Generating Preoperative Finite Element Models to Simulate Facial Appearance and Movements in Orthognathic Surgery.” *International Journal for Numerical Methods in Biomedical Engineering* 42(2), e70144. [doi:10.1002/cnm.70144](https://doi.org/10.1002/cnm.70144); [full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC12873870/).
3. Cong, M., Bao, M., E, J. L., Bhat, K. S., and Fedkiw, R. (2015). “Fully Automatic Generation of Anatomical Face Simulation Models.” *Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation*, 175–183. [doi:10.1145/2786784.2786786](https://doi.org/10.1145/2786784.2786786); [Eurographics record](https://diglib.eg.org/items/f0e4e8eb-dc11-48cd-b4ad-1587800e86ee).
4. Barbarino, G., Jabareen, M., Trzewik, J., and Mazza, E. (2008). “Physically Based Finite Element Model of the Face.” In *Biomedical Simulation*, LNCS 5104, 1–10. [doi:10.1007/978-3-540-70521-5_1](https://doi.org/10.1007/978-3-540-70521-5_1).
5. Weickenmeier, J., Jabareen, M., and Mazza, E. (2015). “Suction Based Mechanical Characterization of Superficial Facial Soft Tissues.” *Journal of Biomechanics* 48(16), 4279–4286. [doi:10.1016/j.jbiomech.2015.10.039](https://doi.org/10.1016/j.jbiomech.2015.10.039); [author PDF](https://weickenmeierlab.com/wp-content/uploads/paper/Weickenmeier_JBIOMECH15.pdf).
6. Ni Annaidh, A., Bruyere, K., Destrade, M., Gilchrist, M. D., and Ottenio, M. (2012). “Characterization of the Anisotropic Mechanical Properties of Excised Human Skin.” *Journal of the Mechanical Behavior of Biomedical Materials* 5(1), 139–148. [doi:10.1016/j.jmbbm.2011.08.016](https://doi.org/10.1016/j.jmbbm.2011.08.016); [PubMed](https://pubmed.ncbi.nlm.nih.gov/22100088/).
