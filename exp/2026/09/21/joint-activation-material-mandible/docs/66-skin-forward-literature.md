# Regional skin parameters for the simple forward model

## Decision

The primary source located in this audit with both regional facial-skin
stiffness parameters and prestress is Flynn et al. (2013). Use its six regional fits as sparse
literature anchors. For the current isotropic membrane, transfer the derived
small-strain tangent modulus and the arithmetic mean of the two reported
prestresses at each site. This preserves regional differences without inventing
unregistered material directions.

These are inverse-model estimates, not direct measurements on the current face.
They came from one volunteer for the regional comparison and must not be called
a measured registered map.

## Primary source and protocol

Flynn, Taberner, Nielsen, and Fels, *Simulating the three-dimensional
deformation of in vivo facial skin*, Journal of the Mechanical Behavior of
Biomedical Materials 28 (2013), 484--494.

- [Publisher DOI](https://doi.org/10.1016/j.jmbbm.2013.03.004)
- [PubMed record, PMID 23566769](https://pubmed.ncbi.nlm.nih.gov/23566769/)
- [Author-uploaded manuscript](https://www.researchgate.net/publication/236138310_Simulating_the_three-dimensional_deformation_of_in_vivo_facial_skin),
  especially Methods pp. 6--10, Table 3 on manuscript p. 23, and Discussion
  pp. 14--17.

Five male volunteers, mean age 26 years with standard deviation 6 years, were
tested at the central right cheek. Six regions were tested repeatedly on one
volunteer. A 4 mm probe applied 16 in-plane and out-of-plane deformation paths
at 0.1 Hz after three preconditioning cycles. The inverse model used a flat
50 mm square, uniform 1.5 mm thick, incompressible two-term isotropic Ogden
shell with one-term QLV viscoelasticity. Material parameters and two in-plane
initial stresses were jointly optimized against three-component probe reaction
forces. Regional fit errors were 16.1--23.3%.

The paper used local axes defined separately at each probe site. Its prestress
axes `X'-Y'` were rotated 30 degrees relative to those local axes at CC, NE, CJ,
and ZYG, and 0 degrees at NL and FH. The underlying local axes differ by site
and are specified graphically in Figure 2. They are not registered to the
current mesh or its world axes.

## Regional values

`mu_1`, `mu_2`, `alpha_1`, `alpha_2`, `sigma_X`, `sigma_Y`, equivalent
prestrain, and fit error are reported in Flynn Table 3. `mu_2` is reported in
Pa; all other stress columns are kPa. The scalar tangent below is derived for
the paper's incompressible Ogden convention:

$$
G_0=\frac12\sum_i \mu_i\alpha_i,\qquad
E_0=3G_0=\frac32\sum_i \mu_i\alpha_i.
$$

It is a zero-stress infinitesimal tangent of the fitted Ogden law, not a
reported or directly measured Young's modulus. The scalar prestress
`sigma_mean` is also derived here as `(sigma_X + sigma_Y) / 2`; the paper did
not report it as a separate quantity.

| Code | Flynn anatomical site | `mu_1` (kPa) | `mu_2` (Pa) | `alpha_1` | `alpha_2` | Derived `E_0` (MPa) | `sigma_X / sigma_Y` (kPa) | Derived `sigma_mean` (kPa) | Reported equivalent prestrain `X / Y` | Fit error |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CC | Central right cheek | 58.27 | 0.14 | 2.334 | 33.081 | 0.204010 | 89.4 / 71.8 | 80.60 | 0.33 / 0.24 | 23.3% |
| NE | Right parotideomasseteric region, near ear | 57.40 | 0.27 | 3.000 | 44.060 | 0.258318 | 84.0 / 77.7 | 80.85 | 0.20 / 0.17 | 18.0% |
| NL | Right cheek near the lips | 41.29 | 0.16 | 1.658 | 54.964 | 0.102701 | 24.2 / 15.9 | 20.05 | 0.13 / 0.04 | 20.6% |
| FH | Centre of forehead | 53.95 | 0.30 | 1.868 | 68.998 | 0.151199 | 34.1 / 26.7 | 30.40 | 0.11 / 0.06 | 19.6% |
| CJ | Centre of right jaw | 57.73 | 0.42 | 2.265 | 34.689 | 0.196160 | 81.3 / 75.4 | 78.35 | 0.30 / 0.27 | 16.1% |
| ZYG | Right zygomatic region | 65.00 | 0.44 | 2.161 | 44.966 | 0.210727 | 64.4 / 58.3 | 61.35 | 0.20 / 0.17 | 22.9% |

The reported equivalent-prestrain values should remain provenance only. The
paper obtains them after its prestress load step but does not define them as
engineering, Green--Lagrange, or logarithmic strain. They must not be inserted
as engineering prestrain in a different constitutive model.

## Transfer to the current membrane

For a current membrane thickness `h_model`, preserving Flynn's fitted 3D stress
gives the isotropic initial resultant

$$
N_0 = \sigma_{\mathrm{mean}} h_{\mathrm{model}} I_2.
$$

At the current `h_model = 1 mm`, a value in kPa has the same numeric value in
N/m. The recommended regional `(E, N_0)` pairs are therefore:

| Region | `E` (MPa) | Isotropic `N_0` at 1 mm (N/m) |
| --- | ---: | ---: |
| CC | 0.204010 | 80.60 |
| NE | 0.258318 | 80.85 |
| NL | 0.102701 | 20.05 |
| FH | 0.151199 | 30.40 |
| CJ | 0.196160 | 78.35 |
| ZYG | 0.210727 | 61.35 |

This convention preserves the paper's fitted 3D stress while using the current
thickness. Preserving the resultant of Flynn's 1.5 mm shell instead would give
1.5 times these `N_0` values. Those are different transfer assumptions and must
not be mixed. The first convention is the smaller, simpler change and is
consistent with using the derived 3D tangent directly.

## Mapping boundary

The current face has no verified correspondence to Flynn's six probe centers or
their tangent axes. A defensible implementation must record six manually
reviewed anchor locations for FH, ZYG, CC, NL, NE, and CJ, mirror the right-side
values to the left as an explicit bilateral-symmetry assumption, and identify
how values are interpolated between anchors. Any values assigned to the nose,
eyelids, lips themselves, ears, neck, or scalp are extrapolations because Flynn
did not test those sites.

For the simple isotropic forward model, a smooth convex interpolation of the six
scalar anchors on the skin surface is preferable to inventing directions. The
receipt should retain the anchor IDs, interpolation rule, and source table.
Calling the resulting field “Flynn-derived regional transfer” is accurate;
calling it a measured subject-specific skin map is not.

## Limits that materially affect interpretation

- Table 3 compares regions in one volunteer; it does not provide population
  means or regional uncertainty. The central-cheek fits across five volunteers
  give derived `E_0 = 0.098704--0.211256 MPa`, `sigma_X = 34.9--89.5 kPa`,
  and `sigma_Y = 27.2--73.8 kPa` (Table 2), demonstrating substantial
  inter-volunteer variation without defining regional uncertainty.
- Stiffness and prestress were inferred jointly, so they are correlated model
  parameters rather than independent measurements.
- The material law was isotropic Ogden plus QLV. Observed anisotropy was assigned
  entirely to prestress. The current stable Neo-Hookean membrane does not retain
  the nonlinear or viscoelastic response.
- The flat single-layer model omitted connections to subdermal tissue and bone.
  The authors state that this probably increased the estimated stiffness and
  in-vivo tension; the zygomatic region was especially affected by geometry and
  the zygomatic ligament.
- The stress directions cannot be transferred from `X'-Y'` to world or mesh
  tangent axes without an explicit site registration. Averaging them discards
  anisotropy but avoids fabricating that registration.

No local primary PDF or tabulated companion dataset was present in the checked
workspace. The numbers above were transcribed from the author manuscript's
Table 3 and checked against the publisher DOI and PubMed record.
