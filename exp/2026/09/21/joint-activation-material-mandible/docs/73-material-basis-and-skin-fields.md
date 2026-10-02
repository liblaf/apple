# Material basis and prescribed skin fields

The diagnostic forward uses literature-informed initial values. None of the studies calibrates this subject or the coupled Stable Neo-Hookean (SNH) model. The figures show the exact prescribed fields used by `simple-skin-forward-002`; this review changes no material parameters or simulation state.

## Parameter evidence

| Tissue | Current input | Primary evidence | Transfer limitation |
| --- | --- | --- | --- |
| Fat | E = 11.2 kPa; nu = 0.46 | [Paluch et al., Table 1](https://www.termedia.pl/doi_ft/10.5114/ada.2018.79778): in-vivo shear-wave elastography (SWE) of deep medial cheek fat in 89 women, 11.2 ± 6.9 kPa. | Apparent wave-derived stiffness; the table labels the quantity “DMCFC strain [kPa].” Strong age dependence. It is not a controlled quasistatic SNH calibration or a universal facial-fat modulus. |
| Muscle | E = 12 kPa; nu = 0.46 | [Ternifi et al.](https://www.scirp.org/pdf/jbise_2019111915335847.pdf): relaxed left zygomaticus major in 15 healthy volunteers. Two SWE probes gave 12.0 ± 4.3 and 18.3 ± 3.7 kPa. | The selected value comes from the first probe. Apparent modulus depends on probe and wave-conversion assumptions; it does not identify passive SNH behavior or activation. |
| Aponeurosis | E = 1.693 MPa; nu = 0.35 | [Tereshenko et al.](https://academic.oup.com/asjopenforum/article/doi/10.1093/asjof/ojaf126/8290344): untreated cervical SMAS/platysma samples from seven facelift patients; ex-vivo uniaxial tension at 3 mm/min, modulus from the linear region, 1.693 ± 0.543 MPa. | A real tensile modulus, but a neck-tissue, direction- and protocol-specific proxy. This is the weakest anatomical transfer to cheek aponeurosis; no isotropic multiaxial SNH fit is available. |
| Skin | Spatial E = 127.382–257.861 kPa; nu = 0.46; h = 1 mm | [Flynn et al., Table 3](https://doi.org/10.1016/j.jmbbm.2013.03.004): six regional inverse Ogden–QLV fits on one volunteer. | The map converts the Ogden fits to a zero-stress tangent modulus, then transfers sparse regional values manually. It is not a measured subject-specific stiffness distribution. |

Poisson ratios are modeling assumptions. Zero bulk baseline stress and zero muscle activation are the requested diagnostic conditions, not measured natural values. Numerical contact settings are not material measurements.

## Skin transfer and interpretation

The incompressible zero-stress Ogden tangent is converted as `E0 = (3/2) sum(mu_i alpha_i)`, with the paper's `mu2` converted from Pa to kPa. The current skin law uses this scalar as its SNH stiffness input; it does not reproduce the fitted nonlinear viscoelastic constitutive law.

Prestress is represented by an isotropic membrane stress resultant:

`N0 = h (sigma_X + sigma_Y) / 2`.

The frozen reference tangent-frame tensor is `diag(N0, N0)`, with no shear. At the assumed `h = 0.001 m`, the numerical range **30.400–80.830 N/m** corresponds to **30.400–80.830 kPa** mean three-dimensional stress. This is a prescribed baseline stress, not an inferred prestrain.

The source uses a 1.5 mm shell. Our 1 mm transfer preserves the fitted three-dimensional stress, not the source shell's resultant. The source model omitted underlying tissue/bone attachments; its authors identify this as a possible reason for overestimated stiffness and tension. The directional stresses and thickness therefore do not constitute a calibrated prestress prescription for this coupled face model.

Eleven manually placed anchors represent five bilateral sites and a central forehead site. Positive normalized Gaussian weights at triangle centroids, with 20 mm width in Euclidean coordinates, generate smooth convex blends. This procedure does not exactly interpolate anchor values: the six raw stiffness values span 102.701–258.318 kPa, while the actual field spans 127.382–257.861 kPa. Raw mean prestress anchors span 20.05–80.85 N/m; the actual blend spans 30.400–80.830 N/m. Bilateral symmetry, isotropic averaging, 1 mm thickness and the blend width are assumptions. Nose, lips, eyelids and scalp values are extrapolated.

See [the detailed skin source audit](66-skin-forward-literature.md) for Table 3 values and the transfer formula.

## Visual outputs and verification

Outputs are in `data/prescribed-skin-field-visuals-001/`:

- `00-skin-fields-overview.png`: front views of both fields.
- `01-skin-stiffness-front-profile.png`: stiffness front and side views, sharing one scale.
- `02-skin-prestress-front-profile.png`: prestress front and side views, sharing one scale.
- `prescribed-skin-fields.vtp`: original skin geometry with per-triangle stiffness, prestress resultant, equivalent mean stress and thickness.
- `summary.json`, `provenance.json`, `sources/`: ranges, source/input hashes and archived rendering implementation.

The renderer verifies all five input array hashes, the prepared-input and reference-skin hashes, and exact triangle ordering through `GlobalPointId`. All 29,899 skin triangles and 15,299 skin points are retained. Constant lighting is disabled so shading cannot be mistaken for a scalar change. Overview and profile images were visually inspected; labels, units and legends are readable and geometry is uncropped.

Input field SHA-256: `2f57719152d7ab48989fc3891eb7f63d170fcd643f83e80346acda381ed220cb`.

Input manifest SHA-256: `3af1785f1a1988d87400b1add60950f5274689834070ebb7b9d8a397888aa38f`.

The tailnet review (private preview omitted) now places the maps first and includes a collapsible material-evidence table. HTTP retrieval verified all three served PNG hashes against the evidence manifest. The existing runtime-only server was reused. Native iOS browser rendering was not tested. Forward convergence and final joint readiness remain false.

## Reproduction and run record

Working directory: `exp/2026/09/21/joint-activation-material-mandible`.

```bash
CHERRIES_NAME='Prescribed skin stiffness and prestress maps' \
CHERRIES_TAGS=joint-inverse,material-priors,visualization \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYVISTA_OFF_SCREEN=true \
uv run --frozen python src/73-visualize-prescribed-skin-fields.py
```

The script requires a new output directory to preserve run provenance. Ruff check passed. The process and Cherries shutdown completed successfully. Matplotlib reported a semibold font substitution to weight 700; visual inspection confirmed readable text.

Comet summary, also recorded in `logs/73-visualize-prescribed-skin-fields.log`:

```text
name: Prescribed skin stiffness and prestress maps
url: https://www.comet.com/liblaf/apple/b9391d86d9154801b0b7307b48665e2b
entrypoint: exp/2026/09/21/joint-activation-material-mandible/src/73-visualize-prescribed-skin-fields.py
git sha: d56fa1b553b287b22b2cf7bb82d46117e34ed6bb
start: 2026-09-21 17:03:22.746858+08:00
end: 2026-09-21 17:03:25.777196+08:00
```

The Git SHA does not identify the dirty experiment files by itself; the source archive and hashes record the implementation. No commit was created.

Review rebuild command:

```bash
DEBUG=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CHERRIES_NAME='Skin material basis and map review' \
CHERRIES_TAGS=joint-inverse,review,materials \
uv run --frozen python src/43-build-review.py \
  --neutral-run-dirs '["data/neutral-convergence-025-contact-spatial80-metric-bfgs-001","data/neutral-convergence-025-contact-spatial80-metric-bfgs-002"]' \
  --convergence-plot-dirs '["data/spatial25-u0200-convergence-visuals"]' \
  --neutral-visual-dir data/spatial25-u0200-shape-visuals \
  --neutral-contact-visual-dir data/spatial25-u0200-contact-visuals-v2 \
  --neutral-lineage-visual-dir data/spatial25-u0200-lineage-visuals \
  --jaw-visual-dir data/jaw-preflight25-diagnostic003-visuals-003
```

The local review rebuild completed with 82 images in its evidence manifest; `logs/43-build-review.log` contains the completion receipt. It includes older diagnostic galleries, which retain their status labels.
