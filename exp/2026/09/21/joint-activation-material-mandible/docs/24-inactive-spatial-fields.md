# Inactive spatial shared-field prototype

`joint_spatial_fields.py` provides the unactivated 80-coordinate fallback from
the accepted exact-rest force audit. It is not imported by the current neutral,
joint, or physics runners. Construction requires explicit paths to the `005`
basis and audit receipt plus the exact mesh cell count; it rejects a basis whose
SHA-256, schema, lineage, support sizes, partition weights, or G/M matrices do
not match the frozen contract.

The flat layout is fat `[0:24]` as `[4,6]`, aponeurosis `[24:48]` as `[4,6]`,
muscle `[48:78]` as `[5,6]`, uniform isotropic skin resultant at `78`, and
global skin log stiffness at `79`. `constant20_to_spatial80` repeats each old
bulk six-vector at every tissue anchor and copies both skin values. This exactly
preserves the old constant stress on every positive-fraction cell and gives zero
spatial roughness. Full stress reconstruction scatters each support-only field
into `[3, 1146517, 3, 3]`, with zeros outside that tissue's support.

Projection clips every normalized symmetric anchor tensor spectrally to
`[-0.9, 10]`. Nonnegative partition weights that sum to one preserve the same
Loewner bounds in every reconstructed cell. The skin constraints and all
material/resultant scales remain those in `joint_fields.py`.

`regularizers()` exposes each tissue's exact `C^T G C` roughness and `C^T M C`
magnitude. `bulk_spatial_roughness` and `bulk_spatial_magnitude` are the audited
mean over the three individually volume-normalized tissues. Roughness is
deliberately absent from `prior_total`: an activating runner must freeze and
record a nonzero coefficient separately. For the approved sensitivity probe,
the exact audited term is

```text
0.5 * 100.0 * regularizers()["bulk_spatial_roughness"]
```

The accepted CPU plus default-device receipt is
[`data/spatial-fields-validation-cpu-v8/summary.json`](../data/spatial-fields-validation-cpu-v8/summary.json).
All 19 checks passed, including full-cell constant embedding, reconstruction
and G/M directional derivatives, all-cell spectral preservation, the uniform
80.6 N/m setter, mismatched-metadata rejection, and unchanged default field and
physics modules. It also constructed the class with `device=None` while the
PyTorch global default was `cuda:0`, verified that every parameter and basis
buffer was on CUDA, and reconstructed the constant field there to `1.11e-16`
maximum error. Maximum lower/upper reconstructed-cell violations were
`9.99e-16`/`7.11e-15`; reconstruction and quadratic directional relative errors
were `5.94e-12` and `7.01e-13`.

```bash
cd exp/2026/09/21/joint-activation-material-mandible
PYTHONWARNINGS=error DEBUG=1 \
CHERRIES_NAME='Inactive spatial shared-field CPU validation v8' \
CHERRIES_TAGS='joint-inverse,spatial-basis,cpu,cuda-default-device,validation,inactive' \
uv run --frozen python src/24-validate-spatial-fields.py \
  --output-dir data/spatial-fields-validation-cpu-v8
```

This receipt validates parameterization and CPU derivatives only. A future
nonlinear probe still requires explicit physics support for full per-cell bulk
stress, a full-face directional derivative receipt, a fresh optimizer history,
and its unchanged equilibrium/deformation gates.
