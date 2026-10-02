# Frozen surface-bumpiness measurement protocol

## Status and scope

This protocol was fixed before inspecting the final endpoints of the tensor active-stress arms. It applies unchanged to Raw6, every PSD arm, and the exact Smile target. The implementation is `src/40-measure-surface.py`.

The measurement is CPU-only post-processing of saved meshes. It performs no forward solve, adjoint solve, optimization, collision test, or control-field modification.

## Frozen implementation

The script reuses the exact surface implementation previously used by `face-actuation-diagnosis/src/41-surface-roughness.py`. That helper loads the verified cotangent-operator and `surface_diagnostics` functions from `face-activation-materials/src/40-audit-face-results.py`. The output records SHA-256 hashes for both earlier sources, the new wrapper, the manifest, the fixture, the skin, every endpoint, and every generated surface map.

The fixture is `face-actuation-diagnosis/data/12-historical-fixture/volume.vtu`; its `skin.vtp` supplies the single rest-surface topology. No endpoint supplies its own surface topology or normals.

At execution, `frozen-surface-map.npz` records:

- skin `GlobalPointId` into the fixture;
- triangular connectivity;
- rest skin and fixture points in metres;
- rest-state area-weighted vertex normals;
- lumped vertex areas;
- `Lip*` seed and intrinsic 10 mm mouth masks;
- membrane-boundary mask and lip distances;
- visible Smile target displacement.

Each array receives a separate SHA-256 hash. Every endpoint must have the same point count and an exactly matching `RestPosition` array. The skin points must exactly equal the fixture points selected by `GlobalPointId`.

## Scalar fields

For each rest-skin vertex \(i\), let \(\mathbf n_i\) be its frozen rest normal, \(\mathbf u_i\) the endpoint displacement, and \(\mathbf u_i^\star\) the Smile target displacement. Measure

$$
d_i=\mathbf n_i\mathbin{\cdot}\mathbf u_i,
\qquad
r_i=\mathbf n_i\mathbin{\cdot}(\mathbf u_i-\mathbf u_i^\star).
$$

Thus \(d\) is normal displacement and \(r\) is normal target residual. Tangential motion is outside this bumpiness measure. The exact target is evaluated as an in-memory endpoint with \(\mathbf u=\mathbf u^\star\); its residual field is zero by construction.

## Surface operator and spatial scales

The rest skin supplies a lumped triangle-area mass matrix \(M\) and piecewise-linear cotangent stiffness \(K\). For scalar field \(x\) and named scale \(s\), calculate

$$
(M+tK)y=Mx,
\qquad
t=s^2/4,
\qquad
h=x-y.
$$

Here \(y\) is the low-pass field and \(h\) is the high-pass field. The named scale is the two-dimensional heat-kernel RMS radius \(\sqrt{4t}\).

The only scales are 2, 5, and 10 mm. The primary scale is 5 mm; 2 and 10 mm are fixed sensitivity views. Natural Neumann/no-flux behavior is retained on the open membrane boundary. A constant-field invariance check must pass at every scale.

## Regions

Every field is measured on:

- `full_face`: all vertices of the frozen rest skin;
- `mouth_10mm`: vertices whose shortest intrinsic rest-surface edge distance from any `GroupName` beginning with `Lip` is at most 10 mm.

The mouth mask is computed once from the fixture and shared by all cases.

## Absolute and normalized measures

For both \(d\) and \(r\), at every scale and on both regions, report the area-weighted high-pass RMS in millimetres. Also report the dimensionless ratio

$$
\eta_{x,s,\Omega} =
\frac{\operatorname{RMS}_{M,\Omega}(h_{x,s})}
{\operatorname{RMS}_{M,\Omega}(x)}.
$$

The numerator and denominator always use the same scalar field and region. This prevents full-face motion from silently normalizing a mouth-only numerator.

The frozen zero guard is

$$
\epsilon_{\rm rms} =
\max\left(10^{-9}\ {\rm mm},
10^{-8}\times\operatorname{targetRMS}_{\rm full\ face}\right).
$$

The ratio is defined only when its denominator is strictly greater than \(\epsilon_{\rm rms}\). Otherwise the JSON value is `null`, `defined` is false, and the receipt retains the numerator, denominator, applied guard, and reason. A zero-motion state and the target's zero residual therefore do not produce a misleading ratio of zero.

## Primary comparison

The primary bumpiness record for every case contains the 5 mm absolute high-pass RMS and normalized ratio for:

| Field | Full face | Mouth within 10 mm |
| --- | ---: | ---: |
| normal displacement | required | required |
| normal target residual | required | required |

The same 2 and 10 mm values remain in the machine-readable sensitivity record. Raw6 is a historical active-strain continuity reference. Its difference from the PSD active-stress arms is not interpreted as an isolated effect of the PSD constraint.

High-pass target detail can be real. A lower displacement high-pass value can accompany lost expression motion, and a lower normalized value does not by itself identify an artifact. Absolute RMS, normalized RMS, total field RMS, fit, and motion must be read together.

## Endpoint manifest

After all runs complete, create `docs/40-measurement-manifest.json` without changing this protocol. It must follow this shape:

```json
{
  "schema_version": 1,
  "fixture_vtu": "../../face-actuation-diagnosis/data/12-historical-fixture/volume.vtu",
  "skin_vtp": "../../face-actuation-diagnosis/data/12-historical-fixture/skin.vtp",
  "scales_mm": [2, 5, 10],
  "primary_scale_mm": 5,
  "mouth_radius_mm": 10,
  "include_target": true,
  "cases": [
    {
      "id": "raw6",
      "label": "Raw6 historical active-strain continuity reference",
      "endpoint_vtu": "../data/CASE-DIRECTORY/final.vtu"
    }
  ]
}
```

Case IDs must contain only lowercase letters, digits, and hyphens. Paths must name immutable final VTUs; names containing `latest` are rejected. Add every PSD arm to the same manifest and run the measurement once so all cases share one fixture map, operator, zero guard, and source receipt.

Run through Cherries with the explicit no-commit profile:

```bash
cd exp/2026/09/07/tensor-active-stress
env COMET_AUTO_LOG_GIT_METADATA=false \
  COMET_AUTO_LOG_GIT_PATCH=false \
  COMET_AUTO_LOG_ENV_DETAILS=false \
  CHERRIES_NAME='Tensor active-stress surface measurements' \
  CHERRIES_TAGS='cpu,face,tensor-active-stress,surface,measurement' \
  .venv/bin/python \
  exp/2026/09/07/tensor-active-stress/src/40-measure-surface.py \
  --manifest exp/2026/09/07/tensor-active-stress/docs/40-measurement-manifest.json
```

The run must start with an empty output directory. The primary artifact is `summary.json`; per-case VTP files retain the scalar low-pass and high-pass maps for later inspection.
