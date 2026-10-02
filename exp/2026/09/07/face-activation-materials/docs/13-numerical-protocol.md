# Four-control face basis protocol

## Purpose

This CPU preparation builds four smooth spatial controls inside each of the 35
named facial-muscle regions. It is reserved for a follow-up if one scalar per
muscle underfits. The controls are kinematic modes for scalar fiber amplitudes
or log-activation tensors. They are not anatomical compartments.

## Construction

For active tetrahedron `i`, the rest-space cell location is its vertex mean and
its mass is

```text
m_i = rest_tetrahedron_volume_i * MuscleFraction_i.
```

Within each region, `m_i` defines a weighted centroid and covariance. Let
`(lambda_major, e_major)` and `(lambda_minor, e_minor)` be the two leading PCA
pairs, with each axis sign fixed by making its largest absolute Cartesian
component positive. The local control order and centers are

```text
0: centroid + sqrt(lambda_major) * e_major
1: centroid - sqrt(lambda_major) * e_major
2: centroid + sqrt(lambda_minor) * e_minor
3: centroid - sqrt(lambda_minor) * e_minor.
```

The Gaussian width and normalized partition weights are

```text
sigma = sqrt((lambda_major + lambda_minor) / 2)
g_ik  = exp(-||x_i - center_k||^2 / (2 sigma^2))
phi_ik = g_ik / sum_l g_il.
```

Thus every `phi_ik` is nonnegative and every row sums to one. Four equal local
control values recover the original regional scalar field, and convex
interpolation preserves any shared lower and upper activation bounds. Global
control IDs are `4 * region_id + local_control_id`. Their lumped masses are
`sum_i(m_i * phi_ik)` over cells in that region.

The builder accepts rest points, tetrahedra, active IDs, region IDs, muscle IDs,
and muscle fractions. It has no expression-target input. Region names are read
only for output labels.

## Recorded run

Working directory:

```text
exp/2026/09/07/face-activation-materials
```

Command:

```bash
DEBUG=1 CHERRIES_NAME='Prepare four-control face basis' \
  CHERRIES_TAGS='face,controls,cpu,pca' \
  uv run python src/12-prepare-controls.py
```

The run wrote `data/12-controls.npz`, `data/12-controls.json`, and
`logs/12-prepare-controls.log`. The NPZ contains 120,020 active-cell rows and
140 global controls. Independent reload checks found a maximum partition error
of `4.44e-16`, uniform-region recovery error of `3.33e-16`, strictly positive
weights in `[2.63e-5, 0.9765]`, and strictly positive parameter masses in
`[1.53e-8, 3.91e-6] m^3`. Lumped parameter mass equals active muscle-volume
mass to floating-point precision.

The JSON records the complete formulas, region labels and diagnostics, source
hashes, hashes of every consumed geometry array, and hashes of the three fixture
files. The run was local-only and performed no forward or GPU solve.
