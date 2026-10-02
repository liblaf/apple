# Chin-based rigid MouthOpen pose estimate

`116-estimate-chin-rigid.py` fits an unrestricted rigid pose to the saved
27-vertex chin patch, from `new-neutral-005` to the transferred `MouthOpen`
target. It uses area-weighted Kabsch alignment and stores the result in the
same convention as `joint_equilibrium.rigid_displacement`:

```text
(x - pivot) @ R(rotvec).T + pivot + translation
```

The saved `pose_rad_m` is

```text
[ 0.17388586765445685, -0.01240702857117505,  0.012728364310625942,
  0.0005163017883831333, -0.005821238348295399, 0.0012228731422144054 ]
```

Its fitted rotation magnitude is 10.014843 degrees and its translation norm
is 5.970662 mm. The area-weighted patch RMS decreases from 23.392064 mm to
0.179841 mm; the largest residual is 0.388676 mm. The fitted rotation has
determinant 1.0000000000000009. The Kabsch singular-value condition number
is 348.7873.

The estimate has no pose cap or regularizer. It is an initialization for a
subsequent stepped carry and forward solve, not a collision or volume-valid
state claim. The receipt pins the blendshape, neutral endpoint, repaired
reference, source protocol, and chin-patch receipt by SHA-256.

The saved receipt also contains `numpy_forward_residual_max_m`, which was a
same-function self-consistency check and therefore carries no independent
validation value. It is retained in this immutable estimate receipt, but was
removed from the reusable helper. A separate native
`joint_equilibrium.rigid_displacement` check should be recorded beside any
pose-path receipt.

Reproduce it from the experiment directory:

```bash
cd exp/2026/09/23/new-neutral
CHERRIES_NAME='Estimate unrestricted rigid MouthOpen chin pose' \
CHERRIES_TAGS='mouthopen,chin,rigid-pose,cpu' \
uv run python src/116-estimate-chin-rigid.py
```

The output directory must not already exist. The script validates its
rotation-vector and pivot translation conversion against a synthetic,
noncommuting rigid transform about a 20.86 m pivot. Its maximum forward
reconstruction error is 1.03e-14 m and its rotation-vector error is
1.77e-14 rad. The complete receipt is
`data/chin-rigid-pose-001/estimate.json`.

Cherries receipt: <https://www.comet.com/liblaf/apple/9bc3f1005fed4f74b2213364ef00fe14>.
