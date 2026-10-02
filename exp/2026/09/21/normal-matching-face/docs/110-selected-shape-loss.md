# Selected 2 mm / 5 degree shape loss

The selected reference length is **13.236093032531715 mm**, with normal weight
one. The intended data objective is
`position_component_mse_mm2 / l_ref_mm**2 + normal_chord_squared`.
The reusable [loss configuration](../data/110-shape-loss-config/loss-config.json)
records the chosen units, conventions, physical length and calibration inputs.

Position loss retains the existing reference-area-weighted vector MSE divided
by three. Thus 2 mm vector position RMS gives 4/3 mm². A 5 degree reference
normal deviation gives squared unit-normal chord error
`4*sin(radians(5)/2)**2 = 0.007610603816508936`. The selected length is
`sqrt((2**2/3) / (4*sin(radians(5)/2)**2))`; both normalized contributions equal
0.007610603816508936. This is a reference-angle calibration, not an exact
conversion from aggregate angle RMS when triangles have different errors.

The [calibration receipt](../data/110-shape-loss-config/calibration.json) passed.
Using the audited neutral losses from `data/90-beta1/protocol.json`, the new
initial contributions are position 0.04940857010416178 and normal
0.0243814646293817, giving position:normal **2.026480806432782:1**. The equivalent
old beta for relative data-term weighting is 0.49346630712002737. This does not
establish the same Adam trajectory: global loss scaling interacts with epsilon,
and any activation-regularization coefficient must be expressed consistently.

The length mode is fixed physical length, preserving the absolute 2 mm / 5
degree calibration. For this neutral skin's axis-aligned bounding-box diagonal
275.42224553811593 mm, the equivalent fraction is 0.048057458128232276. Keeping
that fraction fixed on differently sized faces is a separate scale-invariant
choice, which changes the absolute positional tolerance with face size. It is
recorded as a diagnostic, not selected as the length mode.

This step records the configuration and performs scalar calibration only.
It does not integrate the configuration into the frozen beta runner or launch
inverse-physics optimization. All 97 numerical source records used by the beta-1
run were checked and remain unchanged. The new script reads the prior protocol
and verifies the skin fixture SHA256 without constructing a physics model.

The named Cherries run exited zero after shutdown. Its [Comet record](https://www.comet.com/liblaf/apple/ce9d8f0d9a7848b58866e530d00a0a0a)
is named `Face shape loss: selected 2 mm and 5 degree calibration`; the recorded
Git SHA is `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`. The working tree also
contains uncommitted research work. Exact command, from
`exp/2026/09/21/normal-matching-face`:

```bash
env CHERRIES_NAME='Face shape loss: selected 2 mm and 5 degree calibration' CHERRIES_TAGS='face,normal-matching,reference-length,calibration,configuration,cpu' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python src/110-configure-shape-loss.py
```

Source: [110-configure-shape-loss.py](../src/110-configure-shape-loss.py).
Log: [110-configure-shape-loss.log](../logs/110-configure-shape-loss.log).
Ruff formatting and lint checks passed. No optimization improvement or
convergence claim follows from this configuration-only calculation.
