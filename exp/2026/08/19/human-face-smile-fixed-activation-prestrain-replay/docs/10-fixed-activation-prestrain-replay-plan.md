# Fixed-activation c020 prestrain replay: reviewed execution plan

## Status

The producer is implemented but **has not been executed**.  A source-level
`EXECUTION_APPROVED_AFTER_STATIC_REVIEW = False` gate stops the program before
input reads or CUDA/Warp initialization.  Clearing that gate requires a separate
review decision after both the producer and analyzer have been inspected.

This is a cheap forward-only causal replay.  It contains no inverse optimization,
adjoint solve, backward pass, or muscle-activation update.

## Question

With the corrected homogeneous skin and the exact muscle activation recovered by
the corrected p000 inverse held fixed, does progressively adding raw-plus-uniform
skin prestrain reduce visible/quantitative bumpiness without an unacceptable loss
of target fidelity?

## Frozen baseline

- prepared anatomy:
  `exp/2026/06/17/human-face-smile-prestrain-v2/data/10-human-face-prepared.vtu`,
  SHA-256 `8131d6944b322d7c1e21918688f297e2887b42bf4dbc19ce36259b007e8dc563`;
- corrected IsFace p000 skin:
  `exp/2026/08/18/human-face-smile-plane-stress-skin/data/10-corrected-baseline/skin-isface-e0200-p000.vtp`,
  SHA-256 `4c7ddce893eed4a8d0590042488ae1b35f0cae23383db6bc9814427eb6f7cc6f`;
- corrected best/terminal step-40 result:
  `exp/2026/08/18/human-face-smile-plane-stress-skin/data/20-human-face-smile-skin-no-prestrain-lr3-corrected-isface-e0200-p000-screen.vtu`,
  SHA-256 `c6a0b183675ffb3ec537c1153544b041acd7aa0fdd5216c0cf9a50022d52b0a4`;
- frozen full-volume muscle `ActivationInv` little-endian-f8 SHA-256:
  `4494f1eca2ce6f14c2e87a184d2227c080fbfa4594e7d6e96ced0c0c35c981de`;
- frozen step-40 displacement little-endian-f8 SHA-256:
  `f8ca27d820ff1f4b7afb734d917c9ec1292cd26ab96fc93090277dcc017268fb`.

The result and 41-frame history have already been independently checked to have
bit-identical step-40 displacement and muscle activation.  The producer repeats
the result-side validation before any runtime initialization.

## Material and boundary contract

- Koiter domain: exactly 15,299 points and 29,899 all-vertex-`IsFace`
  triangles;
- skin: homogeneous `E=0.2 MPa`, `nu=0.49`, thickness `1 mm`;
- skin Lamé conversion: plane stress,
  `lambda=E*nu/(1-nu^2)`, `mu=E/(2*(1+nu))`;
- skin energy measure: fixed original reference area, independent of prestrain;
- volume tissues: unchanged 3D Lamé conversion and existing anatomical
  fractions;
- artificial cross-section: every one of the 6,980 incident vertices is fixed to
  exact zero displacement; the resulting model has 33,636 fixed vertices and
  100,908 fixed DoFs;
- muscle activation: the exact corrected p000 step-40 full-cell tensor is copied
  into every fresh forward builder and checked bit-for-bit before and after each
  solve.

## Prestrain definition

For raw target/rest triangle area ratio `R`, the full c020 natural-area ratio is

```text
rho_full = 0.98^2 * clip(R, 0.5, 1)
```

Thus `c020` means 2% **linear** uniform tightening; its uniform-only natural-area
ratio is `0.9604`, not `0.98`.  The floor clamps 31 triangles, representing
`0.0879366%` of corrected skin rest area.

Continuation uses logarithmic natural-area dose:

```text
rho_alpha = numpy.power(rho_full, alpha)
diag      = 1 / numpy.sqrt(rho_alpha) - 1
ActivationInv = [diag, diag, 0]
```

The exact NumPy algorithm is pinned because algebraically equivalent evaluation
orders do not necessarily produce identical bytes.

## Cases and seeds

The ordered primary path is:

1. `c020-continuation-alpha-000`: identity prestrain, seeded from the exact
   corrected baseline displacement;
2. `c020-continuation-alpha-025`: seeded from solved alpha 0;
3. `c020-continuation-alpha-050`: seeded from solved alpha 0.25;
4. `c020-continuation-alpha-075`: seeded from solved alpha 0.5;
5. `c020-continuation-alpha-100`: seeded from solved alpha 0.75;
6. `c020-direct-alpha-100`: a fresh builder at full prestrain, seeded directly
   from the exact corrected baseline displacement.

The final direct replay diagnoses continuation/path sensitivity.  Every alpha is
a fresh model build; only the continuation displacement is transferred.

The alpha-0 replay must differ from the pinned baseline by no more than `1e-3` of
target-displacement RMS on both `SmileLossMask` and the exact corrected-IsFace
point set.  The corresponding absolute thresholds are approximately
`5.310139e-6 m` and `5.310655e-6 m`.  All fixed/cut displacements must remain
bit-exact zero.

## c050 policy

There is deliberately no `c050` CLI option or automatic fallback in this
producer.  Five-percent linear tightening is eligible only as a conditional
second stage after reviewing c020.  If requested, it must use a new isolated
producer/output namespace and a new approval decision; it must not modify or
overwrite the c020 evidence.

## Output contract

After an approved execution, the aggregate files will be:

- `data/10-fixed-activation-prestrain-replay-summary.json`;
- `data/10-fixed-activation-prestrain-replay-table.md`.

Each case will have:

```text
data/10-fixed-activation-prestrain-replay/c020/
  continuation/alpha-{000,025,050,075,100}/
    result.vtu
    skin.vtp
    forward-summary.json
  direct/alpha-100/
    result.vtu
    skin.vtp
    forward-summary.json
```

The producer refuses any existing aggregate/result root and validates exact
result topology, finite values, fixed-zero constraints, muscle activation,
derived skin prestrain, unchanged skin RestArea/Lamé/Fraction fields, and VTK
readback before completing the aggregate.

## Proposed command after approval

Working directory:

```text
/home/liblaf/Projects/liblaf/apple/exp/2026/08/19/human-face-smile-fixed-activation-prestrain-replay
```

Command:

```bash
DEBUG=1 \
CHERRIES_NAME='Fixed-activation c020 prestrain replay' \
CHERRIES_TAGS='human-face,skin,prestrain,fixed-activation,forward,replay,c020,debug' \
uv run --frozen python src/10-fixed-activation-prestrain-replay.py
```

This command was **not run** while preparing this plan.

## Static verification

The following checks passed without executing the experiment:

```bash
uv run --frozen ruff format --check src/10-fixed-activation-prestrain-replay.py
uv run --frozen ruff check --no-fix src/10-fixed-activation-prestrain-replay.py
uv run --frozen python -m py_compile src/10-fixed-activation-prestrain-replay.py
git diff --check -- exp/2026/08/19/human-face-smile-fixed-activation-prestrain-replay
```
