# Joint stress-field parameterization validation

## Purpose

This validation covers the field layer only: the 20 shared coefficients, dense
six-coordinate per-tetrahedron activation, signed/PSD projections, physical
normalization, graph regularization, and gradients. It does not solve a facial
equilibrium problem or establish that the literature proxy prestress is feasible.

## Frozen contract

`SharedFieldParameters.coefficients` is one 20-vector. Coordinates 0:18 are
three constant Frobenius-orthonormal symmetric tensors in fat, aponeurosis, and
muscle order; coordinate 18 is an isotropic tangential skin resultant; coordinate
19 is the log skin-stiffness multiplier. All baseline coordinates initialize to
zero. Dense activation uses six coordinates per active tetrahedron and is
projected to the PSD cone with a fixed cap.

The checked fixed scales are:

| Quantity | Value | Status |
| --- | ---: | --- |
| Fat `mu` | 0.003835616438 MPa | Derived from the configured sensitivity seed |
| Aponeurosis `mu` | 0.6270370370 MPa | Derived from a distant SMAS/platysma proxy |
| Muscle `mu` | 0.004109589041 MPa | Derived from an elastography seed |
| Skin `mu` | 0.06849315068 MPa | Derived from the configured 0.2 MPa skin sensitivity value |
| Activation reference | 0.01232876712 MPa | Modeling scale, three times configured muscle `mu` |
| Activation cap | 0.1232876712 MPa | Modeling cap, ten reference units |
| Skin resultant reference | 68.49315068 N/m = 6.849315068e-5 MPa m | `mu h` with assumed 1 mm thickness |

The optional first skin continuation target is 0.806 N/m, one percent of the
80.6 N/m mean of the two Flynn central-cheek directional proxy values. It is a
modeling continuation target and may increase only after valid neutral
equilibrium. It is not a measured map and is not imposed at initialization.
No measured facial-aponeurosis baseline-stress distribution is claimed.

## Commands and run receipt

Working directory:
`exp/2026/09/21/joint-activation-material-mandible`

Smoke command:

```bash
DEBUG=1 CHERRIES_NAME="Joint field validation smoke" CHERRIES_TAGS="joint-inverse,field-validation,cpu,smoke" uv run python src/06-validate-fields.py --dense-cells 4096 --dense-edges 8192
```

The normal noncommitting `ProfileJoint` run used:

```bash
CHERRIES_NAME="Joint stress field parameterization validation" CHERRIES_TAGS="joint-inverse,field-validation,cpu,noncommitting" uv run python src/06-validate-fields.py
```

It opened Comet experiment
<https://www.comet.com/liblaf/apple/cf59e2ba73f94c55ba243b8f515798eb>
and completed the numerical work, but its shutdown hook spent 4 minutes 51
seconds constructing a repository-wide git patch across roughly 164,000 indexed
paths and a large filtered VTK file. The process was interrupted after the
artifacts and metrics were written; therefore the Comet environment/patch upload
is incomplete. A final debug-profile CPU receipt, with the same full sizes and
current source hashes, completed normally:

```bash
DEBUG=1 CHERRIES_NAME="Joint stress field parameterization validation (local receipt)" CHERRIES_TAGS="joint-inverse,field-validation,cpu,local-receipt" uv run python src/06-validate-fields.py
```

## Results

All 23 checks passed in float64 on CPU. The largest error-to-tolerance ratio was
0.5614, from the regularizer directional finite difference. The dense check used
288,235 cells, 501,409 edges, and 1,729,410 activation coordinates. Its forward
and backward calculation took 0.0400 s with a finite gradient RMS of
2.6829e-9 on this run.

The checks include coordinate round trips and Frobenius norms; rotation
invariance/covariance; PSD lower/upper bounds and idempotence; signed baseline
eigenvalue bounds for every bulk tissue; the exact physical graph smoothness and
effective-volume magnitude formulas; a central directional finite difference;
activation and skin unit normalization; positivity of the exponential skin
multiplier; and the absence of fabricated spatial-smoothness gradients for the
constant baseline basis.

## Evidence and limitations

- `data/field-validation/summary.json` contains the complete check table,
  runtime versions, dense dimensions, and source hashes.
- `data/field-validation/checks.csv` is the flat check table.
- `data/field-validation/research-informed-material-config.json` is the frozen,
  source/status-labeled configuration consumed by the runner.

This validates implementation algebra and differentiability of the field layer.
It does not validate facial geometry, equilibrium convergence, neutral-shape
compatibility, identifiability, or biological transfer of any modulus or
prestress proxy. The fixed activation reference/cap, Poisson ratios, thickness,
prior weights, and continuation fraction remain computational assumptions.

## Initial-equilibrium guard regression

Two fresh synthetic receipts verify the exact-initial-equilibrium guard without
changing the strict Armijo path for non-equilibrium seeds:

- `data/equilibrium-initial-guard-validation/summary.json` has `success: true`.
  The four-tetrahedron implicit-equilibrium check retained a maximum directional
  finite-difference relative error of 5.215654214232551e-10, a maximum component
  relative error of 5.738796039453212e-8, and a maximum two-expression gradient
  isolation error of 2.838650012739081e-16.
- `data/coupled-initial-guard-validation/summary.json` has `success: true`.
  The coupled 20-shared-coordinate, per-tetrahedron activation, skin, and jaw
  check retained a maximum directional finite-difference relative error of
  1.523357748454515e-5 against the fixed 2e-3 acceptance limit, with exactly zero
  measured expression-isolation error.
- The coupled exact-neutral seed had zero displacement and a free-gradient norm
  of 2.2531981134570898e-24 against the declared absolute tolerance of 1e-15.
  Its receipt reports `result: initial_equilibrium`, `steps: 0`, and line search
  `status: not_run`; the regression also asserts that a stale prior
  `forward.last_solution` is cleared. An independently perturbed seed resolved
  by the diagnostic 3-by-3 Newton solve to within 2.9726206198737827e-18 m of the
  reference, from an initial gradient norm of 1.0161582293326268e-6 to
  4.0151664261961495e-20 in two steps.

These are small synthetic implementation checks. They do not establish
full-face equilibrium convergence or admit the joint optimization.
