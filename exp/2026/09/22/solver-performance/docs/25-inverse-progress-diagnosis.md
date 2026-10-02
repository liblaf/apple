# Fixed-stress Smile amplitude diagnosis

## Purpose

This bounded probe tested whether the recorded Hybrid `Smile` step-20 active
stress *direction* can improve the fit at larger amplitudes. It did not change
the completed 20-step run, optimizer state, jaw angle, materials, or contact
law. Each completed scale started from the saved step-20 deformation and used a
fresh full-source-bone and rigid-eye contact equilibrium.

## Command

```bash
DEBUG=1 CHERRIES_NAME='Smile fixed stress amplitude probe' \
CHERRIES_TAGS='smile,inverse,amplitude,contact,hybrid,objective-only' \
uv run --frozen python src/25-probe-stress-amplitude.py \
  --output-dir data/smile-amplitude-probe-001 \
  --forward-wall-seconds 180 --ipc-threads 8
```

The source checkpoint is the completed Hybrid V100 host step-20 `Smile` endpoint.
The forward force threshold was the production value
`1.5192003475221146e-10`; the Hybrid solver transition floor was `1e-7`.

## Results

| stress scale | fit RMS | objective | data term | weighted smoothness | force norm | inversions | active gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 5.110762 mm | 0.990420 | 0.989686 | 0.0007330 | 6.98e-11 | 0 | 16.820 um |
| 3 | 5.060368 mm | 0.976868 | 0.970265 | 0.0065969 | 1.10e-11 | 0 | 16.479 um |

Both completed fields satisfied the unchanged spectral cap and the 0.05
neighbor-RMS budget. Both passed the production force threshold, full contact
audit, active-gap gate, and no-inversion gate. Scale 3 required 48.25 seconds;
its inner CCD fraction reached 0.5348, so the larger field was not an equivalent
cost-free replay.

Scale 10 had not started when the user changed the primary experiment to an
LR=0.3 Hybrid pilot and released GPU 0. The process was terminated before that
scale. The partial receipt is
[`progress.json`](../data/smile-amplitude-probe-001/progress.json) and the
interruption record is [`interrupted.json`](../data/smile-amplitude-probe-001/interrupted.json).

## Interpretation and limits

At three times the saved field, the data term and RMS improved while the
smoothness penalty grew about ninefold, as expected for a quadratic
regularizer. This establishes only that the final field direction has remaining
feasible local amplitude room at scale 3. It does not establish that a larger
Adam learning rate will follow that direction, that scale 10 is feasible, or
that the inverse problem is near an optimum.

No adjoint, outer gradient, or residual/error correction was computed; the
reported objective is an objective-only equilibrium evaluation. The partial
probe should therefore guide the new LR=0.3 pilot rather than substitute for a
matched optimizer trajectory.

## Outer optimizer audit

The completed `0.003` pilot used 1,729,410 stress coordinates for 15,299
observed surface nodes. It accepted 20 updates after 30 candidate evaluations.
The first seven accepted line-search fractions were
`0.5, 0.25, 0.25, 0.25, 0.5, 0.5, 0.5`; all subsequent steps accepted the full
configured Adam proposal. The endpoint is still nonstationary and does not
establish a limitation of the physical model's expressive capacity.

The final maximum stress eigenvalue was about `1.37 kPa`, and none of the
normalized eigenvalues reached the cap of 10. The jaw remained at its lower
bound with a positive gradient, so the zero projected jaw step was consistent
with its bound rather than an ignored variable. Active stress changed both
the FEM state and the observed fit, ruling out a frozen-control/no-op error.

Although the weighted regularization *value* was only `0.000733` against a
data value near `0.9897`, its local gradient was not small. The saved total
gradient minus an exact CPU evaluation of the regularizer gradient gives:

| Gradient measure at update 20 | Data term | Regularizer |
| --- | ---: | ---: |
| Euclidean L2 over stress coordinates | 0.0016695 | 0.0021207 |
| Dual normalized-volume norm | 0.6979 | 1.1039 |

The data/regularizer Euclidean gradient cosine was `-0.0869`. This is evidence
of significant spatial regularizer conditioning; comparing scalar objective
values alone would miss it. A larger learning rate is a useful requested
experiment, but it does not guarantee that the same backtracking policy will
accept larger effective parameter changes.

All ten rejected candidates in the pilot already exceeded their raw Armijo
ceiling. Their unnecessary adjoints cost `173.421 s`, or `14.43%` of the
`1201.988 s` fitting time. A raw-objective rejection before backward avoids
that work without changing the residual-corrected acceptance condition.
The LR `0.3` restart also enables a prior-only rejection before physics when
the nonnegative data term cannot possibly rescue a candidate's objective.
