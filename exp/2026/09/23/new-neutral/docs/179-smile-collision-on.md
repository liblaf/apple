# Collision-on Smile preparation

`178-inverse-smile-rigid.py` is an additive expression runner. Its target is
selected by the explicit `expression_name` configuration field, with the
default `Smile`; its saved protocol includes both the name and array index.
It starts from the independently audited IsFixed neutral with zero Raw6 strain
and zero jaw pose. It uses collision, the preserved active-strain skin field,
all-four-IsFixed tet exclusion, an inversion count limit of 100, and retained
inverted-rest-volume fraction limit of `1e-4`.

The objective uses the existing target-scaled L2, normal, and same-muscle
activation-smoothness construction. Its weights are rebuilt from the Smile
target scale and the retained active-cell physical-volume graph; they are not
borrowed from a MouthOpen state.

`179-run-smile-collision-on.py` fixes the requested Smile pipeline and applies
an explicit UTC deadline (`deadline_iso_utc`, default 13:50 Asia/Shanghai on
2026-09-30). It computes the remaining wall budget at launch, then recomputes
the absolute deadline immediately before the first equilibrium evaluation. The runner writes
`input-preflight.json` before CUDA initialization, asserting byte-bound source
paths, that the IsFixed neutral endpoint reproduces the blendshape neutral, and
that the Smile target equals that neutral plus its selected source delta.

No GPU run was launched while local MouthOpen computation remained active.
