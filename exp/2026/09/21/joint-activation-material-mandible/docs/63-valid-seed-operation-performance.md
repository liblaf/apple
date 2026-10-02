# Valid-seed complete-source operation timing

`58-benchmark-full-skull-model-operations.py` was generalized to load the
hash-bound candidate002 initialization and admission. It independently verifies
the geometry hash and recomputes the volume metrics before timing. The earlier
candidate001 invalid-volume receipts remain unchanged.

The benchmark compares the same extended model and DOF map with complete-source
soft-bone contact disabled and enabled. Both arms use the frozen Spatial80
material state and candidate002 displacement. The protocol is unchanged: one
warmup per arm, five CUDA-synchronized measurements per arm, and alternating
order. It runs fixed-state model operations only; no equilibrium solve occurs.

Median warmed timings from
`data/full-skull-model-operation-benchmark-valid-contended-001/summary.json`:

| Operation | Contact off | Complete-source contact | Ratio |
| --- | ---: | ---: | ---: |
| State build | 1.525 ms | 13.870 ms | 9.09x |
| Energy | 8.124 ms | 12.885 ms | 1.59x |
| Gradient | 1.808 ms | 7.277 ms | 4.02x |
| First HVP | 4.897 ms | 22.164 ms | 4.53x |
| Cached HVP | 5.438 ms | 6.109 ms | 1.12x |
| CCD maximum step | 0.169 ms | 12.961 ms | — |

Contact-off CCD is a no-op; its ratio has no useful physical meaning. The
complete-source CCD cost is about 13 ms.

The complete-source state had 6,184 active soft-bone pairs, minimum active
distance 0.121721 micrometres, and a valid finite contact receipt. The seed had
zero inverted tetrahedra and determinant range 0.250100 to 1.999900.

This was an intentionally contended measurement. Startup GPU utilization was
96%, with two other Python GPU processes present. The ratios describe this
fixed-state run and are not idle-capacity estimates. They do not measure a full
Newton solve, equilibrium convergence, source bone-bone contact, or final-run
readiness.

Evidence:

- summary SHA-256:
  `92667709010849410cf55af495db2700ab8434d25b7fced62abfd12843d0238c`;
- candidate SHA-256:
  `82db42385ec03d8dd5b997b41dbf2ef419d286978994a4b8f8be452af0d75b77`;
- admission SHA-256:
  `d001751d06eedda19aac39486f17eead928b6b9ae3543f232b7de38deebef115`;
- benchmark source SHA-256:
  `dd1bd45585eae76ca99e40e85222eb4df173b7cb8295fae750fe8772f828e0e2`;
- [Comet run](https://www.comet.com/liblaf/apple/6864612e31334897a8d3e5fa32f7d1f7).

Reproduction:

```bash
CHERRIES_NAME=full-skull-model-operation-benchmark-valid \
CHERRIES_TAGS=joint,full-skull,performance,valid-initialization,contended \
uv run exp/2026/09/21/joint-activation-material-mandible/src/58-benchmark-full-skull-model-operations.py \
  --output-dir exp/2026/09/21/joint-activation-material-mandible/data/full-skull-model-operation-benchmark-valid-contended-001
```
