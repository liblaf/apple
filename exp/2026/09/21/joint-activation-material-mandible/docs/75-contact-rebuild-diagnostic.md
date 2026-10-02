# Full-source contact rebuild diagnostic

This CPU-only diagnostic rebuilt the complete-source soft-versus-bone contact
state three times at four exact accepted states from the stopped simple-forward
run 003. It did not solve an equilibrium or use the GPU. Its purpose was to
separate near-zero-gap conditioning from a collision-representation defect.

```bash
CHERRIES_NAME='CPU full-source contact rebuild and weight diagnostic' \
CHERRIES_TAGS='joint-inverse,contact,diagnostic,cpu' \
uv run python src/75-contact-rebuild-diagnostic.py \
  --output-dir data/contact-rebuild-diagnostic-001
```

| Accepted step | Minimum active gap | Improved-max energy (MPa m3) | Improved negative weights | Improved collision counts | IPC energy (MPa m3) | IPC negative weights | IPC collision count |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.121721 um | 2.480e-9 | 951 | 6,186 / 6,188 / 6,190 | 5.240e-9 | 0 | 6,060 |
| 200 | 2.751e-14 m | 4.921e-10 | 744 | 5,515 / 5,518 / 5,520 | 1.917e-9 | 0 | 5,437 |
| 600 | 1.018e-13 m | -9.684e-11 | 664 | 5,292 / 5,293 | 1.363e-9 | 0 | 5,151 |
| 1,000 | 9.955e-15 m | -1.907e-10 | 647 | 5,164 / 5,165 / 5,166 | 1.561e-9 | 0 | 5,018 |

`IMPROVED_MAX_APPROX` therefore has signed area weights and changes equivalent
representatives across identical rebuilds. At the two later states, their sum
makes the nominal barrier energy negative. Standard `IPC` had only positive
weights, deterministic counts, energy repeat ranges below `2.5e-24 MPa m3`,
and relative net-force imbalance below `4.2e-16` in every tested state.

The complete-source standard-IPC directional gradient agreed with central
finite differences to at most `1.152e-6` relative error. This independently
checks the actual surface and area units, rather than only a synthetic fixture.
It does not establish that the subsequent nonlinear forward solve will
converge. Standard IPC is a different, stronger contact objective here, so a
fresh run is required; the invalid run-003 state must not be resumed.

IPCTK 1.6.0 exposes `IPC`, `IMPROVED_MAX_APPROX`, and `OGC`; it has no
`MAX_APPROX` enum. A synthetic point-triangle crossing returned a verified
collision-free fraction with Tight Inclusion caps of 1, 10, 100, 100,000, and
10,000,000 iterations. The smaller caps returned a slightly more conservative
fraction (`0.39375` versus `0.3984375`). This one fixture supports a bounded cap
as a safeguard but is not a general proof of Tight Inclusion behavior.

The receipt is
[`summary.json`](../data/contact-rebuild-diagnostic-001/summary.json) (SHA-256
`8fe091cb7316643da6603753871fcd2157c61881e39b4f593f02c9ec644a3335`).
The script SHA-256 is
`c35600c67dd8a14269fbade274f85e477b40d0dacd2e3f9f73e462df86a450b2`.
The [Comet run](https://www.comet.com/liblaf/apple/2fde088612e84ce7a87c6e1c64951325)
shut down cleanly. Cherries emitted a nonfatal missing-default-output warning
because the explicit `-001` directory differs from its registered default;
the explicit receipt and provenance files are complete.
