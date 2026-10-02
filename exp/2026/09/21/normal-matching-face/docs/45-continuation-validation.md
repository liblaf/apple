# Adam continuation validation

The continuation preserves the original numerical sources, fixture, objective
weights and optimizer settings. The original 100-update study is documented in
[10-results.md](10-results.md); the continuation protocol is in
[40-continuation-protocol.md](40-continuation-protocol.md).

## CPU checkpoint preflight

[35-resume-checks/checks.json](../data/35-resume-checks/checks.json) passed.
All 97 original numerical source records and three fixture hashes match. Each
step-100 checkpoint has exactly the saved controls and displacement, Adam first
and second moments, and counter 100. Tensor storage is independently copied
before restoring the optimizer. An explicit bias-corrected Adam calculation
agrees with the restored optimizer for a deterministic synthetic gradient over
all 1,729,410 controls. No CUDA context is initialized by this check.

The final source-only assertion-format cleanup was checked by rerunning the
preflight into `35-resume-checks-final-source`. Both check receipts have SHA-256
`4f3b09f4cfb3681bd9729ad5b916d0a3b9a2431dd4d9fb0051eec01fe1ba0ea9`.

## One-update GPU smoke and independent audit

All four branches resumed their original update-100 checkpoint and reached
update 101. The smoke process exited 0. The independent CPU audit also exited
0 with both `passed` and `all_completed` true in
[39-verification/checks.json](../data/39-verification/checks.json).

| Branch | Replay maximum displacement difference (m) | Replay physical-gradient RMS relative difference | Explicit Adam q101 maximum absolute error |
| --- | ---: | ---: | ---: |
| Smooth off, L2 | 7.8943e-8 | 3.0370e-5 | 1.1102e-16 |
| Smooth off, L2 + normal | 6.4565e-8 | 6.5684e-5 | 2.2204e-16 |
| Smooth on, L2 | 3.5021e-8 | 1.6509e-5 | 1.1102e-16 |
| Smooth on, L2 + normal | 7.9217e-8 | 3.8815e-5 | 2.2204e-16 |

Replay position-RMS differences are below 3e-9 mm. Each branch retains its
single inverted tetrahedron. Controls, moments and counter remain exactly
unchanged during replay. The auditor checks inherited trace values, solver
receipts and saved checkpoints, recomputes the physical endpoint metrics, and
checks the first new update against the explicit Adam formula.

The original run did not save its full step-100 gradient or adjoint warm start.
Consequently, scalar gradient-RMS agreement is not a test of identical gradient
directions. The finite-tolerance forward replay uses saved u100 rather than
the old u99 seed and is not bitwise identical. The failed initial 1 nm replay
threshold and its receipts are retained; only comparison bounds were changed,
as detailed in the protocol. Failed iterations of the verifier itself are
preserved in the `39-verification-*-audit-contract-mismatch` directories.

The main continuation requires this audited smoke and an exact hash match of
the continuation runner to its smoke-tested snapshot. Its source SHA-256 is
`161e42966b43a7b28109e677d380e5f2c2afb362d4814756bdc8102ed37e5c2b`.
It resumes the original update-100 checkpoints independently of the smoke.

## Final continuation audit

The main continuation completed all four branches at update 200 and exited 0
after Comet shutdown. The independent final CPU audit also exited 0 after
shutdown, with `passed=true` and `all_completed=true` in
[45-verification/checks.json](../data/45-verification/checks.json). It verified
98 source snapshots and current source files, three fixture receipts, exact
inherited history/checkpoints, 100 new successful solve receipts per branch,
restored parent optimizer state, and final Adam counters of 200. The maximum
absolute discrepancy among independently recomputed endpoint metrics was
2.6646e-15. Manual first-update Adam errors were 1.1102e-16 to 2.2204e-16.

All four endpoint geometries contain two inverted tetrahedra. The artifact
audit passing verifies the recorded continuation; it does not certify physical
validity or inverse convergence.

- [Main continuation](https://www.comet.com/liblaf/apple/ad00ed6e78c1407c8d70ee53a642ce45)
- [Final CPU audit](https://www.comet.com/liblaf/apple/d9b8aeacd71b4d0dac3243d9212b487f)

## Commands

Working directory: `exp/2026/09/21/normal-matching-face`.
Use the repository `.venv/bin/python` for all commands.

```bash
DEBUG=1 CHERRIES_NAME='Raw6 Adam resume CPU preflight' CHERRIES_TAGS='raw6,resume,adam,cpu-preflight' python src/35-check-resume.py
DEBUG=1 CHERRIES_NAME='Raw6 face continuation: one-update resume smoke' CHERRIES_TAGS='face,raw6,continuation,adam,resume,smoke' OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python src/40-continue.py --output 39-smoke --steps 101 --require-smoke false
DEBUG=1 CHERRIES_NAME='Raw6 face continuation smoke audit' CHERRIES_TAGS='face,raw6,continuation,cpu,audit,smoke' python src/45-verify-continuation.py --comparison-dir 39-smoke --output 39-verification
```
