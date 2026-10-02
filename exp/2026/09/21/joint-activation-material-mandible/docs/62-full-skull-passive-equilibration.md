# Passive full-skull initialization attempt

The zero-stress initialization attempt was **interrupted without a numerical
receipt**. It is neither a converged preparation nor evidence of numerical
infeasibility. Complete-source collision performance is reported separately in
[63](63-valid-seed-operation-performance.md).

The protocol used the admitted candidate002 displacement, unchanged complete
source bones, canonical passive material stiffnesses, zero baseline and
activation stress, unit skin stiffness multiplier, and zero jaw pose. PNCG had
`rtol=1e-3`, `atol=1e-9`, and at most 5,000 steps. Exact Newton refinement at
`rtol=1e-6`, `atol=1e-12` was conditional on PNCG convergence and the geometry
gates; it was never reached. Bone-bone contact was excluded.

The first phase began at approximately 15:12 on September 21, 2026
(Asia/Shanghai). After about 15 minutes, only source provenance and the protocol
had been saved: no iteration log, displacement checkpoint, phase receipt, or
summary was available. An operational wall-time limit was added **after launch**
to bound this diagnostic. SIGINT was sent at 15:26:59; it did not terminate the
process within about one minute. SIGTERM at 15:28:01 ended the process with code
143. Cherries shutdown did not complete normally. No other process was stopped.

The absence of a receipt does not reveal the current force residual or accepted
iteration count. The next solver attempt needs frequent persistent step/force
telemetry, interruption-safe state capture, and a declared wall-time cap before
launch. The current geometry admission remains valid for initialization only.
Nonzero prestress preparation, complete-source jaw validation, and the final
joint optimization remain incomplete.

Run from `exp/2026/09/21/joint-activation-material-mandible`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
CHERRIES_NAME='Passive complete skull contact equilibration' \
CHERRIES_TAGS=joint-inverse,full-skull,passive,equilibrium \
uv run --frozen python src/62-equilibrate-full-skull-passive.py
```

Evidence is retained under `data/full-skull-passive-equilibrium-001/`, with the
archived source and `logs/62-equilibrate-full-skull-passive.log`.
[Comet metadata](https://www.comet.com/liblaf/apple/e4ae2e0f94fc4ac7ada3ae8e0d35d31e)
is not a completion receipt.

| Artifact | SHA-256 |
| --- | --- |
| Protocol | `d8dbea69c9a37165e76f133cb3cb284718159299ad15e7ff7a135ad4aae65461` |
| Interruption | `6d9c4ae7282ca26a3c70336b6bec2cfd472560f1df42a41f896ea0f17e42e241` |

The experiment used the dirty shared checkout at
`d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; Cherries Git commits were disabled.
