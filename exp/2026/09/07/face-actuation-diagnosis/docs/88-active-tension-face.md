# Target-independent active-tension face diagnostic

## Purpose and fixed comparison

This diagnostic transfers the validated active-tension law from the single tetrahedron to the full face volume without fitting a target. It uses the same fixture, boundary conditions, passive materials, smile-elevator muscle IDs, and reference fibers as the manual no-skin experiment. The activation inverse is identity for every cell. Every gain starts from the same zero-displacement seed; there is no continuation and no inverse optimization.

The muscle energy is the unchanged stable Neo-Hookean passive density plus one term:

$$
\Psi(F)=\Psi_{\mathrm{stable}}(F;\mu,\lambda_{\mathrm{code}})
+\frac{T}{2}\left(f^T F^T F f-1\right),
\qquad T=g(3\mu).
$$

The active first Piola stress is \(T(Ff)\otimes f\), and its tangent action is \(T(dFf)\otimes f\). The full Warp implementation supplies matching energy, stress, Hessian diagonal, Hessian product, and Hessian quadratic paths. The muscle constants remain \(E=0.024\) MPa, \(\nu=0.46\), \(\mu=0.0082191781\) MPa, and \(\lambda_{\mathrm{code}}=0.1027397260\) MPa. Therefore the gain-one tension parameter is \(3\mu=0.0246575342\) MPa.

The selected actuator contains 9,732 tetrahedra and 3,660 vertices across muscle IDs 57, 58, 63, 64, 142, 143, 218, 219, 283, and 284. All selected fixture fibers have unit norm. The fixture contains 1,146,517 tetrahedra, 228,660 vertices, 120,020 active tetrahedra, and 33,636 fixed vertices.

## CPU Warp audit

Before the face launch, the Warp implementation was compiled and checked on one tetrahedron on the CPU. The accepted audit used `ProfileCometNoCommit`, recorded Git SHA `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`, and ran from this experiment directory:

```bash
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
CHERRIES_NAME='Warp active tension CPU assembly audit' \
CHERRIES_TAGS='face,active-tension,warp,cpu,audit,validation' \
.venv/bin/python \
  src/88-active-tension-face.py \
  --audit-only true \
  --output-dir data/88-active-tension-warp-audit
```

The run is recorded at [Comet experiment 2f0ecbbc](https://www.comet.com/liblaf/apple/2f0ecbbcc993473d861f1a4dcd2968f9). The directional energy-gradient error was \(3.03\times10^{-15}\), the Hessian-product error was \(3.57\times10^{-13}\), the Hessian-diagonal error was \(2.42\times10^{-13}\), and the Hessian quadratic differed from the assembled product by \(5.42\times10^{-20}\). Its finite-difference second-energy error was \(2.81\times10^{-9}\). At zero tension, energy, gradient, Hessian diagonal, Hessian product, and Hessian quadratic matched the production passive material exactly. The audit also verified the identity activation-inverse encoding and recorded hashes for the fixture volume, skin, and summary files.

## Full-face command and run identity

The terminal physics run used the following durable service invocation, with output and error appended to `logs/88-active-tension-face-service.log`:

```bash
systemd-run --user \
  --unit=apple-face-active-tension-88 \
  --collect \
  --property=Type=exec \
  --property=StandardOutput=append:exp/2026/09/07/face-actuation-diagnosis/logs/88-active-tension-face-service.log \
  --property=StandardError=append:exp/2026/09/07/face-actuation-diagnosis/logs/88-active-tension-face-service.log \
  --working-directory=exp/2026/09/07/face-actuation-diagnosis \
  --setenv=COMET_AUTO_LOG_GIT_METADATA=false \
  --setenv=COMET_AUTO_LOG_GIT_PATCH=false \
  --setenv=COMET_AUTO_LOG_ENV_DETAILS=false \
  --setenv=CHERRIES_NAME='Target-independent active tension face diagnostic rerun' \
  --setenv=CHERRIES_TAGS='face,active-tension,warp,forward,no-skin,diagnostic' \
  .venv/bin/python \
  src/88-active-tension-face.py
```

The service unit was `apple-face-active-tension-88.service`, invocation ID `51e829e8c21a4810933543e624cdf499`, and launch PID 793581. Cherries ran from 2026-09-07 16:51:40 to 16:55:03 CST and recorded [Comet experiment 76c20908](https://www.comet.com/liblaf/apple/76c20908ab484430a5c6db268fde038d). The run used `ProfileCometNoCommit` and Git SHA `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`.

The Cherries exception hook recorded the final `ForwardConvergenceError` but returned process status 0. Consequently systemd reported `Result=success` and `ExecMainStatus=0`, although the experiment ended with a partial forward failure. The terminal receipt records both statuses so process completion cannot be mistaken for four successful equilibria.

## Results

| Gain | Tension (MPa) | Solver steps | Gradient norm | Min det \(F\) | Inverted tets | Median fiber stretch | Surface RMS (mm) | Surface max (mm) | Smile projection | Lip RMS (mm) | Mean outward lip motion (mm) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 1 | 0 | 1.000000 | 0 | 1.000000 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0.0246575 | 1,177 | \(2.14\times10^{-12}\) | 0.673549 | 0 | 0.950601 | 0.414898 | 2.201443 | 0.045063 | 1.240264 | 0.237229 |
| 3 | 0.0739726 | 3,189 | \(6.28\times10^{-12}\) | 0.159448 | 0 | 0.875791 | 1.033988 | 5.516938 | 0.115819 | 2.989078 | 0.650993 |
| 10 | 0.2465753 | 10,000 | \(1.22\times10^{-9}\) | — | — | — | — | — | — | — | — |

Gains 0, 1, and 3 met the unchanged forward tolerance and have accepted VTU, NPZ, and diagnostics files. Gain 10 reached the fixed 10,000-step budget before meeting that tolerance. It has only a failure receipt. No equilibrium state was exported, and no geometry or motion value is inferred from its final iterate.

The response is target-independent: the smile target is absent from the material energy, control, and equilibrium solve. The reported smile projection uses the target only after equilibrium to describe the direction of the observed motion. From gain 1 to gain 3, the surface RMS rose from 0.415 to 1.034 mm, the lip RMS from 1.240 to 2.989 mm, and the post-hoc smile projection from 0.0451 to 0.1158. The selected-fiber median fell from 0.9506 to 0.8758, consistent with the intended contraction direction.

The fraction-volume-weighted mean active first-Piola norm was 0.02336 MPa at gain 1 and 0.06452 MPa at gain 3. The corresponding median active Cauchy stress along the current fiber was 0.02243 and 0.05788 MPa. These values reflect deformation and mixture weighting; they are not a fitted facial-muscle stress estimate.

Gain 3 remained uninverted, but its minimum determinant was 0.159 while its 1st-percentile determinant was 0.976. This isolates a small region of severe local compression. The experiment deliberately had no post-solve geometry rejection gate, so the state is retained as a converged diagnostic rather than presented as a validated physiological regime.

## Terminal evidence and limitations

The authoritative terminal receipt is `data/93-active-tension-face-outcome.json`, 34,328 bytes, SHA-256 `9433a744a86494558f4b1b35b3b404c4ad4a58356ff76881a3079bbebc1c7e33`. It hashes every existing file under `data/88-active-tension-face/`, the service log, all accepted state endpoints, the gain-10 failure receipt, the fixed run contract, input provenance, and the preserved earlier runner failure. The receipt status is `partial_forward_failure`; its renderable cases are `gain-0000`, `gain-0100`, and `gain-0300`, and its non-renderable case is `gain-1000`.

The source run stopped on gain 10 before its normal `summary.json` and `manifest.json` writers. Their absence is recorded in the terminal receipt and must not be treated as missing evidence for the three accepted cases. Downstream rendering and comparison must consume only the three renderable labels and cite the terminal receipt.

This diagnostic establishes that the bounded active-tension material produces target-independent smile-elevator motion at gains 1 and 3 under the fixed no-skin model. It does not establish physiological calibration, acceptable local distortion at gain 3, convergence at gain 10, behavior with skin, or equivalence to the earlier active-strain control.
