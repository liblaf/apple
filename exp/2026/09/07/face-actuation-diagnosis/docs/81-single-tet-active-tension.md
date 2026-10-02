# Single-tetrahedron active-tension validation

## Purpose

This experiment tests a bounded, target-independent active-tension term before using it on the face mesh. It checks the constitutive derivatives on a generic deformation, compares the proposed law with the rejected scaled active-strain energy difference, and solves a deliberately restricted two-parameter tetrahedron problem over a range of gains.

The proposed muscle energy density is

$$
\Psi(F)=\Psi_{\mathrm{stable}}(F;\mu,\lambda_{\mathrm{code}})
+\frac{T}{2}\left(f^T F^T F f-1\right),
\qquad T=g(3\mu),
$$

where \(f\) is a fixed unit reference fiber and \(g\) is the reported gain. Its active first Piola stress and tangent action are

$$
P_{\mathrm{active}}=T(Ff)\otimes f,
\qquad
dP_{\mathrm{active}}[dF]=T(dFf)\otimes f.
$$

Thus \(T\) is a constant reference-frame second-Piola tension. It is not a prescribed natural contraction, a constant Cauchy tension, or a multiplier on the passive Lamé parameters. Gain zero restores the unchanged passive stable Neo-Hookean energy, stress, and tangent exactly.

## Commands and run records

Both accepted runs used the repository interpreter from this experiment directory, the `ProfileCometNoCommit` profile, and disabled Comet's automatic Git metadata, patch, and environment collection. The following are their exact historical invocations. Although the command path is the same, the source defaults changed between runs and each dataset freezes the source and resolved config that it actually used.

```bash
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
CHERRIES_NAME='Single tetrahedron active tension validation final' \
CHERRIES_TAGS='face,active-tension,single-tet,constitutive,validation' \
.venv/bin/python \
  src/81-single-tet-active-tension.py
```

```bash
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
CHERRIES_NAME='Single tetrahedron active tension extended gain validation' \
CHERRIES_TAGS='face,active-tension,single-tet,constitutive,extended-gain,validation' \
.venv/bin/python \
  src/81-single-tet-active-tension.py
```

The first accepted run recorded gains 0, 0.25, 0.5, 1, and 2 at [Comet experiment 3bf7034e](https://www.comet.com/liblaf/apple/3bf7034e2912407bb9e9721d924d3251). Its executed-source SHA-256 is `c710b4a0ce6e667686cb17f9cd0403c7e3c708012216d43335b9e4012d1a757c` and its resolved-config SHA-256 is `6c0bd30730590ed904cbc428f5a38d7ff4b46d9fc392f67650f28e96414d5578`. The extended run kept those cases and added gains 3 and 10 plus the objectivity audit at [Comet experiment 235f8049](https://www.comet.com/liblaf/apple/235f8049d4b54fbda4d5f0e3bd176b1e). Its executed-source SHA-256 is `aac891afb9ce47128e269940db4c97ec354c058034bf2acb2af3855a31087f10` and its resolved-config SHA-256 is `6b7b8d06bb3a07278ff839f0e3974c88b7356ad27a52cda99603e1a1fdea4ea3`. Both recorded Git SHA `d56fa1b553b287b22b2cf7bb82d46117e34ed6bb`; neither committed or changed Git state.

To replay the extended configuration from the current source without overwriting either accepted dataset, use a new output path and pass the gain list explicitly:

```bash
COMET_AUTO_LOG_GIT_METADATA=false \
COMET_AUTO_LOG_GIT_PATCH=false \
COMET_AUTO_LOG_ENV_DETAILS=false \
CHERRIES_NAME='Single tetrahedron active tension extended replay' \
CHERRIES_TAGS='face,active-tension,single-tet,constitutive,extended-gain,replay' \
.venv/bin/python \
  src/81-single-tet-active-tension.py \
  --output-dir data/81-single-tet-active-tension-replay \
  --gains '[0,0.25,0.5,1,2,3,10]'
```

Two earlier directories ending in `failed-zip-strict` and `failed-source-path` preserve runner-harness failures. They are not constitutive failures and are excluded from the accepted results.

## Constitutive checks

The material constants are the unchanged manual-volume values, \(E=0.024\) MPa and \(\nu=0.46\). They give \(\mu=0.0082191781\) MPa and \(\lambda_{\mathrm{code}}=0.1027397260\) MPa, where the stable polynomial uses \(\lambda_{\mathrm{code}}=\lambda_{\mathrm{classical}}+\mu\). The reference tension is \(3\mu=0.0246575342\) MPa.

At a generic non-diagonal \(F\), central finite differences gave:

| Check | Maximum absolute error |
| --- | ---: |
| Energy gradient versus first Piola stress | \(4.67\times10^{-12}\) MPa |
| Stress difference versus tangent action | \(1.07\times10^{-11}\) MPa |
| Tangent diagonal | \(9.94\times10^{-12}\) MPa |
| Hessian symmetry | \(1.39\times10^{-17}\) MPa |
| Energy second difference versus Hessian quadratic | \(6.43\times10^{-12}\) MPa |

The active material tangent was positive semidefinite with rank 3. Under a superposed spatial rotation, the absolute energy error was \(8.67\times10^{-19}\) MPa, the first-Piola covariance error was \(1.99\times10^{-17}\) MPa, and the tangent-action covariance error was \(6.94\times10^{-18}\) MPa.

The active addition is at least \(-T/2\) and adds a positive-semidefinite tangent. Combined with the coercive stable passive polynomial, the total energy is bounded below. By comparison, the rejected construction

$$
\Psi_{\mathrm{passive}}(F)+10\left[
\Psi_{\mathrm{active\ strain}}(F,A^{-1}_{c50})-
\Psi_{\mathrm{passive}}(F)\right]
$$

has transverse quadratic coefficient \(-4\). Along \(F=\operatorname{diag}(1,s,1/s)\), its sampled energy reached \(-16438.24\) MPa at both \(s=10^{-3}\) and \(s=10^3\), consistent with divergence to negative infinity. The proposed law was \(+4109.58\) MPa at those endpoints.

## Restricted tetrahedron response

The tetrahedron is orthogonal. The origin is fixed, the three edge nodes remain on their reference axes, and the two transverse stretches are constrained equal. The solver therefore searches only the log axial and log transverse stretches. These results show an equilibrium and positive Hessian in that two-dimensional family; they do not establish a full unconstrained global minimum.

| Gain | Axial, unloaded | Transverse, unloaded | det \(F\), unloaded | Axial, 5 kPa load | Transverse, 5 kPa load | det \(F\), 5 kPa load |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.000000 | 1.000000 | 1.000000 | 1.254281 | 0.900111 | 1.016218 |
| 0.25 | 0.825109 | 1.091517 | 0.983043 | 0.962509 | 1.017700 | 0.996884 |
| 0.5 | 0.729445 | 1.153354 | 0.970328 | 0.823311 | 1.092591 | 0.982831 |
| 1 | 0.619467 | 1.238935 | 0.950857 | 0.676887 | 1.192029 | 0.961812 |
| 2 | 0.508946 | 1.346545 | 0.922812 | 0.541343 | 1.312269 | 0.932219 |
| 3 | 0.448404 | 1.417978 | 0.901590 | 0.471040 | 1.390051 | 0.910163 |
| 10 | 0.296794 | 1.652480 | 0.810453 | 0.304290 | 1.638672 | 0.817093 |

All equilibrium residuals were below \(4\times10^{-15}\) MPa. The minimum restricted tangent eigenvalue remained positive in every case. The gain-10 cases show that the intended axial shortening continues without an inversion in this constrained family, while also showing substantial transverse expansion and volume loss. They set a useful stress-test bound; they do not predict safe behavior on the full heterogeneous face mesh.

## Outputs and reproducibility

The original accepted dataset is preserved under `data/81-single-tet-active-tension/`. The extended dataset is under `data/81-single-tet-active-tension-extended/`. Each directory contains `config.json`, `summary.json`, a recursive hash manifest, the exact executed source copy, per-case VTU states, and PNG/PDF figures. Recursive hash verification passed for both accepted datasets. The four extended figures were also inspected visually for legibility and consistent geometry.

That next step was executed by `src/88-active-tension-face.py` with the same passive material constants, smile-elevator cells, reference fibers, no skin, identity activation inverse, and explicit gains. Gains 0, 1, and 3 produced accepted equilibria; gain 10 reached the unchanged 10,000-step budget and has no accepted state. The complete terminal evidence, including hashes and the failure receipt, is recorded in `data/93-active-tension-face-outcome.json` and analyzed in `docs/88-active-tension-face.md`.
