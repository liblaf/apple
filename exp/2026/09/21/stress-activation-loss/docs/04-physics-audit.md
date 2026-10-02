# Signed-stress physics audit

This CPU-only audit covers the experiment-local material before the first
full-face GPU validation. It does not establish forward convergence or inverse
fit quality.

The volume energy implemented in `src/stress_material.py` is

\[
W(F,Q)=\frac{\mu}{2}(\|F\|_F^2-3)-\mu(J-1)
+\frac{\lambda}{2}(J-1)^2+\frac12Q:(F^TF-I),\qquad J=\det F,
\]

with finite symmetric, caller-supplied additive second-Piola stress `Q` in MPa.
Its first Piola stress is

\[
P(F,Q)=P_{\mathrm{SNH}}(F)+FQ.
\]

On 2026-09-21, a double-precision CPU Torch-autograd directional check at a
random orientation-preserving `F` and symmetric signed `Q` gave maximum
componentwise absolute error `1.1102230246251565e-16 MPa` between autograd and
the implemented `P` formula. No full-face GPU solve was run for this audit.

The physics module clears the previous adjoint solution at each solve and its
adjoint receipt records absolute and relative residuals. It rejects exhausted
or non-finite inner Armijo trials and PNCG non-success; the outer runner must
also reject a solved state with `det(F) <= 0`.

Frozen source hashes at this audit:

- `src/stress_material.py`:
  `0a834754bf344d7076cc7ab200db74210fd097111cea72ef107e46183a24e32e`
- `src/stress_physics.py`:
  `333d6df7315d22ee5b0b05357833d0b7fb244f9e83bfb09f758159218c0d7947`

The study regularizer applies to `Qhat = Q / 0.004026845637583893 MPa`. With
the fixed 5 mm graph factor, it is dimensionless; its reported neighbor RMS is
converted back to kPa. The normal and position terms use the fixed
`l_ref = 13.236093032531715 mm` and normal weights 0 or 1.
