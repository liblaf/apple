# Current PNCG Solver

$$
u^\*=\arg\min_{u_{\mathrm{free}}} E(u_{\mathrm{free}})
$$

```text
given u0
for k = 0,1,2,...
  gk  = ∇E(uk)
  dk  = diag(Hk) + λ
  d̃k,i = |dk,i|  if |dk,i| > 0, else meanj:|dk,j|>0 |dk,j|
  Mk⁻¹ = diag(1 / d̃k)

  βk = βDK(gk, gk-1, pk-1, Mk⁻¹)
  pk = -Mk⁻¹ gk + βk pk-1
  if k = 0 or gkᵀpk ≥ 0 or previous step rejected:
       pk = -Mk⁻¹ gk

  qk = pkᵀHkpk + λ ||pk||²
  α0 = min( -(gkᵀpk)/qk, Δ∞ / ||pk||∞ )
  αk = backtracking Armijo(α0)

  uk+1 = uk + αk pk
  stop if ||gk|| ≤ atol + rtol ||g0||
```

$$
\beta_k^{\mathrm{DK}}
=
\frac{g_k^\top M_k^{-1}(g_k-g_{k-1})}{(g_k-g_{k-1})^\top p_{k-1}}
\;-\;
\frac{(g_k-g_{k-1})^\top M_k^{-1}(g_k-g_{k-1})}{\left((g_k-g_{k-1})^\top p_{k-1}\right)^2}
\,(p_{k-1}^\top g_k)
$$

$$
E(u_k+\alpha_k p_k)\le E(u_k)+c_1\alpha_k g_k^\top p_k
$$

$$
\text{terminate} \in \{\text{converged},\ \text{max iters},\ \text{stagnation},\ \text{NaN}\}
$$
