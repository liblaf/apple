## Plan

### Minimization Problem

- use co-rotational elasticity
- compute loss using surface points

$$
\begin{gather}
  \mathcal{L}(\vb{q}) = \norm{\vb{u} - \vb{T}}_F^2 + \text{regularizations} \\
  \vb{u} = \operatorname{Forward}(\vb{q})
\end{gather}
$$

### Steps

- given $\vb{q}$, apply forward physics to get $\vb{u}$
- solve $\displaystyle \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}}$ using CG => $\vb{p}$ (sparse matrix @ dense vector)
- compute $\displaystyle \pdv{E}{\vb{q}}{\vb{u}} \vb{p}$ (sparse matrix @ dense vector)
- compute $\displaystyle \pdv{\mathcal{L}}{\vb{q}}$
- minimize $\mathcal{L}$ based on $\displaystyle \dv{\mathcal{L}}{\vb{q}}$ using L-BFGS-B
