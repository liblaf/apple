## Current

### Minimization Problem

- **use As-Rigid-As-Possible elasticity**
- **compute loss using all points (including internal points)**

$$
\begin{gather}
  \mathcal{L}(\vb{q}) = \norm{\vb{u} - \vb{T}}_F^2 \\
  \vb{u} = \operatorname{Forward}(\vb{q})
\end{gather}
$$

### Steps

- given $\vb{q}$, apply forward physics to get $\vb{u}$
- solve $\displaystyle \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}}$ using CG => $\vb{p}$ (**dense** matrix @ dense vector)
- compute $\displaystyle \pdv{E}{\vb{q}}{\vb{u}} \vb{p}$ (**dense** matrix @ dense vector)
- compute $\displaystyle \pdv{\mathcal{L}}{\vb{q}} = \vb{0}$
- minimize $\mathcal{L}$ based on $\displaystyle \dv{\mathcal{L}}{\vb{q}}$ using **Gradient Descent**

