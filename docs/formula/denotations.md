### Denotations

- $\mathcal{L}$ - loss function
- $\vb{q}$ - parameters (muscle activation, etc.)
- $\vb{T}$ - target shape
- $\vb{u}$ - displacement
- $E$ - total energy

### Minimization Problem

$$
\begin{gather}
  \mathcal{L}(\vb{q}) = \norm{\vb{u} - \vb{T}}_F^2 + \text{regularizations} \\
  \vb{u} = \operatorname{Forward}(\vb{q})
\end{gather}
$$

### Derivative of Loss Function

$$
\begin{gather}
  \dv{\mathcal{L}}{\vb{q}} = \pdv{\mathcal{L}}{\vb{q}} + \pdv{E}{\vb{u}}{\vb{q}} \vb{p} \label{eq:1} \\
  \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}} \label{eq:2}
\end{gather}
$$

