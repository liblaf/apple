# Differentiable Physics

## Annotations

- SYMBOL (SHAPE) - DESCRIPTION
- $\mathbf{F}$ (C,) - deformation gradient
- $\mathbf{q}$ (Q,) - parameters (elastic moduli, muscle activation, etc.)
- $\mathbf{u}$ (U,) - displacement
- $\mathcal{L}$ (scalar) - loss
- $E$ (scalar) - elastic energy

## Derivative of Loss Function

$$
\begin{align*}
  \dv{\mathcal{L}}{\vb{q}} = \pdv{\mathcal{L}}{\vb{q}} + \pdv{E}{\vb{q}}{\vb{u}} \vb{p} \\
  \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}}
\end{align*}
$$

## Denotations

- $\mathcal{L}$ - loss function
- $\vb{q}$ - parameters (muscle activation, etc.)
- $\vb{T}$ - target shape
- $\vb{u}$ - displacement
- $E$ - total energy

## Plan

###### Minimization Problem

Compute loss using surface points.

$$
\begin{gather}
  \mathcal{L}(\vb{q}) = \norm{\vb{u} - \vb{T}}_F^2 + \text{regularizations} \\
  \vb{u} = \operatorname{Forward}(\vb{q})
\end{gather}
$$

###### Derivative of Loss Function

$$
\begin{gather}
  \dv{\mathcal{L}}{\vb{q}} = \pdv{\mathcal{L}}{\vb{q}} + \pdv{E}{\vb{u}}{\vb{q}} \vb{p} \label{eq:1} \\
  \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}} \label{eq:2}
\end{gather}
$$

###### Steps

- given $\vb{q}$, apply forward physics to get $\vb{u}$
- solve $\displaystyle \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}}$ using CG => $\vb{p}$
- compute $\displaystyle \pdv{E}{\vb{q}}{\vb{u}} \vb{p}$ (sparse matrix @ dense vector)
- compute $\displaystyle \pdv{\mathcal{L}}{\vb{q}}$
- minimize $\mathcal{L}$ based on $\displaystyle \dv{\mathcal{L}}{\vb{q}}$ using L-BFGS-B

## Current

###### Minimization Problem

**Compute loss using all points (including internal points).**

$$
\begin{gather}
  \mathcal{L}(\vb{q}) = \norm{\vb{u} - \vb{T}}_F^2 \\
  \vb{u} = \operatorname{Forward}(\vb{q})
\end{gather}
$$

###### Steps

- given $\vb{q}$, apply forward physics to get $\vb{u}$
- solve $\displaystyle \pdv[2]{E}{\vb{u}} \vb{p} = - \pdv{\mathcal{L}}{\vb{u}}$ using CG => $\vb{p}$
- compute $\displaystyle \pdv{E}{\vb{q}}{\vb{u}} \vb{p}$ (**dense** matrix @ dense vector)
- compute $\displaystyle \pdv{\mathcal{L}}{\vb{q}} = \vb{0}$
- minimize $\mathcal{L}$ based on $\displaystyle \dv{\mathcal{L}}{\vb{q}}$ using **Gradient Descent**
