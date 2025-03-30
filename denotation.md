# Denotation

## Material Properties

- `E`: Young's modulus
- `lmbda`: Lamé's first parameter
- `mu`: Lamé's second parameter
- `nu`: Poisson's ratio
- `q`: parameters (e.g. material properties)

## Array Annotations

- `C`: number of cells / elements / faces / tetras / triangles
- `D`: number of Dirichlet boundary conditions
- `F`: number of DoF after applying Dirichlet boundary conditions
- `N`: number of DoF without boundary conditions, usually `N = 3 * V`
- `Q`: number of parameters
- `V`: number of points / vertices

## Variables

- `dV`: volume per element
- `F`: deformation gradient
- `h`: shape function
- `J`: `det(F)`
- `PK1`: first Piola–Kirchhoff stress tensor
- `Psi`: energy density
- `R`, `S`: polar decomposition of `F`
- `u`: displacement
- `W`: energy
- `x`: position
- `X`: undeformed position
