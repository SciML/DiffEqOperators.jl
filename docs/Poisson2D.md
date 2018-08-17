# Solving the 2D Poisson equation using DiffEqOperators

This tutorial demonstrates how to use `DiffEqOperators` to construct finite difference approximations of two-dimensional operators for the solution of elliptic partial differential equations. For simplicity, we consider the Poisson equation on the unit square with homogeneous Dirichlet conditions:

    -Δu = -∂_xx u -∂_yy u = f on [0,1]²,
                        u = 0 on ∂[0,1]².

The following function uses a Kronecker product of one-dimensional differential operators to construct the two-dimensional finite difference approximation of the Laplace operator -Δ using N×N points as a sparse matrix:

```julia
using DiffEqOperators, SparseArrays, LinearAlgebra
function Laplacian2D(N)
    h = 1/N
    D2 = sparse(DerivativeOperator{Float64}(2,2,h,N,:Dirichlet0,:Dirichlet0))
    Id = sparse(I,N,N)
    A = -kron(D2,Id) - kron(Id,D2)
    return A
end
```

This can be used to solve the Poisson equation for f ≡ 1 as follows:

```julia
N = 128
A = Laplacian2D(N)
f = fill(1.0,N*N)
u = reshape(A\f,N,N)
using GR; surface(u)
```

This approach extends to higher dimensions as well; e.g., the Laplacian on the unit cube can be constructed by

```julia
function Laplacian§D(N)
    h = 1/N
    D2 = sparse(DerivativeOperator{Float64}(2,2,h,N,:Dirichlet0,:Dirichlet0))
    Id = sparse(I,N,N)
    A = -kron(D2,Id,Id) - kron(Id,D2,Id) - kron(Id,Id,D2)
    return A
end
```

