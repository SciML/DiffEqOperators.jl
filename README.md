# DiffEqOperators.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqOperators.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/au9knv63u9oh1aie?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/diffeqoperators-jl)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DiffEqOperators.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DiffEqOperators.jl?branch=master)
[![codecov.io](http://codecov.io/github/shivin9/DiffEqOperators.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DiffEqOperators.jl?branch=master)

DiffEqOperators.jl provides a set of pre-defined operators for use with
DifferentialEquations.jl. These operators make it easy to discretize and solve
common partial differential equations.

## Automated Finite Difference Method (FDM) Operators

This library provides lazy operators for finite difference discretizations.
There are two types of `DerivativeOperator`s: the `CenteredDifference` operator
and the `UpwindDifference` operator. The `CenteredDifference` operator utilizes
a central difference scheme while the upwind operator requires a coefficient
to be defined to perform an upwinding difference scheme.

### Operators Constructors

The constructors are as follows:

```julia
CenteredDifference{N}(derivative_order::Int,
                      approximation_order::Int, dx,
                      len::Int, coeff_func=nothing)

UpwindDifference{N}(derivative_order::Int,
                    approximation_order::Int, dx
                    len::Int, coeff_func=nothing)
```

The arguments are:

- `N`: The directional dimension of the discretization. If `N` is not given,
  it is assumed to be 1, i.e. differencing occurs along columns.
- `derivative_order`: the order of the derivative to discretize.
- `approximation_order`: the order of the discretization in terms of O(dx^order).
- `dx`: the spacing of the discretization. If `dx` is a `Number`, the operator
  is a uniform discretization. If `dx` is an array, then the operator is a
  non-uniform discretization.
- `len`: the length of the discretization in the direction of the operator.
- `coeff_func`: An operational argument for a coefficient function `f(du,u,p,t)`
  which sets the coefficients of the operator. If `coeff_func` is a `Number`
  then the coefficients are set to be constant with that number. If `coeff_func`
  is an `AbstractArray` with length matching `len`, then the coefficients are
  constant but spatially dependent.

`N`-dimensional derivative operators need to act against a value of at least
`N` dimensions.

### Example

The 3-dimensional Laplacian is created by:

```julia
N = 64
Dxx = CenteredDifference(2,2,dx,N)
Dyy = CenteredDifference{2}(2,2,dx,N)
Dzz = CenteredDifference{3}(2,2,dx,N)
L = Dxx + Dyy + Dzz

u = rand(N,N,N)
L*u
```

### Derivative Operator Actions

These operators are lazy, meaning the memory is not allocated. Similarly, the
operator actions `*` can be performed without ever building the operator
matrices. Additionally, `mul!(y,L,x)` can be performed for non-allocating
applications of the operator.

### Concretizations

The following concretizations are provided:

- `Array`
- `SparseMatrixCSC`
- `BandedMatrix`
- `BlockBandedMatrix`

Additionally, the function `sparse` is overloaded to give the most efficient
matrix type for a given operator. For one-dimensional derivatives this is a
`BandedMatrix`, while for higher dimensional operators this is a `BlockBandedMatrix`

## Boundary Value Operators

Boundary conditions are implemented through a ghost node approach. The discretized
values `u` should be the interior of the domain so that, for the boundary value
operator `Q`, `Q*u` is the discretization on the closure of the domain. By
using it like this, `L*Q*u` is the `NxN` operator which satisfies the boundary
conditions.

### Periodic Boundary Conditions

The constructor `PeriodicBC` provides the periodic boundary condition operator.

### Robin Boundary Conditions

The variables in l are `[αl, βl, γl]`, and correspond to a BC of the form
`al*u(0) + bl*u'(0) = cl`, and similarly `r` for the right boundary
`ar*u(N) + br*u'(N) = cl`.

```julia
RobinBC(l::AbstractArray{T}, r::AbstractArray{T}, dx::AbstractArray{T}, order = one(T))
```

Additionally, the following helpers exist for the Neumann `u'(0) = α` and
Dirichlet `u(0) = α` cases.

```julia
NeumannBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1)
DirichletBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1)
```

### General Boundary Conditions

Implements a generalization of the Robin boundary condition, where α is a vector
of coefficients. Represents a condition of the form
α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0

```julia
GeneralBC(αl::AbstractArray{T}, αr::AbstractArray{T}, dx::AbstractArray{T}, order = 1)
```

### Concretizations

The following concretizations are provided:

- `Array`
- `SparseMatrixCSC`

Additionally, the function `sparse` is overloaded to give the most efficient
matrix type for a given operator. For these operators it's `SparseMatrixCSC`.

## Operator Compositions

## Matrix-Free Operators

## Jacobian-Vector Product Operators
