# DiffEqOperators.jl

[![Build Status](https://travis-ci.org/JuliaDiffEq/DiffEqOperators.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/DiffEqOperators.jl)
[![Build status](https://ci.appveyor.com/api/projects/status/au9knv63u9oh1aie?svg=true)](https://ci.appveyor.com/project/ChrisRackauckas/diffeqoperators-jl)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/DiffEqOperators.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/DiffEqOperators.jl?branch=master)
[![codecov.io](http://codecov.io/github/shivin9/DiffEqOperators.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/DiffEqOperators.jl?branch=master)

DiffEqOperators.jl provides a set of pre-defined operators for use with
DifferentialEquations.jl. These operators make it easy to discretize and solve
common partial differential equations.

## Automated Finite Difference Method (FDM) Operators

This library provides lazy operators for arbitrary order uniform and non-uniform
finite difference discretizations of arbitrary high derivative order and for
arbitrarily high dimensions.

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
`BandedMatrix`, while for higher dimensional operators this is a `BlockBandedMatrix`.
The concretizations are made to act on `vec(u)`.

## Boundary Condition Operators

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
Dirichlet0BC(T::Type)
DirichletBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1)
Neumann0BC(dx::Union{AbstractVector{T}, T}, order = 1)
NeumannBC(α::AbstractVector{T}, dx::AbstractVector{T}, order = 1)
```

### General Boundary Conditions

Implements a generalization of the Robin boundary condition, where α is a vector
of coefficients. Represents a condition of the form
α[1] + α[2]u[0] + α[3]u'[0] + α[4]u''[0]+... = 0

```julia
GeneralBC(αl::AbstractArray{T}, αr::AbstractArray{T}, dx::AbstractArray{T}, order = 1)
```

### Multidimensional Boundary Conditions

```julia
Q_dim = MultiDimBC(Q, size(u), dim)
```

turns `Q` into a boundary condition along the dimension `dim`. Additionally,
to apply the same boundary values to all dimensions, one can use

```julia
Qx,Qy,Qz = MultiDimBC(YourBC, size(u)) # Here u is 3d
```

Multidimensional BCs can then be composed into a single operator with:

```julia
Q = compose(BCs...)
```

### Operator Actions

The boundary condition operators act lazily by appending the appropriate values
to the end of the array, building the ghost-point extended version for the
derivative operator to act on. This utilizes special array types to not require
copying the interior data.

### Concretizations

The following concretizations are provided:

- `Array`
- `SparseMatrixCSC`

Additionally, the function `sparse` is overloaded to give the most efficient
matrix type for a given operator. For these operators it's `SparseMatrixCSC`.
The concretizations are made to act on `vec(u)`.

## GhostDerivative Operators

When `L` is a `DerivativeOperator` and `Q` is a boundary condition operator,
`L*Q` produces a `GhostDerivative` operator which is the composition of the
two operations.

### Concretizations

The following concretizations are provided:

- `Array`
- `SparseMatrixCSC`
- `BandedMatrix`

Additionally, the function `sparse` is overloaded to give the most efficient
matrix type for a given operator. For these operators it's `BandedMatrix` unless
the boundary conditions are `PeriodicBC`, in which case it's `SparseMatrixCSC`.
The concretizations are made to act on `vec(u)`.

## Matrix-Free Operators

```julia
MatrixFreeOperator(f::F, args::N;
                   size=nothing, opnorm=true, ishermitian=false) where {F,N}
```

A `MatrixFreeOperator` is a linear operator `A*u` where the action of `A` is
explicitly defined by an in-place function `f(du, u, p, t)`.

## Jacobian-Vector Product Operators

```julia
JacVecOperator{T}(f,u::AbstractArray,p=nothing,t::Union{Nothing,Number}=nothing;autodiff=true,ishermitian=false,opnorm=true)
```

The `JacVecOperator` is a linear operator `J*v` where `J` acts like `df/du`
for some function `f(u,p,t)`. For in-place operations `mul!(w,J,v)`, `f`
is an in-place function `f(du,u,p,t)`.

## Operator Compositions

Multiplying two DiffEqOperators will build a `DiffEqOperatorComposition`, while
adding two DiffEqOperators will build a `DiffEqOperatorCombination`. Multiplying
a DiffEqOperator by a scalar will produce a `DiffEqScaledOperator`. All
will inherit the appropriate action.
