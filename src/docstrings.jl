@doc """
    CenteredDifference{N}(n, order, step, len, [coeff_func])
    CenteredDifference{N}(n, order, steps, len, [coeff_func])

See also: [`UpwindDifference`](@ref)
"""
CenteredDifference


@doc """
    calculate_weights(n::Int, x₀::Real, x::Vector)

Return a vector `c` such that `c⋅f.(x)` approximates ``f^{(n)}(x₀)`` for smooth `f`.

The points `x` need not be evenly spaced.

The stencil `c` has the highest approximation order possible given values of `f` at `length(x)` points.  More precisely, if `x` has length `m`, there is a function `g` such that ``g(y) = f(y) + O(y-x₀)^{m-n+?}`` and ``c⋅f.(x) = g'(x₀)``.

The algorithm is due to [Fornberg](https://doi.org/10.1090/S0025-5718-1988-0935077-0), with a [modification](http://epubs.siam.org/doi/pdf/10.1137/S0036144596322507) to improve stability.
"""
calculate_weights


@doc """
    DerivativeOperator{T<:Real,N,Wind,T2,S1,S2<:SVector,T3,F}(<fields>)

Represent a finite difference derivative operator.

These operators implement the `DiffEqBase.AbstractDiffEqLinearOperator` interface.  Therefore `eltype` returns `T`.

These operators can be contracted over an arbitrary dimension, given by the type parameter `N`.

The `CentredDifference` and `UpwindDifference` types serve as hooks to hang constructors, but no objects are actually constructed with those types.  Their constructors return structures of type `DerivativeOperator`, with stencils constructed by `calculate_weights`, and other fields filled out appropriately.

The finite-difference methods are defined for `DerivativeOperator`.  In particular, `*`, which actually takes derivatives.

The key data are three stencils of coefficients, stored in `SVector`s.  The interior stencil, stored in `stencil_coefs::S1`, is the normal one used in the interior of the grid.  The others, `low_boundary_coefs::S2` and `high_boundary_coefs::S2`, are used where the normal stencil would jut out of the grid boundary.  These can have a different length than the interior stencil, hence the two types.

When the operator is applied by `mul!`, these stencils are multiplied by the vector in the `coefficients` field.  It is set to `coeff_func.(1:len)` in `UpwindDifference`, which seems odd, because `len` is not the stencil length.  Scalar multiplication is absorbed into this vector, as required by the `DiffEqBase.AbstractDiffEqLinearOperator` interface.

The `coefficients` field appears to be a more general `DifferentialEquations` thing.  There is an `update_coefficients!` method.

The term “left boundary” is used interchangeably with “low boundary”, and “right” with “high”.

# Type Parameters

Presumably, most of these are to force method specialization on:a) the stencil lengths, which are given by the parameters of the `SVector` types `S1` and `S2`, and b) even or arbitrary grid spacing.

- `T<:Real`: Function range type.
- `N`: Contraction dimension
- `Wind`: Flag `true` for upwind operators, `false` for centered.
- `T2`: Concrete type of `dx` field, which might be `T` or `AbstractVector{T}`.
- `S1`: Concrete type of the interior stencil.  (When is this not an `SVector`?)
- `S2<:SVector`: Concrete type of boundary coefficients.
- `T3`: `Vector{T}` if a `coeff_func` was supplied, `Nothing` otherwise.
- `F`: `typeof(coeff_func)` if a `coeff_func` was supplied, `Nothing` otherwise.

# Fields

- `derivative_order::Int`: The `n` in `f⁽ⁿ⁾(x)`.
- `approximation_order::Int`: The degree of polynomial for which the operator is numerically exact.
- `dx::T2`: Grid step.
- `len::Int`: The number of interior points on the grid.
- `stencil_length::Int`: Length of the interior stencil
- `stencil_coefs::S1`: An `SVector` of the interior stencil coefficients.
- `boundary_stencil_length::Int`:
- `boundary_point_count::Int`:
- `low_boundary_coefs::S2`: An array of `SVector` stencils for the interior points where `stencil_coefs` juts over the left boundary.
- `high_boundary_coefs::S2`: An array of `SVector` stencils for the interior points where `stencil_coefs` juts over the right boundary.
- `coefficients::T3`: Multiplier for the stencil.
- `coeff_func::F`:  God only knows what this function is for.

See also: [`AbstractBC`](@ref), [`CenteredDifference`](@ref), [`UpwindDifference`](@ref)
"""
DerivativeOperator


@doc """
    abstract type AbstractBC{T}

`T` is the range type of the discretised function.

Boundary condition operators extrapolate the discretised function
as illustrated in the README, adding a ghost node at each end such
that an interpolated polynomial satisfies the boundary condition.

Some of these are affine operators with a constant term, not linear
operators, so using `*` and `\\` is a minor abuse of notation.  For example,
when there is a non-zero Dirichlet boundary value, the value at the
ghost node is constant, and does not scale proportional to the
interior function values.  Because boundary condition operators are
not linear, they can not be concretised as matrices.

`GhostDerivativeOperator` somehow concretises as a pair of matrices.

The implementation of `*(::AffineBC, ::Vector)` returns a
`BoundaryPaddedVector`.  This is a structure that stores the ghost
values separately from the original vector.  Presumably this is to
avoid allocations, although the `mul!` method for `GhostDerivativeOperator`
and `Matrix` method allocates a new `BoundaryPaddedArray` for each
column.

In higher dimensional cases, it might return a
`BoundaryPaddedArray`.

The `*` operator for a derivative multiplying a boundary condition
packages them up as a `GhostDerivativeOperator`.  This can not
return a sparse matrix, because the operator is affine not linear.
When a vector is multiplied by one of those, it uses `*(::AbstractBC,
::AbstractVector)` to pad with ghost nodes, then
`mul!(::DerivativeOperator, ::AbstractVector)` to evaluate the
derivative.

See also: [`DerivativeOperator`](@ref), [`PeriodicBC`](@ref) [`NeumannBC`](@ref), [`DirichletBC`](@ref), [`RobinBC`](@ref)
"""
AbstractBC


@doc """
    PeriodicBC{T}()

Only one ghost node is added at each end.  This means that periodic
functions will not be consistent at the interior and near the
boundary with high order stencils.  It is likely to cause other
limitations too.

Currently, periodic boundary conditions are only implemented in one
dimension.

# Type parameters

- `T` is the domain type of the discretised function.

See also: [`AbstractBC`](@ref)
"""
PeriodicBC


@doc """
NeumannBC(α::NTuple{2,T}, dx::Union{AbstractVector{T}, T}, order = 1) where T = RobinBC((zero(T), one(T), α[1]), (zero(T), one(T), α[2]), dx, order)
DirichletBC(αl::T, αr::T) where T = RobinBC((one(T), zero(T), αl), (one(T), zero(T), αr), 1.0, 2.0 )
#specialized constructors for Neumann0 and Dirichlet0
Dirichlet0BC(T::Type) = DirichletBC(zero(T), zero(T))
Neumann0BC(dx::Union{AbstractVector{T}, T}, order = 1) where T = NeumannBC((zero(T), zero(T)), dx, order)


See also: [`AbstractBC`](@ref)
"""
RobinBC
@doc (@doc RobinBC) NeumannBC
@doc (@doc RobinBC) Neumann0BC
@doc (@doc RobinBC) DirichletBC
@doc (@doc RobinBC) Dirichlet0BC
