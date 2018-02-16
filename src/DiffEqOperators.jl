__precompile__()

module DiffEqOperators

import LinearMaps: LinearMap, AbstractLinearMap
import Base: *, getindex
using DiffEqBase, StaticArrays
import DiffEqBase: update_coefficients, update_coefficients!

abstract type AbstractDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end

### Basic Operators
include("diffeqscalar.jl")
include("array_operator.jl")

### Derivative Operators
include("derivative_operators/upwind_operator.jl")
include("derivative_operators/fornberg.jl")
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/boundary_operators.jl")

export DiffEqScalar, DiffEqArrayOperator
export AbstractDerivativeOperator, DerivativeOperator, UpwindOperator, FiniteDifference
end # module
