__precompile__()

module DiffEqOperators

import LinearMaps: LinearMap, AbstractLinearMap
import Base: *, getindex
using DiffEqBase, StaticArrays

abstract type AbstractDiffEqDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end
export PDEOperator

### Basic Operators
include("diffeqscalar.jl")
include("array_operator.jl")

### Derivative Operators
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/fornberg.jl")
include("derivative_operators/boundary_operators.jl")

export DiffEqScalar, DiffEqArrayOperator
export AbstractDiffEqDerivativeOperator, DerivativeOperator
end # module
