__precompile__()

module DiffEqOperators

import LinearMaps: LinearMap
import Base: *, getindex
import DiffEqBase: AbstractDiffEqOperator, AbstractDiffEqLinearOperator, AbstractDiffEqDerivativeOperator
using StaticArrays

# abstract type AbstractDiffEqLinearOperator{T} <: AbstractDiffEqOperator{T} end
# abstract type AbstractDiffEqDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end

export DiffEqOperator

include("linear_operator.jl")
include("fornberg.jl")
include("upwind_operator.jl")
include("boundary_operators.jl")

export DiffEqLinearOperator, DiffEqUpwindOperator
end # module
