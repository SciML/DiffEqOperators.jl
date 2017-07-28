__precompile__()

module PDEOperators

import LinearMaps: LinearMap
import Base: *, getindex
using DiffEqBase, StaticArrays

abstract type AbstractLinearOperator{T} <: AbstractDiffEqOperator{T} end
export PDEOperator

include("linear_operator.jl")
include("fornberg.jl")
include("upwind_operator.jl")
include("boundary_operators.jl")

export AbstractLinearOperator, LinearOperator, UpwindOperator
end # module
