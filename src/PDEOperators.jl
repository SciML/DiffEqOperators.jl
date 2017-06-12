__precompile__()

module PDEOperators

import LinearMaps: LinearMap, AbstractLinearMap
import Base: *
using DiffEqBase, StaticArrays

abstract AbstractLinearOperator{T} <: AbstractDiffEqOperator{T}

export PDEOperator
include("fornberg.jl")

export AbstractLinearOperator, LinearOperator
end # module
