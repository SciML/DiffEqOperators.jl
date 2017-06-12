__precompile__()

module PDEOperators

import LinearMaps: LinearMap, AbstractLinearMap
import Base: *, getindex
using DiffEqBase, StaticArrays

abstract type AbstractLinearOperator{T} <: AbstractDiffEqOperator{T} end

export PDEOperator
include("fornberg.jl")

export AbstractLinearOperator, LinearOperator
end # module
