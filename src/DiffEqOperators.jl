__precompile__()

module DiffEqOperators

import Base: *, /, \, size, getindex, setindex!, Matrix
using DiffEqBase, StaticArrays, LinearAlgebra
import LinearAlgebra: mul!, lmul!, rmul!, axpy!, opnorm, factorize
import DiffEqBase: AbstractDiffEqLinearOperator, update_coefficients!, is_constant

abstract type AbstractDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end

struct DEFAULT_UPDATE_FUNC end
(::DEFAULT_UPDATE_FUNC)(A,u,p,t) = A

### Basic Operators
include("diffeqscalar.jl")
include("array_operator.jl")

### Derivative Operators
include("derivative_operators/fornberg.jl")
include("derivative_operators/upwind_operator.jl")
include("derivative_operators/derivative_irreg_operator.jl")
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/abstract_operator_functions.jl")
include("derivative_operators/boundary_operators.jl")

### Linear Combination of Operators
#include("operator_combination.jl")

export DiffEqScalar, DiffEqArrayOperator
export AbstractDerivativeOperator, DerivativeOperator, UpwindOperator, FiniteDifference
export opnormbound
end # module
