__precompile__()

module DiffEqOperators

import Base: +, -, *, /, \, size, getindex, setindex!, Matrix, convert
using DiffEqBase, StaticArrays, LinearAlgebra
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, axpy!, opnorm, factorize
import DiffEqBase: AbstractDiffEqLinearOperator, update_coefficients!, is_constant

abstract type AbstractDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractDiffEqCompositeOperator{T} <: AbstractDiffEqLinearOperator{T} end

DEFAULT_UPDATE_FUNC(A,u,p,t) = A

### Basic Operators
include("basic_operators.jl")

### Derivative Operators
include("derivative_operators/fornberg.jl")
include("derivative_operators/upwind_operator.jl")
include("derivative_operators/derivative_irreg_operator.jl")
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/abstract_operator_functions.jl")
include("derivative_operators/boundary_operators.jl")

### Composite Operators
include("composite_operators.jl")

export DiffEqScalar, DiffEqArrayOperator
export AbstractDerivativeOperator, DerivativeOperator, UpwindOperator, FiniteDifference
end # module
