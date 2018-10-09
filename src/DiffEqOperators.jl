module DiffEqOperators

import Base: +, -, *, /, \, size, getindex, setindex!, Matrix, convert
using DiffEqBase, StaticArrays, LinearAlgebra
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, axpy!, opnorm, factorize, I
import DiffEqBase: AbstractDiffEqLinearOperator, update_coefficients!, is_constant
using SparseArrays

abstract type AbstractDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractDiffEqCompositeOperator{T} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractMatrixFreeOperator{T} <: AbstractDiffEqLinearOperator{T} end

### Common default methods for the operators
include("common_defaults.jl")

### Basic Operators
include("basic_operators.jl")

### Matrix-free Operators
include("matrixfree_operators.jl")

### Derivative Operators
include("derivative_operators/fornberg.jl")
include("derivative_operators/upwind_operator.jl")
include("derivative_operators/derivative_irreg_operator.jl")
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/abstract_operator_functions.jl")
include("derivative_operators/boundary_operators.jl")

### Composite Operators
include("composite_operators.jl")

# The (u,p,t) and (du,u,p,t) interface
for T in [DiffEqScalar, DiffEqArrayOperator, FactorizedDiffEqArrayOperator, DiffEqIdentity,
  DiffEqScaledOperator, DiffEqOperatorCombination, DiffEqOperatorComposition]
  (L::T)(u,p,t) = (update_coefficients!(L,u,p,t); L * u)
  (L::T)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du,L,u))
end

export MatrixFreeOperator
export DiffEqScalar, DiffEqArrayOperator, DiffEqIdentity, getops
export AbstractDerivativeOperator, DerivativeOperator, UpwindOperator, FiniteDifference
end # module
