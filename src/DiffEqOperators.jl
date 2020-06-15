module DiffEqOperators

import Base: +, -, *, /, \, size, getindex, setindex!, Matrix, convert, ==
using DiffEqBase, StaticArrays, LinearAlgebra
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, axpy!, opnorm, factorize, I
import DiffEqBase: AbstractDiffEqLinearOperator, update_coefficients!, isconstant
using SparseArrays, ForwardDiff, BandedMatrices, NNlib, LazyArrays, BlockBandedMatrices
using LazyBandedMatrices, ModelingToolkit

abstract type AbstractDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractDiffEqCompositeOperator{T} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractMatrixFreeOperator{T} <: AbstractDiffEqLinearOperator{T} end

### Matrix-free Operators
include("matrixfree_operators.jl")
include("jacvec_operators.jl")

### Utilities
include("utils.jl")

### Boundary Padded Arrays
include("boundary_padded_arrays.jl")

### Boundary Operators
include("derivative_operators/BC_operators.jl")
include("derivative_operators/multi_dim_bc_operators.jl")

### Derivative Operators
include("derivative_operators/fornberg.jl")
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/abstract_operator_functions.jl")
include("derivative_operators/convolutions.jl")
include("derivative_operators/concretization.jl")
include("derivative_operators/ghost_derivative_operator.jl")
include("derivative_operators/derivative_operator_functions.jl")
include("derivative_operators/coefficient_functions.jl")

### Composite Operators
include("composite_operators.jl")

include("MOL_discretization.jl")

include("docstrings.jl")

# The (u,p,t) and (du,u,p,t) interface
for T in [DiffEqScaledOperator, DiffEqOperatorCombination, DiffEqOperatorComposition]
  (L::T)(u,p,t) = (update_coefficients!(L,u,p,t); L * u)
  (L::T)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du,L,u))
end

export MatrixFreeOperator
export AnalyticalJacVecOperator, JacVecOperator, getops
export AbstractDerivativeOperator, DerivativeOperator,
       CenteredDifference, UpwindDifference
export DirichletBC, Dirichlet0BC, NeumannBC, Neumann0BC, RobinBC, GeneralBC, MultiDimBC, PeriodicBC,
       MultiDimDirectionalBC, ComposedMultiDimBC,
       compose, decompose, perpsize
export GhostDerivativeOperator
export MOLFiniteDifference
end # module
