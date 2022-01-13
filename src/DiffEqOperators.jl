module DiffEqOperators

using Base: Number
import Base: +, -, *, /, \, size, getindex, setindex!, Matrix, convert, ==
using DiffEqBase, StaticArrays, LinearAlgebra
import LinearAlgebra: mul!, ldiv!, lmul!, rmul!, axpy!, opnorm, factorize, I
import DiffEqBase: update_coefficients!, isconstant
using SciMLBase: AbstractDiffEqLinearOperator, AbstractDiffEqCompositeOperator, DiffEqScaledOperator
import SciMLBase: getops
using SparseDiffTools
using SparseArrays, ForwardDiff, BandedMatrices, NNlib, LazyArrays, BlockBandedMatrices, LoopVectorization
using LazyBandedMatrices
using RuntimeGeneratedFunctions
using Requires
RuntimeGeneratedFunctions.init(@__MODULE__)

abstract type AbstractDiffEqAffineOperator{T} end
abstract type AbstractDerivativeOperator{T} <: AbstractDiffEqLinearOperator{T} end
abstract type AbstractMatrixFreeOperator{T} <: AbstractDiffEqLinearOperator{T} end

### Matrix-free Operators
include("matrixfree_operators.jl")
include("jacvec_operators.jl")

### Utilities
include("utils.jl")

### Exceptions
include("exceptions.jl")

### Boundary Padded Arrays
include("boundary_padded_arrays.jl")

### Boundary Operators
include("derivative_operators/bc_operators.jl")
include("derivative_operators/multi_dim_bc_operators.jl")

### Derivative Operators
include("derivative_operators/fornberg.jl")
include("derivative_operators/derivative_operator.jl")
include("derivative_operators/abstract_operator_functions.jl")
include("derivative_operators/convolutions.jl")
include("derivative_operators/ghost_derivative_operator.jl")
include("derivative_operators/derivative_operator_functions.jl")
include("derivative_operators/coefficient_functions.jl")

### Vector Calculus Operators
include("derivative_operators/vector_calculus_operators.jl")
include("derivative_operators/vector_calculus_convolutions.jl")
include("derivative_operators/vector_algebraic_operations.jl")

### Composite Operators
include("composite_operators.jl")
include("docstrings.jl")

### Concretizations
include("derivative_operators/concretization.jl")

# The (u,p,t) and (du,u,p,t) interface
for T in [DiffEqScaledOperator, DiffEqOperatorCombination, DiffEqOperatorComposition, GhostDerivativeOperator]
  (L::T)(u,p,t) = (update_coefficients!(L,u,p,t); L * u)
  (L::T)(du,u,p,t) = (update_coefficients!(L,u,p,t); mul!(du,L,u))
end

function __init__()
  @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
    export VecJacOperator

    include("vecjac_operators.jl")
  end
end

export MatrixFreeOperator
export AnalyticalJacVecOperator, JacVecOperator, getops
export AbstractDerivativeOperator, DerivativeOperator,
       CenteredDifference, UpwindDifference, CompleteCenteredDifference, CompleteUpwindDifference, nonlinear_diffusion, nonlinear_diffusion!,
       GradientOperator, Gradient, CurlOperator, Curl, DivergenceOperator, Divergence
export DirichletBC, Dirichlet0BC, NeumannBC, Neumann0BC, RobinBC, GeneralBC, MultiDimBC, PeriodicBC,
       MultiDimDirectionalBC, ComposedMultiDimBC
export compose, decompose, perpsize, square_norm, square_norm!, dot_product, dot_product!, cross_product,
       cross_product!

export GhostDerivativeOperator
export BoundaryConditionError
end # module
