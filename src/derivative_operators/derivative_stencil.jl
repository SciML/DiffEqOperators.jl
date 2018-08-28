struct UniformDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: T
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: S
end

struct IrregularDerivativeStencil{T,S<:SVector} <: AbstractDiffEqLinearOperator{T}
    derivative_order    :: Int
    approximation_order :: Int
    dx                  :: Vector{T}
    dimension           :: Int
    stencil_length      :: Int
    stencil_coefs       :: Vector{S}
end

DerivativeStencil = Union{UniformDerivativeStencil, IrregularDerivativeStencil}
